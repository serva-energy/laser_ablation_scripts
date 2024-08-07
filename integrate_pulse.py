#
# Copyright 2024 Fusion Energy Solutions, Inc. dba. Serva Energy
#

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy as np
import pandas as pd

import os
import sys

from scipy.signal import find_peaks

import argparse

class LaserAblationData():

    baseline_shrink_factor = 0.1
    pulse_shrink_factor = 0.3
    pulse_shrink_seconds = 4.0
    minimum_pulse_length_seconds = 0.0

    def __init__(self, filename=None) -> None:
        self.name = None
        self.file_timestamp = None

        self.metadata = {}
        self.timestamps = None
        self.dt = None
        self.isotope_pulse_raw_data = {}
        self.isotope_heights = {}

        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        # read and parse the metadata
        metadata = ""
        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.strip()
                # measurement name and string file timestamp
                if i == 0:
                    self.name, _, self.file_timestamp = line[:-1].partition(":")
                    continue

                # end of metadata
                if line.startswith("Time"):
                    break

                if line:
                    metadata += line

        metadata = metadata.split(";")
        for item in metadata:
            k, _, v = item.partition("=")
            if k:
                self.metadata[k] = v

        # read the data
        df = pd.read_csv(filename, delimiter=',', skiprows=13)

        # sanitize
        df = df.drop(0)
        df = df.dropna(axis=1)

        # parse
        self.timestamps = df['Time'].to_numpy()
        for col in df.columns[1:]:
            self.isotope_pulse_raw_data[col] = df[col].to_numpy(dtype='f8')

        self.dt = np.median(np.diff(self.timestamps))

    def plot(self, plot_all=False, maximized=True, show=True, export_plots=True):
        try:
            baseline_boundaries_indices, pulse_boundaries_indices = self.find_pulse_boundaries()
        except ValueError:
            baseline_boundaries_indices = pulse_boundaries_indices = np.array([], dtype='i8')
        baseline_boundaries = tuple(self.timestamps[baseline_boundaries_indices])
        pulse_boundaries = tuple(self.timestamps[pulse_boundaries_indices])

        isotopes = self.isotope_pulse_raw_data.keys() if plot_all else ['29Si']

        for iso in isotopes:
            x_data = self.timestamps
            y_data = self.isotope_pulse_raw_data[iso]

            pw = pg.plot(x=x_data, y=y_data, symbol='o', pen='b')
            pw.setBackground('w')
            if baseline_boundaries and pulse_boundaries:
                pw.addItem(pg.LinearRegionItem(values=baseline_boundaries, brush='#00ff0040', movable=False))
                pw.addItem(pg.LinearRegionItem(values=pulse_boundaries, brush='#0000ff40', movable=False))
            else:
                pw.setBackground('y')

            # plot seconds derivative
            y_diff = np.diff(y_data)
            pw.plot(x=x_data[:len(y_diff)], y=y_diff, pen='r')

            # plot threshold            
            pw.addItem(pg.InfiniteLine(angle=0, pos=np.mean(y_data), movable=False))

            pw.setWindowTitle(f"{self.name} - {iso}")
            pw.setLabel('bottom', 'Time', 's')
            pw.setLabel('left', 'Counts')
            if show:
                if maximized:
                    pw.showMaximized()
                else:
                    pw.show()

            if export_plots:
                pw.resize(1920,1080)    
                pw.show()
                pw.hide()
                fname = f"{self.name}-{iso}.svg"
                print(f"Saving \"{fname}\".")                
                if fname.endswith('.svg'):
                    exporter = pg.exporters.SVGExporter(pw.getPlotItem())                
                elif fname.endswith('.png'):
                    exporter = pg.exporters.ImageExporter(pw.getPlotItem())
                else:
                    raise RuntimeError(f"Unsupported format: {fname}")
                exporter.export(fname)



    def find_pulse_boundaries(self):
        baseline_boundaries_indices = []
        pulse_boundaries_indices = []
        for isotope in ['29Si']:
            y_data = self.isotope_pulse_raw_data[isotope]

            # METHOD 1
            # find the pulse boundaries using 1st derivative
            y_diff = np.diff(y_data)
            peak_threshold = np.mean(np.abs(y_diff))

            # find first "tall" peak from left
            peaks, _ = find_peaks(y_diff, prominence=peak_threshold)
            pulse_start = peaks[0]

            # find first "tall" negative peak from right
            y_diff = -y_diff
            peaks, _ = find_peaks(y_diff, prominence=peak_threshold)
            pulse_end = peaks[-1]
            pulse_boundaries_indices.append(np.array([pulse_start, pulse_end]))

            # METHOD 2
            # find pulse boundaries using mean value as threshold
            threshold = np.mean(y_data)
            pulse_indices = np.where(y_data>threshold)[0]
            pulse_start = pulse_indices.min()
            pulse_end = pulse_indices.max()            
            pulse_boundaries_indices.append(np.array([pulse_start, pulse_end]))
            
            # check if there is more than one pulse - both methods have to agree
            peaks, _ = find_peaks(y_data, prominence=0.5*y_data.max(), distance=len(y_data)//3)
            if (np.diff(pulse_indices).max() > 1) and (len(peaks) != 1):
                raise ValueError(f"{self.name} {isotope} doesn't contain exactly one pulse.")

            # check if methods disagree
            std = np.std(pulse_boundaries_indices, axis=0,)
            if np.sum(std) > 10:
                # methods disagree
                print(f"Warning: {self.name} {isotope} methods disagree {pulse_boundaries_indices} stddev={np.std(pulse_boundaries_indices, axis=0,)}. ", end="", file=sys.stderr)
                diffs = [arr[1]-arr[0] for arr in pulse_boundaries_indices]
                
                # pick result with narrower pulse                
                pulse_boundaries_indices = [pulse_boundaries_indices[np.argmin(diffs)]]
                print(f"Picking shorter estimate ({pulse_boundaries_indices})", file=sys.stderr)
            else:                
                # methods agree, use mean of results
                pulse_boundaries_indices = [np.mean(pulse_boundaries_indices, axis=0, dtype='i4')]

            # baseline is from start to recording to start of pulse
            baseline_boundaries_indices.append(np.array([0, pulse_boundaries_indices[0][0]])) 

        baseline_boundaries_indices = np.array(baseline_boundaries_indices)
        pulse_boundaries_indices = np.array(pulse_boundaries_indices)

        # shrink by `pulse_shrink_seconds` seconds at the start and at the end        
        sec_to_samples = np.ceil(self.pulse_shrink_seconds/self.dt)
        pulse_boundaries_indices[0][0] += sec_to_samples
        pulse_boundaries_indices[0][1] -= sec_to_samples

        baseline_boundaries_indices = np.mean(baseline_boundaries_indices, axis=0)
        pulse_boundaries_indices = np.mean(pulse_boundaries_indices, axis=0)

        baseline_boundaries_indices = self.shrink_range(baseline_boundaries_indices, self.baseline_shrink_factor)
        pulse_boundaries_indices = self.shrink_range(pulse_boundaries_indices, self.pulse_shrink_factor)

        # check and enforce minimum pulse length
        if self.minimum_pulse_length_seconds:
            pulse_len_s = (pulse_boundaries_indices[1] - pulse_boundaries_indices[0]) * self.dt
            missing = (self.minimum_pulse_length_seconds - pulse_len_s) / self.dt
            # if pulse is too short expand it
            if missing > 0.0:
                print(f"Warning: {self.name} {isotope} pulse length ({pulse_len_s:.3f}s) is too short, expanding to {self.minimum_pulse_length_seconds:.3f}s")
                pulse_boundaries_indices[0] -= (missing//2)
                pulse_boundaries_indices[1] += (missing//2)
        
        # check for region intersection
        if self.check_intersection(baseline_boundaries_indices, pulse_boundaries_indices):
            raise ValueError("Baseline and pulse regions intersect!")

        return baseline_boundaries_indices, pulse_boundaries_indices

    @staticmethod
    def shrink_range(boundary_indices, ratio):
        # Ensure the input is a 2-element array
        assert len(boundary_indices) == 2, "Input must be a 2-element array"

        start, end = boundary_indices
        length = end - start
        shrink_amount = length * ratio

        new_start = start + shrink_amount / 2
        new_end = end - shrink_amount / 2

        return np.array([int(new_start), int(new_end)])

    @staticmethod
    def check_intersection(region1, region2):
        # region1 and region2 should be numpy arrays with two elements [start, end]
        return np.minimum(region1[1], region2[1]) >= np.maximum(region1[0], region2[0])


    def calculate_heights(self):
        def to_range(endpoints):
            return np.arange(endpoints[0], endpoints[1] + 1)
        
        baseline_boundaries_indices, pulse_boundaries_indices = self.find_pulse_boundaries()

        baseline_indices = to_range(baseline_boundaries_indices)
        pulse_indices = to_range(pulse_boundaries_indices)

        for isotope, y_data in self.isotope_pulse_raw_data.items():
            val = np.median(y_data[pulse_indices])
            base = np.median(y_data[baseline_indices])
            if base>val:
                print(f"Warning:{self.name} {isotope} pulse ({val}) is less than baseline ({base}), setting to 0", file=sys.stderr)
            val -= base
            self.isotope_heights[isotope] = np.clip(val, 0, np.inf)


def main():
    parser = argparse.ArgumentParser(description='Process a file or directory of files.')
    parser.add_argument('input', type=str, help='File name or path to a directory of files')
    parser.add_argument('-o', '--output', type=str, default='results.xlsx',
                        help='Output file name (default: results.xlsx)')

    parser.add_argument('--baseline_shrink_factor', type=float, default=0.1,
                        help='Baseline shrink factor (default: 0.1)')
    parser.add_argument('--pulse_shrink_factor', type=float, default=0.0, help='Pulse shrink factor (default: 0.0)')
    parser.add_argument('--pulse_shrink_seconds', type=float, default=4.0, help='Pulse shrink (on either end) in seconds (default: 4.0)')
    parser.add_argument('--minimum_pulse_length', type=float, default=0.0, help="Minimum pulse length in seconds. Expands pulses shorter than this value.")
    parser.add_argument('--plot', action='store_true', help='Visualize the output')
    parser.add_argument('--export-plots', action='store_true', help='Save the plots to files')

    args = parser.parse_args()

    if args.export_plots:
        args.plot = True

    results_df = None

    if args.plot:
        global pg
        pg = None
        try:
            import pyqtgraph as pg
            import pyqtgraph.exporters
            app = pg.mkQApp()
    
        except ImportError:
            print("Please install `pyqtgraph` and `pyside6` modules to use the plot function:")
            print("pip install pyqtgraph pyside6")
            args.plot = False

    # Check if input is a file 
    if os.path.isfile(args.input):
        results_df, abl = process_file(args.input, **vars(args))
        if args.plot:
            abl.plot(plot_all=True, maximized=False, show=not args.export_plots, export_plots=args.export_plots)

    # or directory
    elif os.path.isdir(args.input):
        for filename in sorted(os.listdir(args.input)):
            file_path = os.path.join(args.input, filename)
            if os.path.isfile(file_path):
                results_df, abl = process_file(file_path, results_df=results_df, **vars(args))
                if args.plot:
                    abl.plot(show=not args.export_plots, export_plots=args.export_plots)
    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        sys.exit(1)

    # start app to show plots
    if args.plot and not args.export_plots:
        app.exec()

    if results_df is None:
        print("No results were produced!")
        sys.exit(1)
    
    # Save results_df to output file based on the extension
    print(f"Saving results to `{args.output}`")

    _, file_extension = os.path.splitext(args.output)

    match file_extension:
        case '.xls' | '.xlsx':            
            if os.path.exists(args.output):
                kwargs = { 'mode': 'a', 'if_sheet_exists' : 'replace' }
            else:
                kwargs = { 'mode': 'w'}
            with pd.ExcelWriter(args.output, engine='openpyxl', **kwargs) as writer:
                results_df.to_excel(writer, sheet_name="Results_bg_corrected", index=False, startrow=2, float_format="%.0f")
        case '.csv':
            results_df.to_csv(args.output, index=False, float_format="%.0f")
        case _:
            raise ValueError(f"Unsupported file extension: {file_extension}")


def process_file(file_path, baseline_shrink_factor, pulse_shrink_factor, pulse_shrink_seconds, plot, results_df=None, minimum_pulse_length=0.0, **kwargs):
    print(f"Processing `{file_path}`", end="")

    a = LaserAblationData(file_path)
    print(f" ({a.name})")
    a.baseline_shrink_factor = baseline_shrink_factor
    a.pulse_shrink_factor = pulse_shrink_factor
    a.pulse_shrink_seconds = pulse_shrink_seconds
    a.minimum_pulse_length_seconds = minimum_pulse_length
    try:
        a.calculate_heights()
        if results_df is None:
            results_df = pd.DataFrame(columns=[''] + list(a.isotope_heights.keys()))

        # store results
        results_df.loc[len(results_df)] = [a.name] + list(a.isotope_heights.values())

    except ValueError as e:        
        results_df.loc[len(results_df)] = [a.name] + [""] * len(a.isotope_pulse_raw_data)
        print(f"{e}")


    return results_df, a


if __name__ == "__main__":
    main()
