import numpy as np
import pandas as pd

import os
import sys

from scipy.signal import find_peaks

import argparse


class LaserAblationData():

    baseline_shrink_factor = 0.1
    pulse_shrink_factor = 0.3

    def __init__(self, filename=None) -> None:
        self.name = None
        self.file_timestamp = None

        self.metadata = {}
        self.timestamps = None
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

    def plot(self, plot_all=False, maximized=True):
        baseline_boundaries_indices, pulse_boundaries_indices = self.find_pulse_boundaries()
        baseline_boundaries = tuple(self.timestamps[baseline_boundaries_indices])
        pulse_boundaries = tuple(self.timestamps[pulse_boundaries_indices])

        isotopes = self.isotope_pulse_raw_data.keys() if plot_all else ['29Si']
        for iso in isotopes:
            x_data = self.timestamps
            y_data = self.isotope_pulse_raw_data[iso]

            pw = pg.plot(x=x_data, y=y_data, symbol='o', pen='b')
            pw.addItem(pg.LinearRegionItem(values=baseline_boundaries, brush='#00ff0040', movable=False))
            pw.addItem(pg.LinearRegionItem(values=pulse_boundaries, brush='#0000ff40', movable=False))

            # plot seconds derivative
            y_diff = np.diff(y_data)
            pw.plot(x=x_data[:len(y_diff)], y=y_diff, pen='r')

            pw.setWindowTitle(f"{self.name} - {iso}")
            if maximized:
                pw.showMaximized()

    def find_pulse_boundaries(self):
        baseline_boundaries_indices = []
        pulse_boundaries_indices = []
        for isotope in ['29Si']:
            y_data = self.isotope_pulse_raw_data[isotope]

            # find pulse
            # peaks, props = find_peaks(y_data, prominence=0.5*y_data.max())            
            # if len(peaks) == 1:
            #     baseline_boundaries_indices.append(np.array([0, props['left_bases'][0]]))
            #     pulse_boundaries_indices.append(np.array([props['left_bases'][0], props['right_bases'][0]]))
            # else:
            # TODO throw exception

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
            
            # baseline is from start to recording to start of pulse
            baseline_boundaries_indices.append(np.array([0, pulse_boundaries_indices[0][0]])) 

        baseline_boundaries_indices = np.array(baseline_boundaries_indices)
        pulse_boundaries_indices = np.array(pulse_boundaries_indices)

        # shrink by 4 seconds at the start and at the end
        pulse_boundaries_indices[0][0] += 12
        pulse_boundaries_indices[0][1] -= 12

        baseline_boundaries_indices = np.mean(baseline_boundaries_indices, axis=0)
        pulse_boundaries_indices = np.mean(pulse_boundaries_indices, axis=0)

        baseline_boundaries_indices = self.shrink_range(baseline_boundaries_indices, self.baseline_shrink_factor)
        pulse_boundaries_indices = self.shrink_range(pulse_boundaries_indices, self.pulse_shrink_factor)

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

    def calculate_heights(self):
        baseline_boundaries_indices, pulse_boundaries_indices = self.find_pulse_boundaries()

        for isotope, y_data in self.isotope_pulse_raw_data.items():
            val = np.mean(y_data[pulse_boundaries_indices])
            base = np.mean(y_data[baseline_boundaries_indices])
            val -= base
            self.isotope_heights[isotope] = val


def main():
    parser = argparse.ArgumentParser(description='Process a file or directory of files.')
    parser.add_argument('input', type=str, help='File name or path to a directory of files')
    parser.add_argument('-o', '--output', type=str, default='results.xlsx',
                        help='Output file name (default: results.xlsx)')

    parser.add_argument('--baseline_shrink_factor', type=float, default=0.1,
                        help='Baseline shrink factor (default: 0.1)')
    parser.add_argument('--pulse_shrink_factor', type=float, default=0.0, help='Pulse shrink factor (default: 0.0)')
    parser.add_argument('--plot', action='store_true', help='Visualize the output')

    args = parser.parse_args()

    results_df = None

    if args.plot or args.plotall:
        global pg
        pg = None
        try:
            import pyqtgraph as pg
            app = pg.mkQApp()
    
        except ImportError:
            print("Please install `pyqtgraph` and `pyside6` modules to use the plot function:")
            print("pip install pyqtgraph pyside6")
            args.plot = False

    # Check if input is a file 
    if os.path.isfile(args.input):
        results_df, abl = process_file(args.input, args.baseline_shrink_factor, args.pulse_shrink_factor, args.plot)
        if args.plot:
            abl.plot(plot_all=True, maximized=False)

    # or directory
    elif os.path.isdir(args.input):
        for filename in sorted(os.listdir(args.input)):
            file_path = os.path.join(args.input, filename)
            if os.path.isfile(file_path):
                results_df, abl = process_file(file_path, args.baseline_shrink_factor,
                                          args.pulse_shrink_factor, args.plot, results_df=results_df)
                if args.plot:
                    abl.plot()
    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        sys.exit(1)

    # start app to show plots
    if args.plot:
        app.exec()

    # Save results_df to output file based on the extension
    print(f"Saving results to `{args.output}`")

    _, file_extension = os.path.splitext(args.output)

    match file_extension:
        case '.xls' | '.xlsx':
            results_df.to_excel(args.output, index=False)
        case '.csv':
            results_df.to_csv(args.output, index=False)
        case _:
            raise ValueError(f"Unsupported file extension: {file_extension}")


def process_file(file_path, baseline_shrink_factor, pulse_shrink_factor, plot, results_df=None, ):
    print(f"Processing `{file_path}`", end="")

    a = LaserAblationData(file_path)
    print(f" ({a.name})")
    a.baseline_shrink_factor = baseline_shrink_factor
    a.pulse_shrink_factor = pulse_shrink_factor
    a.calculate_heights()

    if results_df is None:
        results_df = pd.DataFrame(columns=[''] + list(a.isotope_heights.keys()))

    # store results
    results_df.loc[len(results_df)] = [a.name] + list(a.isotope_heights.values())

    return results_df, a


if __name__ == "__main__":
    main()
