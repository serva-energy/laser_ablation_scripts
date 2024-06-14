import numpy as np
import pandas as pd
import pyqtgraph as pg

import os
import glob

from scipy.signal import find_peaks

class LaserAblationData():

    baseline_shrink_factor = 0.1
    pulse_shrink_factor = 0.3

    def __init__(self, filename=None) -> None:
        self.name = None
        self.file_timestamp = None

        self.metadata = {}
        self.timestamps = None
        self.isotopes = {}

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
            self.isotopes[col] = df[col].to_numpy(dtype='f8')

    def plot(self):

        baseline_boundaries_indices, pulse_boundaries_indices = self.find_pulse_boundaries()            
        baseline_boundaries = tuple(self.timestamps[baseline_boundaries_indices])
        pulse_boundaries = tuple(self.timestamps[pulse_boundaries_indices])

        for iso in self.isotopes.keys():
            x_data = self.timestamps
            y_data = self.isotopes[iso]

            pw = pg.plot(x=x_data, y=y_data, symbol='o', pen='b')            
            pw.addItem(pg.LinearRegionItem(values=baseline_boundaries, brush='#00ff0040', movable=False))
            pw.addItem(pg.LinearRegionItem(values=pulse_boundaries, brush='#0000ff40', movable=False))

            pw.setWindowTitle(iso)
            pw.showMaximized()

    def find_pulse_boundaries(self):
        baseline_boundaries_indices = []
        pulse_boundaries_indices = []
        for isotope in self.isotopes.keys():
            y_data = self.isotopes[isotope]

            # find pulse
            peaks, props = find_peaks(y_data, prominence=0.8*y_data.max())
            if len(peaks) == 1:            
                baseline_boundaries_indices.append(np.array([0, props['left_bases'][0]]))
                pulse_boundaries_indices.append(np.array([props['left_bases'][0], props['right_bases'][0]]))
            

            # find the pulse using smoothed 1st derivative
            y_diff = np.diff(np.convolve(y_data, np.ones(21) / 21, mode='same'))            
            baseline_boundaries_indices.append(np.array([0, np.argmax(y_diff)]))
            pulse_boundaries_indices.append(np.array([np.argmax(y_diff), np.argmin(y_diff)]))


        baseline_boundaries_indices = np.array(baseline_boundaries_indices)
        pulse_boundaries_indices = np.array(pulse_boundaries_indices)

        baseline_boundaries_indices = np.mean(baseline_boundaries_indices, axis=0)
        pulse_boundaries_indices = np.mean(pulse_boundaries_indices, axis=0)

        baseline_boundaries_indices = self.shrink_range (baseline_boundaries_indices, self.baseline_shrink_factor)
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

        print (f"{self.name}")
        for isotope, y_data in self.isotopes.items():
            val = np.mean(y_data[pulse_boundaries_indices])
            base = np.mean(y_data[baseline_boundaries_indices])
            val -= base
            print (f"{isotope} {val:.1f}")
        print()

def __debug_plots():    
    # a = LaserAblationData("20240510_Montero_Bullet-Glass_01_1.csv")
    a = LaserAblationData("./20240531BulletGlassOriginals/20240531_Montero_Bullet_Glass_04_45.csv")
    
    app = pg.mkQApp()
    a.plot()
    app.exec()

def __debug_loading():
    for name in glob.glob(os.path.join('./20240531BulletGlassOriginals', '*.csv')):
        a = LaserAblationData(name)

def __debug_calculate():
    a = LaserAblationData("20240510_Montero_Bullet-Glass_01_1.csv")
    a.calculate_heights()

if __name__ == "__main__":
    # __debug_plots()
    # __debug_loading()
    __debug_calculate()