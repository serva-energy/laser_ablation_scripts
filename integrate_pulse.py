import numpy as np
import pandas as pd
import pyqtgraph as pg

import os
import glob

class LaserAblationData():
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
        for iso in self.isotopes.keys():
            pw = pg.plot(x=self.timestamps, y=self.isotopes[iso], symbol='o', pen='b')
            pw.setWindowTitle(iso)


def __debug_plots():    
    a = LaserAblationData("20240510_Montero_Bullet-Glass_01_1.csv")
    
    app = pg.mkQApp()
    a.plot()
    app.exec()

def __debug_loading():
    for name in glob.glob(os.path.join('./20240531BulletGlassOriginals', '*.csv')):
        LaserAblationData(name)

if __name__ == "__main__":
    __debug_plots()
    # __debug_loading()