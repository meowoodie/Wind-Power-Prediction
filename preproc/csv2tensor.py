#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read sample data and convert the raw csv data to a numpy matrix
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_wind_graph(wind_t, locations):
    """
    Plot wind data at a time point on a map, where each point represents a location and the arrow represents the wind at this location.

    wind_t:    [ n_locations, 2 ]
    locations: [ n_locations, 2 ]
    """
    direction_t, speed_t = wind_t[:, 0], wind_t[:, 1]
    lats, lngs           = locations[:, 0], locations[:, 1]
    plt.scatter(lats, lngs, s=)
    plt.show()

if __name__ == "__main__":

    # read wind speed
    with open("../data/sample data/windspeed_1week_data.csv", newline='') as f:
        speeds      = []
        speedreader = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(speedreader):
            if i == 0:
                locations = [ [ float(latlng) for latlng in locstr.strip('"').split(",") ] for locstr in row[1:] ]
                locations = np.array(locations)
            else:
                speeds.append([ float(speed) for speed in row[1:] ])
        speeds      = np.array(speeds)

    # read wind direction
    with open("../data/sample data/winddirection_1week_data.csv", newline='') as f:
        directions      = []
        directionreader = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(directionreader):
            if i == 0:
                continue
            else:
                directions.append([ float(direction) for direction in row ])
        directions      = np.array(directions)

    wind = np.stack([directions, speeds], axis=-1) # [ n_times, n_locations, n_features=2 ]
    # np.save("../data/sample_wind.npy", wind)


    plot_wind_graph(wind[0, :, :], locations)


