#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read sample data and convert the raw csv data to a numpy matrix
"""

import csv
import arrow
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plot_single_wind_graph(t, wind_t, locations, latlim, lnglim, max_speed=15., arrow_length=1., filename="wind-plot"):
    """
    Plot wind data at a time point on a map, where each point represents a location and the arrow represents the wind at this location.

    wind_t:    [ n_locations, 2 ]
    locations: [ n_locations, 2 ]
    """
    direction_t, speed_t = wind_t[:, 0], wind_t[:, 1]
    lats, lngs           = locations[:, 0], locations[:, 1]
    
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(lngs, lats, s=10, c="red")

    # # DEBUG
    # i = 0
    # dir_t = direction_t[i] / 180 * np.pi
    # spd_t = 100
    # ax.text(lngs[i], lats[i], direction_t[i], fontsize=10)
    # y     = [ lats[i], lats[i] + np.cos(dir_t) * 5 ]
    # x     = [ lngs[i], lngs[i] + np.sin(dir_t) * 5 ]
    # ax.plot(x, y, linewidth=2., c="blue", alpha=spd_t / max_speed)

    for i in range(locations.shape[0]):
        dir_t = direction_t[i] / 180 * np.pi
        spd_t = speed_t[i]
        y     = [ lats[i], lats[i] + np.cos(dir_t) * arrow_length ]
        x     = [ lngs[i], lngs[i] + np.sin(dir_t) * arrow_length ]
        ax.plot(x, y, linewidth=2., c="blue", alpha=spd_t / max_speed)

    plt.title("$t=%d$" % t)
    plt.ylim(latlim)
    plt.xlim(lnglim)
    plt.axis("off")
    plt.savefig("../results/plot_wind_k50/%s.png" % filename)

def plot_wind_graphs(wind, locations):
    """
    Plot all wind data.

    wind:      [ n_times, n_locations, 2 ]
    locations: [ n_locations, 2 ]
    """
    mheight = locations[:, 0].max() - locations[:, 0].min()
    mwidth  = locations[:, 1].max() - locations[:, 1].min()
    latlim  = [ locations[:, 0].min() - mheight * .1, locations[:, 0].max() + mheight * .1 ]
    lnglim  = [ locations[:, 1].min() - mwidth * .1 , locations[:, 1].max() + mwidth * .1 ]
    for t in range(wind.shape[0]):
        print("[%s] generating the wind plot at time %d" % (arrow.now(), t))
        plot_single_wind_graph(t, wind[t, :, :], locations, 
            latlim=latlim, lnglim=lnglim, max_speed=15., arrow_length=1., filename="wind-plot-t%d" % t)
            


if __name__ == "__main__":

    # # read wind speed
    # with open("../data/sample data/windspeed_1week_data.csv", newline='') as f:
    #     speeds      = []
    #     speedreader = csv.reader(f, delimiter=',', quotechar='"')
    #     for i, row in enumerate(speedreader):
    #         if i == 0:
    #             locations = [ [ float(latlng) for latlng in locstr.strip('"').split(",") ] for locstr in row[1:] ]
    #             locations = np.array(locations)
    #         else:
    #             speeds.append([ float(speed) for speed in row[1:] ])
    #     speeds = np.array(speeds)

    # # read wind direction
    # with open("../data/sample data/winddirection_1week_data.csv", newline='') as f:
    #     directions      = []
    #     directionreader = csv.reader(f, delimiter=',', quotechar='"')
    #     for i, row in enumerate(directionreader):
    #         if i == 0:
    #             continue
    #         else:
    #             directions.append([ float(direction) for direction in row ])
    #     directions = np.array(directions)

    # wind = np.stack([directions, speeds], axis=-1) # [ n_times, n_locations, n_features=2 ]

    # np.save("../data/sample_wind.npy", wind)
    # np.save("../data/locations.npy", locations)


    wind      = np.load("../data/sample_wind_k50.npy")
    locations = np.load("../data/locations_k50.npy")

    # plot wind data on maps
    plot_wind_graphs(wind, locations)
    
    # generate gif
    images    = []
    filenmaes = [ "../results/plot_wind_k50/wind-plot-t%d.png" % t for t in range(wind.shape[0]) ]
    for filename in filenmaes:
        images.append(imageio.imread(filename))
    print("[%s] generating the wind animation" % arrow.now())
    imageio.mimsave('../results/wind-animation-k50.gif', images)



