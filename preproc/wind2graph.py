#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read preprocessed wind numpy matrix and convert it to a directed graph
"""

import csv
import arrow
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances



def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(ax, 
    start_point, end_point, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), steps=10,
    linewidth=1., alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    path  = mpath.Path(np.stack([start_point, end_point]))
    verts = path.interpolated(steps=steps).vertices
    x, y  = verts[:, 0], verts[:, 1]
    z     = np.linspace(0, 1, len(x))

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

    return lc

def plot_wind_graph(t, locs, edges, latlim, lnglim, filename):
    """
    """
    plt.rc('text', usetex=True)

    fig, ax    = plt.subplots(1, 1)
    # lngs, lats = locs[:, 0], locs[:, 1]
    lats, lngs = locs[:, 0], locs[:, 1]
    ax.scatter(lngs, lats, s=10, c="blue")

    for j, i in zip(*edges):
        # start_point = np.array([locs[j, 1], locs[j, 0]])
        # end_point   = np.array([locs[i, 1], locs[i, 0]])
        start_point = np.array([locs[j, 1], locs[j, 0]])
        end_point   = np.array([locs[i, 1], locs[i, 0]])
        colorline(ax, start_point, end_point, cmap=plt.get_cmap('Reds_r'), alpha=.5, linewidth=1.)
    
    plt.title("$t=%d$" % t)
    plt.ylim(latlim)
    plt.xlim(lnglim)
    plt.axis("off")
    fig.tight_layout()
    plt.savefig("../results/plot_graph/%s.png" % filename)

def plot_dynamic_graphs(locs, dgraph):
    """
    Plot dynamic graphs.

    locs:    [ K, 2 ]
    dgraphs: [ T, K, K ]
    """
    mheight = locs[:, 0].max() - locs[:, 0].min()
    mwidth  = locs[:, 1].max() - locs[:, 1].min()
    latlim  = [ locs[:, 0].min() - mheight * .1, locs[:, 0].max() + mheight * .1 ]
    lnglim  = [ locs[:, 1].min() - mwidth * .1 , locs[:, 1].max() + mwidth * .1 ]

    for t in range(wind.shape[0]):
        print("[%s] generating the wind plot at time %d" % (arrow.now(), t))
        plot_wind_graph(t, locs, np.nonzero(dgraph[t]), latlim=latlim, lnglim=lnglim, filename="graph-plot-%d" % t)

def k_nearest_graph_support(locs, k):
    """
    graph support: binary matrix indicating the k nearest locations in each row
    """
    
    # return a binary (0, 1) vector where value 1 indicates whether the entry is 
    # its k nearest neighbors. 
    def _k_nearest_neighbors(arr, k=k):
        idx  = arr.argsort()[:k]  # [K]
        barr = np.zeros(len(arr)) # [K]
        barr[idx] = 1         
        return barr

    # pairwise distance
    distmat = euclidean_distances(locs) # [K, K]
    # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
    supp    = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
    # set diagonal to be zero (exclude self-loop)
    np.fill_diagonal(supp, 0)
    return supp

def angle2north(loc_j, loc_i):
    """
    return the angle between the direction from j to i and the due north direction, ranging from 0 to 360. 
    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
    """
    dir_nor = np.array([0, 1])
    dir_ji  = loc_i - loc_j
    dir_nor = dir_nor / np.linalg.norm(dir_nor)
    dir_ji  = dir_ji / np.linalg.norm(dir_ji)

    x1, y1  = dir_nor[0], dir_nor[1]
    x2, y2  = dir_ji[0], dir_ji[1]

    dot     = x1*x2 + y1*y2 # dot product
    det     = x1*y2 - y1*x2 # determinant
    angle   = (- np.arctan2(det, dot) / np.pi * 180 + 360) % 360
    return angle


if __name__ == "__main__":

    wind = np.load("../data/sample_wind.npy") # [ T, K, 2 ]
    locs = np.load("../data/locations.npy")   # [ K, 2 ]

    n_locs = locs.shape[0]
    n_time = wind.shape[0]
    print("[%s] %d locations and %d time slots." % (arrow.now(), n_locs, n_time)) 

    # a matrix contains the i's relative position to j for each pair of (i, j), 
    locs = np.flip(locs, 1)                   # inter-change the latitude (y-axis) and longitude (x-axis)
    rpos = np.zeros((n_locs, n_locs))
    for j in range(n_locs):
        for i in range(n_locs):
            if j != i:               
                rpos[j, i] = angle2north(locs[j], locs[i])
    locs = np.flip(locs, 1)                   # inter-change the latitude (y-axis) and longitude (x-axis)
    
    # graph support
    supp = k_nearest_graph_support(locs, k=100)
    np.save("../data/gsupp.npy", supp)

    # temporal dynamic graph: 
    # at time t, a unique directed graph is decided by the wind directions, 
    # where the wind blows from j to i
    dgraph = np.zeros((n_time, n_locs, n_locs))
    for t in tqdm(range(n_time)):
        for j, i in zip(*np.nonzero(supp)):
            angle = abs(wind[t, j, 0] - rpos[j, i])
            if angle <= 15 or 360 - angle <= 15:
                dgraph[t, j, i] = 1
    np.save("../data/dgraph.npy", dgraph)

    # plot dynamic graphs on the map
    plot_dynamic_graphs(locs, dgraph)

    # # generate gif
    # images    = []
    # filenmaes = [ "../results/plot_graph/graph-plot-%d.png" % t for t in range(550) ]
    # for filename in filenmaes:
    #     images.append(imageio.imread(filename))
    # print("[%s] generating the graph animation" % arrow.now())
    # imageio.mimsave('../results/graph-animation.gif', images)


            
                
