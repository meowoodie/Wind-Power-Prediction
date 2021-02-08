#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import random
import numpy as np


def dataloader(
    dgraphpath="../data/dgraph.npy", 
    windpath="../data/sample_wind.npy", 
    gsupppath="../data/gsupp.npy", 
    locspath="../data/locations.npy", 
    mugpath="../data/gauss_mean_n4.npy", N=4):
    """
    load data matrices and perform basic preprocessing.
    """
    print("[%s] loading data matrices." % arrow.now())
    # load data
    locs   = np.load(locspath)         # [ K, 2 ]
    dgraph = np.load(dgraphpath)       # [ T, K, K ]
    wind   = np.load(windpath)         # [ T, K, 2 ]
    gsupp  = np.load(gsupppath)        # [ K, K ]
    muG    = np.load(mugpath)          # [ T/N, K, K ]
    speeds = wind[:, :, 1]             # [ T, K ]
    
    # truncate data to make the number of time units is divisible by N.
    n_time = int(dgraph.shape[0] / N) * N
    dgraph = dgraph[:n_time, :, :]
    wind   = wind[:n_time, :, :]
    speeds = speeds[:n_time, :]
    print("[%s] first %d time units are selected." % (arrow.now(), n_time))

    # take average every four time units (1 hour)
    avg_speeds = avg(speeds, N=N)          # [ T/N, K ]
    inds       = [ t for t in range(dgraph.shape[0]) if t % N == int(N/2) ]
    avg_dgraph = dgraph[inds, :, :]
    print("[%s] speeds shape: %s, dgraph shape:%s." % (arrow.now(), speeds.shape, dgraph.shape))
    print("[%s] first %d rows of original speeds:" % (arrow.now(), 2*N))
    print("%s" % speeds[:2*N, :])
    print("[%s] first %d rows of averaged speeds:" % (arrow.now(), 2))
    print("%s" % avg_speeds[:2, :])

    # include diagonal entries
    np.fill_diagonal(gsupp, 1)
    for t in range(dgraph.shape[0]):         
        np.fill_diagonal(dgraph[t], 1)
    
    return avg_dgraph, avg_speeds, gsupp, muG, locs

def avg(mat, N=4):
    """
    calculate sample average for every N steps. 
    reference:
    https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
    """
    cum = np.cumsum(mat,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = mat.shape[0] % N
    if remainder != 0:
        if remainder < mat.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result