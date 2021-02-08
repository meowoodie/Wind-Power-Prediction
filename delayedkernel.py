#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pre-compute the mean of the delayed Hawkes kernel given locations and their corresponding wind speed. 
"""

import torch
import utils
import arrow
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm



def haversine_distance(loc1, loc2):
    """
    return distance (m) between two geolocations 
    https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4
    """
    lat1, lon1, lat2, lon2 = loc1[0], loc1[1], loc2[0], loc2[1]
    r            = 6371
    phi1         = np.radians(lat1)
    phi2         = np.radians(lat2)
    delta_phi    = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a            = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res          = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2) * 1000


if __name__ == "__main__":

    # meta info
    N      = 4
    t_unit = 15 * N # minutes
    depth  = 50     # time units

    # load data matrices
    dgraph, speeds, gsupp, muG, locs = utils.dataloader(N=4)
    n_times, n_locations = speeds.shape[0], speeds.shape[1]

    # pairwise distance between locations
    D = np.zeros((n_locations, n_locations))
    for j in range(n_locations):
        for i in range(n_locations):
            D[j, i] = haversine_distance(locs[i], locs[j])

    # pre-computation for Gaussian kernel mean
    # count = 0
    G = np.zeros((n_times, n_locations, n_locations))
    for t in tqdm(range(n_times)):
        for j, i in zip(*np.nonzero(dgraph[t])):
            speed      = speeds[t, j] if speeds[t, j] > 0 else 1e-5
            # if speeds[t, j] > 0:
            G[t, j, i] = D[j, i] / (speed * 60 * t_unit)
            # else: 
            #     count += 1

    # NOTE: there are 29 entries in speeds being zero. 
    np.save("../data/gauss_mean.npy", G)

    # print(G[G > 0], count)
    # plt.hist(G[G>0].flatten(), bins=100)
    # plt.show()
    
    