#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge data in the vicinity by their geolocations
"""

import arrow
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def plot_merged_locs(labels, locs, merged_locs, filename="merged_locations"):
    """
    Plot merged locations on the map.
    """
    latlow, latup = locs[:, 0].min(), locs[:, 0].max()
    lnglow, lngup = locs[:, 1].min(), locs[:, 1].max()
    width, height = lngup - lnglow, latup - latlow
    latlow, latup = latlow - height * 0.1, latup + height * 0.1
    lnglow, lngup = lnglow - width * 0.1, lngup + width * 0.1
    heightwidthratio = float(height / width)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # Create a Stamen terrain background instance.
    stamen_terrain = cimgt.Stamen('terrain-background')

    fig = plt.figure(figsize=(10, heightwidthratio * 10))

    # Create a GeoAxes in the tile's projection.
    ax  = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([lnglow, lngup, latlow, latup], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)

    # colorbar
    cm = plt.cm.get_cmap('gist_ncar')
    # scatter plot
    for label in range(len(set(labels))):
        idx = np.where(labels == label)[0]
        sct = ax.scatter(locs[idx, 1], locs[idx, 0], 
            alpha=0.3, s=10, c=labels[idx], cmap=cm, vmin=0, vmax=len(set(labels)), 
            marker="o", edgecolors="black", lw=1, transform=ccrs.PlateCarree())
        sct = ax.scatter(merged_locs[label, 1], merged_locs[label, 0], 
            alpha=1., s=100, c=[label], cmap=cm, vmin=0, vmax=len(set(labels)), 
            marker="*", edgecolors="black", lw=1, transform=ccrs.PlateCarree())

    fig.tight_layout()
    fig.savefig("../results/%s.pdf" % filename)

def plot_merged_mae(
    labels, locs, data, merged_data, 
    vmax=None, vmin=None,
    filename="merged_mae", titlename="wind", is_pct=False):
    """
    Plot merged mean absolute error on the map.
    """
    latlow, latup = locs[:, 0].min(), locs[:, 0].max()
    lnglow, lngup = locs[:, 1].min(), locs[:, 1].max()
    width, height = lngup - lnglow, latup - latlow
    latlow, latup = latlow - height * 0.1, latup + height * 0.1
    lnglow, lngup = lnglow - width * 0.1, lngup + width * 0.1
    heightwidthratio = float(height / width)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # Create a Stamen terrain background instance.
    stamen_terrain = cimgt.Stamen('terrain-background')

    fig = plt.figure(figsize=(10, heightwidthratio * 10))

    # Create a GeoAxes in the tile's projection.
    ax  = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([lnglow, lngup, latlow, latup], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)

    maes = []
    for label in range(len(set(labels))):
        idx = np.where(labels == label)[0]
        if is_pct:
            mae = abs(data[:, idx] - merged_data[:, [label]]).mean(axis=0) / data[:, idx].mean(axis=0)
        else:
            mae = abs(data[:, idx] - merged_data[:, [label]]).mean(axis=0)
        maes.append(mae)
    if vmax is None or vmin is None:
        max_mae, min_mae = max([ max(mae) for mae in maes ]), min([ min(mae) for mae in maes ])
    else:
        max_mae, min_mae = vmax, vmin

    # colorbar
    cm = plt.cm.get_cmap('Reds')
    # rescale marker size to the range below
    mins, maxs = 5, 300
    # scatter plot
    for label in range(len(set(labels))):
        idx  = np.where(labels == label)[0]
        mae  = maes[label]
        size = (mae - min_mae) / (max_mae - min_mae)
        size = size * (maxs - mins) + mins
        sct  = ax.scatter(locs[idx, 1], locs[idx, 0], 
            alpha=0.3, s=size, c=mae, cmap=cm, vmin=0, vmax=max_mae, 
            marker="o", edgecolors="black", lw=1, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sct, ax=ax, pad=0.02)
    if is_pct:
        cbar.set_label('Mean absolute error percentage')
    else:
        cbar.set_label('Mean absolute error')

    plt.title(titlename)
    fig.tight_layout()
    fig.savefig("../results/%s.pdf" % filename)

def merge_wind_data(labels, wind, locs, merged_locs):
    """
    Merge wind data based on the clusters
    """
    # NOTE: TAKE AVEAGE
    _K       = len(set(labels))
    T, K, _  = wind.shape
    upd_wind = np.zeros((T, _K, 2))
    for label in range(len(set(labels))):
        idx = np.where(labels == label)[0]
        upd_wind[:, label, :] = wind[:, idx, :].mean(axis=1)
        # upd_wind[:, label, 0] = wind[:, idx, 0].mean()
        # upd_wind[:, label, 1] = wind[:, idx, 1].mean()
    return upd_wind

    # # NOTE: TAKE THE LOCATION NEAREST TO THE CLUSTER CENTER
    # dist     = euclidean_distances(merged_locs, locs)
    # _K       = len(set(labels))
    # T, K, _  = wind.shape
    # upd_wind = np.zeros((T, _K, 2))
    # for label in range(len(set(labels))):
    #     idx = dist[label].argmin()
    #     upd_wind[:, label, :] = wind[:, idx, :]
    #     # upd_wind[:, label, 0] = wind[:, idx, 0].mean()
    #     # upd_wind[:, label, 1] = wind[:, idx, 1].mean()
    # return upd_wind


if __name__ == "__main__":

    wind = np.load("../data/sample_wind.npy") # [ T, K, 2 ]
    locs = np.load("../data/locations.npy")   # [ K, 2 ]

    n_clusters = 300
    kmeans     = KMeans(n_clusters=n_clusters, random_state=0).fit(locs)
    labels     = kmeans.labels_
    upd_locs   = kmeans.cluster_centers_
    upd_wind   = merge_wind_data(labels, wind, locs, upd_locs)
    
    np.save("../data/locations_k%d.npy" % n_clusters, upd_locs)
    np.save("../data/sample_wind_k%d.npy" % n_clusters, upd_wind)

    # plot_merged_locs(labels, locs, upd_locs, filename="merged_locations_k%d" % n_clusters) 
    # plot_merged_mae(labels, locs, wind[:, :, 1], upd_wind[:, :, 1], 
    #     vmax=0.4, vmin=0.,   
    #     filename="merged_speed_mae_k%d" % n_clusters, titlename="Wind speed", is_pct=True)
    # plot_merged_mae(labels, locs, wind[:, :, 0], upd_wind[:, :, 0], 
    #     vmax=40, vmin=0.,
    #     filename="merged_direction_mae_k%d" % n_clusters, titlename="Wind direction")