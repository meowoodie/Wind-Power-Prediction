#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import arrow
import utils
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import offset_copy
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from spregress_backup import SpatioTemporalRegressor
from spregress import SpatioTemporalRegressor, SpatioTemporalDelayedRegressor

def pred_linechart(pred, true, filename):
    """
    Plot prediction trajectory against ground truth. 
    """
    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(pred))
    ground = np.zeros(len(pred))

    with PdfPages("../results/result_lineplots/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.fill_between(x, true, ground, where=true >= ground, facecolor='green', alpha=0.2, interpolate=True, label="Real")
        # ax.plot(x, pred, linewidth=3, color="green", alpha=1, label="In-sample estimation")
        ax.plot(x, pred, linewidth=3, color="green", alpha=1, label="Out-of-sample prediction")

        plt.xlabel(r"Time index")
        plt.ylabel(r"Wind speed (m/s)")
        plt.legend(fontsize=15) # loc='upper left'
        plt.title(filename)
        fig.tight_layout()      # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def mae_heatmap(pred, true, filename):

    nt, nlocs = pred.shape
    ae        = abs(pred - true)
    mae_locs  = ae.mean(0)
    mae_time  = ae.mean(1)
    loc_order = np.argsort(mae_locs)
    ae        = ae[:, loc_order]
    rev_loc_order = np.flip(loc_order)
    mae_locs  = mae_locs[rev_loc_order]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 10}
    plt.rc('font', **font)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width       = 0.15, 0.65
    bottom, height    = 0.15, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_imshow = [left, bottom, width, height]
    rect_week   = [left, bottom_h, width, 0.12]
    rect_state  = [left_h, bottom, 0.12, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8, 8))

    ax_imshow = plt.axes(rect_imshow)
    ax_locs   = plt.axes(rect_state)
    ax_time   = plt.axes(rect_week)

    # no labels
    ax_locs.xaxis.set_major_formatter(nullfmt)
    ax_time.yaxis.set_major_formatter(nullfmt)

    # the absolute error matrix for each location
    cmap = matplotlib.cm.get_cmap('magma')
    img  = ax_imshow.imshow(ae, cmap=cmap, vmin=0, vmax=ae.max(), extent=[0,nt,0,50], aspect=float(nt)/50.)
    ax_imshow.get_xaxis().set_ticks([])
    ax_imshow.get_yaxis().set_ticks([])
    ax_imshow.set_xlabel("Time")
    ax_imshow.set_ylabel("Location")

    # the error vector for states and weeks
    # mae_locs[mae_locs > 2.] = 2.
    ax_locs.plot(mae_locs, np.arange(nlocs), c="red", linewidth=2, linestyle="-", alpha=.7)
    ax_time.plot(mae_time, c="blue", linewidth=2, linestyle="-", alpha=.7)

    ax_locs.get_yaxis().set_ticks([])
    ax_locs.get_xaxis().set_ticks([])
    ax_locs.set_xlabel("MAE")
    ax_locs.set_ylim(0, 50)
    ax_time.get_xaxis().set_ticks([])
    ax_time.get_yaxis().set_ticks([])
    ax_time.set_ylabel("MAE")
    ax_time.set_xlim(0, nt)
    plt.figtext(0.81, 0.133, '0')
    plt.figtext(0.91, 0.133, '%.2f' % max(mae_locs))
    plt.figtext(0.135, 0.81, '0')
    plt.figtext(0.110, 0.915, '%.2f' % max(mae_time))

    cbaxes = fig.add_axes([left_h, height + left + 0.01, .03, .12])
    cbaxes.get_xaxis().set_ticks([])
    cbaxes.get_yaxis().set_ticks([])
    cbaxes.patch.set_visible(False)
    cbar = fig.colorbar(img, cax=cbaxes)
    cbar.set_ticks([0, ae.max()])
    cbar.set_ticklabels([0, ae.max()])
    cbar.ax.set_ylabel('AE', rotation=270, labelpad=-1)

    fig.tight_layout()
    fig.savefig("../results/%s.pdf" % filename)
    
def mae_map(pred, true, locs, filename):
    """
    Plot prediction mean absolute error over the map.
    """
    mae = abs(pred - true).mean(0)

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

    # rescale marker size
    mins, maxs = 5, 300
    size = (mae - mae.min()) / (mae.max() - mae.min())
    size = size * (maxs - mins) + mins
    # colorbar
    cm   = plt.cm.get_cmap('Reds')
    # scatter plot
    sct  = ax.scatter(locs[:, 1], locs[:, 0], alpha=0.5, s=size, c=mae, cmap=cm, edgecolors="black", lw=1, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sct, ax=ax, pad=0.02)
    cbar.set_label('Mean absolute error')

    fig.tight_layout()
    fig.savefig("../results/%s.pdf" % filename)



if __name__ == "__main__":

    torch.manual_seed(12)

    # load data matrices
    dgraph, speeds, gsupp, locs = utils.dataloader(rootpath="../data/data_k50", N=24)

    model = SpatioTemporalRegressor(speeds, dgraph, gsupp, d=50)
    model.load_state_dict(torch.load("saved_models/in-sample-exp-kernel-k50-t24-d10.pt"))

    model.eval()
    loss, pred0, pred1 = model() 
    print("[%s] Loaded model: final loss %.3e." % (arrow.now(), loss.item()))
    pred = (pred0 + pred1).detach().numpy().transpose()
    print(pred.shape)
    
    # # plots
    # for i in range(50):
    #     pred_linechart(pred[:, i], speeds[:, i], filename="Turbine %d" % i)
    # pred_linechart(pred.mean(1), speeds.mean(1), filename="Average")

    # # MAE map
    # mae_map(pred, speeds, locs, filename="MAE map")

    # MAP heatmap
    mae_heatmap(pred, speeds, filename="MAE heatmap")

