#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import arrow
import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from spregress_backup import SpatioTemporalRegressor
from spregress import SpatioTemporalRegressor, SpatioTemporalDelayedRegressor

def pred_linechart(data, true, filename):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(data))
    ground = np.zeros(len(data))

    with PdfPages("../results/result_lineplots/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.fill_between(x, true, ground, where=true >= ground, facecolor='green', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data, linewidth=3, color="green", alpha=1, label="In-sample estimation")

        plt.xlabel(r"Time index")
        plt.ylabel(r"Wind speed (m/s)")
        plt.legend(fontsize=15) # loc='upper left'
        plt.title(filename)
        fig.tight_layout()      # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

if __name__ == "__main__":

    torch.manual_seed(12)

    # load data matrices
    dgraph, speeds, gsupp, muG, _ = utils.dataloader(N=4)

    # load model
    # model = SpatioTemporalRegressor(speeds, dgraph, gsupp, d=50)
    # # model.load_state_dict(torch.load("saved_models/backup-in-sample-exp-kernel.pt"))
    # model.load_state_dict(torch.load("saved_models/backup-in-sample-gauss-kernel.pt"))

    model = SpatioTemporalRegressor(speeds, dgraph, gsupp, d=50)
    model.load_state_dict(torch.load("saved_models/in-sample-exp-kernel.pt"))

    model.eval()
    loss, pred0, pred1 = model() 
    print("[%s] Loaded model: final loss %.3e." % (arrow.now(), loss.item()))
    pred = (pred0 + pred1).detach().numpy()
    print(pred.shape)
    
    # plots
    for i in range(20):
        pred_linechart(pred[i], speeds[:, i], filename="Turbine %d" % i)
    pred_linechart(pred.mean(0), speeds.mean(1), filename="Average")

