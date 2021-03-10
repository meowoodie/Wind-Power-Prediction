#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Out-of-Sample
"""

import torch
import arrow
import random
import utils
import numpy as np
import torch.optim as optim
from scipy.stats import norm
from spregress import train, SpatioTemporalRegressor
from plotpred import pred_linechart, mae_map, mae_heatmap



class SpatioTemporalRegressivePredictor(SpatioTemporalRegressor):
    """
    PyTorch module for one-step ahead prediction using spatio-temporal regressive model
    """
    def __init__(self, speeds, dgraph, gsupp, d=20):
        """
        Denote the number of time units as T, the number of locations as K
        Args:
        - speeds: wind speed observations [ T, K ]
        - dgraph: dynamic graph           [ T, K, K ]
        - gsupp:  graph support           [ K, K ]
        - d:      memory depth            scalar
        """
        SpatioTemporalRegressor.__init__(self, speeds, dgraph, gsupp, d)
    
    def forward(self):
        """
        customized forward function: return the one-step ahead prediction for time T+1
        """
        # convert sparse alpha representation to dense matrix
        self.Alpha = torch.sparse.FloatTensor(
            self.coords, self.Alpha_nonzero,
            torch.Size([self.K, self.K])).to_dense() # [ K, K ]
        pred0 = self._base(self.T)                   # [ K ]
        pred1 = self._pred(self.T)                   # [ K ]
        return pred0 + pred1


if __name__ == "__main__":
    
    torch.manual_seed(12)

    # load data matrices
    N     = 4
    d     = 10
    wsize = 50

    dgraph, speeds, gsupp, locs = utils.dataloader(rootpath="../data/data_k50", N=N)
    T, K  = speeds.shape

    # # training
    # # for t in range(d, T-1):
    # for t in range(wsize, T-1):
    #     _dgraph, _speeds = dgraph[t-wsize:t, :, :], speeds[t-wsize:t, :]
    #     model = SpatioTemporalRegressor(_speeds, _dgraph, gsupp, d=d)
    #     if t > wsize:
    #         model.load_state_dict(torch.load("saved_models/out-of-sample-[i=%d]-exp-kernel-k50-t%d-d%d.pt" % (t - 1, N, d)))
    #         print("[%s] Model at t = %d has been loaded as initialization" % (arrow.now(), (t - 1)))
    #     train(model, niter=5000, lr=1e-2, log_interval=500, modelname="out-of-sample-[i=%d]-exp-kernel-k50-t%d-d%d" % (t, N, d))
    
    # evaluation
    preds = []
    # for t in range(d, T-1):
    for t in range(wsize, 168):
        _dgraph, _speeds = dgraph[t-wsize:t, :, :], speeds[t-wsize:t, :]
        model = SpatioTemporalRegressivePredictor(_speeds, _dgraph, gsupp, d=d)
        model.load_state_dict(torch.load("saved_models/out-of-sample-[i=%d]-exp-kernel-k50-t%d-d%d.pt" % (t, N, d)))
        pred  = model().detach().numpy()
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    print(preds.shape)

    # # plot
    # for i in range(K):
    #     pred_linechart(preds[:, i], speeds[wsize+1:T, i], filename="Turbine %d" % i)
    # pred_linechart(preds.mean(1), speeds[wsize+1:T,:].mean(1), filename="Average")
    # mae_map(preds, speeds[wsize+1:169,:], locs, filename="out-of-sample MAE map")
    mae_heatmap(preds, speeds[wsize+1:169,:], filename="out-of-sample MAE heatmap")