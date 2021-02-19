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
    dgraph, speeds, gsupp, locs = utils.dataloader(rootpath="../data/data_k50", N=24)
    d    = 10
    T, K = speeds.shape

    # # training
    # for t in range(d, T-1):
    #     _dgraph, _speeds = dgraph[:t, :, :], speeds[:t, :]
    #     model = SpatioTemporalRegressor(_speeds, _dgraph, gsupp, d=d)
    #     model.load_state_dict(torch.load("saved_models/out-of-sample-[i=%d]-exp-kernel-k50-t24-d10.pt" % t))
    #     train(model, niter=2000, lr=1e-2, log_interval=500, modelname="out-of-sample-[i=%d]-exp-kernel-k50-t24-d10" % t)
    
    # evaluation
    preds = []
    for t in range(d, T-1):
        _dgraph, _speeds = dgraph[:t, :, :], speeds[:t, :]
        model = SpatioTemporalRegressivePredictor(_speeds, _dgraph, gsupp, d=d)
        model.load_state_dict(torch.load("saved_models/out-of-sample-[i=%d]-exp-kernel-k50-t24-d10.pt" % t))
        pred  = model().detach().numpy()
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    print(preds.shape)

    # plot
    for i in range(K):
        pred_linechart(preds[:, i], speeds[d+1:T, i], filename="Turbine %d" % i)
    pred_linechart(preds.mean(1), speeds[d+1:T,:].mean(1), filename="Average")
    mae_map(preds, speeds[d+1:T,:], locs, filename="One-step ahead MAE map")
    # mae_heatmap(preds, speeds[d+1:T,:], filename="One-step ahead MAE heatmap")
