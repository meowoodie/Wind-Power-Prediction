#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatio-temporal regressive model for wind power prediction
"""

import torch
import arrow
import random
import utils
import numpy as np
import torch.optim as optim
from scipy.stats import norm



def train(model, niter=1000, lr=1e-1, log_interval=50, modelname="in-sample"):
    """training procedure for one epoch"""
    # define model clipper to enforce inequality constraints
    clipper = NonNegativeClipper()  

    # initial loss without training
    model.eval()
    loss, _, _ = model()
    print("[%s] Initial Loss: %.3e" % (arrow.now(), loss.item()))

    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    losses     = []
    optimizer  = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    for _iter in range(niter):
        try:
            model.train()
            optimizer.zero_grad()            # init optimizer (set gradient to be zero)
            loss, _, _ = model()
            # objective function
            loss.backward(retain_graph=True) # gradient descent
            optimizer.step()                 # update optimizer
            print(model.Alpha)
            # model.apply(clipper)
            # log training output
            losses.append(loss.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train epoch: %d\tLoss: %.3e" % (arrow.now(), 
                    _iter / log_interval, 
                    sum(losses) / len(losses)))
                losses = []
                torch.save(model.state_dict(), "saved_models/%s.pt" % modelname)
        except KeyboardInterrupt:
            break



class NonNegativeClipper(object):
    """
    References:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/3
    """

    def __init__(self):
        pass

    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Alpha'):
            Alpha = module.Alpha.data
            module.Alpha.data = torch.clamp(Alpha, min=0.)
        if hasattr(module, 'Beta'):
            Beta  = module.Beta.data
            module.Beta.data  = torch.clamp(Beta, min=0.)
        if hasattr(module, 'Mu'):
            Mu  = module.Mu.data
            module.Mu.data    = torch.clamp(Mu, min=0.)



class SpatioTemporalRegressor(torch.nn.Module):
    """
    PyTorch module for spatio-temporal regressive model for wind power prediction
    """

    def __init__(self, speeds, dgraph, gsupp, d=20):
        """
        Denote the number of time units as T, the number of locations as K
        Args:
        - speeds: wind speed observations [ T, K ]
        - dgraph: dynamic graph           [ T, K, K ]
        - gsupp:  graph support           [ K, K ]
        - muG:    mean of gaussian kernel [ T, K, K ]
        - d:      memory depth            scalar
        """
        torch.nn.Module.__init__(self) 
        # configurations
        self.T, self.K = speeds.shape
        self.d         = d
        # data
        self.speeds = torch.Tensor(speeds)                                    # [ T, K ]
        self.speeds = torch.transpose(self.speeds, 0, 1)                      # [ K, T ] transpose
        self.dgraph = torch.Tensor(dgraph)                                    # [ T, K, K ]
        # parameters
        self.base   = self.speeds.mean(1) / 3 # + torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, 1)) # [ K ]
        self.Beta   = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3)) # [ K ]

        # non-zero entries of alpha (spatio dependences)
        self.n_nonzero     = len(np.where(gsupp == 1)[0])
        self.coords        = torch.LongTensor(np.where(gsupp == 1))
        self.Alpha_nonzero = torch.nn.Parameter(torch.randn((self.n_nonzero), requires_grad=True))
    
    def _base(self, _t):
        """
        Background rate at time `t`
        """
        return self.base

    def _pred(self, _t):
        """
        Wind prediction at time _t for all K locations.
        Args:
        - _t:   index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - pred: a vector of predictions at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            depth = self.d if _t >= self.d else _t
            preds = []
            for tau in range(_t-depth, _t):
                # getting data at past time tau
                graph  = self.dgraph[tau, :, :].clone()            # [ K, K ]
                X_tau  = self.speeds[:, tau].clone()               # [ K ] 
                # calculate self-exciting effects for each location i
                A      = graph * self.Alpha
                kernel = self.__exp_kernel(_t, tau, self.Beta)     # [ K ]
                pred   = torch.mm(
                    torch.transpose(A, 0, 1), 
                    (X_tau * kernel).unsqueeze(1)).squeeze()       # [ K, 1 ]
                pred   = torch.nn.functional.softplus(pred)        # [ K ]
                preds.append(pred)
            preds = torch.stack(preds, 0)
            preds = preds.sum(0)
        else:
            preds = torch.zeros(self.K)
        return preds
        
    def _l2_loss(self):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)
        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   a list of historical conditional intensity values at time t = tau, ..., t
        """
        self.Alpha = torch.sparse.FloatTensor(
            self.coords, self.Alpha_nonzero,
            torch.Size([self.K, self.K])).to_dense()           # [ K, K ]
        # pred values from 0 to T
        pred0 = [ self._base(t) for t in np.arange(self.T) ]   # ( T, [ K ] )
        pred1 = [ self._pred(t) for t in np.arange(self.T) ]   # ( T, [ K ] )
        pred0 = torch.stack(pred0, dim=1)                      # [ K, T ]
        pred1 = torch.stack(pred1, dim=1)                      # [ K, T ]
        # l2 loss function
        loss  = torch.square(self.speeds - pred1 - pred0).mean()
        return loss, pred0, pred1

    def forward(self):
        """
        customized forward function
        """
        # calculate l2 loss and prediction
        return self._l2_loss()

    @staticmethod
    def __exp_kernel(_t, tau, beta):
        """
        Args:
        - beta:  decaying rate [ K ]
        - _t:    time index    scalar
        - depth: time depth    scalar
        """
        return beta * torch.exp(- (_t - tau) * beta)
        


class SpatioTemporalDelayedRegressor(SpatioTemporalRegressor):
    """
    PyTorch module for spatio-temporal delayed regressive model for wind power prediction
    """
    def __init__(self, speeds, dgraph, gsupp, muG, d=20):
        """
        Denote the number of time units as T, the number of locations as K
        Args:
        - speeds: wind speed observations [ T, K ]
        - dgraph: dynamic graph           [ T, K, K ]
        - gsupp:  graph support           [ K, K ]
        - muG:    mean of gaussian kernel [ T, K, K ]
        - d:      memory depth            scalar
        """
        SpatioTemporalRegressor.__init__(self, speeds, dgraph, gsupp, d)
        # mean for Gaussian delayed kernel
        self.muG = torch.Tensor(muG) # [ T, K, K ]
        # unregister beta in SpatioTemporalRegressor
        self.Beta.requires_grad = False
    
    def _pred(self, _t):
        """
        Wind prediction at time _t for all K locations.
        Args:
        - _t:   index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - pred: a vector of predictions at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            depth = self.d if _t >= self.d else _t
            preds = []
            for tau in range(_t-depth, _t):
                # getting data at past time tau
                graph  = self.dgraph[tau, :, :].clone() # [ K, K ]
                mu     = self.muG[tau, :, :].clone()    # [ K, K ]
                X_tau  = self.speeds[:, tau].clone()    # [ K ] 
                # calculate delayed effects for each location i
                A      = graph * self.Alpha
                kernel = self.__trunc_gaussian_kernel(_t, tau, mu) # [ K, K ]
                pred   = torch.mm(
                    torch.transpose(A * kernel, 0, 1), 
                    X_tau.unsqueeze(1)).squeeze()                  # [ K, 1 ]
                pred   = torch.nn.functional.softplus(pred)        # [ K ]
                preds.append(pred)
            preds = torch.stack(preds, 0)
            preds = preds.sum(0)
        else:
            preds = torch.zeros(self.K)
        return preds

    @staticmethod
    def __trunc_gaussian_kernel(t, tau, mu_mat, sigma=20):
        """
        Args: 
        - t:      current time index              scalar
        - tau:    past time index                 scalar
        - mu_mat: mean matrix for gaussian kernel [ K, K ]
        - sigma:  sigma for gaussian kernel
        """
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
            torch.exp((-1/2) * torch.square((t - tau - mu_mat)/sigma))




if __name__ == "__main__":
    
    torch.manual_seed(12)

    # load data matrices
    dgraph, speeds, gsupp, muG, _ = utils.dataloader(N=4)
    
    # training
    model = SpatioTemporalRegressor(speeds, dgraph, gsupp, d=20)
    train(model, niter=1000, lr=1e0, log_interval=2, modelname="in-sample-exp-kernel")

    # model = SpatioTemporalDelayedRegressor(speeds, dgraph, gsupp, muG=muG, d=20)
    # train(model, niter=1000, lr=5e1, log_interval=2, modelname="in-sample-gauss-kernel")