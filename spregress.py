#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatio-temporal regressive model for wind power prediction
"""

import torch
import arrow
import random
import numpy as np
import torch.optim as optim



def train(model, niter=1000, lr=1e-1, log_interval=50):
    """training procedure for one epoch"""
    # define model clipper to enforce inequality constraints
    clipper = NonNegativeClipper()  

    # initial loss without training
    model.eval()
    loss, _, _ = model()
    print("[%s] Initial Loss: %.3e" % (arrow.now(), loss.item()))

    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    losses  = []
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    for _iter in range(niter):
        try:
            model.train()
            optimizer.zero_grad()            # init optimizer (set gradient to be zero)
            loss, _, _ = model()
            # objective function
            loss.backward(retain_graph=True) # gradient descent
            optimizer.step()                 # update optimizer
            model.apply(clipper)
            # log training output
            losses.append(loss.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train batch: %d\tLoss: %.3e" % (arrow.now(), 
                    _iter / log_interval, 
                    sum(losses) / len(losses)))
                losses = []
                torch.save(model.state_dict(), "saved_models/in-sample.pt")
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
    PyTorch Module for Spatio-temporal regressive model for wind power prediction
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
        torch.nn.Module.__init__(self) 
        # configurations
        self.T, self.K = speeds.shape
        self.d         = d
        # data
        self.speeds = torch.Tensor(speeds)                                    # [ T, K ]
        self.speeds = torch.transpose(self.speeds, 0, 1)                      # [ K, T ] transpose
        self.dgraph = torch.Tensor(dgraph)
        # parameters
        self.base   = self.speeds.mean(1) / 10 + torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, 1)) # [ K ]
        self.Beta   = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3))  # [ K ]

        # non-zero entries of alpha (spatio dependences)
        n_nonzero          = len(np.where(gsupp == 1)[0])
        print(n_nonzero)
        coords             = torch.LongTensor(np.where(gsupp == 1))
        self.Alpha_nonzero = torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True))
        self.Alpha         = torch.sparse.FloatTensor(coords, 
            self.Alpha_nonzero, torch.Size([self.K, self.K])).to_dense()      # [ K, K ]
    
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
            depth  = self.d if _t >= self.d else _t
            # current time and the past 
            t      = torch.ones(depth, dtype=torch.int32) * _t        # [ d ]
            tp     = torch.arange(_t-depth, _t)                       # [ d ]
            # self-exciting effect
            kernel = self.__exp_kernel(self.Beta, t, tp, self.K)      # [ K, d ]
            Xt     = self.speeds[:, _t-depth:_t].clone()              # [ K, d ]
            graph  = self.dgraph[_t, :, :].clone()                    # [ K, K ]
            pred   = torch.mm(self.Alpha * graph, Xt * kernel).sum(1) # [ K ]
            pred   = torch.nn.functional.softplus(pred)               # [ K ]
        else:
            pred   = torch.zeros(self.K)
        return pred
        
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
        # calculate data log-likelihood
        return self._l2_loss()

    @staticmethod
    def __exp_kernel(Beta, t, tp, K):
        """
        Args:
        - Beta:  decaying rate [ K ]
        - t, tp: time index    [ t ]
        """
        delta_t = t - tp                              # [ t ]
        delta_t = delta_t.unsqueeze(0).repeat([K, 1]) # [ K, t ]
        Beta    = Beta.unsqueeze(1)                   # [ K, 1 ]
        return Beta * torch.exp(- delta_t * Beta)



if __name__ == "__main__":
    
    torch.manual_seed(12)

    # load data
    dgraph = np.load("../data/dgraph.npy")
    wind   = np.load("../data/sample_wind.npy")
    gsupp  = np.load("../data/gsupp.npy")
    speeds = wind[:, :, 1]
    # speeds[speeds > 15] == speeds.mean()
    print(speeds.shape, dgraph.shape, gsupp.shape)

    # training
    model = SpatioTemporalRegressor(speeds, dgraph, gsupp)
    train(model, niter=1000, lr=1e-6, log_interval=2)
    print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/in-sample.pt")