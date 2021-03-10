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
            optimizer.zero_grad() # init optimizer (set gradient to be zero)
            loss, _, _ = model()
            # objective function
            # loss.backward(retain_graph=True) # gradient descent
            loss.backward()       # gradient descent
            optimizer.step()      # update optimizer
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
            depth  = self.d if _t >= self.d else _t
            graph  = self.dgraph[_t-depth:_t, :, :].clone()                           # [ d, K, K ]
            A      = self.Alpha.unsqueeze(0).repeat(depth, 1, 1) * graph              # [ d, K, K ]

            # kernel = self.__exp_kernel(self.Beta, _t, depth, self.K)                  # [ K, d ]
            # kernel = torch.transpose(kernel, 0, 1).unsqueeze(-1).repeat(1, 1, self.K) # [ d, K, K ]
            kernel = self.__exp_kernel(self.Beta, _t, depth, self.K)                  # [ d, K, K ]

            Xt     = self.speeds[:, _t-depth:_t].clone()                              # [ K, d ]
            Xt     = torch.transpose(Xt, 0, 1).unsqueeze(-1).repeat(1, 1, self.K)     # [ d, K, K ]
            pred   = (A * Xt * kernel).sum(0).sum(0)                                  # [ K ]
            pred   = torch.nn.functional.softplus(pred)                               # [ K ]
        else:
            pred   = torch.zeros(self.K)
        return pred
        
    def _l2_loss(self):
        """
        L2 loss
        """
        # convert sparse alpha representation to dense matrix
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
    def __exp_kernel(beta, _t, depth, K):
        """
        Args:
        - beta:  decaying rate [ K ]
        - _t:    time index    scalar
        - depth: time depth    scalar
        """
        # # current time and the past 
        # t       = torch.ones(depth, dtype=torch.int32) * _t # [ d ]
        # tp      = torch.arange(_t-depth, _t)                # [ d ]
        # delta_t = t - tp                                    # [ d ]
        # delta_t = delta_t.unsqueeze(0).repeat([K, 1])       # [ K, d ]
        # beta    = beta.unsqueeze(1)                         # [ K, 1 ]
        # return beta * torch.exp(- delta_t * beta)

        # current time and the past 
        t       = torch.ones(depth, dtype=torch.int32) * _t             # [ d ]
        tp      = torch.arange(_t-depth, _t)                            # [ d ]
        delta_t = t - tp                                                # [ d ]
        delta_t = delta_t.unsqueeze(-1).unsqueeze(-1).repeat([1, K, K]) # [ d, K, K ]
        beta    = beta.unsqueeze(1).unsqueeze(0)                        # [ 1, K, 1 ]
        return beta * torch.exp(- delta_t * beta)
        


class SpatioTemporalDelayedRegressor(SpatioTemporalRegressor):
    """
    PyTorch module for spatio-temporal delayed regressive model for wind power prediction
    """
    # def __init__(self, speeds, dgraph, gsupp, muG, d=20):
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
        SpatioTemporalRegressor.__init__(self, speeds, dgraph, gsupp, d)
        # mean for Gaussian delayed kernel
        self.muG                = 10 # torch.Tensor(muG) # [ T, K, K ]
        # non-zero entries of sigma (variance for gaussian kernel)
        self.Sigma_nonzero      = torch.nn.Parameter(torch.Tensor(self.n_nonzero).uniform_(10, 20), requires_grad=True)
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
            depth  = self.d if _t >= self.d else _t
            graph  = self.dgraph[_t-depth:_t, :, :].clone()                           # [ d, K, K ]
            A      = self.Alpha.unsqueeze(0).repeat(depth, 1, 1) * graph              # [ d, K, K ]
            # mu     = self.muG[_t-depth:_t, :, :].clone()                              # [ d, K, K ]
            # kernel = self.__trunc_gaussian_kernel(mu, self.Sigma, _t, depth, self.K)  # [ d, K, K ]
            kernel = self.__trunc_gaussian_kernel(self.muG, self.Sigma, _t, depth, self.K)  # [ d, K, K ]
            Xt     = self.speeds[:, _t-depth:_t].clone()                              # [ K, d ]
            Xt     = torch.transpose(Xt, 0, 1).unsqueeze(-1).repeat(1, 1, self.K)     # [ d, K, K ]
            pred   = (A * Xt * kernel).sum(0).sum(0)                                  # [ K ]
            pred   = torch.nn.functional.softplus(pred)                               # [ K ]
        else:
            pred   = torch.zeros(self.K)
        return pred

    def _update_l2_loss(self):
        """
        L2 loss
        """
        # convert sparse alpha representation to dense matrix
        self.Sigma = torch.sparse.FloatTensor(
            self.coords, self.Sigma_nonzero,
            torch.Size([self.K, self.K])).to_dense() # [ K, K ]
        # calculate l2 loss
        return self._l2_loss()
    
    def forward(self):
        """
        customized forward function
        """
        # calculate l2 loss and prediction
        return self._update_l2_loss()

    @staticmethod
    def __trunc_gaussian_kernel(Mu, Sigma, _t, depth, K):
        """
        Args: 
        - _t:      current time index             scalar
        - tau:    past time index                 scalar
        - mu_mat: mean matrix for gaussian kernel [ K, K ]
        - sigma:  sigma for gaussian kernel
        """
        # current time and the past 
        t       = torch.ones(depth, dtype=torch.int32) * _t             # [ d ]
        tp      = torch.arange(_t-depth, _t)                            # [ d ]
        delta_t = t - tp                                                # [ d ]
        delta_t = delta_t.unsqueeze(-1).unsqueeze(-1).repeat([1, K, K]) # [ d, K, K ]
        Sigma   = Sigma.unsqueeze(0).repeat(depth, 1, 1)                # [ d, K, K ]
        return (1 / (np.sqrt(2 * np.pi) * Sigma)) * \
            torch.exp((-1/2) * torch.square((delta_t - Mu)/Sigma))




if __name__ == "__main__":
    
    torch.manual_seed(12)

    N = 4
    d = 10

    # load data matrices
    dgraph, speeds, gsupp, locs = utils.dataloader(rootpath="../data/data_k50", N=N)
    T, K  = speeds.shape
    
    # training
    model = SpatioTemporalRegressor(speeds, dgraph, gsupp, d=d)
    # train(model, niter=5000, lr=1e-2, log_interval=500, modelname="in-sample-exp-kernel-k50-t%d-d%d" % (N, d))
    model.load_state_dict(torch.load("saved_models/in-sample-exp-kernel-k50-t%d-d%d.pt" % (N, d)))

    # visualization
    from plotpred import pred_linechart, mae_map, mae_heatmap
    
    _, pred0, pred1 = model()
    preds = (pred0 + pred1).detach().numpy().transpose()

    # for i in range(K):
    #     pred_linechart(preds[:, i], speeds[:, i], filename="Turbine %d" % i)
    # pred_linechart(preds.mean(1), speeds.mean(1), filename="Average")
    mae_heatmap(preds, speeds, filename="in-sample MAE heatmap")
    # mae_map(preds, speeds, locs, filename="insample MAE map")