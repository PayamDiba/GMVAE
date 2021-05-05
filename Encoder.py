"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict


class encoder(nn.Module):

    def __init__(self, n_site, n_clone, shared_fc_chans, p_drop, device, n_dim = 2):

        super(encoder,self).__init__()
        self.shared_mean_, self.shared_var_ = self._fc_shared(n_site, shared_fc_chans, p_drop)
        self.clonal_mean_, self.clonal_var_ = self._fc_clonal(shared_fc_chans[-1], n_clone, n_dim)
        self.n_clone_ = n_clone

    def _fc_shared(self, n_site, channels, p_drop):
        mean_layers = []
        var_layers = []
        in_channels = n_site
        for i,c in enumerate(channels):
            mean_layers += [nn.Linear(in_channels, c)]
            mean_layers += [nn.BatchNorm1d(c)]
            mean_layers += [nn.PReLU(c)]

            var_layers += [nn.Linear(in_channels, c)]
            var_layers += [nn.BatchNorm1d(c)]
            var_layers += [nn.PReLU(c)]
            in_channels = c

            if (i+1) % 2 == 1:
                mean_layers += [nn.Dropout(p = p_drop)]
                var_layers += [nn.Dropout(p = p_drop)]

        return nn.Sequential(*mean_layers), nn.Sequential(*var_layers)


    def _fc_clonal(self, in_channels, n_clone, n_dim):

        mean_layers = []
        var_layers = []

        for clone in range(n_clone):
            mean_layers.append(('mean_clone_1_{}'.format(clone), nn.Linear(in_channels, in_channels//4)))
            mean_layers.append(('mean_clone_2_{}'.format(clone), nn.BatchNorm1d(in_channels//4)))
            mean_layers.append(('mean_clone_3_{}'.format(clone), nn.PReLU(in_channels//4)))
            mean_layers.append(('mean_clone_4_{}'.format(clone), nn.Linear(in_channels//4, n_dim)))

            var_layers.append(('var_clone_1_{}'.format(clone), nn.Linear(in_channels, in_channels//4)))
            var_layers.append(('var_clone_2_{}'.format(clone), nn.BatchNorm1d(in_channels//4)))
            var_layers.append(('var_clone_3_{}'.format(clone), nn.PReLU(in_channels//4)))
            var_layers.append(('var_clone_4_{}'.format(clone), nn.Linear(in_channels//4, n_dim)))

        mean_layers = nn.Sequential(OrderedDict(mean_layers))
        var_layers = nn.Sequential(OrderedDict(var_layers))

        return mean_layers, var_layers


    def forward(self,x):

        shared_mean = self.shared_mean_(x)
        shared_var = self.shared_var_(x)
        clonal_mean = []
        clonal_var = []
        for c in range(self.n_clone_):
            tmp_mean = shared_mean
            tmp_var = shared_var
            for l in range(1,4):
                tmp_mean = self.clonal_mean_._modules["mean_clone_{}_{}".format(l,c)](tmp_mean)
                tmp_var = self.clonal_var_._modules["var_clone_{}_{}".format(l,c)](tmp_var)

            clonal_mean.append(self.clonal_mean_._modules["mean_clone_4_{}".format(c)](tmp_mean))
            clonal_var.append(self.clonal_var_._modules["var_clone_4_{}".format(c)](tmp_var))
            #clonal_mean.append(self.clonal_mean_._modules["mean_clone_1_{}".format(c)](tmp_mean))
            #clonal_var.append(self.clonal_var_._modules["var_clone_1_{}".format(c)](tmp_var))

        clonal_mean = torch.stack(clonal_mean, dim = 1) #nBatch * nClone * ndim
        clonal_var = torch.stack(clonal_var, dim = 1) #nBatch * nClone * ndim
        return clonal_mean, clonal_var
