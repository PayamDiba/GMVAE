"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class nn_decoder(nn.Module):

    def __init__(self, n_dim, n_site, fc_mid_channels):

        super(nn_decoder,self).__init__()
        self.fc_ = self._fc_layer(n_dim, n_site, fc_mid_channels)


    def _fc_layer(self, n_dim, n_site, mid_channels):

        layers = []
        in_channels = n_dim
        for i,mc in enumerate(mid_channels):
            layers += [nn.Linear(in_channels, mc)]
            layers += [nn.LayerNorm(mc)]
            layers += [nn.ReLU()]
            in_channels = mc

            if (i+1) % 2 == 0:
                layers += [nn.Dropout(p = 0)]

        layers += [nn.Linear(in_channels, n_site)]
        #layers += [nn.Sigmoid()]

        return nn.Sequential(*layers)


    def forward(self,x):

        recon = self.fc_(x)
        return recon

class tree_decoder(nn.Module):

    def __init__(self, n_clone, n_site, device):

        super(tree_decoder,self).__init__()
        self.tree_pre_mask, self.mask_ = self._tree_mat(n_clone)
        self.fc_ = self._fc_layer(n_clone, n_site)
        self.device_ = device
        self.mask_ = self.mask_.to(self.device_)
        self.n_clone_ = n_clone
        self.tree = torch.zeros_like(self.tree_pre_mask).to(device)

    def _tree_mat(self, n_clone):
        tree = torch.normal(10, 1, size=(n_clone, n_clone), requires_grad = True)
        tree = nn.parameter.Parameter(tree, requires_grad = True)
        mask = torch.ones(size = (n_clone, n_clone), requires_grad = False)
        mask = torch.tril(mask, diagonal = -1)

        return tree, mask

    def _fc_layer(self, n_clone, n_site):

        layers = []
        layers += [nn.Linear(n_clone, n_site)]
        layers += [nn.BatchNorm1d(n_site)]
        layers += [nn.Sigmoid()]

        return nn.Sequential(*layers)


    def forward(self,x):

        ones = torch.eye(self.n_clone_, requires_grad = False).to(self.device_)
        self.tree = nn.functional.sigmoid(self.tree_pre_mask - 10) * self.mask_
        self.tree = self.tree + ones

        input_fc = torch.matmul(x, self.tree)
        recon = self.fc_(input_fc)

        return recon
