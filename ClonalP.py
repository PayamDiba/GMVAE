"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class clonalP(nn.Module):

    def __init__(self, n_site, n_clone, fc_midChannels):

        super(clonalP,self).__init__()
        self.n_site_ = n_site
        self.n_clone_ = n_clone
        self.fc_ = self._fc_layers(fc_midChannels)

    def _fc_layers(self, mid_channels):

        layers = []
        in_channels = self.n_site_
        for i,mc in enumerate(mid_channels):
            layers += [nn.Linear(in_channels, mc)]
            layers += [nn.BatchNorm1d(mc)]
            layers += [nn.PReLU(mc)]
            in_channels = mc

        layers += [nn.Linear(in_channels, self.n_clone_)]
        layers += [nn.BatchNorm1d(self.n_clone_)]
        layers += [nn.Softmax(dim = -1)] #make it a probability vector
        return nn.Sequential(*layers)


    def forward(self,x):
        p = self.fc_(x)
        return p
