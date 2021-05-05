"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from Encoder import encoder
from Decoder import nn_decoder, tree_decoder
from ClonalP import clonalP
from torch.distributions.multivariate_normal import MultivariateNormal
import itertools
from utils import metric_bce, metric_mse, metric_varKL, metric_entropy
from utils import init_weights
import numpy as np

class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(min = 0)
            module.weight.data=w

class vae(object):

    def __init__(self, flags):
        cl_mc = [int(i) for i in flags.cl_mc.split(',')]
        en_shared_c = [int(i) for i in flags.en_shared_c.split(',')]
        de_mc = [int(i) for i in flags.de_mc.split(',')]

        self.nIter_ = 0
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clonalP_ = clonalP(n_site = flags.nsite,
                                n_clone = flags.nclone,
                                fc_midChannels = cl_mc)

        self.encoder_ = encoder(n_site = flags.nsite,
                                n_clone = flags.nclone,
                                shared_fc_chans = en_shared_c,
                                p_drop = flags.en_pd,
                                device = self.device_,
                                n_dim = flags.ndim)

        self.nn_decoder_ = nn_decoder(n_dim = flags.ndim,
                                      n_site = flags.nsite,
                                      fc_mid_channels = de_mc)

        self.tree_decoder_ = tree_decoder(n_clone = flags.nclone,
                                          n_site = flags.nsite,
                                          device = self.device_)

        self.clonalP_.to(self.device_)
        self.encoder_.to(self.device_)
        self.nn_decoder_.to(self.device_)
        self.tree_decoder_.to(self.device_)
        self.recon_loss_tree = 0.0
        self.tree_loss = 0.0

        if flags.init:
            init_weights(self.clonalP_, flags.init, 0.02)
            init_weights(self.encoder_, flags.init, 0.02)
            init_weights(self.nn_decoder_, flags.init, 0.02)
            init_weights(self.tree_decoder_, flags.init, 0.02)

        self.optim_vae_ = torch.optim.Adam(itertools.chain(self.encoder_.parameters(),
                                                           self.nn_decoder_.parameters(),
                                                           self.clonalP_.parameters()), lr = flags.lr_vae)
        self.optim_tree_ = torch.optim.Adam(self.tree_decoder_.parameters(), lr = flags.lr_tree)
        self.flags_ = flags
        self.n_clone_ = flags.nclone

    def _forward_vae(self, input):

        p_clones = self.clonalP_(input)
        clonal_mean, clonal_log_var = self.encoder_(input)
        clonal_z = self._sample_z(clonal_mean, clonal_log_var, self.flags_.nsample) #nBatch * nClone * nSample * nDim
        clonal_z = clonal_z.reshape(-1,self.flags_.ndim)
        decoded = self.nn_decoder_(clonal_z)
        decoded = decoded.reshape(-1, self.flags_.nclone, self.flags_.nsample, self.flags_.nsite)
        decoded = decoded.mean(dim = 2)
        recon_nn = 0
        for cl in range(self.n_clone_):
            recon_nn += p_clones[:,cl][:,None] * decoded[:,cl,:]

        #input_tree_decoder = p_clones.detach().clone()
        #recon_tree = self.tree_decoder_(input_tree_decoder)
        return clonal_mean, clonal_log_var, p_clones, recon_nn#, recon_tree

    def _forward_tree(self, input):
        recon_tree = self.tree_decoder_(input)
        return recon_tree

    def _sample_z(self, mean, log_var, n_sample):

        clonal_samples = []
        for cl in range(self.n_clone_):
            curr_mean = mean[:,cl,:]
            curr_log_var = log_var[:,cl,:]
            curr_mean = curr_mean.squeeze(dim = 1)
            curr_log_var = curr_log_var.squeeze(dim = 1)
            curr_std = torch.exp(0.5*curr_log_var) # standard deviation
            normal_dist = MultivariateNormal(torch.zeros(self.flags_.ndim),torch.eye(self.flags_.ndim))

            curr_samples = []
            for _ in range(n_sample):
                eps = normal_dist.sample((curr_std.shape[0],)).to(self.device_)
                z = curr_mean + (eps * curr_std)
                curr_samples.append(z)

            curr_samples = torch.stack(curr_samples, dim = 1) #nBatch * nSample * nDim
            clonal_samples.append(curr_samples)

        return torch.stack(clonal_samples, dim = 1) #nBatch * nClone * nSample * nDim

    def _set_lossW(self):

        if self.flags_.slow_var:
            self.varW = min(1, 1 * self.nIter_/self.flags_.epoch_vae * self.flags_.varW)
        else:
            self.varW = self.flags_.varW

        if self.flags_.slow_entr:
            self.entrW = min(1, 1 * self.nIter_/self.flags_.epoch_vae * self.flags_.entrW)
        else:
            self.entrW = self.flags_.entrW

        if self.flags_.slow_tree:
            self.treeW = (self.nIter_ - self.flags_.epoch_vae)/self.flags_.epoch_tree * self.flags_.treeW

    def _var_loss(self, mean, log_var, p_clones):

        klD = 0
        for cl in range(self.n_clone_):
            curr_mean = mean[:,cl,:]
            curr_var = log_var[:,cl,:]
            curr_mean = curr_mean.squeeze(dim = 1)
            curr_var = curr_var.squeeze(dim = 1)
            klD += -0.5 * torch.sum(p_clones[:,cl][:,None]*(1 + curr_var - curr_mean.pow(2) - curr_var.exp()))
        return klD


    def _recon_loss(self, recon_mut, true_mut, type = 'bce'):
        if type == 'bce':
            recL = nn.BCEWithLogitsLoss(reduction='sum')
        elif type == 'mse':
            recL = nn.MSELoss(reduction='sum')
        return recL(recon_mut, true_mut)


    def _entropic_loss(self, p_clones):

        p_sums = torch.sum(p_clones, dim = -1)
        assert torch.allclose(p_sums, torch.ones_like(p_sums)), p_sums
        entropy = -torch.sum((p_clones) * torch.log(p_clones + 1e-7), dim = -1)
        return entropy.sum()

    def _tree_loss(self):
        ret = torch.pow(self.tree_decoder_.tree_pre_mask - 10, 2)
        ret = -1 * torch.sum(ret, (0,1))
        return ret


    def _total_loss_vae(self, clonal_mean, clonal_log_var, p_clones, recon_nn, true_mut):

        var_loss = self._var_loss(clonal_mean, clonal_log_var, p_clones)
        recon_loss = self._recon_loss(recon_nn, true_mut, type = self.flags_.rec_loss)
        entropy_loss = self._entropic_loss(p_clones)

        self.var_loss = var_loss.item()
        self.recon_loss_nn = recon_loss.item()
        self.entropy_loss = entropy_loss.item()

        l2_reg = 0
        for name, param in self.encoder_.named_parameters():
            l2_reg += torch.norm(param)


        return self.varW * var_loss + recon_loss + self.entrW * entropy_loss + 0*l2_reg

    def _total_loss_tree(self, recon_tree, true_mut):
        # true_mut here can be recon_nn
        recon_loss = self._recon_loss(recon_tree, true_mut, type = self.flags_.rec_loss)
        tree_loss = self._tree_loss()

        self.recon_loss_tree = recon_loss.item()
        self.tree_loss = tree_loss.item()

        l1_reg = 0
        for name, param in self.tree_decoder_.named_parameters():
            if 'tree' in name:
                l1_reg += torch.norm(param, p = 1)

        return recon_loss + self.treeW * tree_loss + 0.1 * l1_reg


    def _backwardNN(self, clonal_mean, clonal_log_var, p_clones, recon_nn, true_mut):

        self.optim_vae_.zero_grad()
        loss = self._total_loss_vae(clonal_mean, clonal_log_var, p_clones, recon_nn, true_mut)
        loss.backward()
        self.optim_vae_.step()

    def _backwardTree(self,recon_tree, true_mut):

        self.optim_tree_.zero_grad()
        loss = self._total_loss_tree(recon_tree, true_mut)
        loss.backward()
        self.optim_tree_.step()

    def _freeze(self, model = 'vae'):
        if model == 'tree':
            for p in self.encoder_.parameters():
                p.requires_grad_(True)
            for p in self.nn_decoder_.parameters():
                p.requires_grad_(True)
            for p in self.clonalP_.parameters():
                p.requires_grad_(True)
            for p in self.tree_decoder_.parameters():
                p.requires_grad_(False)

        elif model == 'vae':
            for p in self.encoder_.parameters():
                p.requires_grad_(False)
            for p in self.nn_decoder_.parameters():
                p.requires_grad_(False)
            for p in self.clonalP_.parameters():
                p.requires_grad_(False)
            for p in self.tree_decoder_.parameters():
                p.requires_grad_(True)

    def train_step(self, input):

        self._set_lossW()
        input = input.to(self.device_)
        self.clonalP_.train()
        self.encoder_.train()
        self.nn_decoder_.train()
        self.tree_decoder_.train()

        if self.nIter_ < self.flags_.epoch_vae:
            self._freeze('tree')
            clonal_mean, clonal_log_var, p_clones, recon_nn = self._forward_vae(input)
            self._backwardNN(clonal_mean = clonal_mean,
                             clonal_log_var = clonal_log_var,
                             p_clones = p_clones,
                             recon_nn = recon_nn,
                             true_mut = input)



        else:
            constraints=weightConstraint()
            self.tree_decoder_._modules['fc_'].apply(constraints)

            clonal_mean, clonal_log_var, p_clones, recon_nn = self._forward_vae(input)
            self._freeze('vae')
            true_mut_tree = recon_nn.detach().clone()
            input_tree = p_clones.detach().clone()
            input_tree = torch.argmax(input_tree, axis = 1)
            input_tree = torch.nn.functional.one_hot(input_tree, num_classes = self.n_clone_)
            input_tree = input_tree.float()
            input_tree = input_tree.to(self.device_)
            recon_tree = self._forward_tree(input_tree)
            self._backwardTree(recon_tree = recon_tree,
                               true_mut = true_mut_tree)


        return self.var_loss, self.entropy_loss, self.recon_loss_nn, self.recon_loss_tree, self.tree_loss, self.tree_decoder_.tree

    def evaluate(self, input):

        input = input.to(self.device_)
        with torch.no_grad():
            self.clonalP_.eval()
            self.encoder_.eval()
            self.nn_decoder_.eval()
            self.tree_decoder_.eval()

            clonal_mean, clonal_log_var, p_clones, recon_nn = self._forward_vae(input)

            input_tree = p_clones.detach().clone()
            recon_tree = self._forward_tree(input_tree)

            if self.flags_.rec_loss == 'bce':
                recon_loss_nn = metric_bce(recon_nn, input)
                recon_loss_tree = metric_bce(recon_tree, recon_nn.detach().clone())
            elif self.flags_.rec_loss == 'mse':
                recon_loss_nn = metric_mse(recon_nn, input)
                recon_loss_tree = metric_mse(recon_tree, recon_nn.detach().clone())
            entropy_loss = metric_entropy(p_clones)
            var_loss = self._var_loss(clonal_mean, clonal_log_var, p_clones)

        return p_clones, clonal_mean, recon_nn, recon_tree, recon_loss_nn, recon_loss_tree, entropy_loss, var_loss

    def get_group_mut(self, input):
        """
        evaluates the output of fc2 layer in tree decoder
        """
        input = input.to(self.device_)
        with torch.no_grad():
            self.tree_decoder_.eval()
            layer = self.tree_decoder_.fc_
            ret = layer(input)
            ret = ret.detach()
        return ret

    def save(self, path):
        torch.save({'nIter': self.nIter_,
                    'encoder_state_dict': self.encoder_.state_dict(),
                    'nn_decoder_state_dict': self.nn_decoder_.state_dict(),
                    'tree_decoder_state_dict': self.tree_decoder_.state_dict(),
                    'clonalP_state_dict': self.clonalP_.state_dict(),
                    'optimizer_vae_state_dict': self.optim_vae_.state_dict(),
                    'optimizer_tree_state_dict': self.optim_tree_.state_dict(),
                    }, path + '/checkpoint_' + str(self.nIter_) + '.tar')

    def load(self, path):
        checkpoint = torch.load(path)
        self.encoder_.load_state_dict(checkpoint['encoder_state_dict'])
        self.nn_decoder_.load_state_dict(checkpoint['nn_decoder_state_dict'])
        self.tree_decoder_.load_state_dict(checkpoint['tree_decoder_state_dict'])
        self.clonalP_.load_state_dict(checkpoint['clonalP_state_dict'])
        self.optim_vae_.load_state_dict(checkpoint['optimizer_vae_state_dict'])
        self.optim_tree_.load_state_dict(checkpoint['optimizer_tree_state_dict'])
        self.nIter_ = checkpoint['nIter']
