import torch
import torch.nn.functional as F
import os
import pickle

def metric_bce(pred, gt):
    return F.binary_cross_entropy_with_logits(pred, gt, reduction='sum')

def metric_mse(pred, gt):
    return F.mse_loss(pred, gt, reduction='sum')

def metric_entropy(p):
    entropy = -torch.sum(p * torch.log(p + 1e-7), dim = -1)
    return entropy.mean()

def metric_varKL(mean, log_var):
    klD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return klD

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class runningMean(object):
    def __init__(self,index):
        self.index_ = index
        self._reset()

    def append(self, dict, size):
        self.n_ += size
        for n in dict.keys():
            self.data_[n] += dict[n]

    def aggregate(self):
        ret = {}
        for n in self.data_.keys():
            ret[n] = self.data_[n] / self.n_

        self._reset()
        return ret

    def _reset(self):
        self.data_ = {}
        self.n_ = 0
        for n in self.index_:
            self.data_[n] = 0

class process_model(object):
    def __init__(self):
        losses = ['var','recon_nn', 'recon_tree','entropy']
        self.trainManager_ = runningMean(losses)
        self.testManager_ = runningMean(losses)
        self.info = {'epoch' : [],
                     'train_var_loss' : [],
                     'train_recon_loss_nn' : [],
                     'train_recon_loss_tree' : [],
                     'train_entropy_loss' : [],
                     'test_recon_loss_nn': [],
                     'test_recon_loss_tree': [],
                     'tree' : []}

        self.train_latent = None
        self.train_p_clone = None
        self.train_recon_nn = None
        self.train_recon_tree = None
        self.test_latent = None
        self.test_p_clone = None
        self.test_recon_nn = None
        self.test_recon_tree = None
        self.clonal_mut = None

    def trackTraining(self, var_loss, recon_loss_nn, recon_loss_tree, entropy, size):
        dict = {'var': var_loss,
                'recon_nn': recon_loss_nn,
                'recon_tree': recon_loss_tree,
                'entropy': entropy}
        self.trainManager_.append(dict, size)

    def trackTesting(self, var_loss, recon_loss_nn, recon_loss_tree, entropy, size):
        dict = {'var': var_loss,
                'recon_nn': recon_loss_nn,
                'recon_tree': recon_loss_tree,
                'entropy': entropy}
        self.testManager_.append(dict, size)

    def update(self, epoch, tree):
        train_loss = self.trainManager_.aggregate()
        test_loss = self.testManager_.aggregate()
        self.info['epoch'].append(epoch)
        self.info['train_var_loss'].append(train_loss['var'])
        self.info['train_recon_loss_nn'].append(train_loss['recon_nn'])
        self.info['train_recon_loss_tree'].append(train_loss['recon_tree'])
        self.info['train_entropy_loss'].append(train_loss['entropy'])
        self.info['test_recon_loss_nn'].append(test_loss['recon_nn'])
        self.info['test_recon_loss_tree'].append(test_loss['recon_tree'])
        self.info['tree'].append(tree)

    def post_proc(self, model, trainLoader, testLoader, proc):
        """
        proc is a list of integers, it showing a seperate processing:
        1: reconstrcuted mutation, clonal probability, and latent dim for train data
        2: reconstrcuted mutation, clonal probability, and latent dim for test data
        3: obtain clonal mutations
        """

        for p in proc:
            if p == 1:
                self.train_latent, self.train_p_clone, self.train_recon_nn, self.train_recon_tree = self._reconstrcut(model, trainLoader)
            elif p == 2:
                self.test_latent, self.test_p_clone, self.test_recon_nn, self.test_recon_tree = self._reconstrcut(model, testLoader)
            elif p == 3:
                self.clonal_mut = self._clonal_mut(model)

        self.info['train_latent'] = self.train_latent
        self.info['train_p_clone'] = self.train_p_clone
        self.info['train_recon_nn'] = self.train_recon_nn
        self.info['train_recon_tree'] = self.train_recon_tree
        self.info['test_latent'] = self.test_latent
        self.info['test_p_clone'] = self.test_p_clone
        self.info['test_recon_nn'] = self.test_recon_nn
        self.info['test_recon_tree'] = self.test_recon_tree
        self.info['clonal_mut'] = self.clonal_mut


    def _reconstrcut(self, model, dataloader):
        latent_mean = []
        p_clones = []
        recon_nn = []
        recon_tree = []
        for input in dataloader:
            input = input.float()
            pc, clonal_mean, rec_nn, rec_tree, _, _, _, _ = model.evaluate(input)
            latent_mean.append(clonal_mean)
            p_clones.append(pc)
            recon_nn.append(rec_nn)
            recon_tree.append(rec_tree)

        latent_mean = torch.cat(latent_mean, dim = 0).cpu().numpy()
        p_clones = torch.cat(p_clones, dim = 0).cpu().numpy()
        recon_nn = torch.cat(recon_nn, dim = 0).cpu().numpy()
        recon_tree = torch.cat(recon_tree, dim = 0).cpu().numpy()

        return latent_mean, p_clones, recon_nn, recon_tree

    def _clonal_mut(self, model):
        single_mut = []
        n_clone = model.n_clone_
        for i in range(n_clone):
            curr = torch.zeros(n_clone)
            curr[i] = 1
            single_mut.append(curr)

        single_mut = torch.stack(single_mut, dim = 0).to(model.device_)
        input = torch.cat([single_mut, torch.round(model.tree_decoder_.tree)], dim = 0)
        clonal_mut = model.get_group_mut(input)
        return clonal_mut.cpu().numpy()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.info, f, protocol=pickle.HIGHEST_PROTOCOL)

def init_weights(model, type = 'normal', scale = 0.02):
    """
    Initializes the neural network weights.
    The function was taken from :
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def initilizer(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, scale)
            elif type == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight.data, gain = scale)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scale)
            torch.nn.init.constant_(m.bias.data, 0.0)

    model.apply(initilizer)
