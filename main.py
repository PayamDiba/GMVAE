from MyModel import vae
from DataTools import getDataLoader
import argparse
import os
from DataTools import split
from utils import make_dir, process_model
from tqdm import tqdm, trange
import pandas as pd

def main():
    """
    Define flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl_mc', type=str, default=None, help='chanels of middle layers in clonal network, comma seperated integer values', required = True)
    parser.add_argument('--en_shared_c', type=str, default=None, help='chanels of encoder shared fully connected layer, comma seperated integer values', required = True)
    parser.add_argument('--de_mc', type=str, default=None, help='chanels of middle layers in nn decoder, comma seperated integer values', required = True)
    parser.add_argument('--en_pd', type=float, default=0.25, help='dropout probability | Default: 0.25', required = False)
    parser.add_argument('--ndim', type=int, default=2, help='size of the latent space | Default: 2', required = False)
    parser.add_argument('--nclone', type=int, default=None, help='number of clones. Also shows the number of mutation groups/blocks', required = True)
    parser.add_argument('--nsite', type=int, default=None, help='number of clones. Also shows the number of mutation groups/blocks', required = True)
    parser.add_argument('--lr_tree', type=float, default=None, help='learning rate', required = True)
    parser.add_argument('--lr_vae', type=float, default=None, help='learning rate', required = True)
    parser.add_argument('--init', type=str, default=None, help='Specify the type of initialization | Default: None', required = False)
    parser.add_argument('--rec_loss', type=str, default='bce', help='reconstruction loss type | Options: bce (default), mse', required = False)
    parser.add_argument('--entrW', type=float, default=1, help='relative weight of the entropy loss | Default: 1', required = False)
    parser.add_argument('--treeW', type=float, default=1, help='relative weight of the tree loss | Default: 1', required = False)
    parser.add_argument('--varW', type=float, default=1, help='relative weight of the tree loss | Default: 1', required = False)
    parser.add_argument('--i', type=str, default=None, help='path to input mutation matrix', required = True)
    parser.add_argument('--o', type=str, default=None, help='output directory', required = True)
    parser.add_argument('--restore', type=int, default=None, help='restore the model from provided epoch', required = False)
    parser.add_argument('--sr', type=float, default=0.8, help='training data split ratio | Default: 0.8', required = False)
    parser.add_argument('--seed', type=int, default=12345, help='seed for random data slplitting | Default: 12345', required = False)
    parser.add_argument('--bs', type=int, default=32, help='mini-batch size | Default: 32', required = False)
    parser.add_argument('--nw', type=int, default=2, help='Number of workers for building data loader | Default: 2', required = False)
    parser.add_argument('--epoch_vae', type=int, default=100, help='Number of training epochs. Note that if model is restored, it will know what was the last epoch. So increase this number if require more epochs after restroing | Default: 100', required = False)
    parser.add_argument('--epoch_tree', type=int, default=100, help='Number of training epochs. Note that if model is restored, it will know what was the last epoch. So increase this number if require more epochs after restroing | Default: 100', required = False)
    parser.add_argument('--save_freq', type=int, default=2, help='Saving frequency | Default: 2', required = False)
    parser.add_argument('--post_proc', type=str, default=None, help='post processing of model, specify options with comma seperated integer values | most complete analysis is 1,2,3; Default: None', required = False)
    parser.add_argument('--slow_tree', type=int, default=None, help='...', required = False)
    parser.add_argument('--slow_var', type=int, default=None, help='...', required = False)
    parser.add_argument('--slow_entr', type=int, default=None, help='...', required = False)
    parser.add_argument('--nsample', type=int, default = 1, help='Number of samples to be taken from latent space', required = False)


    FLAGS = parser.parse_args()

    """
    make directories and split data if necessary
    """
    input_dir = FLAGS.i.split('/')
    input_dir = '/'.join(input_dir[:-1])
    split_dir = input_dir + '/split_data'
    if not os.path.exists(split_dir + '/train.tab'):
        if FLAGS.restore:
            raise ValueError("trained model can be restored only when split data is available")
        else:
            make_dir(split_dir)
            split(in_dir = FLAGS.i, out_dir = split_dir, train_ratio = FLAGS.sr, seed = FLAGS.seed)

    if not FLAGS.restore:
        make_dir(FLAGS.o)
        make_dir(FLAGS.o + '/checkpoints')

    """
    define model
    """
    model = vae(FLAGS)
    if FLAGS.restore:
        path_checkpoint = FLAGS.o + '/checkpoints/checkpoint_' + str(FLAGS.restore) + '.tar'
        model.load(path_checkpoint)

    """
    prepare data
    """
    trainLoader = getDataLoader(data_dir = split_dir, split = 'train', batch_size = FLAGS.bs, n_workers = FLAGS.nw, shuffle = True)
    testLoader = getDataLoader(data_dir = split_dir, split = 'test', batch_size = FLAGS.bs, n_workers = FLAGS.nw, shuffle = False)

    """
    train the model
    """
    processor = process_model()


    for epoch in range(model.nIter_, FLAGS.epoch_vae + FLAGS.epoch_tree):
        pbar = tqdm(trainLoader, 'epoch: {}'.format(epoch), total=len(trainLoader), position=0, leave=True)
        for input in pbar:
            input = input.float()
            bs = input.shape[0]
            var_loss, entropy_loss, recon_loss_nn, recon_loss_tree, tree_loss, tree = model.train_step(input) # recon_loss is already in "sum" reduction
            processor.trackTraining(var_loss, recon_loss_nn, recon_loss_tree, entropy_loss, bs)
            pbar.set_postfix_str(f'{recon_loss_nn:.3} recon loss nn, {recon_loss_tree:.3} recon loss tree, {var_loss:.3} var loss, {entropy_loss:.3} entropy, {tree_loss:.3} tree loss',refresh=False)

        model.nIter_ += 1
        for input in testLoader:
            input = input.float()
            bs = input.shape[0]
            _, _, _, _, recon_loss_nn, recon_loss_tree, entropy_loss, var_loss = model.evaluate(input) # recon_loss is already in "sum" reduction
            processor.trackTesting(var_loss.cpu().item(), recon_loss_nn, recon_loss_tree, entropy_loss.cpu().item(), bs)
        processor.update(epoch, tree.detach().cpu().numpy())

        """
        Save model
        """
        if (epoch+1)%FLAGS.save_freq == 0:
            model.save(FLAGS.o + '/checkpoints')

    if FLAGS.post_proc:
        proc = [int(i) for i in FLAGS.post_proc.split(',')]
        trainLoader = getDataLoader(data_dir = split_dir, split = 'train', batch_size = FLAGS.bs, n_workers = FLAGS.nw, shuffle = False)
        processor.post_proc(model, trainLoader, testLoader, proc)
    processor.save(FLAGS.o + '/results.dict')


if __name__ == "__main__":
    main()
