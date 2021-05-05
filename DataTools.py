import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def split(in_dir, out_dir, train_ratio = 0.8, seed = 12345):

    df = pd.read_csv(in_dir, header = 0, index_col = 0, sep = '\t')
    train_df = df.sample(frac = train_ratio, random_state = seed)
    test_df = df.drop(train_df.index)

    train_df.to_csv(out_dir + '/train.tab', header = True, index = True, sep = '\t')
    test_df.to_csv(out_dir + '/test.tab', header = True, index = True, sep = '\t')


def getDataLoader(data_dir, split, batch_size, n_workers, shuffle):
    if split == 'train':
        dataset = MutationDataset(data_dir + '/train.tab')
    elif split == 'test':
        dataset = MutationDataset(data_dir + '/test.tab')
    else:
        raise ValueError('requested split {} is not implemented'.format(split))

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = n_workers)


class MutationDataset(Dataset):

    def __init__(self, data):
        self.df_ = pd.read_csv(data, header = 0, index_col = 0, sep = '\t')

    def __len__(self):
        return self.df_.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ret = self.df_.iloc[idx].values
        #ret = np.expand_dims(ret, axis = 0)
        return ret
