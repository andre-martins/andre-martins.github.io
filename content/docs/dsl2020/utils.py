import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def read_data(filepath, partitions=None):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    x_seq = []
    y_seq = []
    X = []
    y = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\t\n")
            fields = line.split("\t")
            letter = fields[1]
            last = int(fields[2]) == -1
            if letter in labels:
                k = labels[letter]
            else:
                k = len(labels)
                labels[letter] = k
            partition = int(fields[5])
            if partitions is not None and partition not in partitions:
                continue
            x = np.array([float(v) for v in fields[6:]])
            x_seq.append(x)
            y_seq.append(k)
            if last:
                X.append(x_seq)
                y.append(y_seq)
                x_seq = []
                y_seq = []
    ll = ["" for k in labels]
    for letter in labels:
        ll[labels[letter]] = letter
    return X, y, ll


def pairwise_features(x_i):
    """
    x_i (n_features)
    """
    feat_size = x_i.shape[0]
    ix = np.triu_indices(feat_size)
    return np.array(np.outer(x_i, x_i)[ix])


def collate_samples(samples):
    """This function will glue a list of tensors in a single batch tensor
    using padding for samples smaller than max_seq_length.
    Padding values are equal to 26 and padding indexes are equal to -1.
    """
    pad_index = 26
    pad_value = -1
    batch_size = len(samples)
    max_seq_length = max([x.shape[0] for x, y in samples])
    input_shape = samples[0][0].shape[1:]
    X_shape = (batch_size, max_seq_length, *input_shape)
    y_shape = (batch_size, max_seq_length)
    X = torch.zeros(X_shape, dtype=torch.float).fill_(pad_value)
    Y = torch.zeros(y_shape, dtype=torch.long).fill_(pad_index)
    for i, (x, y) in enumerate(samples):
        seq_len = x.shape[0]
        X[i, :seq_len] = x
        Y[i, :seq_len] = y
    return X, Y


class OCRDataset(Dataset):
    def __init__(self, path, part='train', train_labels=None, feature_function=None):
        if part == 'train':
            self.X, self.y, self.labels = read_data(path, partitions=set(range(8)))
        elif part == 'dev':
            self.X, self.y, _ = read_data(path, partitions={8})
            self.labels = train_labels
        elif part == 'test':
            self.X, self.y, _ = read_data(path, partitions={9})
            self.labels = train_labels
        self.feature_function = feature_function

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.feature_function is not None:
            X = [self.feature_function(x) for x in self.X[idx]]
        else:
            X = self.X[idx]
        return torch.tensor(X).float(), torch.tensor(self.y[idx]).long()


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('{}.pdf'.format(name), bbox_inches='tight')
