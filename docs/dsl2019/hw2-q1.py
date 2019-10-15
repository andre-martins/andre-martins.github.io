#!/usr/bin/env python

import argparse
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt


class OCRDataset(Dataset):
    """Binary OCR dataset."""

    def __init__(self, path, dev_fold=8, test_fold=9):
        """
        path: location of OCR data
        """
        label_counter = count()
        labels = defaultdict(lambda: next(label_counter))
        X = []
        y = []
        fold = []
        with open(path) as f:
            for line in f:
                tokens = line.split()
                pixels = [int(t) for t in tokens[6:]]
                letter = labels[tokens[1]]
                fold.append(int(tokens[5]))
                X.append(pixels)
                y.append(letter)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        fold = torch.tensor(fold, dtype=torch.long)
        # boolean masks, not indices
        train_idx = (fold != dev_fold) & (fold != test_fold)
        dev_idx = fold == dev_fold
        test_idx = fold == test_fold

        self.X = X[train_idx]
        self.y = y[train_idx]

        self.dev_X = X[dev_idx]
        self.dev_y = y[dev_idx]

        self.test_X = X[test_idx]
        self.test_y = y[test_idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)
        """
        super(LogisticRegression, self).__init__()
        # Implement me!

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        raise NotImplementedError


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super(FeedforwardNetwork, self).__init__()
        # Implement me!

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        raise NotImplementedError


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    raise NotImplementedError


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_sizes', type=int, default=200)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    opt = parser.parse_args()

    dataset = OCRDataset(opt.data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]  # 128

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes, n_feats,
            opt.hidden_size, opt.layers,
            opt.activation, opt.dropout)

    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    plot(epochs, train_mean_losses, ylabel='Loss', name='training-loss')
    plot(epochs, valid_accs, ylabel='Accuracy', name='validation-accuracy')


if __name__ == '__main__':
    main()
