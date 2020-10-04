#!/usr/bin/env python

# Deep Structured Learning Homework 1
# Fall 2019

import argparse
import random
import os
from itertools import count
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(path, feature_rep=None, bias=False, dev_fold=8, test_fold=9):
    """
    path: location of OCR data
    feature_rep: a function or None. Use it to transform the binary pixel
        representation into something more interesting in Q 2.2a
    bias: whether to add a bias term as an extra feature dimension
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
    X = np.array(X, dtype='int8')
    y = np.array(y, dtype='int8')

    # postprocess X into a feature representation
    if feature_rep is not None:
        X = feature_rep(X)
    if bias:
        bias_vector = np.ones((X.shape[0], 1), dtype=int)
        X = np.hstack((X, bias_vector))

    fold = np.array(fold, dtype='int8')
    # boolean masks, not indices
    train_ix = (fold != dev_fold) & (fold != test_fold)
    dev_ix = fold == dev_fold
    test_ix = fold == test_fold

    train_X, train_y = X[train_ix], y[train_ix]
    dev_X, dev_y = X[dev_ix], y[dev_ix]
    test_X, test_y = X[test_ix], y[test_ix]

    return {"train": (train_X, train_y),
            "dev": (dev_X, dev_y),
            "test": (test_X, test_y)}


def custom_features(X):
    """
    X (n_examples x n_features)
    returns (n_examples x ???): It's up to you to define an interesting feature
        representation. One idea: pairwise pixel features (see the handout).
    """
    # Q2.2 a
    raise NotImplementedError


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Question 2.1 b
        raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        l2_penalty (float): BONUS
        """
        # Question 2.2 b
        raise NotImplementedError


class MLP(object):
    # Q3. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        raise NotImplementedError

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        raise NotImplementedError


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument("-custom_features", action="store_true",
                        help="""Whether to use the custom_features() function
                        to postprocess the binary pixel features, as in Q2.2,
                        part (a).""")
    parser.add_argument('-bias', action='store_true',
                        help="""Whether to add an extra bias feature to all
                        samples in the dataset. In an MLP, where there can be
                        biases for each neuron, adding a bias feature to the
                        input is not sufficient.""")
    opt = parser.parse_args()

    configure_seed(seed=42)

    feature_function = custom_features if opt.custom_features else None
    data = load_data(opt.data, bias=opt.bias, feature_rep=feature_function)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 26
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        # Q3. Be sure to experiment with different values for hidden_size.
        hidden_size = 1  # tune me!
        model = MLP(n_classes, n_feats, hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(train_X, train_y)
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
