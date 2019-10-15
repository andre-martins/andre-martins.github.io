import argparse

import numpy as np
from hw2_decoder import viterbi, forward_backward
import matplotlib.pyplot as plt


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
            line = line.rstrip('\t\n')
            fields = line.split('\t')
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
    ll = ['' for k in labels]
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


class LinearSequenceModel(object):
    def __init__(self, n_classes, n_features, feature_function=None):
        self.W_unigrams = np.zeros((n_classes, n_features))
        self.W_bigrams = np.zeros((n_classes, n_classes))  # curr, prev.
        self.W_start = np.zeros(n_classes)
        self.W_stop = np.zeros(n_classes)
        self.feature_function = feature_function

    @property
    def weights(self):
        return [self.W_unigrams, self.W_bigrams, self.W_start, self.W_stop]

    @property
    def n_classes(self):
        return self.W_unigrams.shape[0]

    def train_epoch(self, X_train, y_train, **kwargs):
        value = 0  # number of mistakes for perceptron, average loss for CRF
        for xseq, yseq in zip(X_train, y_train):
            if self.feature_function is not None:
                xseq = [self.feature_function(x) for x in xseq]
            value += self.update_weight(xseq, yseq, **kwargs)
        return value

    def update_weight(self, xseq, yseq, **kwargs):
        raise NotImplementedError

    def evaluate(self, X, y):
        """Evaluate model on data."""
        correct = 0
        total = 0
        for xseq, yseq in zip(X, y):
            if self.feature_function is not None:
                xseq = [self.feature_function(x) for x in xseq]
            emission_scores = np.zeros((len(xseq), self.n_classes))
            transition_scores = np.zeros(
                (len(xseq) - 1, self.n_classes, self.n_classes)
            )
            initial_scores = np.zeros(self.n_classes)
            final_scores = np.zeros(self.n_classes)
            initial_scores[:] = self.W_start
            for t in range(len(xseq)):
                emission_scores[t] = self.W_unigrams.dot(xseq[t])
                if t > 0:
                    transition_scores[t - 1] = self.W_bigrams
            final_scores[:] = self.W_stop
            yseq_hat = viterbi(
                initial_scores,
                transition_scores,
                final_scores,
                emission_scores)
            correct += sum([yseq[t] == yseq_hat[t] for t in range(len(yseq))])
            total += len(yseq)
        return correct / total


class StructuredPerceptron(LinearSequenceModel):
    def update_weight(self, xseq, yseq, **kwargs):
        """
        xseq (list): list of np.arrays, each of size (n_feat)
        yseq (list): list of class labels (int)
        """
        raise NotImplementedError


class CRF(LinearSequenceModel):
    def update_weight(self, xseq, yseq, l2_decay=None, learning_rate=None):
        """
        xseq (list): list of np.arrays, each of size (n_feat)
        yseq (list): list of class labels (int)
        l2_decay (float): l2 regularization constant
        learning_rate (float)
        """
        assert l2_decay is not None and learning_rate is not None
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=["perceptron", "crf"])
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument("-no_pairwise", action="store_true",
                        help="""If you pass this flag, the model will use
                        binary pixel features instead of pairwise ones. For
                        submission, you only need to report results with
                        pairwise features, but using binary pixels can be
                        helpful for debugging because models can be trained
                        much faster and with less memory.""")
    opt = parser.parse_args()

    model = opt.model
    l2_decay = opt.l2_decay
    learning_rate = opt.learning_rate

    np.random.seed(42)

    print('Loading data...')
    feature_function = pairwise_features if not opt.no_pairwise else None
    X_train, y_train, labels = read_data(opt.data, partitions=set(range(8)))
    X_dev, y_dev, _ = read_data(opt.data, partitions={8})
    X_test, y_test, _ = read_data(opt.data, partitions={9})

    n_classes = len(labels)
    if feature_function is not None:
        n_features = len(feature_function(X_train[0][0]))
    else:
        n_features = len(X_train[0][0])

    print('Training %s model...' % opt.model)
    accuracies = []

    model_cls = StructuredPerceptron if model == "perceptron" else CRF
    clf = model_cls(n_classes, n_features, feature_function)
    total = sum(len(xseq) for xseq in X_train)

    for epoch in range(1, opt.epochs + 1):
        train_order = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in train_order]
        y_train = [y_train[i] for i in train_order]

        # learning rate and regularization should be ignored if the model
        # is a structured perceptron
        epoch_value = clf.train_epoch(
            X_train, y_train, learning_rate=learning_rate, l2_decay=l2_decay)
        acc = clf.evaluate(X_dev, y_dev)
        accuracies.append(acc)
        if model == 'perceptron':
            num_mistakes = epoch_value
            print('Epoch: %d, Mistakes: %d, Dev accuracy: %f' % (
                    epoch, num_mistakes, acc))
        else:
            avg_loss = epoch_value / total
            reg_term = sum(0.5 * l2_decay * (W * W).sum() for W in clf.weights)
            print('Epoch: %d, Loss: %f, Obj: %f, Dev accuracy: %f' % (
                  epoch, avg_loss, avg_loss + reg_term, acc))

    plt.plot(range(1, opt.epochs + 1), accuracies, 'bo-')
    plt.title('Dev accuracy')
    name = '%s_%s_%s' % (
        model, l2_decay, learning_rate) if model == "crf" else model
    plt.savefig('%s.pdf' % name)

    print('Evaluating...')
    test_acc = clf.evaluate(X_test, y_test)
    print('Test accuracy: %f' % test_acc)


if __name__ == "__main__":
    main()
