import torch
from torch import nn


class LinearChainCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        num_classes (int): number of classes in your tagset.
        pad_index (int, optional): integer representing the pad symbol in your tagset.
            If not None, the model will apply constraints for PAD transitions.
            NOTE: there is no need to use padding if you use batch_size=1.
    """

    def __init__(self, num_classes, pad_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes))
        self.initials = nn.Parameter(torch.randn(self.num_classes))
        self.finals = nn.Parameter(torch.randn(self.num_classes))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.initials, -0.1, 0.1)
        nn.init.uniform_(self.finals, -0.1, 0.1)
        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero
        if self.pad_index is not None:
            # no transitions from padding
            self.transitions.data[self.pad_index, :] = -10000.0
            # no transitions to padding
            self.transitions.data[:, self.pad_index] = -10000.0
            # except if we are in a pad position
            self.transitions.data[self.pad_index, self.pad_index] = 0.0

    def forward(self, emissions, mask=None):
        """Run the CRF layer to get predictions."""
        return self.decode(emissions, mask=mask)

    def neg_log_likelihood(self, emissions, tags, mask=None):
        """
        Compute the negative log-likelihood of a sequence of tags given a sequence of
        emissions scores.
        
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels)
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len)
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len)
        
        Returns:
             torch.Tensor: the sum of neg log-likelihoods of each sequence in the batch.
                Shape of ([])
        """
        scores = self.compute_scores(emissions, tags, mask=mask)
        partition = self.compute_log_partition(emissions, mask=mask)
        nll = -torch.sum(scores - partition) 
        return nll

    def decode(self, emissions, mask=None):
        """
        Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels).
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len)
        
        Returns:
            list of lists: the best viterbi sequence of labels for each batch.
        """
        viterbi_path = self.viterbi(emissions, mask=mask)
        return viterbi_path

    def compute_scores(self, emissions, tags, mask=None):
        """
        Compute the scores for a given batch of emissions with their tags.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        raise NotImplementedError

    def compute_log_partition(self, emissions, mask=None):
        """
        Compute the partition function in log-space using the forward-algorithm.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        raise NotImplementedError

    def viterbi(self, emissions, mask=None):
        """
        Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        raise NotImplementedError
