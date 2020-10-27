"""
hw2_decoder.py

This module is for implementations of the Viterbi and Forward-Backward
algorithms, which are used in Questions 1. Forward-Backward is provided
for you; you will need to implement Viterbi for Question 1.
In Question 2 you will adapt these algorithms to PyTorch.
"""

import numpy as np


def viterbi(initial_scores, transition_scores, final_scores, emission_scores):
    """Computes the viterbi trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    Your solution should return:
    - best_path: (length) array containing the most likely state sequence
    """
    raise NotImplementedError


def _forward(initial_scores, transition_scores, final_scores, emission_scores):
    """Compute the forward trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    """
    length = emission_scores.shape[0]
    num_states = initial_scores.shape[0]

    # Forward variables.
    forward = np.full((length, num_states), -np.inf)
    forward[0] = emission_scores[0] + initial_scores

    # Forward loop.
    for i in range(1, length):
        for state in range(num_states):
            forward[i, state] = np.logaddexp.reduce(
                forward[i - 1] + transition_scores[i - 1, state])
            forward[i, state] += emission_scores[i, state]

    # Termination.
    log_likelihood = np.logaddexp.reduce(forward[length - 1] + final_scores)

    return log_likelihood, forward


def _backward(
        initial_scores, transition_scores, final_scores, emission_scores):
    """Compute the backward trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    """
    length = emission_scores.shape[0]
    num_states = initial_scores.shape[0]

    # Backward variables.
    backward = np.full((length, num_states), -np.inf)

    # Initialization.
    backward[length-1, :] = final_scores

    # Backward loop.
    for i in range(length - 2, -1, -1):
        for state in range(num_states):
            backward[i, state] = np.logaddexp.reduce(
                backward[i + 1] +
                transition_scores[i, :, state] +
                emission_scores[i + 1])

    # Termination.
    log_likelihood = np.logaddexp.reduce(
        backward[0, :] + initial_scores + emission_scores[0, :])

    return log_likelihood, backward


def forward_backward(
        initial_scores, transition_scores, final_scores, emission_scores):
    log_likelihood, forward = _forward(
        initial_scores, transition_scores, final_scores, emission_scores)

    log_likelihood, backward = _backward(
        initial_scores, transition_scores, final_scores, emission_scores)

    emission_posteriors = np.exp(forward + backward - log_likelihood)
    transition_posteriors = np.zeros_like(transition_scores)
    length = np.size(emission_scores, 0)  # Length of the sequence.
    # num_states = np.size(initial_scores)  # Number of states.
    for i in range(1, length):
        # bp: afaik, the multiplication by np.ones is unnecessary
        # transition_posteriors[i - 1] =
        # np.exp(forward[i - 1: i].transpose() +
        # transition_scores[i - 1] +
        # emission_scores[i: i + 1] +
        # backward[i: i + 1] -
        # log_likelihood)
        fw_t = forward[i - 1: i].T
        bw = backward[i]
        tr = transition_scores[i - 1]
        em = emission_scores[i]
        transition_posteriors[i - 1] = fw_t + tr + em + bw
    transition_posteriors = np.exp(transition_posteriors - log_likelihood)
    # the transition_posteriors aren't even used in Q2...

    return emission_posteriors, transition_posteriors, log_likelihood
