import numpy as np
import torch

EPSILON = 1e-14


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    idx = torch.where(y_1hot==1)
    loss = -torch.mean(torch.log(X[idx]+epsilon))
    return loss
    # raise NotImplementedError


def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    _exp = torch.exp(X)
    _sum = torch.sum(_exp,axis=0)
    return _exp/_sum
    # raise NotImplementedError


def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    Z = X - torch.max(X,axis=0,keepdim=True)[0]
    _exp = torch.exp(Z)
    _sum = torch.sum(_exp,axis=0)
    return _exp/_sum
    # raise NotImplementedError


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    zeros = torch.zeros_like(X)
    return torch.maximum(X,zeros)
    # raise NotImplementedError


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)




def average_cross_entropy(logits, targets):
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
    return loss

