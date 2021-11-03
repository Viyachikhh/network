import numpy as np


def relu(h):
    return np.where(h > 0, h, 0)


def softmax(h):
    exp_h = np.exp(h)
    return exp_h / exp_h.sum(axis=0, keepdims=True)


def sigmoid(h):
    return 1 / (1 + np.exp(-h))


def tanh(h):
    return np.tanh(h)


def derivative_relu(h):
    return np.where(h > 0, 1, 0)


def derivative_softmax(h):
    """
    IN PROGRESS...
    """
    softmax_values = softmax(h)
    result = np.zeros((h.shape[1], h.shape[1]))
    for i in range(h.shape[1]):
        for j in range(h.shape[1]):
            result[i, j] = softmax_values[i] * (1 - softmax_values[i]) if i == j else \
                softmax_values[i] * softmax_values[j]
    return result


def derivative_sigmoid(h):
    return sigmoid(h) * (1 - sigmoid(h))


def derivative_tanh(h):
    return 1 - tanh(h) ** 2
