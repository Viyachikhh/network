import numpy as np


def relu(h):
    return np.where(h > 0, h, 0)


def softmax(h):
    exp_h = np.exp(h)
    return exp_h / exp_h.sum(axis=1).reshape(-1, 1)


def sigmoid(h):
    return 1 / (1 + np.exp(-h))


def tanh(h):
    return np.tanh(h)


def derivative_relu(h):
    return np.where(h > 0, 1, 0)


def derivative_softmax(h):
    pass


def derivative_sigmoid(h):
    return sigmoid(h) * (1 - sigmoid(h))


def derivative_tanh(h):
    return 1 - tanh(h) ** 2
