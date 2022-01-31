import numpy as np

from abc import ABC, abstractmethod


class Activation(ABC):

    @abstractmethod
    def __call__(self, inputs):
        pass

    @abstractmethod
    def _activation_(self, inputs):
        pass

    @abstractmethod
    def derivative(self, other_input):
        pass


class ReLU(Activation):

    def __init__(self):
        self.h = None

    def __call__(self, inputs):
        self.h = inputs
        return self._activation_(self.h)

    def _activation_(self, inputs):
        return np.where(inputs > 0, inputs, 0)

    def derivative(self, other_input):
        return np.where(other_input > 0, 1, 0)


class LeakyReLU(Activation):

    def __init__(self):
        self.h = None
        self.coeff = 0.01

    def __call__(self, inputs):
        self.h = inputs
        return self._activation_(self.h)

    def _activation_(self, inputs):
        return np.where(inputs > 0, inputs, self.coeff * inputs)

    def derivative(self, other_input):
        return np.where(other_input > 0, 1, self.coeff)


class Sigmoid(Activation):

    def __init__(self):
        self.h = None

    def __call__(self, inputs):
        self.h = inputs
        return self._activation_(self.h)

    def _activation_(self, inputs):
        return 1 / (1 + np.exp(inputs))

    def derivative(self, other_input):
        return self._activation_(other_input) * (1 - self._activation_(other_input))


class Tanh(Activation):

    def __init__(self):
        self.h = None

    def __call__(self, inputs):
        self.h = inputs
        return self._activation_(self.h)

    def _activation_(self, inputs):
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def derivative(self, other_input):
        return 1 - self._activation_(other_input) ** 2


class Softmax(Activation):

    def __init__(self):
        self.h = None

    def __call__(self, inputs):
        self.h = inputs
        return self._activation_(self.h)

    def _activation_(self, inputs):
        exp_h = np.exp(inputs - np.max(inputs))
        return exp_h / exp_h.sum(axis=0, keepdims=True)

    def derivative(self, other_input):
        return other_input * (1 - other_input)

