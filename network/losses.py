import numpy as np

from abc import ABC, abstractmethod

"""
Перенести на Tensor
"""


class Loss(ABC):

    def __init__(self):
        self.labels = None
        self.predictions = None

    @abstractmethod
    def calculate_loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def gradient_loss(self):
        pass


class CategoricalCrossEntropy(Loss):

    def __init__(self):
        super().__init__()

    def calculate_loss(self, y_true, y_pred, without_memory=False):
        if not without_memory:
            self.labels = y_true
            self.predictions = y_pred
        return (-1 / y_true.shape[1]) * np.sum(y_true * np.log(y_pred + 1e-5))

    def gradient_loss(self):
        return self.predictions - self.labels


class MSE(Loss):

    def __init__(self):
        super().__init__()

    def calculate_loss(self, y_true, y_pred, without_memory=False):
        if not without_memory:
            self.labels = y_true
            self.predictions = y_pred
        return 1 / y_true.shape[1] * np.sum((y_true - y_pred) ** 2)

    def gradient_loss(self):
        return (2 / self.labels.shape[1]) * (self.predictions - self.labels)