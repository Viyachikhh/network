import numpy as np

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    def __init__(self):
        pass


class NAG(BaseOptimizer):

    def __init__(self, beta, learning_rate):
        super().__init__()
        self.beta = beta
        self.learning_rate = learning_rate
