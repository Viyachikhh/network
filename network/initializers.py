import numpy as np

from abc import ABC, abstractmethod


class BaseInitializer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prev_shape, shape, size):
        pass


class XavierWeights(BaseInitializer):

    def __init__(self):
        self.initializer = np.random.uniform

    def __call__(self, prev_shape, shape, size):
        return self.initializer(low=-np.sqrt(6 / (prev_shape + shape)),
                                high=np.sqrt(6 / (prev_shape + shape)), size=size)


class ZerosInitializer(BaseInitializer):

    def __init__(self):
        self.initializer = np.zeros

    def __call__(self, shape, size, prev_shape=None):
        return self.initializer(shape)
