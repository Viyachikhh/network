import numpy as np
from network.activations import *
from abc import ABC, abstractmethod

DICT_ACTIVATIONS = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu, 'softmax': softmax}
DICT_DERIVATIVES = {'tanh': derivative_tanh, 'sigmoid': derivative_sigmoid,
                    'relu': derivative_relu, 'softmax': derivative_softmax}


class Layer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, inputs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def update_weights_and_history(self, dZ, learning_rate, beta):
        pass

    @abstractmethod
    def get_gradients(self, dZ):
        pass


class DenseLayer(Layer):

    def __init__(self, layer_size, prev_layer_size, activation=None):
        """
        initialization of object DenseLayer
        """
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.weights = np.random.uniform(low=-np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         high=np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         size=(self.layer_size, self.prev_layer_size))
        self.bias = np.zeros((self.layer_size, 1))
        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def update_weights_and_history(self, dZ, learning_rate, beta):
        gradW, gradb = self.get_gradients(dZ)
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)

    def get_gradients(self, dZ):
        db = (1 / self.cache[0].shape[1]) * dZ.sum(axis=1, keepdims=True)
        dW = (1 / self.cache[0].shape[1]) * (dZ @ self.cache[0].T)
        return dW, db

    def __call__(self, inputs):
        h = (self.weights @ inputs) + self.bias
        result = h if self.activation is None else self.activation(h)
        self.cache = (inputs, result)
        return result

    def __str__(self):
        return f'Dense_layer, w.shape = {self.weights.shape}, b.shape = {self.bias.shape}'


class Conv2DLayer(Layer):

    def __init__(self, count_filters, filter_size, padding=True, input_shape=(28, 28, 1), activation=None):
        self.weights = np.random.uniform(low=-np.sqrt(6 / (1 + count_filters)),
                                         high=np.sqrt(6 / (1 + count_filters)),
                                         size=(filter_size, filter_size, count_filters))
        self.bias = np.zeros((count_filters, 1))
        self.padding = padding
        self.input_shape = input_shape
        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def __call__(self, inputs):
        h = convolution(inputs, self.weights) + self.bias
        result = h if self.activation is None else self.activation(h)
        self.cache = (inputs, result)
        return result

    def get_gradients(self, dZ):
        val = convolution(self.cache[0], self.cache[1])
        pass

    def update_weights_and_history(self, dZ, learning_rate, beta):
        pass

    def __str__(self):
        return 'Conv_layer'


class FlattenLayer(Layer):

    def __init__(self):
        self.cache = None

    def get_gradients(self, dZ):
        pass

    def update_weights_and_history(self, dZ, learning_rate, beta):
        pass

    def __call__(self, inputs):
        sizes = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        result = inputs.reshape((inputs.shape[0], sizes))
        self.cache = (inputs, result)
        return result

    def __str__(self):
        return 'Flatten'


class NeuralNetwork(object):
    def __init__(self, your_layers: list):

        for i in range(len(your_layers)):
            vars(self)[f'layer_{i}'] = your_layers[i]

    def __call__(self, inputs):
        layers = list(vars(self).keys())
        for i in range(len(layers)):
            layer = getattr(self, layers[i])
            x = layer(inputs) if i == 0 else layer(x)
        return x

    def back_propagation(self, y_true, y_pred, batch_size, learning_rate=0.005, beta=0.9):
        layers = list(vars(self).keys())[::-1]

        for i, layer in enumerate(layers):
            if i == 0:
                dZ = y_pred - y_true
            else:
                weights = getattr(self, layers[i - 1]).weights
                layer_activation = getattr(self, layer).activation
                activation = layer_activation.__name__ if layer_activation is not None else None
                derivate_act = DICT_DERIVATIVES.get(activation, None)
                if derivate_act is not None:
                    derivative_values = derivate_act(getattr(self, layer).cache[1])
                else:
                    derivative_values = np.ones(getattr(self, layer).cache[1].shape)
                dZ = (weights.T @ dZ) * derivative_values
            getattr(self, layers[i]).update_weights_and_history(dZ, learning_rate, beta)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))


def convolution(mass, fil, padding=True):
    repeat = np.array([fil] * mass.shape[0])  # ~0.03
    if padding:
        filling = ((0, 0), (fil.shape[1] // 2, fil.shape[1] // 2), (fil.shape[2] // 2, fil.shape[2] // 2), (0, 0))
        pad_mass = np.pad(mass, filling, 'constant', constant_values=0)
        result = np.zeros((mass.shape[0], mass.shape[1], mass.shape[2], fil.shape[-1]))
    else:
        pad_mass = mass
        result = np.zeros((mass.shape[0], pad_mass.shape[1] - fil.shape[0] + 1,
                           pad_mass.shape[2] - fil.shape[1] + 1, fil.shape[-1]))
    for i in range(fil.shape[0] // 2, mass.shape[1] - fil.shape[0] // 2):
        for j in range(fil.shape[1] // 2, mass.shape[2] - fil.shape[1] // 2):
            box = pad_mass[:, i - fil.shape[0] // 2: i + 1 + fil.shape[0] // 2,
                  j - fil.shape[1] // 2: j + 1 + fil.shape[1] // 2]
            output = np.expand_dims(np.einsum('ijkh, ijkl -> il', box, repeat), axis=(0, 1))
            result[:, i - fil.shape[0] // 2, j - fil.shape[1] // 2, :] = output
    return result
