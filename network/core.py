import numpy as np
from network.activations import *


dict_activations = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu, 'softmax': softmax}
dict_derivatives = {'tanh': derivative_tanh, 'sigmoid': derivative_sigmoid,
                    'relu': derivative_relu, 'softmax': derivative_softmax}


class DenseLayer(object):

    def __init__(self, layer_size, prev_layer_size, activation='softmax'):
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.weights = np.random.uniform(low=-np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         high=np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         size=(self.layer_size, self.prev_layer_size))
        self.bias = np.zeros((self.layer_size, 1))
        self.history = (0, 0)
        self.activation = dict_activations.get(activation, None)

    def update_weights_and_history(self, gradW, gradb, learning_rate, beta):
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)

    def __call__(self, inputs):
        h = (self.weights @ np.swapaxes(inputs, 0, 1)) + self.bias
        h = np.swapaxes(h, 0, 1)
        return h if self.activation is None else self.activation(h)


class NeuralNetwork(object):
    def __init__(self, dense_layer_count=2, neural_counts=(128, 11), activations=('relu', 'softmax')):
        assert dense_layer_count == len(neural_counts)
        assert dense_layer_count == len(activations)
        self.history_outputs = []
        for i in range(dense_layer_count):
            vars(self)[f'layer_{i}'] = DenseLayer(neural_counts[i],
                                                    784 if i == 0 else neural_counts[i-1], activations[i])

    def __call__(self, inputs):
        self.history_outputs = [inputs]
        dense = list(vars(self).keys())[1:]
        for i in range(len(dense)):
            layer = getattr(self, dense[i])
            x = layer(inputs) if i == 0 else layer(x)
            self.history_outputs.append(x)
        result = self.history_outputs[-1]
        del self.history_outputs[-1]
        return result

    def back_propagation(self, y_true, y_pred, batch_size, learning_rate=0.005, beta=0.9):
        layers = list(vars(self).keys())[:0:-1]
        for i, layer in enumerate(layers):
            if i == 0:
                dZ = y_pred - y_true
            else:
                weights = getattr(self, layers[i-1]).weights
                activation = getattr(self, layer).activation.__name__
                dZ = (dZ @ weights) * dict_derivatives[activation](self.history_outputs[-i])
            db = np.expand_dims((1 / batch_size) * dZ.sum(axis=0), axis=-1)
            dW = (1 / batch_size) * (np.swapaxes(dZ, 0, 1) @ self.history_outputs[-(i+1)])
            getattr(self, layers[i]).update_weights_and_history(dW, db, learning_rate, beta)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))

