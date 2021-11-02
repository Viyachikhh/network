import numpy as np
from network.activations import *

DICT_ACTIVATIONS = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu, 'softmax': softmax}
DICT_DERIVATIVES = {'tanh': derivative_tanh, 'sigmoid': derivative_sigmoid,
                    'relu': derivative_relu, 'softmax': derivative_softmax}


class DenseLayer(object):

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
    def __init__(self, *your_layers, input_shape=784, dense_layer_count=0,
                 neural_counts=None, activations=None):
        if dense_layer_count != 0:
            assert dense_layer_count == len(neural_counts)
            assert dense_layer_count == len(activations)
        self.history_outputs = []
        for i in range(dense_layer_count):
            vars(self)[f'layer_{i}'] = DenseLayer(neural_counts[i],
                                                  input_shape if i == 0 else neural_counts[i - 1], activations[i])
        for i in range(dense_layer_count, dense_layer_count + len(your_layers)):
            vars(self)[f'layer_{i}'] = your_layers[i - dense_layer_count]

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
                weights = getattr(self, layers[i - 1]).weights
                layer_activation = getattr(self, layer).activation
                activation = layer_activation.__name__ if layer_activation is not None else None
                derivate_act = DICT_DERIVATIVES.get(activation, None)
                if derivate_act is not None:
                    derivative_values = derivate_act(self.history_outputs[-i])
                else:
                    derivative_values = np.ones(self.history_outputs[-i].shape)
                dZ = (dZ @ weights) * derivative_values
            db = np.expand_dims((1 / batch_size) * dZ.sum(axis=0), axis=-1)
            dW = (1 / batch_size) * (np.swapaxes(dZ, 0, 1) @ self.history_outputs[-(i + 1)])
            getattr(self, layers[i]).update_weights_and_history(dW, db, learning_rate, beta)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))
