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

    def update_weights_and_history(self, gradW, gradb, learning_rate=0.004, beta=0.9):
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)

    def __call__(self, inputs):
        h = (self.weights @ inputs.T) + self.bias
        h = h.T
        return h if self.activation is None else self.activation(h)


class NeuralNetwork(object):
    def __init__(self, n_classes=11, dense_layer_count=1, neural_counts=[128], activations=['relu']):
        assert dense_layer_count == len(neural_counts)
        assert dense_layer_count == len(activations)
        self.history_outputs = []
        #self.dense_layer_count = dense_layer_count
        for i in range(dense_layer_count):
            setattr(self, f'layer_{i}', DenseLayer(neural_counts[i],
                                                     784 if i == 0 else neural_counts[i-1], activations[i]))
        self.output = DenseLayer(n_classes, prev_layer_size=neural_counts[-1], activation='softmax')

    def __call__(self, inputs):
        self.history_outputs = [inputs]
        dense = list(vars(self).keys())[1:]
        for i in range(len(dense) - 1):
            layer = getattr(self, dense[i])
            x = layer(inputs) if i == 0 else layer(x)
            self.history_outputs.append(x)
        result = self.output(x)
        return result  #

    def back_propagation(self, y_true, y_pred, batch_size):
        layers = list(vars(self).keys())[::-1][:-1]
        for i, layer in enumerate(layers):
            if i == 0:
                dZ = y_pred - y_true
                #print('dZ', dZ.shape)
            else:
                #print('else, dz Shape', dZ.shape)
                weights = getattr(self, layers[i-1]).weights
                #print('w', weights.shape)
                activation = getattr(self, layer).activation.__name__
                print(activation)
                dZ = (dZ @ weights) * dict_derivatives[activation](self.history_outputs[-i])
                #print('dZ', dZ.shape)
            db = (1 / batch_size) * dZ.sum(axis=0).reshape(-1, 1) # (11,)
            dW = (1 / batch_size) * (dZ.T @ self.history_outputs[-(i+1)])
            #print(f'dW = {dW.shape}, db = {db.shape}')
            getattr(self, layers[i]).update_weights_and_history(dW, db)



def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))

