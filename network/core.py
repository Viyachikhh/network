import numpy as np


class DenseLayer(object):
    def __init__(self, layer_size, prev_layer_size, activation='softmax'):
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.weights = np.random.uniform(low=-np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         high=np.sqrt(6 / (self.layer_size + self.prev_layer_size)),
                                         size=(self.layer_size, self.prev_layer_size))
        self.bias = np.zeros((self.layer_size, 1))
        self.history = (0, 0)
        self.activation = activation

    def update_weights_and_history(self, gradW, gradb, learning_rate=0.004, beta=0.9):
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)

    def __call__(self, inputs):
        h = (self.weights @ inputs.T) + self.bias
        h = h.T
        if self.activation != 'softmax':
            return np.where(h > 0, h, 0)
        else:
            exp_h = np.exp(h)
            return exp_h / exp_h.sum(axis=1).reshape(-1, 1)


class NeuralNetwork(object):
    def __init__(self, n_classes=11, neural_count=128):
        self.layer_1 = DenseLayer(neural_count, prev_layer_size=784, activation='relu')
        self.output = DenseLayer(n_classes, prev_layer_size=neural_count, activation='softmax')

    def __call__(self, inputs):
        state1 = self.layer_1(inputs)
        state2 = self.output(state1)
        return inputs, state1, state2

    def back_propagation(self, y_true, states, input_shape):

        dZ2 = states[-1] - y_true
        db2 = (1 / input_shape) * dZ2.sum(axis=0).reshape(-1, 1) # (11,)
        dW2 = (1 / input_shape) * (dZ2.T @ states[-2])  # (11, 128)
        self.output.update_weights_and_history(dW2, db2)

        dZ = (dZ2 @ self.output.weights) * derivative_relu(states[-2])  # (60000, 128)
        db = (1 / input_shape) * dZ.sum(axis=0).reshape(-1, 1)  # (128,)
        dW = (1 / input_shape) * (dZ.T @ states[-3])  # (128, 784)
        self.layer_1.update_weights_and_history(dW, db)


def derivative_relu(z):
    return np.where(z > 0, 1, 0)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))

