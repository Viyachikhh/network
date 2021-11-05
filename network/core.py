
from network.activations import *
from abc import ABC, abstractmethod
from network.utils import col2im_indices, im2col_indices

DICT_ACTIVATIONS = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu, 'softmax': softmax}
DICT_DERIVATIVES = {'tanh': derivative_tanh, 'sigmoid': derivative_sigmoid,
                    'relu': derivative_relu, 'softmax': derivative_softmax}
REV_ACTIVATIONS = {value: key for key, value in DICT_ACTIVATIONS.items()}


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
    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
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

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        gradW, gradb, gradZ = self.get_gradients(dZ, activation_next=activation_next)
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)
        return gradZ

    def get_gradients(self, dZ, activation_next=None):
        db = (1 / self.cache[0].shape[1]) * dZ.sum(axis=1, keepdims=True)
        dW = (1 / self.cache[0].shape[1]) * (dZ @ self.cache[0].T)
        derivative = DICT_DERIVATIVES.get(activation_next, None)
        derivative_values = derivative(self.cache[0]) if derivative is not None else self.cache[0]
        next_dZ = (self.weights.T @ dZ) * derivative_values
        return dW, db, next_dZ

    def __call__(self, inputs):
        h = (self.weights @ inputs) + self.bias
        self.cache = (inputs, h)
        result = h if self.activation is None else self.activation(h)
        return result

    def __str__(self):
        return f'Dense_layer, w.shape = {self.weights.shape}, b.shape = {self.bias.shape}'


class Conv2DLayer(Layer):

    def __init__(self, count_filters, inp_channels, filter_size, stride=1,
                 padding=0, activation=None):
        self.weights = np.random.uniform(low=-np.sqrt(6 / (1 + count_filters * inp_channels * filter_size ** 2)),
                                         high=np.sqrt(6 / (1 + count_filters * inp_channels * filter_size ** 2)),
                                         size=(count_filters, inp_channels, filter_size, filter_size))
        self.bias = np.zeros((count_filters, 1))
        self.padding = padding
        self.stride = stride
        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def __call__(self, inputs):
        """
        if self.padding != 0:
            filling = ((0, 0),
                       (self.weights.shape[0] // 2, self.weights.shape[0] // 2),
                       (self.weights.shape[1] // 2, self.weights.shape[1] // 2),
                       (0, 0))
            pad_mass = np.pad(inputs, filling, 'constant', constant_values=0)
        else:
            pad_mass = inputs
        result = np.zeros((inputs.shape[0], pad_mass.shape[1] - self.weights.shape[0] + 1,
                           pad_mass.shape[2] - self.weights.shape[1] + 1, self.weights.shape[-1]))

        h = self.convolution(pad_mass) + self.bias
        self.cache = (pad_mass, h)
        result = h if self.activation is None else self.activation(h)
        """
        X_in_col = im2col_indices(inputs, self.weights.shape[2], self.weights.shape[3],
                                  padding=self.padding, stride=self.stride)
        W_in_col = self.weights.reshape(self.weights.shape[0], -1)
        h = (W_in_col @ X_in_col) + self.bias
        height_output = (inputs.shape[2] - self.weights.shape[2] + 2 * self.padding) // self.stride + 1
        width_output = (inputs.shape[3] - self.weights.shape[3] + 2 * self.padding) // self.stride + 1

        h = h.reshape(self.weights.shape[0], height_output, width_output, inputs.shape[0])
        h = h.transpose(3, 0, 1, 2)

        result = self.activation(h) if self.activation is not None else h
        self.cache = (inputs, X_in_col, result)

        return result

    def get_gradients(self, dZ, activation_next=None):
        db = np.sum(dZ, axis=(0, 2, 3))
        db = db.reshape(self.weights.shape[0], -1)

        dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(self.weights.shape[0], -1)

        dW = dZ_reshaped @ self.cache[1].T
        dW = dW.reshape(self.weights.shape)

        W_reshaped = self.weights.reshape(self.weights.shape[0], -1)

        dZ_next_col = W_reshaped.T @ dZ_reshaped
        dZ_next = col2im_indices(dZ_next_col, self.cache[0].shape, self.weights.shape[2],
                                 self.weights.shape[3], padding=self.padding, stride=self.stride)

        derivative = DICT_DERIVATIVES.get(activation_next, None)
        derivative_values = derivative(self.cache[0]) if derivative is not None else self.cache[0]

        dZ_next = dZ_next * derivative_values

        return dW, db, dZ_next

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        gradW, gradb, gradZ = self.get_gradients(dZ, activation_next=activation_next)
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)
        return gradZ

    def convolution(self, inputs):
        """
        height_filter = self.weights.shape[0] // 2
        width_filter = self.weights.shape[1] // 2

        result = np.zeros((inputs.shape[0], inputs.shape[1] - self.weights.shape[0] + 1,
                           inputs.shape[2] - self.weights.shape[1] + 1, self.weights.shape[-1]))

        for i in range(height_filter, inputs.shape[1] - height_filter):
            for j in range(width_filter, inputs.shape[2] - width_filter):
                box = inputs[:, i - height_filter: i + 1 + height_filter, j - width_filter: j + 1 + width_filter]
                output = np.expand_dims(np.einsum('ijkl, jkmh -> ih', box, self.weights), axis=(0, 1))
                result[:, i - height_filter, j - width_filter, :] = output

        print('conv_res', result.shape)
        return result
        """
        pass

    def convolution_reverse(self, dZ):
        """
        dW = np.zeros(self.weights.shape)

        for i in range(dW.shape[0]):
            for j in range(dW.shape[1]):
                box = self.cache[0][:, i:i + self.cache[1].shape[1],
                      j:j + self.cache[1].shape[2], :]
                output = np.einsum('ijkl, ijkh -> ilh', box, dZ)
                dW[i, j, :, :] = output.mean(axis=0)

        dZ_next = np.zeros(self.cache[0].shape)

        print(dW.shape)
        height = self.weights.shape[0] // 2
        width = self.weights.shape[1] // 2

        filling = ((0, 0), (height, height), (width, width), (0, 0))
        padded = np.pad(dZ, filling, 'constant', constant_values=0)

        for i in range(height, padded.shape[1] - height, self.stride):
            for j in range(width, padded.shape[2] - width, self.stride):
                slice_dZ = dZ[:, i - height: i + height + 1, j - width: j + 1 + width, :]
                output = np.einsum('ijkl, mnop -> mijk', self.weights, slice_dZ)
                dZ_next[:, i - height: i + height + 1, j - width: j + 1 + width, :] += output
        return dW, dZ_next
        """
        pass

    def __str__(self):
        return 'Conv_layer'


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.cache = None

    def __call__(self, inputs):
        X_reshaped = inputs.reshape(inputs.shape[0] * inputs.shape[1], 1, inputs.shape[2], inputs.shape[3])

        X_in_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        self.X_col_size = X_in_col

        self.max_indexes = np.argmax(X_in_col, axis=0)
        result = X_in_col[self.max_indexes, range(self.max_indexes.size)]
        result = result.reshape(inputs.shape[2] // self.pool_size, inputs.shape[3] // self.pool_size,
                           inputs.shape[0], inputs.shape[1]).transpose(2, 3, 0, 1)

        self.cache = (inputs, result)

        return result

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        dZ_next = self.get_gradients(dZ)
        return dZ_next

    def get_gradients(self, dZ):
        dZ_col = np.zeros_like(self.X_col_size)

        dZ_flat = dZ.transpose(2, 3, 0, 1).ravel()

        dZ_col[self.max_indexes, range(self.max_indexes.size)] = dZ_flat

        shape = (self.cache[0].shape[0] * self.cache[0].shape[1], 1, self.cache[0].shape[2], self.cache[0].shape[3])
        dZ_next = col2im_indices(dZ_col, shape, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        dZ_next = dZ_next.reshape(self.cache[0].shape[0], self.cache[0].shape[1],
                                  self.cache[0].shape[2], self.cache[0].shape[3])
        return dZ_next

    def __str__(self):
        pass


class FlattenLayer(Layer):

    def __init__(self):
        self.cache = None

    def get_gradients(self, dZ):
        dZ_next = dZ.reshape(self.cache[0].shape)
        print(dZ_next.shape)
        return dZ_next

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        dZ_next = self.get_gradients(dZ)
        return dZ_next

    def __call__(self, inputs):
        sizes = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        result = inputs.reshape((sizes, inputs.shape[0]))
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

    def back_propagation(self, y_true, y_pred, learning_rate=0.005, beta=0.9):
        layers = list(vars(self).keys())[::-1]

        dZ = y_pred - y_true
        for i, layer in enumerate(layers):
            if i != len(layers) - 1:
                try:
                    activation = REV_ACTIVATIONS.get(getattr(self, layers[i + 1]).activation, None)
                except AttributeError:
                    activation = None
            else:
                activation = None
            dZ = getattr(self, layers[i]).update_weights_and_history(dZ, learning_rate, beta, activation)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred + 1e-5))
