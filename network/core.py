from network.activations import *
from abc import ABC, abstractmethod

from network.utils import col2im_indices, im2col_indices, getWindows, xavier_uniform_generator
from network.initializers import XavierWeights

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
    def build_weights(self, prev_size):
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

    def __init__(self, layer_size, activation=None):
        """
        initialization of object DenseLayer
        """
        self.layer_size = layer_size
        self.prev_layer_size = None
        self.weights_initializer = XavierWeights()
        self.bias = np.zeros((self.layer_size, 1))
        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def build_weights(self, prev_size):
        return self.weights_initializer(prev_size, self.layer_size, (self.layer_size, prev_size))

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
        if 'weights' not in vars(self).keys():
            prev_size = inputs.shape[0]
            self.weights = self.build_weights(prev_size)

        h = (self.weights @ inputs) + self.bias
        self.cache = (inputs, h)
        result = h if self.activation is None else self.activation(h)
        return result

    def __str__(self):
        return f'Dense_layer, w.shape = {self.weights.shape}, b.shape = {self.bias.shape}'


class Conv2DLayer(Layer):

    def __init__(self, count_filters, filter_size, stride=1, padding=False, activation=None):
        self.prev_channels = None
        self.weights_initializer = XavierWeights()
        self.bias = np.zeros(count_filters)

        self.count_filters = count_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride

        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def build_weights(self, prev_size):
        return self.weights_initializer(prev_size, self.count_filters * prev_size * self.filter_size ** 2,
                                      (self.count_filters, prev_size, self.filter_size, self.filter_size))

    def __call__(self, inputs):
        if 'weights' not in vars(self).keys():
            prev_size = inputs.shape[1]
            self.weights = self.build_weights(prev_size)
            self.padding_size = (self.weights.shape[2] // 2 if self.padding else 0,
                                 self.weights.shape[3] // 2 if self.padding else 0)

        height_output = (inputs.shape[2] - self.weights.shape[2] + 2 * self.padding_size[0]) // self.stride + 1
        width_output = (inputs.shape[3] - self.weights.shape[3] + 2 * self.padding_size[0]) // self.stride + 1

        # размер массива - (batch_size, in_channels, height_picture, width_picture, filter_height, filter_width)
        windows = getWindows(inputs, (inputs.shape[0], self.weights.shape[0], height_output, width_output),
                             filter_size=self.weights.shape[2], padding=self.padding_size[0], stride=self.stride)

        Z = np.einsum('ijklmn, ojmn -> iokl', windows, self.weights)  # свёртка входных данных с фильтром
        Z += self.bias[None, :, None, None]
        self.cache = (windows, inputs)
        result = Z if self.activation is None else self.activation(Z)
        return result

    def get_gradients(self, dZ, activation_next=None):
        padding = self.weights.shape[2] - 1 if not self.padding else self.padding_size[0]
        input_windows, inputs = self.cache
        windows_dZ = getWindows(dZ, inputs.shape, filter_size=self.weights.shape[2], padding=padding, stride=1,
                                dilate=self.stride - 1)  # dilate отвечает за количество нулей между элементами, если
        # stride != 1
        weights_flipped = self.weights[:, :, ::-1, ::-1]  # для вычисления производной по входным нужно повернуть фильтр

        dW = np.einsum('ijklmn, iokl -> ojmn', input_windows, dZ)  # свёртка входных данных и dZ
        dZ_prev = np.einsum('ijklmn, jomn -> iokl', windows_dZ, weights_flipped)  # полная свёртка dZ и весов
        derivative = DICT_DERIVATIVES.get(activation_next, None)
        derivative_values = derivative(inputs) if derivative is not None else np.ones(inputs.shape)
        dZ_prev *= derivative_values
        db = np.sum(dZ, axis=(0, 2, 3))
        return dW, db, dZ_prev

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        """
        Собственно, обновление весов
        """
        gradW, gradb, gradZ = self.get_gradients(dZ, activation_next=activation_next)
        vdw = beta * self.history[0] - learning_rate * gradW
        vdb = beta * self.history[1] - learning_rate * gradb
        self.weights += vdw
        self.bias += vdb
        self.history = (vdw, vdb)
        return gradZ

    def __str__(self):
        return f'Conv_layer with filter_size = {self.weights.shape[2]} and count_filters = {self.weights.shape[0]}'


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2, input_channels=None):
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.cache = None

    def __call__(self, inputs):
        X_reshaped = inputs.reshape(inputs.shape[0] * inputs.shape[1], 1, inputs.shape[2], inputs.shape[3])

        # разворачивание в двумерный массив
        X_in_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        self.X_col_size = X_in_col

        self.max_indexes = np.argmax(X_in_col, axis=0)  # индексы максимальных элементов
        result = X_in_col[self.max_indexes, range(self.max_indexes.size)]

        # деалем массив из двумерного обратно в четырёхмерный
        result = result.reshape(inputs.shape[2] // self.pool_size, inputs.shape[3] // self.pool_size,
                                inputs.shape[0], inputs.shape[1]).transpose(2, 3, 0, 1)

        self.cache = (inputs, result)

        return result

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        dZ_next = self.get_gradients(dZ)
        return dZ_next

    def get_gradients(self, dZ):
        dZ_col = np.zeros_like(self.X_col_size)

        # транспонирование для получения нужных размерностей и разворачивание
        dZ_flat = dZ.transpose(2, 3, 0, 1).ravel()

        # индексы, где стояли максимальные элементы, заполняем элементами из dZ
        dZ_col[self.max_indexes, range(self.max_indexes.size)] = dZ_flat

        # получение нужной размерности данных для следующего слоя для расчёта производных
        shape = (self.cache[0].shape[0] * self.cache[0].shape[1], 1, self.cache[0].shape[2], self.cache[0].shape[3])
        dZ_next = col2im_indices(dZ_col, shape, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        dZ_next = dZ_next.reshape(self.cache[0].shape[0], self.cache[0].shape[1],
                                  self.cache[0].shape[2], self.cache[0].shape[3])
        return dZ_next

    def __str__(self):
        pass

    def build_weights(self, prev_size):
        pass


class FlattenLayer(Layer):

    def __init__(self):
        self.cache = None

    def get_gradients(self, dZ):
        dZ_next = np.swapaxes(dZ, 0, 1)
        dZ_next = dZ_next.reshape(self.cache[0].shape)
        return dZ_next

    def update_weights_and_history(self, dZ, learning_rate, beta, activation_next=None):
        dZ_next = self.get_gradients(dZ)
        return dZ_next

    def __call__(self, inputs):
        sizes = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        result = inputs.reshape((inputs.shape[0], sizes))
        result = np.swapaxes(result, 0, 1)
        self.cache = (inputs, result)
        return result

    def __str__(self):
        return 'Flatten'

    def build_weights(self, prev_size):
        pass


class NeuralNetwork(object):
    def __init__(self, *your_layers):

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
                except AttributeError:  # на случай, если в слое не предусмотрена функция активации
                    activation = None
            else:
                activation = None
            dZ = getattr(self, layers[i]).update_weights_and_history(dZ, learning_rate, beta, activation)

    def add_layer(self, *layers):
        count = len(vars(self))
        for i in range(len(layers)):
            vars(self)[f'layers {count + i}'] = layers[i]


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[1]) * np.sum(y_true * np.log(y_pred + 1e-5))
