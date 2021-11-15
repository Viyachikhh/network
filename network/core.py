import numpy as np

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
                 padding=False, activation=None):
        self.weights = np.random.uniform(low=-np.sqrt(6 / (1 + count_filters * inp_channels * filter_size ** 2)),
                                         high=np.sqrt(6 / (1 + count_filters * inp_channels * filter_size ** 2)),
                                         size=(count_filters, inp_channels, filter_size, filter_size))
        self.bias = np.zeros((count_filters, 1, 1))
        self.padding = padding
        self.stride = stride
        self.history = (0, 0)
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None
        self.padding_size = (self.weights.shape[2] // 2 if self.padding else 0,
                             self.weights.shape[3] // 2 if self.padding else 0)

    def __call__(self, inputs):

        if self.padding:  # заполнение нулями
            filling = ((0, 0),
                       (0, 0),
                       (self.padding_size[0], self.padding_size[0]),
                       (self.padding_size[1], self.padding_size[1]))
            pad_mass = np.pad(inputs, filling, 'constant', constant_values=0)
        else:
            pad_mass = inputs

        h = self.convolution(pad_mass) + self.bias
        self.cache = (pad_mass, inputs.shape, h)
        result = h if self.activation is None else self.activation(h)

        return result

    def get_gradients(self, dZ, activation_next=None):
        db = np.sum(dZ, axis=(0, 2, 3))
        db = np.expand_dims((1 / self.cache[0].shape[0]) * db.reshape(self.weights.shape[0], -1), axis=-1)

        dW, dZ_next = self.convolution_reverse(dZ)  # производные

        derivative = DICT_DERIVATIVES.get(activation_next, None)
        derivative_values = derivative(self.cache[0]) if derivative is not None else self.cache[0]

        if self.padding:  # если мы заполняли нулями, берём срез, иначе в последних двух размерностях будут нули
            dZ_next = dZ_next * derivative_values[:, :, self.padding_size[0]: -self.padding_size[0],
                                self.padding_size[1]: -self.padding_size[1]]
        else:
            dZ_next = dZ_next * derivative_values
        return dW, db, dZ_next

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

    def convolution(self, inputs):
        # функция, разбивающая 4-хмерный массив на окна, делая его шестимерным, количество окон вдоль одной картинки -
        # кол-во каналов * (ширина картинки - ширина фильтра + 1) * (высота картинки - высота фильтра + 1), окна такого
        # же размера, как и фильтр)
        windows = np.lib.stride_tricks.sliding_window_view(inputs,
                                                           [self.weights.shape[2], self.weights.shape[3]],
                                                           axis=(-2, -1))[:, :, ::self.stride, ::self.stride, :, :]
        # (batch_size, channels, height, width)

        # вариант реализации свёртки через einsum - перемножение происходит между осями, обозначенные одинаковой буквой,
        # оси, буквы которых не написаны после стрелки - вдоль них происходит суммирование, (m,n) - ширина и высота
        # фильтра. Собственно, для этого я и разбивал на окна с помощью функции выше. Ну, и чтобы не двигаться в цикле
        # по массиву, а сразу перемножать.
        result = np.einsum('ijklmn, ojmn -> iokl', windows, self.weights)
        return result

    def convolution_reverse(self, dZ):
        #  Когда считаем производные весов в свёрточном слое, производная тоже будет свёрткой. Для того, чтобы сделать
        #  аналогичные преобразования, как в функции convolution, используем такую же функцию, используя свёртку на том,
        #  что было на входе, с тем, что поступило на вход back propagation (dZ)

        dZ_stride = np.zeros((dZ.shape[0], dZ.shape[1],
                              dZ.shape[2] * self.stride,
                              dZ.shape[3] * self.stride))

        dZ_stride[:, :, ::self.stride, ::self.stride] = dZ
        windows_res = np.lib.stride_tricks.sliding_window_view(self.cache[0],
                                                               [dZ_stride.shape[2], dZ_stride.shape[3]],
                                                               axis=(-2, -1))

        # аналогично перемножаем нужные оси, чтобы получить производную весов
        print(windows_res.shape, dZ_stride.shape)
        dW = np.einsum('ijklmn, iomn -> ojkl', windows_res, dZ_stride)
        dW *= (1 / self.cache[0].shape[0])

        dZ_padding = self.weights.shape[2] // 2

        weights_flipped = self.weights[:, :, ::-1, ::-1]
        # производная по тому, что было на входе - полная свёртка dZ и фильтра, который повернули на 180 градусов

        if self.padding:  # если мы заполняли нулями на входе
            # ещё раз, чтобы не потерять размерности
            filling = ((0, 0), (0, 0), (dZ_padding, dZ_padding), (dZ_padding, dZ_padding))
        else:
            # иначе, придётся заполнять два раза
            filling = ((0, 0), (0, 0), (2 * dZ_padding, 2 * dZ_padding), (2 * dZ_padding, 2 * dZ_padding))

        padded = np.pad(dZ_stride, filling, 'constant', constant_values=0)

        # аналогичным образом, разбиваем на окна для нормальной возможности умножения
        dZ_windows = np.lib.stride_tricks.sliding_window_view(padded, [padded.shape[2] - self.weights.shape[2] + 1,
                                                                       padded.shape[3] - self.weights.shape[3] + 1],
                                                              axis=(-2, -1))

        # расчёт производной dZ для следующего слоя в back propagation
        dZ_next = np.einsum('ijklmn, jokl -> iomn', dZ_windows, weights_flipped)
        #assert dZ_next.shape == self.cache[1], f'{dZ_next.shape}, {self.cache[1]}'
        return dW, dZ_next

    def __str__(self):
        return 'Conv_layer'


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2):
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


class FlattenLayer(Layer):

    def __init__(self):
        self.cache = None

    def get_gradients(self, dZ):
        dZ_next = np.swapaxes(dZ, 0, 1)
        dZ_next = dZ_next.reshape(self.cache[0].shape)
        # print(dZ_next.shape)
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
                except AttributeError:  # на случай, если в слое не предусмотрена функция активации
                    activation = None
            else:
                activation = None
            dZ = getattr(self, layers[i]).update_weights_and_history(dZ, learning_rate, beta, activation)


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[1]) * np.sum(y_true * np.log(y_pred + 1e-5))
