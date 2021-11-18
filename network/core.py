from abc import ABC, abstractmethod
from collections import OrderedDict
from timeit import default_timer


from network.activations import *
from network.utils import col2im_indices, im2col_indices, getWindows
from network.initializers import XavierWeights
from network.optimizers import Momentum

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
    def update_weights_and_history(self, dZ, optimizer, layer_string_name, activation_next=None):
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
        self.activation = DICT_ACTIVATIONS.get(activation, None)
        self.cache = None

    def build_weights(self, prev_size):
        return self.weights_initializer(prev_size, self.layer_size, (self.layer_size, prev_size))

    def update_weights_and_history(self, dZ, optimizer, layer_string_name, activation_next=None):
        gradW, gradb, gradZ = self.get_gradients(dZ, activation_next=activation_next)
        parameters = optimizer.apply_gradients(layer_string_name, {'gradW': gradW, 'gradb': gradb})
        self.weights += parameters['gradW']
        self.bias += parameters['gradb']
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

    def update_weights_and_history(self, dZ, optimizer, layer_string_name, activation_next=None):
        """
        Собственно, обновление весов
        """
        gradW, gradb, gradZ = self.get_gradients(dZ, activation_next=activation_next)
        parameters = optimizer.apply_gradients(layer_string_name, {'gradW': gradW, 'gradb': gradb})
        self.weights += parameters['gradW']
        self.bias += parameters['gradb']
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

    def update_weights_and_history(self, dZ, optimizer, layer_string_name, activation_next=None):
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

    def update_weights_and_history(self, dZ, optimizer, layer_string_name, activation_next=None):
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

        self.layers_dict = OrderedDict()
        for i in range(len(your_layers)):
            self.layers_dict[f'layer_{i}'] = your_layers[i]

        self.optimizer = Momentum(learning_rate=0.004, beta=0.9, nesterov=True)

    def __call__(self, inputs):
        for i in range(len(self.layers_dict)):
            layer = self.layers_dict[f'layer_{i}']
            x = layer(inputs) if i == 0 else layer(x)
        return x

    def back_propagation(self, y_true, y_pred):
        layers = list(self.layers_dict.keys())[::-1]
        dZ = y_pred - y_true
        for i, layer in enumerate(layers):
            if i != len(layers) - 1:
                try:
                    activation = REV_ACTIVATIONS.get(self.layers_dict[layers[i + 1]].activation, None)
                except AttributeError:  # на случай, если в слое не предусмотрена функция активации
                    activation = None
            else:
                activation = None
            dZ = self.layers_dict[layers[i]].update_weights_and_history(dZ=dZ, optimizer=self.optimizer,
                                                                        activation_next=activation,
                                                                        layer_string_name=f'layer_{i}')

    def add_layer(self, *layers):
        cur_length = len(self.layers_dict)
        for i in range(len(layers)):
            self.layers_dict[f'layer_{i + cur_length}'] = layers[i]

    def fit(self, data, labels, count_epochs=200, size_of_batch=32, val=None):
        train_loss = []
        if val is not None:
            valid_loss = []
        ind = np.arange(data.shape[0])
        for epoch in range(count_epochs):
            np.random.shuffle(ind)  # перемешивание данных
            data = data[ind]
            labels = labels[ind]
            print(f'epoch {epoch + 1}')
            rand_int = np.random.randint(0, data.shape[0] - size_of_batch + 1)
            start = default_timer()
            pred = self(data[rand_int:rand_int + size_of_batch])  # генерирование предсказаний
            print(f'time = {default_timer() - start}')  # сколько времени занимала одна эпоха обучения
            loss = categorical_cross_entropy(labels[rand_int:rand_int + size_of_batch].T, pred)
            self.back_propagation(labels[rand_int:rand_int + size_of_batch].T, pred)
            print(f'loss = {loss}', end=', ')
            train_loss.append(loss)
            if val is not None:  # на валидационной
                val_pred = self(val[0])
                val_loss = categorical_cross_entropy(val[1].T, val_pred)
                valid_loss.append(val_loss)
                print(f'validation loss = {val_loss}')
        return train_loss, valid_loss if val is not None else train_loss


def categorical_cross_entropy(y_true, y_pred):
    return (-1 / y_true.shape[1]) * np.sum(y_true * np.log(y_pred + 1e-5))
