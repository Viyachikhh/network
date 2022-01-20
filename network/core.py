import logging
import sys

from collections import OrderedDict
from timeit import default_timer

from network.activations import *
from network.utils import col2im_indices, im2col_indices, getWindows
from network.initializers import XavierWeights, ZerosInitializer
from network.optimizers import Momentum
from network.losses import CategoricalCrossEntropy

from network.utils import get_activation

AVAILABLE_ACTIVATIONS = ['tanh', 'sigmoid', 'relu', 'softmax']
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Layer(ABC):
    count = 0

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
    def update_weights(self, dZ, optimizer):
        pass

    @abstractmethod
    def get_gradients(self, dZ):
        pass


class DenseLayer(Layer):

    def __init__(self, layer_size, activation=None, name=None):
        """
        initialization of object DenseLayer
        """
        self.layer_size = layer_size
        self.prev_layer_size = None
        self.weights_initializer = XavierWeights()
        self.bias_initializer = ZerosInitializer()
        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def build_weights(self, prev_size):
        return self.weights_initializer(prev_size, self.layer_size, (self.layer_size, prev_size)), \
               self.bias_initializer(self.layer_size, size=None)

    def update_weights(self, dZ, optimizer):
        derivative = dZ
        if self.activation_name == 'softmax':
            pass
        elif self.activation is not None:
            derivative *= self.activation.derivative(self.cache[1])

        dW, db, dZ_next = self.get_gradients(derivative)
        parameters = optimizer.apply_gradients(self.name, {'dW': dW, 'db': db})
        self.weights += parameters['dW']
        self.bias += parameters['db']
        return dZ_next

    def get_gradients(self, dZ):
        db = (1 / self.cache[0].shape[1]) * dZ.sum(axis=1)
        dW = (1 / self.cache[0].shape[1]) * (dZ @ self.cache[0].T)
        next_dZ = (self.weights.T @ dZ)
        return dW, db, next_dZ

    def __call__(self, inputs):
        if 'weights' not in vars(self).keys():
            prev_size = inputs.shape[0]
            self.weights, self.bias = self.build_weights(prev_size)

        h = (self.weights @ inputs) + self.bias[:, None]

        result = h if (self.activation_name is None or self.activation_name not in AVAILABLE_ACTIVATIONS) \
            else self.activation(h)
        self.cache = (inputs, result)

        return result

    def __str__(self):
        return f'Dense_layer, w.shape = {self.weights.shape}, b.shape = {self.bias.shape}'


class Conv2DLayer(Layer):

    def __init__(self, count_filters, filter_size, stride=1, padding=False, activation=None, name=None):
        self.prev_channels = None
        self.weights_initializer = XavierWeights()
        self.bias_initializer = ZerosInitializer()

        self.count_filters = count_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride

        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def build_weights(self, prev_size):
        return self.weights_initializer(prev_size, self.count_filters * prev_size * self.filter_size ** 2,
                                        (self.count_filters, prev_size, self.filter_size, self.filter_size)), \
               self.bias_initializer(self.count_filters, size=None)

    def __call__(self, inputs):
        if 'weights' not in vars(self).keys():
            prev_size = inputs.shape[1]
            self.weights, self.bias = self.build_weights(prev_size)
            self.padding_size = (self.weights.shape[2] // 2 if self.padding else 0,
                                 self.weights.shape[3] // 2 if self.padding else 0)

        height_output = (inputs.shape[2] - self.weights.shape[2] + 2 * self.padding_size[0]) // self.stride + 1
        width_output = (inputs.shape[3] - self.weights.shape[3] + 2 * self.padding_size[0]) // self.stride + 1

        # array size - (batch_size, in_channels, height_picture, width_picture, filter_height, filter_width)
        windows = getWindows(inputs, (inputs.shape[0], self.weights.shape[0], height_output, width_output),
                             filter_size=self.weights.shape[2], padding=self.padding_size[0], stride=self.stride)

        Z = np.einsum('ijklmn, ojmn -> iokl', windows, self.weights)  # conv inputs with filter
        Z += self.bias[None, :, None, None]

        result = Z if (self.activation_name is None or self.activation_name not in AVAILABLE_ACTIVATIONS) \
            else self.activation(Z)

        self.cache = (windows, inputs, result)
        return result

    def get_gradients(self, dZ):
        padding = self.weights.shape[2] - 1 if not self.padding else self.padding_size[0]
        input_windows, inputs, _ = self.cache
        windows_dZ = getWindows(dZ, inputs.shape, filter_size=self.weights.shape[2], padding=padding, stride=1,
                                dilate=self.stride - 1)  # dilate is a number of zeros between elems, if stride != 1
        weights_flipped = self.weights[:, :, ::-1, ::-1]  # for computing derivative flip filter by 180 degrees

        dW = np.einsum('ijklmn, iokl -> ojmn', input_windows, dZ)  # convolution input_size and dZ
        # full convolution window_dZ and flipped weights
        dZ_prev = np.einsum('ijklmn, jomn -> iokl', windows_dZ, weights_flipped)
        db = np.sum(dZ, axis=(0, 2, 3))
        return dW, db, dZ_prev

    def update_weights(self, dZ, optimizer):
        derivative = dZ
        if self.activation is not None:
            derivative *= self.activation.derivative(self.cache[2])
        dW, db, dZ_next = self.get_gradients(derivative)
        parameters = optimizer.apply_gradients(self.name, {'dW': dW, 'db': db})
        self.weights += parameters['dW']
        self.bias += parameters['db']
        return dZ_next

    def __str__(self):
        return f'Conv_layer with filter_size = {self.weights.shape[2]} and count_filters = {self.weights.shape[0]}'


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2, input_channels=None, name=None):
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def __call__(self, inputs):
        X_reshaped = inputs.reshape(inputs.shape[0] * inputs.shape[1], 1, inputs.shape[2], inputs.shape[3])

        # unfold input to 2D-size
        X_in_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        self.X_col_size = X_in_col

        self.max_indexes = np.argmax(X_in_col, axis=0)  # indexes of max elements
        result = X_in_col[self.max_indexes, range(self.max_indexes.size)]

        # 2D tensor -> 4D tensor
        result = result.reshape(inputs.shape[2] // self.pool_size, inputs.shape[3] // self.pool_size,
                                inputs.shape[0], inputs.shape[1]).transpose(2, 3, 0, 1)

        self.cache = (inputs, result)

        return result

    def update_weights(self, dZ, optimizer, activation_next=None):
        dZ_next = self.get_gradients(dZ)
        return dZ_next

    def get_gradients(self, dZ):
        dZ_col = np.zeros_like(self.X_col_size)

        # Transpose dZ
        dZ_flat = dZ.transpose(2, 3, 0, 1).ravel()

        # Indexes, with max elements, fill with dZ
        dZ_col[self.max_indexes, range(self.max_indexes.size)] = dZ_flat

        # getting correct dimension for next layer in backward
        shape = (self.cache[0].shape[0] * self.cache[0].shape[1], 1, self.cache[0].shape[2], self.cache[0].shape[3])
        dZ_next = col2im_indices(dZ_col, shape, self.pool_size, self.pool_size, padding=0, stride=self.pool_size)
        dZ_next = dZ_next.reshape(self.cache[0].shape[0], self.cache[0].shape[1],
                                  self.cache[0].shape[2], self.cache[0].shape[3])
        return dZ_next

    def __str__(self):
        return f'MaxPooling layer with pool_size = {self.pool_size}'

    def build_weights(self, prev_size):
        pass


class FlattenLayer(Layer):

    def __init__(self, name=None):
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def get_gradients(self, dZ):
        dZ_next = np.swapaxes(dZ, 0, 1)
        dZ_next = dZ_next.reshape(self.cache[0].shape)
        return dZ_next

    def update_weights(self, dZ, optimizer, activation_next=None):
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


class RecLayer(Layer):

    def __init__(self, n_units, name=None, activation='tanh', return_seq=False):
        """
        W shape = (prev_channels, self.n_units)
        V, U shape = (self.n_units, self.n_units)
        b, c shapes = (self.n_units,)
        """
        self.n_units = n_units
        self.prev_channels = None
        self.weights_initializer = XavierWeights()
        self.bias_initializer = ZerosInitializer()
        self.return_seq = return_seq
        self.expanding = False

        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def __call__(self, inputs):

        if len(inputs.shape) == 3:
            inputs = np.swapaxes(inputs, 0, -1)

        if len(inputs.shape) == 2:
            self.expanding = True
            inputs = np.expand_dims(inputs, axis=1)

        hidden_states = np.zeros((self.n_units, inputs.shape[1] + 1, inputs.shape[-1]))
        outputs = np.zeros((self.n_units, inputs.shape[1], inputs.shape[-1]))

        softmax = Softmax()

        if 'U' not in vars(self).keys():
            prev_size = inputs.shape[0]
            self.W, self.U, self.V, self.b, self.c = self.build_weights(prev_size)

        concat_weights = np.concatenate([self.U, self.W], axis=-1)

        for t in range(inputs.shape[1]):

            concat_input = np.concatenate([hidden_states[:, t - 1], inputs[:, t]], axis=0)
            h = concat_weights @ concat_input + self.b[:, None]
            hidden_states[:, t] = self.activation(h) if self.activation is not None else h
            o = self.V @ hidden_states[:, t] + self.c[:, None]
            outputs[:, t] = softmax(o)

        self.cache = (inputs, hidden_states, outputs)

        if self.return_seq:
            result = np.swapaxes(outputs, 0, -1)
        else:
            result = outputs[:, -1]

        return result

    def build_weights(self, prev_size):
        W = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, prev_size))
        U = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units))
        V = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units))
        b = self.bias_initializer(self.n_units, size=None)
        c = self.bias_initializer(self.n_units, size=None)
        return W, U, V, b, c

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        dW, dU, db, dV, dc, dZ_next = self.get_gradients(dZ)
        parameters = optimizer.apply_gradients(self.name, {'dW': dW, 'db': db,
                                                           'dV': dV, 'dc': dc,
                                                           'dU': dU})
        self.W += parameters['dW']
        self.V += parameters['dV']
        self.U += parameters['dU']
        self.b += parameters['db']
        self.c += parameters['dc']
        return dZ_next

    def get_gradients(self, dZ):

        inputs, hidden_states, outputs = self.cache
        seq_count = inputs.shape[1]

        if len(dZ.shape) == 2:
            dZ = np.expand_dims(dZ, axis=1)

        dZ_next = np.zeros_like(inputs)

        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        dh_next = np.zeros_like(hidden_states[:, 0])
        softmax = Softmax()

        d_out = softmax.derivative(dZ[:, -1])

        for t in reversed(range(seq_count)):
            if seq_count == 1:
                pass
            else:
                if dZ.shape[1] == 1:
                    d_out = np.zeros_like(d_out)
                else:
                    d_out = softmax.derivative(np.swapaxes(dZ[:, t], 0, 1))

            dV += d_out @ np.swapaxes(hidden_states[:, t], 0, 1)
            dc += np.sum(d_out, axis=-1)

            dh = np.swapaxes(self.V, 0, 1) @ d_out + dh_next
            dh_rec = self.activation.derivative(hidden_states[:, t]) * dh

            db += np.sum(dh_rec, axis=-1)

            dW += dh_rec @ np.swapaxes(inputs[:, t], 0, 1)
            dU += dh_rec @ np.swapaxes(hidden_states[:, t+1], 0, 1)

            dh_next = np.swapaxes(self.U, 0, 1) @ dh_rec
            dZ_next[:, t] = np.swapaxes(self.W, 0, 1) @ dh_rec

        if (not self.return_seq and dZ_next.shape[1] != 1) or self.return_seq:
            dZ_next = np.swapaxes(dZ_next, 0, -1)
        else:
            dZ_next = dZ_next[:, -1]

        return dW, dU, db, dV, dc, dZ_next


class LSTM(Layer):
    def __init__(self, n_units, name=None, activation='tanh', return_seq=False):
        """
        W_i, W_f, W_c, W_0  shapes = (self.n_units + prev_shape, self.n_units)
        W_y shape = (self.units, self.units)
        b_i, b_f, b_c, b_o, b_y shapes = (self.n_units,)
        """
        self.n_units = n_units
        self.prev_channels = None
        self.weights_initializer = XavierWeights()
        self.bias_initializer = ZerosInitializer()
        self.return_seq = return_seq
        self.expanding = False

        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1

    def __call__(self, inputs):

        if len(inputs.shape) == 3:
            inputs = np.swapaxes(inputs, 0, -1)

        if len(inputs.shape) == 2:
            self.expanding = True
            inputs = np.expand_dims(inputs, axis=1)

        if 'W_i' not in vars(self).keys():
            prev_size = inputs.shape[0]
            self.W_i, self.W_f, self.W_c, self.W_o, \
                self.b_i, self.b_f, self.b_c, self.b_o = self.build_weights(prev_size)

        f_history = np.zeros((self.n_units, inputs.shape[1], inputs.shape[-1]))
        i_history = np.zeros((self.n_units, inputs.shape[1], inputs.shape[-1]))
        o_history = np.zeros((self.n_units, inputs.shape[1], inputs.shape[-1]))
        c_history = np.zeros((self.n_units, inputs.shape[1], inputs.shape[-1]))
        c_current_history = np.zeros((self.n_units, inputs.shape[1] + 1, inputs.shape[-1]))
        hidden_states = np.zeros((self.n_units, inputs.shape[1] + 1, inputs.shape[-1]))

        sigmoid = Sigmoid()
        tanh = Tanh()

        for t in range(inputs.shape[1]):
            concat_input = np.concatenate([hidden_states[:, t - 1], inputs[:, t]], axis=0)

            #print(concat_input.shape, self.W_i.shape)

            i_history[:, t] = sigmoid(self.W_i @ concat_input + self.b_i[:, None])
            f_history[:, t] = sigmoid(self.W_f @ concat_input + self.b_f[:, None])
            c_history[:, t] = tanh(self.W_c @ concat_input + self.b_c[:, None])
            o_history[:, t] = sigmoid(self.W_o @ concat_input + self.b_o[:, None])

            c_current_history[:, t] = f_history[:, t] * c_current_history[:, t - 1] + i_history[:, t] * c_history[:, t]
            hidden_states[:, t] = o_history[:, t] * tanh(c_current_history[:, t])

        self.cache = (inputs, c_current_history, hidden_states, i_history, f_history, c_history, o_history)

        if self.return_seq:
            result = np.swapaxes(hidden_states[:, :-1], 0, -1)
        else:
            result = hidden_states[:, -1]

        return result

    def build_weights(self, prev_size):
        W_i = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units + prev_size))
        W_f = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units + prev_size))
        W_c = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units + prev_size))
        W_o = self.weights_initializer(prev_size, self.n_units, size=(self.n_units, self.n_units + prev_size))
        b_i = self.bias_initializer(self.n_units, size=None)
        b_f = self.bias_initializer(self.n_units, size=None)
        b_c = self.bias_initializer(self.n_units, size=None)
        b_o = self.bias_initializer(self.n_units, size=None)

        return W_i, W_f, W_c, W_o, b_i, b_f, b_c, b_o

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):

        dW_i, dW_f, dW_c, dW_o, db_i, db_f, db_c, db_o, dZ_next = self.get_gradients(dZ)
        parameters = optimizer.apply_gradients(self.name, {'dW_i': dW_i, 'db_i': db_i,
                                                           'dW_f': dW_f, 'db_f': db_f,
                                                           'dW_c': dW_c, 'db_c': db_c,
                                                           'dW_o': dW_o, 'db_o': db_o})
        self.W_i += parameters['dW_i']
        self.W_f += parameters['dW_f']
        self.W_c += parameters['dW_c']
        self.W_o += parameters['dW_o']
        self.b_i += parameters['db_i']
        self.b_f += parameters['db_f']
        self.b_c += parameters['db_c']
        self.b_o += parameters['db_o']

        return dZ_next

    def get_gradients(self, dZ):

        inputs, c_current_history, hidden_states, i_history, f_history, c_history, o_history = self.cache
        seq_count = inputs.shape[1]

        dW_i, dW_f, dW_c, dW_o = np.zeros_like(self.W_i), np.zeros_like(self.W_f), np.zeros_like(
            self.W_c), np.zeros_like(self.W_o)
        db_i, db_f, db_c, db_o = np.zeros_like(self.b_i), np.zeros_like(self.b_f), np.zeros_like(
            self.b_c), np.zeros_like(self.b_o)

        sigmoid = Sigmoid()
        tan_h = Tanh()

        if len(dZ.shape) == 2:
            dZ = np.expand_dims(dZ, axis=1)

        d_out = dZ[:, -1]

        dh_next = np.zeros_like(hidden_states[:, 0])
        dc_next = np.zeros_like(c_current_history[:, 0])
        dZ_next = np.zeros_like(inputs)

        for t in reversed(range(seq_count)):

            if seq_count == 1:
                pass
            else:
                if dZ.shape[1] == 1:
                    d_out = np.zeros_like(d_out)
                else:
                    d_out = np.swapaxes(dZ[:, t], 0, 1)

            dh = d_out + dh_next

            concat_input = np.concatenate([hidden_states[:, t - 1], inputs[:, t]], axis=0)
            # ----------------
            dho = tan_h(c_current_history[:, t]) * dh
            dho *= sigmoid.derivative(o_history[:, t])

            dW_o += dho @ np.swapaxes(concat_input, 0, 1)
            db_o += np.sum(dho, axis=-1)
            dZ_o = np.swapaxes(self.W_o, 0, 1) @ dho

            # ----------------

            dc = o_history[:, t] * dh * tan_h.derivative(c_current_history[:, t])
            dc += dc_next

            # ----------------

            dhf = c_current_history[:, t - 1] * dc
            dhf *= sigmoid.derivative(f_history[:, t])

            dW_f += dhf @ np.swapaxes(concat_input, 0, 1)
            db_f += np.sum(dhf, axis=-1)
            dZ_f = np.swapaxes(self.W_f, 0, 1) @ dhf

            # ----------------

            dhi = c_history[:, t] * dc
            dhi *= sigmoid.derivative(i_history[:, t])

            dW_i += dhi @ np.swapaxes(concat_input, 0, 1)
            db_i += np.sum(dhi, axis=-1)
            dZ_i = np.swapaxes(self.W_i, 0, 1) @ dhi

            # ----------------

            dhc = i_history[:, t] * dc
            dhc *= tan_h.derivative(c_history[:, t])

            dW_c += dhc @ np.swapaxes(concat_input, 0, 1)
            db_c += np.sum(dhc, axis=-1)
            dZ_c = np.swapaxes(self.W_c, 0, 1) @ dhc

            # ----------------------

            d_concat = dZ_i + dZ_f + dZ_o + dZ_c

            dZ_next[:, t] = d_concat[self.n_units:]

            dh_next = d_concat[:self.n_units]

            dc_next = f_history[:, t] * dc

        if (not self.return_seq and dZ_next.shape[1] != 1) or self.return_seq:
            dZ_next = np.swapaxes(dZ_next, 0, -1)
        else:
            dZ_next = dZ_next[:, -1]

        return dW_i, dW_f, dW_c, dW_o, db_i, db_f, db_c, db_o, dZ_next


class Bidirectional(Layer):
    def __init__(self, base_layer, mode='sum'):

        self.common_layer = base_layer
        self.common_layer.name += '_past_based'

        self.reverse_layer = base_layer
        self.reverse_layer.name += '_future_based'

        self.mode = mode

    def __call__(self, inputs):
        past_based_output = self.common_layer(inputs)
        future_based_output = self.reverse_layer(inputs[:, ::-1])
        return past_based_output + future_based_output[:, ::-1] if self.mode == 'sum' else \
            np.concatenate([past_based_output, future_based_output[:, ::-1]], axis=1)

    def build_weights(self, prev_size):
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):

        if self.mode == 'sum':
            dZ_past = dZ
            dZ_future = dZ[:, ::-1]
        else:
            dZ_past = dZ[:, :dZ.shape[1] // 2]
            dZ_future = dZ[:, dZ.shape[1] // 2:]

        dZ_1_next = self.common_layer.update_weights(dZ_past, optimizer)
        dZ_2_next = self.reverse_layer.update_weights(dZ_future, optimizer)

        return dZ_1_next + dZ_2_next[:, ::-1]

    def get_gradients(self, dZ):
        pass


class NeuralNetwork:
    def __init__(self, *your_layers):

        self.layers_dict = OrderedDict()
        for i in range(len(your_layers)):
            self.layers_dict[f'layer_{i}'] = your_layers[i]

        self.optimizer = Momentum(learning_rate=0.004, beta=0.9, nesterov=True)
        self.loss = CategoricalCrossEntropy()

    def __call__(self, inputs):
        x = inputs
        for i in range(len(self.layers_dict)):
            layer = self.layers_dict[f'layer_{i}']
            x = layer(x)
        return x

    def back_propagation(self):
        layers = list(self.layers_dict.keys())[::-1]
        dZ = self.loss.gradient_loss()
        for i, layer in enumerate(layers):
            dZ = self.layers_dict[layers[i]].update_weights(dZ=dZ, optimizer=self.optimizer)

    def add_layer(self, *layers):
        cur_length = len(self.layers_dict)
        for i in range(len(layers)):
            self.layers_dict[f'layer_{i + cur_length}'] = layers[i]

    def fit(self, data, labels, count_epochs=200, size_of_batch=32, validation_set=None):
        train_loss = []
        valid_loss = []

        for epoch in range(count_epochs):
            rand_int = np.random.randint(data.shape[0], size=size_of_batch)  # generate random examples
            logging.info(f'  epoch {epoch + 1}')
            start = default_timer()
            predictions = self(data[rand_int])
            logging.info(f'  time = {default_timer() - start}')
            loss = self.loss.calculate_loss(labels[rand_int].T, predictions)
            self.back_propagation()
            logging.info(f'  loss = {loss}')
            train_loss.append(loss)
            if validation_set is not None:  # on validation
                val_prediction = self(validation_set[0])
                val_loss = self.loss.calculate_loss(validation_set[1].T, val_prediction, without_memory=True)
                valid_loss.append(val_loss)
                logging.info(f'  validation loss = {val_loss}')
                test_real = np.argmax(validation_set[1].T, axis=0)
                test_prediction = np.argmax(val_prediction, axis=0)
                logging.info(f'  accuracy = {np.mean(test_prediction == test_real)} ')

        return train_loss, valid_loss if validation_set is not None else train_loss
