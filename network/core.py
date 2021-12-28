from collections import OrderedDict
from timeit import default_timer

from network.activations import *
from network.utils import col2im_indices, im2col_indices, getWindows
from network.initializers import XavierWeights, ZerosInitializer
from network.optimizers import Momentum
from network.losses import CategoricalCrossEntropy

from network.utils import get_activation

AVAILABLE_ACTIVATIONS = ['tanh', 'sigmoid', 'relu', 'softmax']


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

        gradW, gradb, gradZ = self.get_gradients(derivative)
        parameters = optimizer.apply_gradients(self.name, {'gradW': gradW, 'gradb': gradb})
        self.weights += parameters['gradW']
        self.bias += parameters['gradb']
        return gradZ

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

        Z = np.einsum('ijklmn, ojmn -> iokl', windows, self.weights)  # свёртка входных данных с фильтром
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
        gradW, gradb, gradZ = self.get_gradients(derivative)
        parameters = optimizer.apply_gradients(self.name, {'gradW': gradW, 'gradb': gradb})
        self.weights += parameters['gradW']
        self.bias += parameters['gradb']
        return gradZ

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

    def __init__(self, n_units, name, activation='tanh'):
        """
        W shape = (prev_channels, self.n_units)
        V, U shape = (self.n_units, self.n_units)
        b, c shapes = (self.n_units,)
        """
        self.n_units = n_units
        self.prev_channels = None
        self.weights_initializer = XavierWeights()
        self.bias_initializer = ZerosInitializer()

        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.cache = None
        self.name = f'layer_{Layer.count}' if name is None else name
        Layer.count += 1
        pass

    def __call__(self, inputs):

        """
        hidden_states = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.n_units))
        outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))

        concat_weights = np.concatenate([U, W], axis=0)

        for t in range(inputs.shape[1]):
            concat_input = np.concatenate([hidden_states[:, t-1], inputs[:, t]],axis=-1)

            h = concat_input @ concat_weights + b[:, None]
            hidden_states[:, t] = self.activation(h) if self.activation is not None else h

            o = hidden_states[:, t] @ V + c[:, None]
            outputs[:, t] = softmax(o)
        
        self.cache = (inputs, hidden_states, outputs)
        return outputs
        """
        pass

    def build_weights(self, prev_size):
        """
        Generate weights with considering of n_units and prev shape
        """
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        """
        Update weights
        """
        pass

    def get_gradients(self, dZ):
        """

        _, seq_count, _ = dZ.shape  # (batch, seq_count, features)
        dZ_next = np.zeros_like(dZ)

        inputs, hidden_states, outputs = self.cache

        dW = np.zeros_like(W)
        dV = np.zeros_like(V)
        dU = np.zeros_like(U)

        dh_next = np.zeros_like(hidden_states[:, 0])

        for t in reversed(range(seq_count)):

            d_out = np.copy(dZ[:, t])
            d_out = Softmax().derivative(d_out)

            dV += np.swap_axes(hidden_states[:, t], 0, 1) @ d_out
            dc += np.sum(d_out)

            dh = np.swap_axes(V, 0, 1) @ d_out + dh_next
            dh_rec = self.activation.derivative(hidden_states[:, t]) * dh

            db += np.sum(dh_rec)
            dW += np.swap_axes(inputs[:, t-1], 0, 1) @ dh_rec
            dU += np.swap_axes(hidden_states[:, t-1], 0, 1) @ dh_rec

            dh_next = np.swap_axes(U, 0, 1) @ dh_rec
            dZ_next[:, t] = np.swap_axes(W, 0, 1) @ dh_rec

        return dW, dU, db, dV, dc, dZ_next
        """
        pass


class LSTM(Layer):
    def __init__(self):
        """

        W_i, W_f, W_c, W_0  shapes = (self.n_units + prev_shape, self.n_units)

        b_i, b_f, b_c, b_o, c shapes = (self.n_units,)
        V shape = (self.n_units, self.n_units)

        """
        pass

    def __call__(self, inputs):
        """

        f_history = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))
        i_history = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))
        o_history = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))
        c_history = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))

        c_current_history = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.n_units))
        hidden_states = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.n_units))
        outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.n_units))

        for t in range(inputs.shape[1]):

            concat_input = np.concatenate([hidden_states[:, t-1], inputs[:, t]],axis=-1)

            i_history[:, t] = sigmoid(W_i @ concat_input + b_i)
            f_history[:, t] = sigmoid(W_f @ concat_input + b_f)
            c_history[:, t] = tanh(W_c @ concat_input + b_c)
            o_history[:, t] = sigmoid(W_o @ concat_input + b_o)

            c_current_history[:, t] = f_history[:, t] * c_current_history[:, t-1] + i_history[:, t] * c_history[:, t]
            hidden_states[:, t] = o_history[:, t] * tanh(c_current_history[:, t])

            o = hidden_states[:, t] @ V + c[:, None]
            outputs[:, t] = softmax(o)

        self.cache = (inputs, c_current_history, outputs, hidden_states, i_history, f_history, c_history, o_history)
        return o_history

        """
        pass

    def build_weights(self, prev_size):
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        pass

    def get_gradients(self, dZ):
        """
        _, seq_count, _ = dZ.shape
        inputs, c_current_history, hidden_states, i_history, f_history, c_history, o_history = self.cache

        dW_i, dW_f, dW_c. dW_o, dW_y = np.zeros_like(W_i), np.zeros_like(W_f), np.zeros_like(W_c), np.zeros_like(W_o),
                                        np.zeros_like(W_y)

        dh_next = np.zeros_like(hidden_states[:, 0])
        dc_next = np.zeros_like(c_current_history[:, 0])

        for t in reversed(range(seq_count)):

            d_out = np.copy(dZ[:, t])
            d_out = softmax.derivative(d_out)

            dW_y += np.swap_axes(h, 0, 1) @ d_out
            db_y += np.sum(d_out)

            dh = d_out @ np.swap_axes(W_y, 0, 1) + dh_next

            dho = tan_h(c) * dh
            dho = sigmoid.derivative(o_history[:, t]) * dho

            dc = ho * dh * tan_h.derivative(c)
            dc += dc_next

            dhf = c_current_history[:, t-1] * dc
            dhf = sigmoid.derivative(f_history[:, t]) * dhf

            dhi = hc * dc
            dhi = sigmoid.derivative(i_history[:, t]) * dhi

            dhc = hi * dc
            dhc = tan_h.derivative(c_history[:, t]) * dhc

            dW_f = np.swap_axes(inputs[:,t], 0, 1) @ dhf
            db_f = dhf
            dZ_f = dhf @ np.swap_axes(W_f, 0, 1)

            dWi = np.swap_axes(inputs[:,t], 0, 1) @ dhi
            dbi = dhi
            dZ_i = dhi @ np.swap_axes(W_i, 0, 1)

            dW_o = np.swap_axes(inputs[:,t], 0, 1) @ dho
            db_o = dho
            dZ_o = dho @ np.swap_axes(W_o, 0, 1)

            dW_c = np.swap_axes(inputs[:,t], 0, 1) @ dhc
            db_c = dhc
            dZ_c = dhc @ np.swap_axes(W_c, 0, 1)

            dZ_next = dZ_o + dZ_o + dZ_o + dZ_o
            dh_next = dX[:, :H]
            # Gradient for c_old in c = hf * c_old + hi * hc
            dc_next = hf * dc




        """
        pass


class BiRecLayer(Layer):
    def __init__(self):
        """
                self.layer = RecLayer(name='BiRec_1')
                self.layer2 = Rec_Layer(name='BiRec_2')
                """
        pass

    def __call__(self, inputs):
        """

                output1 = self.layer(inputs)
                output2 = self.layer2(inputs[:, ::-1])

                return output1 + output2[:, ::-1]

                """
        pass

    def build_weights(self, prev_size):
        """
             Useless
             """
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        pass

    def get_gradients(self, dZ):
        pass


class BiLSTM(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs):

        pass

    def build_weights(self, prev_size):
        """
        Useless
        """
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        pass

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

    def fit(self, data, labels, count_epochs=200, size_of_batch=32, val=None):
        train_loss = []
        valid_loss = []

        for epoch in range(count_epochs):
            rand_int = np.random.randint(data.shape[0], size=size_of_batch)  # generate random examples
            print(f'epoch {epoch + 1}')
            start = default_timer()
            pred = self(data[rand_int])
            print(f'time = {default_timer() - start}')
            loss = self.loss.calculate_loss(labels[rand_int].T, pred)
            self.back_propagation()
            print(f'loss = {loss}', end=', ')
            train_loss.append(loss)
            if val is not None:  # on validation
                val_pred = self(val[0])
                val_loss = self.loss.calculate_loss(val[1].T, val_pred, without_memory=True)
                valid_loss.append(val_loss)
                print(f'validation loss = {val_loss}')
                test_real = np.argmax(val[1].T, axis=0)
                test_pred = np.argmax(val_pred, axis=0)
                print(f'accuracy = {np.mean(test_pred == test_real)} ')

        return train_loss, valid_loss if val is not None else train_loss
