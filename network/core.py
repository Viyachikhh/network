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
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        cycle with count of cells state:
            h(t) = activation(weights_input @ inputs + weights_h @ h(t-1) + bias_h)
            y(t) = softmax(weights_output @ h(t) + bias_y
        

        return self.outputs
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

        _, timestaps, _ = dZ.shape (batch, timestaps,

        dW_inputs
        dW_outputs
        dW_state

        for i in reversed(range(timestaps)):
            dW_outputs += dZ[:, t] @ self.states[:, t]
            grad_wrt_state = dZ[:, t] @ (weights_output) * self.activation.backward(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state @ weights_input
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                dW_inputs += grad_wrt_state.T @ (self.layer_input[:, t_])
                dW_state += grad_wrt_state.T @ dot(self.states[:, t_-1])
                grad_wrt_state = (grad_wrt_state @ self.W) * self.activation.backward(self.state_input[:, t_-1])
        """
        pass


class LSTM(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

    def build_weights(self, prev_size):
        pass

    def __str__(self):
        pass

    def update_weights(self, dZ, optimizer):
        pass

    def get_gradients(self, dZ):
        pass


class BiRecLayer(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

    def build_weights(self, prev_size):
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