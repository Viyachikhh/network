import numpy as np

from deepnetwork.utils import getWindows


class Graph:

    def __init__(self):
        self.variables = dict()
        self.operations = dict()

    def add_node(self, layer):

        pass

    def forward_pass(self):

        pass

    def backward_pass(self, dZ):

        pass

    def sort_graph(self):

        pass


class Operation:

    def forward(self, tensor):
        pass

    def backward(self, tensor, grad):
        pass


class Tensor:

    count_parameters = 0

    def __init__(self, value, history_tensor=None, gradient=None, operation=None, name=None):
        self._value = value

        if gradient:
            self.requires_grad = True
            self.grad = gradient
        else:
            self.requires_grad = False
            self.grad = np.zeros_like(gradient)

        self.operation = operation
        self.history_update = history_tensor
        self.shape = self._value.shape if isinstance(self.value, np.ndarray) else 1
        self.name = f"Tensor/{Tensor.count_parameters}" if name is None else name

        Tensor.count_parameters += 1

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if not self.requires_grad:
            raise AttributeError('Can\'t reassign constant')
        else:
            self._value = new_value
            self.grad = np.zeros_like(self.grad)

    def __add__(self, other):
        return Add().forward([self, other])

    def __mul__(self, other):
        return Multiply().forward([self, other])

    def __neg__(self):
        return Sub().forward([0, self])

    def __pow__(self, other):
        return Pow().forward([self, other])

    def __sub__(self, other):
        return Sub().forward([self, other])

    def __truediv__(self, other):
        return Divide().forward([self, other])

    def __matmul__(self, other):
        return Matmul().forward([self, other])

    def __repr__(self):
        return f'name: {self.name}, value: {self.value}, gradient: {self.grad}'

    def convolution(self, other, padding_size, strides):
        return Convolution().forward([self, other], padding_size=padding_size, strides=strides)

    def backward(self, optimizer=None, gradient=None):
        if gradient is None:
            self.grad = np.ones(self.value.shape) if isinstance(self.value, np.ndarray) else 1
            gradient = self.grad

        if self.operation:
            gradient = self.operation.backward(self.history_update, gradient)
        if self.history_update:
            for i in range(len(gradient)):
                tensor = self.history_update[i]
                tensor.grad = gradient[i]
                tensor.backward(gradient=gradient[i])


class Add(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Add/{Add.count_operations}'
        Add.count_operations += 1

    def forward(self, tensors):
        if isinstance(tensors[0], (float, int)):
            left_value = tensors[0]
            right_value = tensors[1].value
        elif isinstance(tensors[1], (float, int)):
            left_value = tensors[0].value
            right_value = tensors[1]
        else:
            left_value = tensors[0].value
            right_value = tensors[1].value
        return Tensor(left_value + right_value, tensors, operation=self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0], (float, int)) or isinstance(tensors[1], (float, int)):
            return [grad]
        return [grad, grad]


class Sub(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Sub/{Sub.count_operations}'
        Sub.count_operations += 1

    def forward(self, tensors):
        if isinstance(tensors[0], (float, int)):
            left_value = tensors[0]
            right_value = tensors[1].value
        elif isinstance(tensors[1], (float, int)):
            left_value = tensors[0].value
            right_value = tensors[1]
        else:
            left_value = tensors[0].value
            right_value = tensors[1].value
        return Tensor(left_value - right_value, tensors, operation=self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0], (float, int)):
            return [-grad]
        elif isinstance(tensors[1], (float, int)):
            return [grad]
        return [grad, -grad]


class Multiply(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Multiply/{Multiply.count_operations}'
        Multiply.count_operations += 1

    def forward(self, tensors):
        if isinstance(tensors[0], (float, int)):
            left_value = tensors[0]
            right_value = tensors[1].value
        elif isinstance(tensors[1], (float, int)):
            left_value = tensors[0].value
            right_value = tensors[1]
        else:
            left_value = tensors[0].value
            right_value = tensors[1].value
        return Tensor(left_value * right_value, tensors, operation=self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0], (float, int)):
            return [grad * tensors[1].value]
        elif isinstance(tensors[1], (float, int)):
            return [grad * tensors[0].value]
        return [grad * tensors[1].value, grad * tensors[0].value]


class Pow(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Pow/{Pow.count_operations}'
        Pow.count_operations += 1

    def forward(self, tensors):
        if isinstance(tensors[0], (float, int)):
            left_value = tensors[0]
            right_value = tensors[1].value
        elif isinstance(tensors[1], (float, int)):
            left_value = tensors[0].value
            right_value = tensors[1]
        else:
            left_value = tensors[0].value
            right_value = tensors[1].value
        return Tensor(np.power(left_value, right_value), tensors, operation=self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0], (float, int)):
            return [grad * np.power(tensors[0], tensors[1].value) * np.log(tensors[0])]
        elif isinstance(tensors[1], (float, int)):
            return [grad * np.power(tensors[0].value, tensors[1] - 1) * tensors[1]]
        return [grad * np.power(tensors[0], tensors[1].value) * np.log(tensors[0]),
                grad * np.power(tensors[0].value, tensors[1] - 1) * tensors[1]]


class Matmul(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Matmul/{Matmul.count_operations}'
        Matmul.count_operations += 1

    def forward(self, tensors):
        return Tensor(tensors[0].value @ tensors[1].value, tensors, operation=self)

    def backward(self, tensors, grad):
        return [grad @ tensors[1].value.T, tensors[0].value.T @ grad]


class Divide(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Divide/{Divide.count_operations}'
        Divide.count_operations += 1

    def forward(self, tensors):
        if isinstance(tensors[0], (float, int)):
            left_value = tensors[0]
            right_value = tensors[1].value
        elif isinstance(tensors[1], (float, int)):
            left_value = tensors[0].value
            right_value = tensors[1]
        else:
            left_value = tensors[0].value
            right_value = tensors[1].value
        return Tensor(left_value / right_value, tensors, operation=self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0], (float, int)):
            return [grad * (1 / tensors[1].value)]
        elif isinstance(tensors[1], (float, int)):
            return [-grad * tensors[0].value * (1 / np.power(tensors[1].value, 2))]

        return [grad * (1 / tensors[1].value), -grad * tensors[0].value * (1 / np.power(tensors[1].value, 2))]


class Convolution(Operation):

    count_operations = 0

    def __init__(self):
        self.name = f'Convolution/{Convolution.count_operations}'
        self.cache = None
        Convolution.count_operations += 1

    def forward(self, tensors, padding_size=None, strides=None, dilate=None):

        height_output = (tensors[0].value.shape[2] - tensors[1].value.shape[2] + 2 * padding_size) // strides + 1
        width_output = (tensors[0].value.shape[3] - tensors[1].value.shape[3] + 2 * padding_size) // strides + 1

        output_shape = (tensors[0].value.shape[0], tensors[1].value.shape[1], height_output, width_output)
        windows_input = getWindows(tensors[0].value, output_shape, filter_size=tensors[1].value.shape[2],
                                   padding=padding_size, stride=strides)
        self.cache = (windows_input, padding_size, strides)

        return Tensor(np.einsum('ijklmn, ojmn -> iokl', self.cache[0], tensors[1].value), tensors, operation=self)

    def backward(self, tensors, grad):

        window_grad = getWindows(grad, tensors[0].value.shape, filter_size=tensors[1].value.shape[2],
                                 padding=self.cache[1], stride=1, dilate=self.cache[-1] - 1)
        weights_flipped = tensors[1].value[::-1, ::-1]

        dW = np.einsum('ijklmn, iokl -> ojmn', self.cache[0], grad)
        dZ_prev = np.einsum('ijklmn, jomn -> iokl', window_grad, weights_flipped)

        return [dZ_prev, dW]
