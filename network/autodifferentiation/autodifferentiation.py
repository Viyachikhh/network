import numpy as np


from typing import List, NamedTuple, Callable, Optional, Union

"""
В дальнейшем, заменить все np.ndarray на Tensor 
"""


class Dependency(NamedTuple):  # для того, чтобы следить за историй градиентоа
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable):
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Graph:
    def __init__(self):
        self.variables = dict()
        self.operations = dict()
        global _g
        _g = self

    def forward_pass(self):
        """
        Выполняется прямой проход графа
        """
        pass

    def backward_pass(self, dZ):
        """
        Выполняется обратный обход графа
        :param dZ: Градиент ошибки
        """
        pass

    def sort_graph(self):
        """
        Топологическая сортировка графа
        """


class Node:
    def __init__(self):
        pass


#  -------------элементы графа-------------

class Operation(Node):

    def __init__(self, left_value, right_value=None, name='Operation '):
        super().__init__()
        self.input_values = [left_value, right_value]
        self.name = name

    def __call__(self):
        pass


class Tensor(Node):
    """
    Класс  переменной
    Каждая переменная будет хранить по итогу значение градиента
    """
    count_parameters = 0

    def __init__(self, value, requires_grad=False, depend_on: List[Dependency] = None, name=None):
        super().__init__()
        self._value = value

        self.requires_grad = requires_grad
        self.grad = None

        self.depend_on = depend_on
        self.shape = self._value.shape
        self.name = f"Variable/{Tensor.count_parameters}" if name is None else name

        _g.variables.update({self: None})
        Tensor.count_parameters += 1

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if self.requires_grad is None:
            raise AttributeError('Can\'t reassign constant')
        else:
            self._value = new_value
            self.grad = 0

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __neg__(self, other):
        pass

    def __pow__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __matmul__(self, other):
        pass

    def backward(self, dVar):
        """
        Обновление переменной с учётом градиента

        :return:
        """
        pass


#  -------операции над элементами---------


class Add(Operation):
    """
    x+y, град. = dZ, dZ
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Add/{Add.count_parameters}' if name is None else name
        Add.count_parameters += 1

    def __call__(self):
        pass


class Multiply(Operation):
    """
    x*y, град. = dZ*y, dZ*x
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Multiply/{Multiply.count_parameters}' if name is None else name
        Multiply.count_parameters += 1

    def __call__(self):
        pass


class Power(Operation):
    """
    x^y, град. = dZ * x^(y-1), dZ * x^y * ln(x)
    """

    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Power/{Power.count_parameters}' if name is None else name
        Power.count_parameters += 1

    def __call__(self):
        pass


class Matmul(Operation):
    """
    x@y, град. = dZ @ y.T, x.T @ dZ
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Matmul/{Matmul.count_parameters}' if name is None else name
        Matmul.count_parameters += 1

    def __call__(self):
        pass


class Divide(Operation):
    """
    x/y, град. = dZ * (1/y) , dZ * (-x) * (1 / y^2)
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Divide/{Divide.count_parameters}' if name is None else name
        Divide.count_parameters += 1

    def __call__(self):
        pass


class Convolution(Operation):
    """
    Мб и нет необходимости, если обычную свёртку приводить к матричному умножения
    Я думаю
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__(left_value, right_value, name)
        self.name += f'Convolution/{Divide.count_parameters}' if name is None else name
        Convolution.count_parameters += 1

    def __call__(self):
        pass
