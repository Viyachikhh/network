import numpy as np


class Graph:
    def __init__(self):
        self.constants = dict()  # ключи - вершины графа, чтобы запоминать порядок, значения пусть будут None
        self.variables = dict()
        self.placeholder = dict()
        self.operations = dict()
        global _g
        _g = self

    def reset_counts(self, root):
        if hasattr(root, 'count_parameters'):
            root.count_parameters = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del _g
        except:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()

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
        :param graph:
        :return:
        """


class Node:
    def __init__(self):
        pass


#  -------------элементы графа-------------

class Operation(Node):

    def __init__(self, name='Operation '):
        _g.operations.update({self: None})
        super().__init__()
        self.value = None
        self.input = []
        self.gradients = None
        self.name = name


class Placeholder(Node):
    count_parameters = 0

    def __init__(self, value, name):
        _g.placeholder.update({self: None})
        super().__init__()
        self.value = None
        self.gradient = None
        self.name = f"Placeholder/{Placeholder.count_parameters}" if name is None else name
        Placeholder.count_parameters += 1


class Constant(Node):
    count_parameters = 0

    def __init__(self, value, name):
        _g.constants.update({self: None})
        super().__init__()
        self._value = value
        self.gradient = 0
        self.name = f"Constant/{Constant.count_parameters}" if name is None else name
        Constant.count_parameters += 1

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        raise AttributeError('Can\'t reassigned constant')


class Variable(Node):
    """
    Класс  переменной
    Каждая переменная будет хранить по итогу значение градиента
    """
    count_parameters = 0

    def __init__(self, value, name):
        _g.variables.update({self: None})
        super().__init__()
        self._value = value
        self.gradient = None  # Здесь будет храниться градиент переменной,
        # он же будет отсюда вытаскиваться в оптимизационный алгоритм
        self.name = f"Variable/{Variable.count_parameters}" if name is None else name
        Variable.count_parameters += 1

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def update_variable_with_gradients(self, dVar):
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
        super().__init__()
        self.name += f'Add/{Add.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Add.count_parameters += 1

    def forward(self):
        return self.input_value[0].value + self.input_value[1].value

    def backward(self, dZ):
        self.input_value[0].gradient = dZ.value
        self.input_value[1].gradient = dZ.value


class Multiply(Operation):
    """
    x*y, град. = dZ*y, dZ*x
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__()
        self.name += f'Mul/{Multiply.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Multiply.count_parameters += 1

    def forward(self):
        return self.input_value[0].value * self.input_value[1].value

    def backward(self, dZ):
        self.input_value[0].gradient = dZ.value * self.input_value[1].value
        self.input_value[1].gradient = dZ.value * self.input_value[0].value


class Power(Operation):
    """
    x^y, град. = dZ * x^(y-1), dZ * x^y * ln(x)
    """

    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__()
        self.name = f'Pow/{Power.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Power.count_parameters += 1

    def forward(self):
        return self.input_value[0].value ** self.input_value[1].value

    def backward(self, dZ):
        self.input_value[0].gradient = dZ.value * self.input_value[0].value ** (self.input_value[1].value - 1)
        self.input_value[1].gradient = dZ.value * (self.input_value[0].value ** self.input_value[1].value) * \
                                       np.log(self.input_value[0].value)


class Matmul(Operation):
    """
    x@y, град. = dZ @ y.T, x.T @ dZ
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__()
        self.name = f'Matmul/{Matmul.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Matmul.count_parameters += 1

    def forward(self):
        return self.input_value[0].value @ self.input_value[1].value

    def backward(self, dZ):
        self.input_value[0].gradient = dZ.value @ self.input_value[1].value.T
        self.input_value[1].gradient = self.input_value[0].value.T @ dZ.value


class Divide(Operation):
    """
    x/y, град. = dZ * (1/y) , dZ * (-x) * (1 / y^2)
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__()
        self.name = f'Divide/{Divide.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Divide.count_parameters += 1

    def forward(self):
        return self.input_value[0].value / self.input_value[1].value

    def backward(self, dZ):
        self.input_value[0].gradient = dZ.value * (1 / self.input_value[1].value)
        self.input_value[1].gradient = dZ.value * (-self.input_value[0].value) * (1 / self.input_value[1].value ** 2)


class Convolution(Operation):
    """
    Мб и нет необходимости, если обычную свёртку приводить к матричному умножения
    Я думаю
    """
    count_parameters = 0

    def __init__(self, left_value, right_value, name=None):
        super().__init__()
        self.name = f'Convolution/{Convolution.count_parameters}/' if name is None else name
        self.input_value = [left_value, right_value]
        Convolution.count_parameters += 1

    def forward(self):
        pass

    def backward(self, dZ):
        pass


class Exp(Operation):
    count_parameters = 0

    def __init__(self, value, name=None):
        super().__init__()
        self.name = f'Exp/{Exp.count_parameters}/' if name is None else name
        self.input_value = value
        Exp.count_parameters += 1

    def forward(self):
        return np.exp(self.input_value)

    def backward(self, dZ):
        self.input_value.gradient = dZ.value * np.exp(self.input_value.value)


def overloading_for_Node(func, self, other):
    """
    Функция для создания конструкторов операций
    :param other: Variable/Constants
    :param func: Operation
    :param self: Variable
    :return: объект операций для конкретных переменных или констант
    """
    if isinstance(other, Node):
        return func(self, other)

    elif isinstance(other, float) or isinstance(other, int):  # случай с константой
        return func(self, Constant(other))

    elif other is None:
        return func(self)


Node.__add__ = lambda self, other: overloading_for_Node(Add, self, other)
Node.__mul__ = lambda self, other: overloading_for_Node(Multiply, self, other)
Node.__div__ = lambda self, other: overloading_for_Node(Divide, self, other)
Node.__neg__ = lambda self: overloading_for_Node(Multiply, self, Constant(-1, name=None))
Node.__pow__ = lambda self, other: overloading_for_Node(Power, self, other)
Node.__matmul__ = lambda self, other: overloading_for_Node(Matmul, self, other)
Node.convolution = lambda self, other: overloading_for_Node(Convolution, self, other)
Node.exp = lambda self: overloading_for_Node(Exp, self, other=None)
