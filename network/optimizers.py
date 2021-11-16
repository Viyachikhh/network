

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Входные данные для оптимизатора - список кортежей, состоящих из истории и градиента параметра.
    Если мы не используем исторрию (оптимизационный алгоритм не предполагает использование истории) - передаём None.
    """
    def __init__(self):
        pass


class NAG(BaseOptimizer):

    def __init__(self, beta, learning_rate):
        super().__init__()
        self.beta = beta
        self.learning_rate = learning_rate

    def apply_gradients(self, parameters):
        """
        :parameter: список кортежей вида (параметр, градиент)
        """
        res = []
        for i in range(len(parameters)):
            velocity = self.beta * parameters[i][0] - self.learning_rate * parameters[i][1]
            vd_param = self.beta * velocity - self.learning_rate * parameters[i][1]
            res.append(vd_param)
        return res


class Momentum(BaseOptimizer):

    def __init__(self, beta, learning_rate):
        super().__init__()
        self.beta = beta
        self.learning_rate = learning_rate

    def apply_gradients(self, parameters):
        """
        :parameter: список кортежей вида (параметр, градиент)
        """
        res = []
        for i in range(len(parameters)):
            vd_param = self.beta * parameters[i][0] - self.learning_rate * parameters[i][1]
            res.append(vd_param)
        return res


class Adam(BaseOptimizer):

    def __init__(self, beta1, beta2, learning_rate):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate

    def apply_gradients(self, parameters):
        pass
