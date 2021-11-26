import numpy as np
"""
Перенести на Tensor
"""

class BaseOptimizer(object):
    """
    Входные данные для оптимизатора - список кортежей, состоящих из истории и градиента параметра.
    Если мы не используем исторрию (оптимизационный алгоритм не предполагает использование истории) - передаём None.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, layer_name, parameters):
        res = []
        for i in range(len(parameters)):
            value = -self.learning_rate * parameters[i]
            res.append(value)
        return res


class Momentum(BaseOptimizer):

    def __init__(self, beta, learning_rate, nesterov=True):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity_parameters = {}
        self.nesterov = nesterov

    def apply_gradients(self, layer_name, parameters):
        """
        :parameter: список градиентов параметров
        """
        res = {}
        for key, grad in parameters.items():
            name_parameter = layer_name + '_velocity_' + key
            if name_parameter not in self.velocity_parameters.keys():
                self.velocity_parameters[name_parameter] = 0
            velocity = self.beta * self.velocity_parameters[name_parameter] - self.learning_rate * grad

            if self.nesterov:
                vd_param = self.beta * velocity - self.learning_rate * grad
            else:
                vd_param = velocity
            self.velocity_parameters[name_parameter] = vd_param
            res[key] = vd_param
        return res


class Adam(BaseOptimizer):

    def __init__(self, beta1, beta2, learning_rate, nesterov=False):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.nesterov = nesterov
        self.parameters_adam = {'iteration': 0}

    def apply_gradients(self, layer_name, parameters):
        res = {}
        iteration = 0
        for key, grad in parameters.items():
            name_parameter = layer_name + '_velocity_' + key
            if name_parameter not in self.parameters_adam.keys():
                self.parameters_adam[name_parameter] = [0, 0]  # 1-е список - истории 1-го и 2-го моментов
                                                                    # 2-е значение  - номер итерации
            m = self.beta1 * self.parameters_adam[name_parameter][0] + (1 - self.beta1) * grad
            v = self.beta2 * self.parameters_adam[name_parameter][1] + (1 - self.beta2) * grad ** 2
            if self.nesterov:
                self.parameters_adam[name_parameter][0] = m + (1 - self.beta1) * grad
            else:
                self.parameters_adam[name_parameter][0] = m

            self.parameters_adam[name_parameter][1] = v
            iteration += 1
            m_cor = m / (1 - self.beta1 ** iteration)
            v_cor = v / (1 - self.beta2 ** iteration)

            param_update = -self.learning_rate * (m_cor / (np.sqrt(v_cor) + 1e-6))
            res[key] = param_update

        return res

