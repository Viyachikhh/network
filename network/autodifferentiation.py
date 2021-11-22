import numpy as np


class Graph:

    def __init__(self):
        pass


class Node:

    def __init__(self):
        pass


class Operation(Node):
    """
    add
    mul
    matmul
    conv (потому что свёртка отдельно через einsum)
    power(x^a,...)
    exp(a^x,...)
    divide
    """
    def __init__(self):
        super().__init__()


class Placeholder(Node):
    def __init__(self):
        super().__init__()


class Constants(Node):
    def __init__(self):
        super().__init__()


class Variables(Node):
    def __init__(self):
        super().__init__()
