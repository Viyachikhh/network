from __init__ import MNIST, np


def to_categorical(label):
    list_ = [0.] * 11
    list_[label] = 1.
    return list_


def load():
    loader = MNIST('/home/_viyachikhh/python-mnist/data/')
    X_train, y_train = loader.load_training()
    X_train = np.array(X_train)
    X_train = X_train / 255. - 0.5
    y_train = np.array(list(map(lambda x: to_categorical(x), y_train)))
    X_test, y_test = loader.load_testing()
    X_test = np.array(X_test)
    X_test = X_test / 255. - 0.5
    y_test = np.array(list(map(lambda x: to_categorical(x), y_test)))
    return X_train, y_train, X_test, y_test
