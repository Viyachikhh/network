import numpy as np
from mnist import MNIST


def to_categorical(label):
    list_ = [0.] * 11
    list_[label] = 1.
    return list_


def load(reshape=False):
    loader = MNIST('/home/_viyachikhh/python-mnist/data/')
    X_train, y_train = loader.load_training()
    X_train = np.array(X_train)
    X_train = X_train / 255. - 0.5
    if not reshape:
        X_train = np.expand_dims(X_train.reshape(-1, 28, 28), axis=1)
        #plt.imshow(X_train[46])
        #plt.savefig('img.jpg')
    y_train = np.array(list(map(lambda x: to_categorical(x), y_train)))
    X_test, y_test = loader.load_testing()
    X_test = np.array(X_test)
    X_test = X_test / 255. - 0.5
    if not reshape:
        X_test = np.expand_dims(X_test.reshape(-1, 28, 28), axis=1)
    y_test = np.array(list(map(lambda x: to_categorical(x), y_test)))
    return X_train, y_train, X_test, y_test
