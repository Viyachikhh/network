import numpy as np
import matplotlib.pyplot as plt


from deepnetwork.core import NeuralNetwork, DenseLayer, Conv2DLayer, FlattenLayer, MaxPoolingLayer
from deepnetwork.optimizers import Adam
from deepnetwork.load import load


def make_model():
    """
    Function to make model
    """
    conv = Conv2DLayer(4, filter_size=3, padding=True, activation='relu', stride=1)
    pool = MaxPoolingLayer()
    conv2 = Conv2DLayer(8, filter_size=3, padding=True, activation='relu', stride=1)
    pool2 = MaxPoolingLayer()
    flatten = FlattenLayer()
    h_1 = DenseLayer(128, 'relu')
    end_network = DenseLayer(11, 'softmax')
    network = NeuralNetwork(conv, pool, conv2, pool2, flatten, h_1, end_network)
    network.optimizer = Adam(beta1=0.90, beta2=0.999, learning_rate=0.004)
    return network


X_train, y_train, X_val, y_val = load(reshape=False)


epochs = 200
batch_size = 512


nn = make_model()

train_errors, valid_errors = nn.fit(data=X_train, labels=y_train,
                                    count_epochs=epochs, validation_set=(X_val, y_val), size_of_batch=batch_size)


plt.plot(train_errors, label='train')
plt.plot(valid_errors, label='test')
plt.legend()
plt.savefig('graph_best.png')
