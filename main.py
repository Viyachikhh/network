import numpy as np
from matplotlib import pyplot as plt

from network.core import categorical_cross_entropy, NeuralNetwork, DenseLayer, Conv2DLayer, FlattenLayer, MaxPoolingLayer
from network.load import load


def train_model(model, data, labels, epochs=200, batch_size=32, val=None):
    train_loss = []
    if val is not None:
        valid_loss = []
    ind = np.arange(data.shape[0])
    for epoch in range(epochs):
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        print(f'epoch {epoch + 1}')
        rand_int = np.random.randint(0, X_train.shape[0] - batch_size + 1)
        pred = model(X_train[rand_int:rand_int + batch_size])
        print(y_train[rand_int:rand_int + batch_size].T.shape, pred.shape)

        loss = categorical_cross_entropy(y_train[rand_int:rand_int + batch_size].T, pred)
        network.back_propagation(y_train[rand_int:rand_int + batch_size].T, pred)
        print(f'loss = {loss}', end=', ')
        train_loss.append(loss)
        if val is not None:
            val_pred = model(val[0])
            val_loss = categorical_cross_entropy(val[1].T, val_pred)
            valid_loss.append(val_loss)
            print(f'validation loss = {val_loss}')
    return train_loss, valid_loss if val is not None else train_loss


X_train, y_train, X_val, y_val = load(reshape=False)


epochs = 200
batch_size = 512


conv = Conv2DLayer(3, 1, filter_size=3, padding=1, activation='relu')
pool = MaxPoolingLayer()
flatten = FlattenLayer()
h_1 = DenseLayer(128, 588, 'relu')
#h_2 = DenseLayer(64, 128, 'relu')
end_network = DenseLayer(11, 128, 'softmax')
network = NeuralNetwork([conv, pool, flatten, h_1, end_network])

train_errors, valid_errors = train_model(network, X_train, y_train, batch_size=batch_size, val=(X_val, y_val))


plt.plot(train_errors, label='train')
plt.plot(valid_errors, label='test')
plt.legend()

