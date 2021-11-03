import numpy as np
from matplotlib import pyplot as plt

from network.core import categorical_cross_entropy, NeuralNetwork, DenseLayer
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
        pred = model(X_train[rand_int:rand_int + batch_size].T)
        loss = categorical_cross_entropy(y_train[rand_int:rand_int + batch_size], pred.T)
        network.back_propagation(y_train[rand_int:rand_int + batch_size].T, pred, batch_size)
        print(f'loss = {loss}', end=', ')
        train_loss.append(loss)
        if val is not None:
            val_pred = model(val[0].T)
            val_loss = categorical_cross_entropy(val[1], val_pred.T)
            valid_loss.append(val_loss)
            print(f'validation loss = {val_loss}')
    return train_loss, valid_loss if val is not None else train_loss


X_train, y_train, X_val, y_val = load()


epochs = 200
batch_size = 512

activations = ['relu', 'softmax']
h_1 = DenseLayer(256, X_train.shape[1], 'relu')
h_2 = DenseLayer(128, 256, 'relu')
pre_end = DenseLayer(64, 128, 'relu')
end_network = DenseLayer(11, 64, 'softmax')
network = NeuralNetwork([h_1, h_2, pre_end, end_network])

#train_errors, valid_errors = train_model(network, X_train, y_train, val=(X_val, y_val), batch_size=batch_size)


#plt.plot(train_errors, label='train')
#plt.plot(valid_errors, label='test')
#plt.legend()
