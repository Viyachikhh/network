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
        pred = model(X_train[rand_int:rand_int + batch_size])
        loss = categorical_cross_entropy(y_train[rand_int:rand_int + batch_size], pred)
        network.back_propagation(y_train[rand_int:rand_int + batch_size], pred, batch_size)
        if val is not None:
            val_pred = model(val[0])
            val_loss = categorical_cross_entropy(val[1], val_pred)
            valid_loss.append(val_loss)
            print(f'validation loss = {val_loss}', end=', ')
        print(f'loss = {loss}')
        train_loss.append(loss)
    return train_loss, valid_loss if val is not None else train_loss


X_train, y_train, X_val, y_val = load()


epochs = 200
batch_size = 512

neural_counts = [512, 256, 128, 256]
activations = ['relu'] * 4
#h_1 = DenseLayer(256, 784, 'relu')
#h_2 = DenseLayer(128, 256, 'relu')
pre_end = DenseLayer(64, neural_counts[-1], 'relu')
end_network = DenseLayer(11, 64, 'softmax')
network = NeuralNetwork(pre_end, end_network, input_shape=X_train.shape[1],
                        dense_layer_count=4, neural_counts=neural_counts, activations=activations)

train_errors, valid_errors = train_model(network, X_train, y_train, val=(X_val, y_val), batch_size=batch_size)


plt.plot(train_errors, label='train')
plt.plot(valid_errors, label='test')
plt.legend()
