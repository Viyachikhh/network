import numpy as np
from matplotlib import pyplot as plt

from network.core import np, categorical_cross_entropy, NeuralNetwork
from network.load import load

X_train, y_train, X_val, y_val = load()

epochs = 25

network = NeuralNetwork()
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f'epoch {epoch}')
    pred = network(X_train)
    loss = categorical_cross_entropy(y_train, pred[-1])
    network.back_propagation(y_train, pred, X_train.shape[0])
    val_pred = network(X_val)
    val_loss = categorical_cross_entropy(y_val, val_pred[-1])
    digits_train = np.argmax(pred[-1], axis=1)
    digits_test = np.argmax(val_pred[-1], axis=1)
    print(f'loss = {loss}, val_loss = {val_loss}')
    train_loss.append(loss)
    valid_loss.append(val_loss)

plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='test')
plt.legend()
plt.savefig('graph.png')
