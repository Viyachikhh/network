import numpy as np
from matplotlib import pyplot as plt

from network.core import categorical_cross_entropy, NeuralNetwork
from network.load import load

X_train, y_train, X_val, y_val = load()

epochs = 200
batch_size = X_train.shape[0]

neural_counts = [128, 64, 32]
activations = ['relu'] * 2 + ['softmax']
network = NeuralNetwork()
train_loss = []
valid_loss = []
ind = np.arange(X_train.shape[0])
for epoch in range(epochs):
    np.random.shuffle(ind)
    print(f'epoch {epoch + 1}')
    #X_train = X_train[ind]
    #y_train = y_train[ind]
    loss = 0.
    for i in range(0, len(ind), batch_size):
        pred = network(X_train[i:i+batch_size])
        loss += categorical_cross_entropy(y_train[i:i+batch_size], pred)
        network.back_propagation(y_train[i:i+batch_size], pred, batch_size)
        #digits_train = np.argmax(pred[-1], axis=1)
        #digits_real = np.argmax(y_train[i:i+batch_size], axis=1)
        #print(digits_train.shape)
        #print('count of correct predictions = ', round(100 * np.sum(digits_real == digits_train) / batch_size, 3), ' %')
    val_pred = network(X_val)
    val_loss = categorical_cross_entropy(y_val, val_pred[-1])
    print(f'loss = {loss}, val_loss = {val_loss}')
    train_loss.append(loss)
    valid_loss.append(val_loss)

plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='test')
plt.legend()
plt.savefig('graph.png')
