import numpy as np
from matplotlib import pyplot as plt

from timeit import default_timer

from network.core import categorical_cross_entropy, NeuralNetwork, DenseLayer, Conv2DLayer, FlattenLayer, MaxPoolingLayer
from network.load import load


def make_model():
    """
    Function to make model
    """
    conv = Conv2DLayer(4, 1, filter_size=3, padding=False, activation='relu', stride=2)
    pool = MaxPoolingLayer()
    conv2 = Conv2DLayer(8, 4, filter_size=3, padding=True, activation='relu', stride=2)
    pool2 = MaxPoolingLayer()
    flatten = FlattenLayer()
    h_1 = DenseLayer(128, 64, 'relu')
    end_network = DenseLayer(11, 128, 'softmax')
    network = NeuralNetwork([conv, pool, flatten, h_1, end_network])
    return network


def train_model(model, data, labels, epochs=200, batch_size=32, val=None, learning_rate=0.016/4, beta=0.9):
    train_loss = []
    if val is not None:
        valid_loss = []
    ind = np.arange(data.shape[0])
    for epoch in range(epochs):
        np.random.shuffle(ind)  # перемешивание данных
        data = data[ind]
        labels = labels[ind]
        print(f'epoch {epoch + 1}')
        rand_int = np.random.randint(0, X_train.shape[0] - batch_size + 1)  # индекс, чтобы обучиться до случайной подвыборки
        start = default_timer()
        pred = model(X_train[rand_int:rand_int + batch_size])  # генерирование предсказаний
        print(f'time = {default_timer() - start}')  # сколько времени занимала одна эпоха обучения
        loss = categorical_cross_entropy(y_train[rand_int:rand_int + batch_size].T, pred)
        model.back_propagation(y_train[rand_int:rand_int + batch_size].T, pred, learning_rate=learning_rate, beta=beta)
        print(f'loss = {loss}', end=', ')
        train_loss.append(loss)
        if val is not None:  # на валидационной
            val_pred = model(val[0])
            val_loss = categorical_cross_entropy(val[1].T, val_pred)
            valid_loss.append(val_loss)
            print(f'validation loss = {val_loss}')
    return train_loss, valid_loss if val is not None else train_loss


X_train, y_train, X_val, y_val = load(reshape=False)


epochs = 200
batch_size = 1024


nn = make_model()

train_errors, valid_errors = train_model(nn, X_train, y_train, batch_size=batch_size, val=(X_val, y_val))


plt.plot(train_errors, label='train')
plt.plot(valid_errors, label='test')
plt.legend()
#plt.savefig('2conv_2pol_dense.png')
