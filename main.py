from network import np, categorical_cross_entropy, NeuralNetwork
from network import load

X_train, y_train, X_test, y_test = load()

epochs = 200

network = NeuralNetwork()
for epoch in range(epochs):
    print(f'epoch {epoch}')
    pred = network(X_train)
    loss = categorical_cross_entropy(y_train, pred[-1])
    network.back_propagation(y_train, pred, X_train.shape[0])
    val_pred = network(X_test)
    val_loss = categorical_cross_entropy(y_test, val_pred[-1])
    digits_train = np.argmax(pred[-1], axis=1)
    digits_test = np.argmax(val_pred[-1], axis=1)
    print(f'loss = {loss}, val_loss = {val_loss}')
