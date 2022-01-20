# Package installing

1) Download from repository

2) Create venv

3) Inside venv write command: pip install <path_to_repository>

# Watching

Modules: 

1) core - module with layers and NeuralNetwork object. There are fully connected layer, convolution layer and max pooling ( only 2D), Flatten and rnn layers (base recurrent and LSTM).

2) intializers - module with random weights initializers for layers.

3) activations - module with activations for layers.

4) losses - module with losses for neural networks.

5) utils - module for some custom thing for other modules.

6) load - module for fast loading MNIST dataset if you want to check workability of convolution layers.

7) optimizer - module with optimizers for neural network (There are Momentum and Adam)


# How to use

1) Import from core all necessary layers that you want to use(for CNN - Conv2DLayer, MaxPoolingLayer, Flatten and Dense) and object NeuralNetwork

2) Push layers in NeuralNetwork like: NeuralNetwork(Conv1, MaxPool1,..., flatten, ... , Dense(act='softmax')) - it's only example

3) If you want to check on MNIST, please enter in code: X_train, y_train, X_val, y_val = load(reshape=False). 
Reshape - if you want to work with reshaped vectors(not 28x28, but 784)

4) You can also import from optimizer and losses if you want to check workability of another losses/optimizers.

5) You can change them, by getting object variables 'optimizer' and 'loss'

6) If you want to fit network, call method 'fit':def <b>fit</b>(<i>self, data, labels, count_epochs=200, size_of_batch=32, validation_set=None</i>)

Validation set - tuple of validation data and validation labels


# Docker container

If you want use docker, you should know, that script <i>main.py</i> starts, when you start docker container. If
you want to check neural network metrics, enter to cmd "sudo docker logs <container_name>".

P.S I'm not a strong user of docker(not yet), so there are maybe some troubles (on my computer container work)



