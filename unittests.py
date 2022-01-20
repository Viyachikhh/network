import numpy as np
import unittest

from network.autograd.structure import *
from network.core import RecLayer, LSTM, Bidirectional
from network.optimizers import Momentum

N_UNITS = 12
BATCH_SIZE = 512
TIME_STAPS = 7
FEATURES = 4
OPT = Momentum(0.9, 0.004)


class TestRecurrentLayers(unittest.TestCase):

    def test_rnn(self):
        input_shape1 = (FEATURES, BATCH_SIZE)
        input_shape2 = (BATCH_SIZE, TIME_STAPS, FEATURES)
        output_shape = (N_UNITS, BATCH_SIZE)
        inputs1 = np.random.random(input_shape1)
        inputs2 = np.random.random(input_shape2)
        dZ = np.random.random(output_shape)
        layer = RecLayer(n_units=N_UNITS)
        layer2 = RecLayer(n_units=N_UNITS)
        self.assertEqual(layer(inputs1).shape, output_shape, 'incorrect rnn forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape1, 'incorrect rnn backward')
        self.assertEqual(layer2(inputs2).shape, output_shape, 'incorrect rnn timestap forward')
        self.assertEqual(layer2.update_weights(dZ, OPT).shape, input_shape2, 'incorrect rnn timestap backward')

    def test_rnn_time_staps(self):
        input_shape = (BATCH_SIZE, TIME_STAPS, FEATURES)
        output_shape = (BATCH_SIZE, TIME_STAPS, N_UNITS)
        inputs = np.random.random(input_shape)
        dZ = np.random.random(output_shape)
        layer = RecLayer(n_units=N_UNITS, return_seq=True)
        self.assertEqual(layer(inputs).shape, output_shape, 'incorrect rnn timestap forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape, 'incorrect rnn timestap backward')

    def test_lstm(self):
        input_shape1 = (FEATURES, BATCH_SIZE)
        input_shape2 = (BATCH_SIZE, TIME_STAPS, FEATURES)
        output_shape = (N_UNITS, BATCH_SIZE)
        inputs1 = np.random.random(input_shape1)
        inputs2 = np.random.random(input_shape2)
        dZ = np.random.random(output_shape)
        layer = LSTM(n_units=N_UNITS)
        layer2 = LSTM(n_units=N_UNITS)
        self.assertEqual(layer(inputs1).shape, output_shape, 'incorrect lstm forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape1, 'incorrect lstm backward')
        self.assertEqual(layer2(inputs2).shape, output_shape, 'incorrect lstm timestap forward')
        self.assertEqual(layer2.update_weights(dZ, OPT).shape, input_shape2, 'incorrect lstm timestap backward')

    def test_lstm_time_staps(self):
        input_shape = (BATCH_SIZE, TIME_STAPS, FEATURES)
        output_shape = (BATCH_SIZE, TIME_STAPS, N_UNITS)
        inputs = np.random.random(input_shape)
        dZ = np.random.random(output_shape)
        layer = LSTM(n_units=N_UNITS, return_seq=True)
        self.assertEqual(layer(inputs).shape, output_shape, 'incorrect lstm timestap forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape, 'incorrect lstm timestap backward')


if __name__ == '__main__':
    unittest.main()
"""

case = 0

if case == 0:
    return_seq = False
    input_shape = (BATCH_SIZE, TIME_STAPS, FEATURES)
    output_shape = (N_UNITS, BATCH_SIZE)
elif case == 1:
    return_seq = False
    input_shape = (FEATURES, BATCH_SIZE)
    output_shape = (N_UNITS, BATCH_SIZE)
else:
    return_seq = True
    input_shape = (BATCH_SIZE, TIME_STAPS, FEATURES)
    output_shape = (BATCH_SIZE, TIME_STAPS, N_UNITS)


inputs = np.random.random(input_shape)
dZ = np.random.random(output_shape)
print(inputs.shape, dZ.shape)
layer = Bidirectional(LSTM(n_units=N_UNITS, return_seq=return_seq))
pred1 = layer(inputs)
pred2 = layer.update_weights(dZ, OPT)
print(pred1.shape, pred2.shape)

"""