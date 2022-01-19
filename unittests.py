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
        input_shape = (FEATURES, BATCH_SIZE)
        output_shape = (N_UNITS, BATCH_SIZE)
        inputs = np.random.random(input_shape)
        dZ = np.random.random(output_shape)
        layer = RecLayer(n_units=N_UNITS)
        self.assertEqual(layer(inputs).shape, output_shape, 'incorrect rnn forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape, 'incorrect rnn backward')

    def test_rnn_time_staps(self):
        input_shape = (BATCH_SIZE, TIME_STAPS, FEATURES)
        output_shape = (BATCH_SIZE, TIME_STAPS, N_UNITS)
        inputs = np.random.random(input_shape)
        dZ = np.random.random(output_shape)
        layer = RecLayer(n_units=N_UNITS, return_seq=True)
        self.assertEqual(layer(inputs).shape, output_shape, 'incorrect rnn timestap forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape, 'incorrect rnn timestap backward')

    def test_lstm(self):
        input_shape = (FEATURES, BATCH_SIZE)
        output_shape = (N_UNITS, BATCH_SIZE)
        inputs = np.random.random(input_shape)
        dZ = np.random.random(output_shape)
        layer = LSTM(n_units=N_UNITS)

        self.assertEqual(layer(inputs).shape, output_shape, 'incorrect lstm forward')
        self.assertEqual(layer.update_weights(dZ, OPT).shape, input_shape, 'incorrect lstm backward')

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

