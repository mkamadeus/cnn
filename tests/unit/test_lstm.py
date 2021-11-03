import numpy as np
from lembek.layer import LSTM
from icecream import ic


def test_lstm():
    # fico
    weights = np.array([[[0.7, 0.45]], [[0.95, 0.8]], [[0.45, 0.25]], [[0.6, 0.4]]])

    recurrent_weights = np.array([[[0.1]], [[0.8]], [[0.15]], [[0.25]]])

    biases = np.array([[[0.15]], [[0.65]], [[0.2]], [[0.1]]])

    layer = LSTM(size=1, input_size=(2, 2), weights=weights, recurrent_weights=recurrent_weights, biases=biases)
    result = layer.run(inputs=np.array([[1, 2], [0.5, 3]]))
    ic(result)
    expected = np.array([0.7719811058])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None


def test_lstm_2():
    # fico
    weights = np.array([[[0.03]], [[0.5]], [[0.3]], [[0.02]]])

    recurrent_weights = np.array([[[0.06]], [[0.25]], [[0.4]], [[0.04]]])

    biases = np.array([[[0.002]], [[0.01]], [[0.05]], [[0.001]]])

    layer = LSTM(size=1, input_size=(2, 2), weights=weights, recurrent_weights=recurrent_weights, biases=biases)
    result = layer.run(inputs=np.array([[0.1]]))
    ic(result)
    expected = np.array([0.02057522921])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None
