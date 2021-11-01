import numpy as np
from lembek.layer import LSTM
from icecream import ic


def test_lstm():
    weights = np.array([[[0.7, 0.45]], [[0.95, 0.8]], [[0.45, 0.25]], [[0.6, 0.4]]])

    recurrent_weights = np.array([[[0.1]], [[0.8]], [[0.15]], [[0.25]]])

    biases = np.array([[[0.15]], [[0.65]], [[0.2]], [[0.1]]])

    layer = LSTM(size=1, input_size=(2, 2), weights=weights, recurrent_weights=recurrent_weights, biases=biases)
    result = layer.run(inputs=np.array([[1, 2], [0.5, 3]]))
    ic(result)
    expected = np.array([0.7719811058])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None
