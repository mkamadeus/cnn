from lembek.layer import LSTM, Dense
import numpy as np
from icecream import ic

from lembek.sequential import Sequential


def test_lstm_3():
    # LSTM
    weights = np.array(
        [[[0.1, 0.1], [0.1, 0.1]], [[0.2, 0.2], [0.2, 0.2]], [[0.3, 0.3], [0.3, 0.3]], [[0.1, 0.1], [0.1, 0.1]]]
    )

    recurrent_weights = np.array(
        [[[0.2, 0.2], [0.2, 0.2]], [[0.3, 0.3], [0.3, 0.3]], [[0.1, 0.1], [0.1, 0.1]], [[0.2, 0.2], [0.2, 0.2]]]
    )

    biases = np.array([[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]], [[0.7], [0.8]]])

    lstm = LSTM(size=2, input_size=(2, 2), weights=weights, recurrent_weights=recurrent_weights, biases=biases)

    # Dense
    dense_weights = np.array([[0, 0], [0.4, 0.7], [0.5, 0.9]])
    dense = Dense(size=2, input_size=2, weights=dense_weights, activation="linear")

    # seq
    model = Sequential([lstm, dense])
    lstm_result = model.layers[0].run(inputs=np.array([[1, 2], [3, 4]]))
    lstm_expected = np.array([0.718112824, 0.7396938705])
    assert np.testing.assert_array_almost_equal(lstm_result, lstm_expected, decimal=6) is None

    dense_result = model.layers[1].run(inputs=lstm_result)
    dense_expected = np.array([0.6570920649, 1.16840346])
    ic(dense_result)
    assert np.testing.assert_array_almost_equal(dense_result, dense_expected, decimal=6) is None

