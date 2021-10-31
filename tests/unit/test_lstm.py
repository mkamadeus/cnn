import numpy as np
from lembek.layer import LSTM
from icecream import ic


def test_lstm():
    layer = LSTM(size=1, input_size=(2, 2))
    result = layer.run(inputs=np.array([[1, 2], [0.5, 3]]))
    ic(result)
    expected = np.array([[0.7719811058]])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None
