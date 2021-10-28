import numpy as np
from cnn.layer import LSTM
from cnn.activations import softmax_derivative
from icecream import ic


def test_dense():
    layer = LSTM(
        size=3,
        input_size=(2,2)
    )
    result = layer.run(inputs=np.array([[1, 2], [0.5, 3]]))
    print(result)
    expected = np.array([1.517633098])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None

