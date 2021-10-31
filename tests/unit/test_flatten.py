import numpy as np
from lembek.layer import Flatten


def test_flatten():
    layer = Flatten()
    result = layer.run(inputs=np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9])))
    print(result)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None


def test_backprop_flatten():
    layer = Flatten()
    layer.run(inputs=np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9])))
    result = layer.compute_delta(delta=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    expected = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))

    assert np.testing.assert_array_almost_equal(result, expected) is None
