import numpy as np
from cnn.layer import DenseLayer


def test_dense():
    layer = DenseLayer(
        size=3,
        weights=np.array(
            [
                [0.3, 0.3, 0.3, 0.3],
                [0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1],
            ]
        ),
    )
    result = layer.run(inputs=np.array([1, 2, 3]))
    expected = np.array([0.890903, 0.802183, 0.668187])

    assert np.testing.assert_array_almost_equal(result, expected) is None
