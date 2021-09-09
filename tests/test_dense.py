import numpy as np
from cnn.layer import DenseLayer


def test_dense():
    layer = DenseLayer(
        size=3,
        weights=np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]
        ),
    )
    result = layer.run(inputs=np.array([1, 2, 3]))
    expected = np.array([14, 14, 14])

    assert np.testing.assert_array_equal(result, expected) is None
