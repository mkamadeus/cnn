import numpy as np
from cnn.layer import Dense


def test_dense():
    layer = Dense(
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


def test_dense_backpropagation():
    layer = Dense(
        size=2,
        learning_rate=0.5,
        weights=np.array(
            [
                [0.6, 0.4, 0.45],
                [0.6, 0.5, 0.55],
            ]
        ),
    )
    result = layer.run(inputs=np.array([0.59327, 0.59688]))
    expected_result = np.array([0.75137, 0.77293])

    assert np.testing.assert_array_almost_equal(result, expected_result, decimal=5) is None

    delta = layer.compute_delta(np.array([0.01, 0.99]))
    expected_delta = np.array(
        [
            [0.13850, 0.08217, 0.08267],
            [-0.03810, -0.02260, -0.02274],
        ]
    )

    assert np.testing.assert_array_almost_equal(delta, expected_delta, decimal=5) is None

    updated_weights = layer.update_weight()
    expected_weights = np.array(
        [
            [0.53075, 0.35892, 0.40867],
            [0.61905, 0.51130, 0.56137],
        ]
    )

    assert np.testing.assert_array_almost_equal(updated_weights, expected_weights, decimal=5) is None
