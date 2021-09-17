from cnn.layer.detector import Detector
import numpy as np


def test_relu_detector():
    array_input = np.array([[[-9, 32], [14, -6]]])

    layer = Detector("relu")

    result = layer.run(array_input)

    expected = np.array([[[0, 32], [14, 0]]])

    assert np.testing.assert_array_equal(result, expected) is None


def test_linear_detector():
    array_input = np.array([[[-9, 32], [14, -6]]])

    layer = Detector("linear")

    result = layer.run(array_input)

    expected = np.array([[[-9, 32], [14, -6]]])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sigmoid_detector():
    array_input = np.array([[[-9, 32], [14, -6]]])

    layer = Detector("sigmoid")

    result = layer.run(array_input)

    expected = np.array([[[0.000123394576, 1], [0.9999991685, 0.002472623157]]])

    assert np.testing.assert_array_almost_equal(result, expected) is None
