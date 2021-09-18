from cnn.layer import Detector
import numpy as np
import json


def test_detector_3x3x3_linear_single_instance():
    with open("data/test/detector/input-01.json", "r") as f:
        inputs = np.array(json.loads(f.read()))
    with open("data/test/detector/expected-01.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Detector(
        activation="linear"
    )
    result = layer.run(inputs=inputs)
    assert np.testing.assert_array_almost_equal(result, expected, decimal=0) is None


def test_detector_2x2x2_relu_2_instances():
    with open("data/test/detector/input-02.json", "r") as f:
        inputs = np.array(json.loads(f.read()))
    with open("data/test/detector/expected-02.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Detector(
        activation="relu"
    )
    result = layer.run(inputs=inputs)
    assert np.testing.assert_array_almost_equal(result, expected, decimal=0) is None


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
