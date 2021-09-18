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
