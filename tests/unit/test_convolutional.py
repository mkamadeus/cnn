from cnn.layer.convolutional import Convolutional
from cnn.layer import Dense
import json
import numpy as np
from icecream import ic
from cnn.activations import softmax_derivative

def test_convolutional_with_filters_defined_3():
    with open("data/single_input/02/input.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/single_input/02/kernel.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/single_input/02/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Convolutional(
        input_shape=(1, 3, 3), padding=0, filter_count=1, kernel_shape=(2, 2), stride=1, filters=filters
    )

    result = layer.run(inputs)

    assert result.shape == (1, 2, 2)
    assert np.testing.assert_array_equal(result, expected) is None

def test_convolutional_with_filters_defined_2():
    with open("data/single_input/01/input.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/single_input/01/kernel.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/single_input/01/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Convolutional(
        input_shape=(1, 5, 5), padding=0, filter_count=1, kernel_shape=(3, 3), stride=1, filters=filters
    )

    result = layer.run(inputs)

    assert result.shape == (1, 3, 3)
    assert np.testing.assert_array_equal(result, expected) is None

def test_convolutional_with_random_filters_matrix_1():
    with open("data/multiple_inputs/01/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    layer = Convolutional(input_shape=(3, 3, 3), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1)
    result = layer.run(inputs[0])

    assert result.shape == (2, 2, 2)

def test_convolutional_with_filters_defined_1():
    with open("data/multiple_inputs/01/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/01/kernel.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/01/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Convolutional(
        input_shape=(3, 3, 3), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1, filters=filters
    )

    result = layer.run(inputs[0])

    assert result.shape == (2, 2, 2)
    assert np.testing.assert_array_equal(result, expected[0]) is None

def test_convolutional_with_random_filters_matrix_2():
    with open("data/mnist/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    layer = Convolutional(input_shape=(2, 28, 28), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1)

    result = layer.run(inputs)

    assert result.shape == (2, 27, 27)

def test_convolutional_with_filters_defined_4():
    with open("data/single_input/03/input.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/single_input/03/kernel.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/single_input/03/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    layer = Convolutional(
        input_shape=(1, 5, 5), padding=1, filter_count=1, kernel_shape=(3, 3), stride=2, filters=filters
    )

    result = layer.run(inputs)

    assert result.shape == (1, 3, 3)
    assert np.testing.assert_array_equal(result, expected) is None

    expected = np.array(
        [
            [0.07985382],
            [0.01014405],
        ]
    )
    assert np.testing.assert_allclose(result, expected, rtol=1e-06) is None