from cnn.layer.pooling import MaxPooling
from cnn.layer import Detector, Convolutional, Flatten, Dense, Output
from cnn import Sequential
import json
import numpy as np
from icecream import ic

# test taken from https://gdl.cinvestav.mx/amendez/uploads/%20TechnicalPapers/A%20beginner%E2%80%99s%20tutorial%20for%20CNN.pdf
# with bias weight = 0


def test_cnn_backprop():
    with open("data/multiple_inputs/02/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/filters.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/weight01.json", "r") as f:
        weights_1 = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/weight02.json", "r") as f:
        weights_2 = np.array(json.loads(f.read()))

    with open("data/single_input/04/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    assert inputs.shape == (1, 1, 5, 5)
    assert filters.shape == (2, 1, 3, 3)
    assert weights_1.shape == (3, 2)
    assert weights_2.shape == (3, 10)

    model_2 = Sequential()
    model_2.add(
        Convolutional(
            input_shape=(1, 5, 5),
            padding=0,
            filter_count=2,
            kernel_shape=(3, 3),
            stride=1,
            filters=filters,
        )
    )
    model_2.add(Detector(activation="relu"))
    model_2.add(MaxPooling(size=(3, 3), stride=1))
    model_2.add(Flatten())
    model_2.add(Dense(size=2, input_size=2, weights=weights_1, activation="relu"))
    model_2.add(Dense(size=10, input_size=2, weights=weights_2, activation="softmax"))
    model_2.add(Output(size=10))

    target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    model_2.forward_phase(inputs[0])
    result = model_2.backward_phase(target)

    ic(result)

    assert np.testing.assert_array_almost_equal(result, expected) is None

    # TODO: backprop here
