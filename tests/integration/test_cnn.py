from cnn.layer import Detector, Convolutional
from cnn import Sequential
import json
import numpy as np


# test taken from https://docs.google.com/spreadsheets/d/1GP0Db9h7kXpdE4FgDJFYF27kSq26baTBE-jaGzRB_a4/edit
# with bias weight = 0
def test_cnn_1():
    with open("data/multiple_inputs/01/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/01/kernel.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/01/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    assert inputs.shape == (1, 3, 3, 3)
    assert filters.shape == (2, 3, 2, 2)

    model_1 = Sequential()
    model_1.add(
        Convolutional(
            input_shape=(3, 3, 3),
            padding=0,
            filter_count=2,
            kernel_shape=(2, 2),
            stride=1,
            filters=filters,
        )
    )
    model_1.add(Detector(activation="linear"))
    result = model_1.run(inputs=inputs)

    assert np.testing.assert_array_equal(result, expected) is None
