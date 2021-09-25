from cnn.layer.pooling import Pooling
from cnn.layer import Detector, Convolutional, Flatten, Dense
from cnn import Sequential
import json
import numpy as np


# test taken from https://gdl.cinvestav.mx/amendez/uploads/%20TechnicalPapers/A%20beginner%E2%80%99s%20tutorial%20for%20CNN.pdf
# with bias weight = 0

with open("data/multiple_inputs/02/inputs.json", "r") as f:
    inputs = np.array(json.loads(f.read()))

with open("data/multiple_inputs/02/filters.json", "r") as f:
    filters = np.array(json.loads(f.read()))

with open("data/multiple_inputs/02/weight01.json", "r") as f:
    weights_1 = np.array(json.loads(f.read()))

with open("data/multiple_inputs/02/weight02.json", "r") as f:
    weights_2 = np.array(json.loads(f.read()))

with open("data/multiple_inputs/02/result.json", "r") as f:
    expected = np.array(json.loads(f.read()))

target = np.array(
    [
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        1.00e00,
    ]
)


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
model_2.add(Pooling(size=(3, 3), mode="max", stride=1))
model_2.add(Flatten())
model_2.add(Dense(size=2, input_size=2, weights=weights_1, activation="relu"))
model_2.add(Dense(size=2, input_size=2, weights=weights_2, activation="softmax"))

model_2.forward_phase(input_data=inputs[0])
model_2.backward_phase(target=target)
