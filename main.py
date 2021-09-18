from cnn import Sequential
from cnn.layer import Convolutional, Detector, Pooling, Flatten
from icecream import ic
import json
import numpy as np

# load inputs
with open("data/multiple_inputs/01/inputs.json", "r") as f:
    inputs = np.array(json.loads(f.read()))
    ic(inputs)
    ic(inputs.shape)

with open("data/multiple_inputs/01/kernel.json", "r") as f:
    filters = np.array(json.loads(f.read()))
    ic(filters)
    ic(filters.shape)

# sequential model
model = Sequential()
model.add(
    Convolutional(input_shape=(3, 3, 3), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1, filters=filters)
)
model.add(Detector(activation="linear"))
model.add(Pooling(size=(2, 2), stride=1))
model.add(Flatten(size=(2, 2)))


# TODO: this is a single instance input. How about multiple instances?

result = model.run(inputs=inputs)
model.summary()
ic(result)
ic(result.shape)
