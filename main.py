from cnn import Sequential
from cnn.layer import Convolutional, Detector, Pooling, Flatten, Dense
from icecream import ic
import json
import numpy as np
from mlxtend.data import mnist_data

# tqdm.pandas()
ic.disable()

slicing_factor = 10

# Preprocess data
train_x, train_y = mnist_data()
train_x = train_x[:slicing_factor]
train_y = train_y[:slicing_factor]
train_x = train_x.reshape((len(train_x), 1, 28, 28))
print(f"Training shape: f{train_x.shape}")

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
    Convolutional(input_shape=(1, 28, 28), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1, filters=filters)
)
model.add(Detector(activation="linear"))
model.add(Pooling(size=(2, 2), stride=1))
model.add(Flatten())
model.add(Dense(size=10, activation="relu"))
model.add(Dense(size=10, activation="softmax"))

result = model.run(inputs=train_x)

model.mean_squared_error(train_y, result)
model.summary()
ic(result)
ic(result.shape)
