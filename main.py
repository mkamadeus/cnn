from cnn import Sequential
from cnn.layer import Convolutional, Detector, Pooling, Flatten, Dense
from icecream import ic
import json
import numpy as np
from keras.datasets import mnist
from tqdm.auto import tqdm

tqdm.pandas()
ic.disable()


(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X[:1000]
print("Train: X=%s, y=%s" % (train_X.shape, train_y.shape))

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
model.add(Dense(size=2, activation="relu"))
model.add(Dense(size=2, activation="softmax"))


# TODO: this is a single instance input. How about multiple instances?
# print(train_X[1])
inp = np.reshape(train_X, (len(train_X), 1, 28, 28))
print(inp.shape)
result = model.run(inputs=inp)

model.summary()
ic(result)
ic(result.shape)
