from lembek import Sequential
from lembek.layer import Convolutional, Detector, Flatten, Dense, Output
from lembek.layer.pooling import MaxPooling
from lembek.utils import load_model
from icecream import ic
import json
import numpy as np
from mlxtend.data import mnist_data


def script_1():
    slicing_factor = 2

    # Preprocess data
    print("Loading MNIST Dataset...")
    train_x, train_y = mnist_data()
    train_x = train_x[:slicing_factor]
    train_y = train_y[:slicing_factor]
    train_x = train_x.reshape((len(train_x), 1, 28, 28))
    print(f"Training shape: {train_x.shape}")

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
        Convolutional(
            input_shape=(1, 28, 28), padding=0, filter_count=2, kernel_shape=(2, 2), stride=1, filters=filters
        )
    )
    model.add(Detector(activation="linear"))
    model.add(MaxPooling(size=(2, 2), stride=1))
    model.add(Flatten())
    model.add(Dense(size=10, input_size=1352, activation="sigmoid"))
    model.add(Dense(size=10, input_size=10, activation="softmax"))
    model.add(Output(size=10, activation="softmax"))

    result = model.run(inputs=train_x)

    model.mean_squared_error(train_y, result)
    # model.summary()
    ic(result)
    ic(result.shape)


def script_2():
    # load inputs
    with open("data/multiple_inputs/02/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/filters.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    model = Sequential()
    model.add(
        Convolutional(input_shape=(1, 5, 5), padding=0, filter_count=2, kernel_shape=(3, 3), stride=1, filters=filters)
    )
    model.add(MaxPooling(size=(3, 3), stride=1))
    model.add(Flatten())
    model.add(Dense(size=2, input_size=2, weights=np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -4.0]]), activation="relu"))
    model.add(
        Dense(
            size=10,
            input_size=2,
            weights=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.01],
                    [0.02, 0.03, 0.02, 0.02, 0.01, 0.02, 0.07, 0.08, 0.05, 0.01],
                ]
            ),
            activation="softmax",
        )
    )
    model.add(Output(size=10, activation="softmax"))
    model.forward_phase(inputs[0])
    model.save("other-model")

    return model.output


def script_3():
    with open("data/multiple_inputs/02/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    model = load_model("other-model")
    model.forward_phase(inputs[0])
    return model.output


ic(script_3())
