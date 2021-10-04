from cnn.layer.pooling import MaxPooling
from cnn.layer import Detector, Convolutional, Flatten, Dense, detector
from cnn import Sequential
import json
import numpy as np
from icecream import ic
from cnn.activations import softmax_derivative

# test taken from https://gdl.cinvestav.mx/amendez/uploads/%20TechnicalPapers/A%20beginner%E2%80%99s%20tutorial%20for%20CNN.pdf
# with bias weight = 0
# def test_cnn_backprop():
#     with open("data/multiple_inputs/02/inputs.json", "r") as f:
#         inputs = np.array(json.loads(f.read()))

#     with open("data/multiple_inputs/02/filters.json", "r") as f:
#         filters = np.array(json.loads(f.read()))

#     with open("data/multiple_inputs/02/weight01.json", "r") as f:
#         weights_1 = np.array(json.loads(f.read()))

#     with open("data/multiple_inputs/02/weight02.json", "r") as f:
#         weights_2 = np.array(json.loads(f.read()))

#     with open("data/multiple_inputs/02/result.json", "r") as f:
#         expected = np.array(json.loads(f.read()))

#     assert inputs.shape == (1, 1, 5, 5)
#     assert filters.shape == (2, 1, 3, 3)
#     assert weights_1.shape == (3, 2)
#     assert weights_2.shape == (3, 10)

#     model_2 = Sequential()
#     model_2.add(
#         Convolutional(
#             input_shape=(1, 5, 5),
#             padding=0,
#             filter_count=2,
#             kernel_shape=(3, 3),
#             stride=1,
#             filters=filters,
#         )
#     )
#     model_2.add(Detector(activation="relu"))
#     model_2.add(MaxPooling(size=(3, 3), stride=1))
#     model_2.add(Flatten())
#     model_2.add(Dense(size=2, input_size=2, weights=weights_1, activation="relu"))
#     model_2.add(Dense(size=2, input_size=2, weights=weights_2, activation="softmax"))
#     result = model_2.run(inputs=inputs)

#     assert np.testing.assert_array_almost_equal(result, expected) is None

    # TODO: backprop here

def test_convolutional_backprop():
    with open("data/multiple_inputs/02/inputs.json", "r") as f:
        inputs = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/filters.json", "r") as f:
        filters = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/weight01.json", "r") as f:
        weights_2 = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/weight02.json", "r") as f:
        weights_3 = np.array(json.loads(f.read()))

    with open("data/multiple_inputs/02/result.json", "r") as f:
        expected = np.array(json.loads(f.read()))

    assert inputs.shape == (1, 1, 5, 5)
    assert filters.shape == (2, 1, 3, 3)
    assert weights_2.shape == (3, 2)
    assert weights_3.shape == (3, 10)

    conv1 = Convolutional(
        input_shape=(1, 5, 5),
        padding=0,
        filter_count=2,
        kernel_shape=(3, 3),
        stride=1,
        filters=filters,
    )

    # ic(inputs.shape)
    # ic(weights_2)
    # ic(weights_3)

    conv1_res = conv1.run(inputs.reshape(1, 5, 5))

    detector1 = Detector(activation="relu")
    detector1_res = detector1.run(conv1_res)

    pooling1 = MaxPooling(size=(3, 3), stride=1)
    pooling1_res = pooling1.run(detector1_res)

    flatten1 = Flatten()
    flatten1_res = flatten1.run(pooling1_res)

    dense1 = Dense(size=2, input_size=2, weights=weights_2, activation="relu")
    # X3
    dense1_res = dense1.run(flatten1_res)
    ic(dense1_res)

    dense2 = Dense(size=2, input_size=2, weights=weights_3, activation="softmax")
    dense2_res = dense2.run(dense1_res)

    delta1 = softmax_derivative(
        output=dense2_res,
        target_class=9,
    )
    
    result_delta1 = dense2.compute_delta(delta1.reshape(len(delta1), 1))
    ic(delta1.reshape(len(delta1), 1))

    ic(result_delta1)
    # model_2.add(Dense(size=2, input_size=2, weights=weights_1, activation="relu"))
    # model_2.add(Dense(size=2, input_size=2, weights=weights_2, activation="softmax"))
    # result = model_2.run(inputs=inputs)

    # assert np.testing.assert_array_almost_equal(result, expected) is None


    # with open("data/multiple_inputs/02/filters.json", "r") as f:
    #     filters = np.array(json.loads(f.read()))

    # res_dense = [
    #     [0.07985382],
    #     [0.01014405]
    # ]

    # layer_conv = Convolutional(
    #     input_shape=(1, 5, 5),
    #     padding=0,
    #     filter_count=2,
    #     kernel_shape=(3, 3),
    #     stride=1,
    #     filters=filters,
    # )

    # ic(layer_conv.compute_delta(res_dense))