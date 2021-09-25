import json
from cnn.layer import Dense
import numpy as np

output = np.array(
    [
        9.855924e-01,
        1.420001e-02,
        2.045880e-04,
        2.947620e-06,
        4.246811e-08,
        6.118632e-10,
        8.815475e-12,
        6.118632e-10,
        4.246811e-08,
        1.829905e-15,
    ]
)
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

delta_output = output - target
with open("data/multiple_inputs/02/weight02.json", "r") as f:
    weights_2 = np.array(json.loads(f.read()))

d = Dense(size=10, input_size=2, weights=weights_2, activation="relu")
d.input = np.array([424, -172])
d.output = np.array([424, 0])
d.compute_delta(delta_output)
