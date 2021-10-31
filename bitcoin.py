import numpy as np
import pandas as pd
from icecream import ic
from tqdm import tqdm

import lembek
from lembek.layer.lstm import LSTM

WINDOW_SIZE = 7


def preprocess():
    # preprocess types and stuff
    train = pd.read_csv("data/bitcoin/train.csv", parse_dates=["Date"], thousands=",")
    train = train.sort_values("Date").reset_index(drop=True)
    train = train.drop(columns=["Date", "Volume"])

    # windowing
    windows = np.empty(shape=(0, WINDOW_SIZE, 5))
    labels = np.empty(shape=(0, 1, 5))
    for i in tqdm(range(0, len(train) - WINDOW_SIZE - 1)):
        # window = []
        window = train.loc[i : i + WINDOW_SIZE - 1].to_numpy().reshape(1, 7, 5)
        # ic(window)
        windows = np.append(windows, window, axis=0)

        label = train.iloc[i + WINDOW_SIZE].to_numpy().reshape(1, 1, 5)
        # ic(label)
        labels = np.append(labels, label, axis=0)
        # break

    ic(windows.shape)
    ic(windows)
    ic(labels.shape)
    ic(labels)

    return windows, labels


x_train, y_train = preprocess()

# model definition
layer = LSTM(size=5, input_size=(WINDOW_SIZE, 5))
layer.run(x_train[0])
