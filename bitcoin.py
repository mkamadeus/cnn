import numpy as np
import pandas as pd
from icecream import ic
from tqdm import tqdm
from lembek.layer.dense import Dense
from lembek.layer.lstm import LSTM
from lembek.sequential import Sequential

WINDOW_SIZE = 32
COLUMN_COUNT = 6
STRIDE = 1

ic.disable()


def preprocess():
    # preprocess types and stuff
    train = pd.read_csv("data/bitcoin/train.csv", parse_dates=["Date"], thousands=",")
    train = train.sort_values("Date").reset_index(drop=True)
    train = train.loc[243:]
    train["Volume"] = train["Volume"].str.replace(",", "").astype(np.int32)
    print(train)
    train = train.drop(columns=["Date"])
    train = train.reset_index(drop=True)
    ic(train.head())
    COLUMN_COUNT = len(train.columns)

    # windowing
    windows = np.empty(shape=(0, WINDOW_SIZE, COLUMN_COUNT))
    labels = np.empty(shape=(0, 1, COLUMN_COUNT))
    for i in tqdm(range(0, len(train) - WINDOW_SIZE, STRIDE)):
        window = train.iloc[i : i + WINDOW_SIZE].to_numpy().reshape(1, WINDOW_SIZE, COLUMN_COUNT)
        ic(window)
        windows = np.append(windows, window, axis=0)

        label = train.iloc[i + WINDOW_SIZE].to_numpy().reshape(1, 1, COLUMN_COUNT)
        ic(label)
        labels = np.append(labels, label, axis=0)

    ic(windows.shape)
    ic(windows)
    ic(labels.shape)
    ic(labels)

    return np.array(windows), np.array(labels)


x_train, y_train = preprocess()
print(x_train.shape, y_train.shape)
print()

# model definition
lstm_layer = LSTM(size=6, input_size=(WINDOW_SIZE, COLUMN_COUNT))
dense_layer = Dense(size=6, input_size=6, activation="linear")

model = Sequential(layers=[lstm_layer, dense_layer])
model.summary(input_shape=None)
print()

print(f"window size: {WINDOW_SIZE}\tstride size: {STRIDE}")
for i, window in enumerate(x_train):
    result = model.forward_phase(window)
    print(f"Window {i}: {result}")
