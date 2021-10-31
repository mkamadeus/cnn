from mlxtend.data import mnist_data
from lembek.layer import Convolutional, Detector, Flatten, Dense, Output
from lembek.layer.pooling import MaxPooling
from lembek.sequential import Sequential
from icecream import ic
import pandas as pd
import idx2numpy

# constants
SLICING_FACTOR = 10

# disable logger
ic.disable()

print("Loading MNIST dataset...")
train_x, train_y = mnist_data()

# train_x = train_x[:SLICING_FACTOR]
train_x = train_x.reshape((len(train_x), 1, 28, 28)) / 255
print(train_x[0])

# train_y = train_y[:SLICING_FACTOR]

# input shape (1,28,28)
print(f"Dataset loaded with shape {train_x.shape}")

model = Sequential(epoch=10, learning_rate=0.01)
# model.add(Convolutional(input_shape=(1, 28, 28), padding=0, stride=1, filter_count=2, kernel_shape=(3, 3)))
# model.add(Detector(activation="sigmoid"))
# model.add(MaxPooling(size=(2, 2), stride=2))

# model.add(Convolutional(input_shape=(6, 14, 14), padding=0, stride=1, filter_count=16, kernel_shape=(5, 5)))
# model.add(Detector(activation="sigmoid"))
# model.add(AveragePooling(size=(2, 2), stride=2))

model.add(Flatten())
model.add(Dense(input_size=784, size=25, activation="relu"))
model.add(Dense(input_size=25, size=15, activation="relu"))
model.add(Dense(input_size=15, size=10, activation="softmax"))
# model.add(Dense(input_size=50, size=25, activation="sigmoid"))
# model.add(Dense(input_size=25, size=10, activation="softmax"))
model.add(Output(size=10, error_mode="logloss"))

print("Sequential model summary")
model.summary(input_shape=(1, 28, 28))

model.stochastic_run(train_x, train_y)
model.save("dense")
test_data = idx2numpy.convert_from_file("t10k-images.idx3-ubyte") / 255.0
reshaped = test_data.reshape(len(test_data), 1, 28, 28)

prediction = model.predict(reshaped)

print(f"Prediction: {prediction}")
print(f"True: {train_y}")

submission = pd.DataFrame({"id": list(range(1, 10001)), "labels": prediction.flatten()})

submission.to_csv("resulthehe.csv", index=False)
