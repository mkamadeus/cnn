from mlxtend.data import mnist_data
from cnn.layer import Convolutional, Detector, Flatten, Dense, Output
from cnn.layer.pooling import AveragePooling
from cnn.sequential import Sequential
from icecream import ic

# constants
SLICING_FACTOR = 10

# disable logger
ic.disable()

print("Loading MNIST dataset...")
train_x, train_y = mnist_data()

train_x = train_x[:SLICING_FACTOR]
train_x = train_x.reshape((len(train_x), 1, 28, 28))

train_y = train_y[:SLICING_FACTOR]

# input shape (1,28,28)
print(f"Dataset loaded with shape {train_x.shape}")

model = Sequential(learning_rate=1e-5)
model.add(Convolutional(input_shape=(1, 28, 28), padding=2, stride=1, filter_count=6, kernel_shape=(5, 5)))
model.add(Detector(activation="sigmoid"))
model.add(AveragePooling(size=(2, 2), stride=2))

model.add(Convolutional(input_shape=(6, 14, 14), padding=0, stride=1, filter_count=16, kernel_shape=(5, 5)))
model.add(Detector(activation="sigmoid"))
model.add(AveragePooling(size=(2, 2), stride=2))

model.add(Flatten())
model.add(Dense(input_size=400, size=120, activation="sigmoid"))
model.add(Dense(input_size=120, size=84, activation="sigmoid"))
model.add(Dense(input_size=84, size=10, activation="softmax"))
model.add(Output(size=10, error_mode="logloss"))

print("Sequential model summary")
model.summary(input_shape=(1, 28, 28))

model.stochastic_run(train_x, train_y)
print(f"Prediction: {model.predict(train_x)}")
