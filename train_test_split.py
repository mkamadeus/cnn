from mlxtend.data import mnist_data
from cnn.layer import Convolutional, Detector, Flatten, Dense, Output
from cnn.layer.pooling import MaxPooling
from cnn.sequential import Sequential
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# constants
SLICING_FACTOR = 10

# disable logger
ic.disable()

print("Loading MNIST dataset...")
X, y = mnist_data()

# train_x = train_x[:SLICING_FACTOR]
X = X.reshape((len(X), 1, 28, 28)) / 255
# print(X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# train_y = train_y[:SLICING_FACTOR]

# input shape (1,28,28)
print(f"Dataset loaded with shape {X_train.shape}")

model = Sequential(epoch=2)
model.add(Convolutional(input_shape=(1, 28, 28), padding=0, stride=1, filter_count=2, kernel_shape=(3, 3)))
model.add(Detector(activation="sigmoid"))
model.add(MaxPooling(size=(2, 2), stride=2))

# model.add(Convolutional(input_shape=(6, 14, 14), padding=0, stride=1, filter_count=16, kernel_shape=(5, 5)))
# model.add(Detector(activation="sigmoid"))
# model.add(AveragePooling(size=(2, 2), stride=2))

model.add(Flatten())
model.add(Dense(input_size=338, size=10, activation="softmax"))
# model.add(Dense(input_size=50, size=25, activation="sigmoid"))
# model.add(Dense(input_size=25, size=10, activation="softmax"))
model.add(Output(size=10, error_mode="logloss"))

print("Sequential model summary")
model.summary(input_shape=(1, 28, 28))

model.stochastic_run(X_train, y_train)
model.save("model")
print(f"Prediction: {model.predict(X_train)}")
print(f"True: {y_train}")


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))