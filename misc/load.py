from mlxtend.data import mnist_data
import pickle
from icecream import ic
import glob

ic.disable()

filenames = glob.glob("*.picl")
newest_filename = filenames[-1]
print(newest_filename)
with open(newest_filename, "rb") as f:
    model = pickle.load(f)

print("Loading MNIST dataset...")
train_x, train_y = mnist_data()

# train_x = train_x[:SLICING_FACTOR]
train_x = train_x.reshape((len(train_x), 1, 28, 28)) / 255
print(train_x[0])

model.learning_rate = 0.01
model.epoch = 5
model.stochastic_run(train_x, train_y)

print("Predicting data...")
print(f"Prediction: {model.predict(train_x)}")
print(f"True: {train_y}")
