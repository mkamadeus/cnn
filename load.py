from mlxtend.data import mnist_data
import pickle
from icecream import ic

ic.disable()

with open("model.picl", "rb") as f:
    model = pickle.load(f)

print("Loading MNIST dataset...")
train_x, train_y = mnist_data()

# train_x = train_x[:SLICING_FACTOR]
train_x = train_x.reshape((len(train_x), 1, 28, 28)) / 255
print(train_x[0])

model.learning_rate = 0.5
model.stochastic_run(train_x, train_y)
