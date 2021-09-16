from keras.datasets import mnist
import json

data = mnist.load()
data = data[:2]

print(data)

# with open('data/mnist/inputs.json') as f:
#     f.write(json.dumps(data))
