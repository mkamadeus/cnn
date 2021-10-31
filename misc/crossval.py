from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from lembek.utils import load_model
from mlxtend.data import mnist_data
from icecream import ic

ic.disable()

print("Loading MNIST dataset...")
train_x, train_y = mnist_data()

# train_x = train_x[:SLICING_FACTOR]
train_x = train_x.reshape((len(train_x), 1, 28, 28)) / 255

kf = KFold(n_splits=10, random_state=1, shuffle=True)

model = load_model("1633690450-model")
i = 0
models_arr = []
accuracy_arr = []
for train_index, test_index in kf.split(train_x):
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    # Train the model
    model.stochastic_run(X_train, y_train)  # Training the model
    models_arr.append(model)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracy_arr.append(accuracy)
    print(f"Accuracy for the fold no. {i} on the test set: {accuracy}")
    i += 1

best_model = models_arr[max(accuracy_arr)]
