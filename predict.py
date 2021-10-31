import idx2numpy
from lembek.utils import load_model
import pandas as pd
from icecream import ic

ic.disable()

test_data = idx2numpy.convert_from_file("t10k-images.idx3-ubyte") / 255.0
reshaped = test_data.reshape(len(test_data), 1, 28, 28)

model = load_model("1633693286-model")
print("prediction")
prediction = model.predict(reshaped)

submission = pd.DataFrame({"id": list(range(1, 10001)), "labels": prediction.flatten()})

submission.to_csv("result.csv", index=False)
