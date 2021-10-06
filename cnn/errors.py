import numpy as np


def log_loss(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    for i in range(0, predicted.shape[0]):
        predicted[i] = min(max(1e-15, predicted[i]), 1 - 1e-15)
    err = np.seterr(all="ignore")
    score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    np.seterr(
        divide=err["divide"],
        over=err["over"],
        under=err["under"],
        invalid=err["invalid"],
    )
    if isinstance(score, np.ndarray):
        score[np.isnan(score)] = 0
    else:
        if np.isnan(score):
            score = 0

    return np.mean(score)


def mean_sum_squared_error(actual, predicted):
    sum_square_error = 0.0
    for i in range(len(actual)):
        sum_square_error += (actual[i] - predicted[i]) ** 2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    return mean_square_error
