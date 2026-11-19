import numpy as np

X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

y = np.array([5.0, 8.0, 11.0, 14.0])

w = np.array([1.0, 2.0])
b = 0.5


def predict(X, w, b):
    y_pred = X@w+b
    return y_pred


def mse_loss(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)


y_pred = predict(X, w, b)
loss = mse_loss(y_pred, y)

print("Predictions:", y_pred)
print("MSE:", loss)