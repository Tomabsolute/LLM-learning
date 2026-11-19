import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

y = np.array([
    2.8, 5.1, 6.9, 9.2,
    11.1, 12.8, 15.2, 17.1
])


def loocv_linear_regression(x, y):
    if (not len(x) == len(y)) or (x.ndim !=1) or (y.ndim != 1) or (len(x)<3):
        raise ValueError
    x_1 = np.column_stack([np.ones(len(x)),x])
    MSE = []
    predictions = []
    for i in range(len(x_1)):
        l = list(range(len(x_1)))
        l.pop(i)
        x_train = x_1[l]
        x_val = x_1[i]
        y_train = y[l]
        y_val = y[i]

        beta = np.linalg.lstsq(x_train,y_train,rcond = 0)[0]
        y_pred = x_val@beta
        MSE.append((y_val-y_pred)**2)
        predictions.append(y_pred)
    MSE = np.array(MSE)
    mse = np.mean(MSE)
    return dict(mse=mse,predictions = np.array(predictions),squared_errors = MSE)
        



result = loocv_linear_regression(x, y)

print("LOOCV MSE:", result["mse"])
print("Predictions:", result["predictions"])
print("Squared errors:", result["squared_errors"])
