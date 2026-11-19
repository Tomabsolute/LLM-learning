import numpy as np

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [5.0, 5.0]
])

beta = np.array([0.8, -0.5])
intercept = -0.2

def sigmoid(z):
    exp_neg_abs_z = np.exp(-np.abs(z))
    return np.where(
        z >= 0,
        1 / (1 + exp_neg_abs_z),
        exp_neg_abs_z / (1 + exp_neg_abs_z),
    )


def predict_logistic(X, beta, intercept, threshold=0.5):
    if 0 <= threshold <= 1:
        pass
    else:
        raise ValueError("threshold is not in the spectrum")
    if X.shape[1] != beta.shape[0]:
        raise ValueError("长度不匹配")
    z = X @ beta + intercept
    p = sigmoid(z)
    y = np.where(p >= threshold,1,0)
    return (p,y)


probabilities, predictions = predict_logistic(
    X, beta, intercept
)

print("Probabilities:", probabilities)
print("Predictions:", predictions)

print(sigmoid(np.array([-1000, 0, 1000])))

prob, pred = predict_logistic(
    X, beta, intercept, threshold=0.7
)

print(prob)
print(pred)
