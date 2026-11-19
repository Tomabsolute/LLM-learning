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

learning_rate = 0.01


def gradient_step(X, y, w, b, learning_rate):
    y_pred = X@w+b
    n = len(y)
    L_w = 2*X.T@(y_pred-y)/n
    L_b = 2*np.mean(y_pred-y)
    w_new = w-learning_rate*L_w
    b_new = b-learning_rate*L_b
    return w_new,b_new


w_new, b_new = gradient_step(X, y, w, b, learning_rate)

print("New weights:", w_new)
print("New bias:", b_new)