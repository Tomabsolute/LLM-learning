import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = np.array([3.0, 5.0, 7.0, 9.0])

w = np.array([0.0])
b = 0.0

def train_linear_regression(X, y, w, b, learning_rate=0.01, epochs=1000):
    losses = []
    n = len(y)
    w_new = w.copy()
    b_new = b
    for i in range(epochs):
        y_pred = X@w_new+b_new
        loss = np.mean((y-y_pred)**2)
        losses.append(loss)
        L_w = 2*X.T@(y_pred-y)/n
        L_b = 2*np.mean(y_pred-y)
        w_new = w_new-learning_rate*L_w
        b_new = b_new-learning_rate*L_b
    return w_new,b_new,losses


w_new, b_new, losses = train_linear_regression(X, y, w, b)

print("Weights:", w_new)
print("Bias:", b_new)
print("Initial loss:", losses[0])
print("Final loss:", losses[-1])

plt.plot(range(1,1001),losses)
plt.xlabel = "epochs"
plt.ylabel = "losses"
plt.title("The losses of model train")
plt.legend()
plt.show()