import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

y = np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])

w = np.array([0.0])
b = 0.0


def train_minibatch(
    X, y, w, b,
    learning_rate=0.01,
    epochs=500,
    batch_size=2,
    seed=42
):
    if batch_size <= 0 or epochs <= 0 or (len(X) != len(y)) or (len(X) == 0):
        raise ValueError
    rng = np.random.default_rng(seed)
    w_new = w.copy()
    b_new = b
    losses = []
    n = len(X)//batch_size
    m = len(X)%batch_size
    if m:
        count = n+1
    else:
        count = n
    for i in range(epochs):
        index = rng.permutation(len(X))
        X_1 = X[index]
        y_1 = y[index]
        for j in range(count):
            X_batch = X_1[j*batch_size:(j+1)*batch_size,:]
            y_batch = y_1[j*batch_size:(j+1)*batch_size]
            y_pred = X_batch@w_new+b_new
            L_w = 2*X_batch.T@(y_pred-y_batch)/len(y_batch)
            L_b = 2*np.mean(y_pred-y_batch)
            w_new = w_new-learning_rate*L_w
            b_new = b_new-learning_rate*L_b
        y_pred = X_1@w_new+b_new
        loss = np.mean((y_pred-y_1)**2)
        losses.append(loss)
    return (w_new,b_new,losses)

w_new, b_new, losses = train_minibatch(X, y, w, b)

print("Weights:", w_new)
print("Bias:", b_new)
print("Initial loss:", losses[0])
print("Final loss:", losses[-1])
