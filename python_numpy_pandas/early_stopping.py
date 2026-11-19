import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = np.array([3.0, 5.0, 7.0, 9.0])

w = np.array([0.0])
b = 0.0

def train_with_early_stopping(
    X, y, w, b,
    learning_rate=0.01,
    epochs=10000,
    tol=1e-6,
    patience=10
):
    w_new = w.copy()
    b_new = b
    losses = []
    count = 0
    for i in range(epochs):
        y_pred = X@w_new+b_new
        loss = np.mean((y-y_pred)**2)
        losses.append(loss)
        if i>=1:
            if losses[i-1]-losses[i]<tol:
                count+=1
                if count == patience:
                    break
            else:
                count = 0
        n = len(y)
        L_w = 2*X.T@(y_pred-y)/n
        L_b = 2*np.mean(y_pred-y)
        w_new = w_new-learning_rate*L_w
        b_new = b_new-learning_rate*L_b
    return(w_new,b_new,losses)

w_new, b_new, losses = train_with_early_stopping(X, y, w, b)

print("Weights:", w_new)
print("Bias:", b_new)
print("Epochs trained:", len(losses))
print("Final loss:", losses[-1])