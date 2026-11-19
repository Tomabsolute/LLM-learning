import numpy as np

X = np.array([
    [0.2, 1.0],
    [0.5, 1.5],
    [1.0, 0.8],
    [1.5, 1.2],
    [2.0, 2.0],
    [2.5, 1.8],
    [3.0, 2.5],
    [3.5, 3.0]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
beta = np.zeros(X.shape[1])
b = 0.0

def sigmoid(z):
    exp_neg_abs_z = np.exp(-np.abs(z))
    return np.where(
        z >= 0,
        1 / (1 + exp_neg_abs_z),
        exp_neg_abs_z / (1 + exp_neg_abs_z),
    )


def train_logistic(
    X, y, beta, b,
    learning_rate=0.1,
    epochs=1000
):  
    if epochs<=0 or learning_rate <=0:
        raise ValueError
    if len(X.shape) != 2:
        raise ValueError
    if X.shape[0] != y.shape[0] or X.shape[1] != beta.shape[0]:
        raise ValueError()
    beta_new = beta.copy()
    b_new = b
    L = []
    for _ in range(epochs):
        p = sigmoid(X@beta_new+b_new)
        L_beta = np.mean((p-y)*X.T,axis = 1)
        L_b = np.mean(p-y)
        beta_new -= learning_rate*L_beta
        b_new -= learning_rate*L_b
        losses = -np.mean(-np.logaddexp(0,-X@beta_new-b_new)+(1-y)*(-X@beta_new-b_new))
        L.append(losses)
    return beta_new,b_new,L


beta_new, b_new, losses = train_logistic(
    X, y, beta, b
)

print("Beta:", beta_new)
print("Intercept:", b_new)
print("Initial loss:", losses[0])
print("Final loss:", losses[-1])
print("Training accuracy:",
      np.mean((X @ beta_new + b_new >= 0) == y))