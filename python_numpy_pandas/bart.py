import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

K = 5
sigma = 0.3
tau = 0.5

n = 120
X = rng.uniform(-2, 2, size=(n, 2))

y = (
    1.5 * (X[:, 0] > 0)
    - 1.0 * (X[:, 1] > 0.5)
    + rng.normal(0, 0.3, n)
)

idx = rng.permutation(n)

X_train = X[idx[:96]]
X_test = X[idx[96:]]

y_train = y[idx[:96]]
y_test = y[idx[96:]]

def generate_tree_candidates(X):
    pass

if __name__ == "__main__":
    