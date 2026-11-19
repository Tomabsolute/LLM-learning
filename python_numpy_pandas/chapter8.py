import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recursive_binary_splitting import build_tree,tree_predict,count_leaves
from cost_complexity_pruning import prune_tree

rng = np.random.default_rng(2026)

n = 400
p = 8

X = rng.normal(size=(n, p))

y = (
    2
    + 2.5 * (X[:, 0] > 0)
    - 1.8 * (X[:, 1] > 0.5)
    + 1.2 * np.sin(2 * X[:, 2])
    + 0.8 * X[:, 0] * X[:, 1]
    + rng.normal(0, 0.6, n)
)

if __name__ == "__main__":
    max_depth = 10
    min_samples_split = 6
    min_samples_leaf = 3
    k = 5
    alphas = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    rng = np.random.default_rng(42)
    index = rng.permutation(len(X))

    ntrain = int(np.ceil(len(X)*0.8))
    X_train = X[index][:ntrain]
    X_test = X[index][ntrain:]
    Y_train = y[index][:ntrain]
    Y_test = y[index][ntrain:]

    X_1 = np.array_split(X_train,k)
    Y_1 = np.array_split(Y_train,k)
    CV_MSE = []
    for alpha in alphas:
        MSE = []
        for i in range(k):
            X_train_1 = np.concatenate(X_1[:i]+X_1[i+1:])
            X_val_1 = X_1[i]
            Y_train_1 = np.concatenate(Y_1[:i]+Y_1[i+1:])
            Y_val_1 = Y_1[i]

            tree = build_tree(X_train_1,Y_train_1,max_depth,min_samples_split,min_samples_leaf)
            pruned_tree = prune_tree(tree,alpha)
            y_pred = tree_predict(X_val_1,pruned_tree)
            mse = np.mean((Y_val_1-y_pred)**2)
            MSE.append(mse)
        CV_MSE.append(np.mean(MSE))
    idx = np.argmin(CV_MSE)
    best_alpha = alphas[idx]
    tree = build_tree(X_train,Y_train,max_depth,min_samples_split,min_samples_leaf)
    pruned_tree = prune_tree(tree,alpha)
    y_pred = tree_predict(X_test,pruned_tree)
    test_mse = np.mean((Y_test-y_pred)**2)

    plt.figure()
    positions = np.arange(len(alphas))
    plt.plot(positions,CV_MSE,label = "CV MSE")
    plt.xticks(positions,labels = [str(a) for a in alphas])
    plt.legend()
    plt.show()
