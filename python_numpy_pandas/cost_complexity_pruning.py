import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression_tree import best_split, node_rss
from recursive_binary_splitting import build_tree, tree_predict, count_leaves

rng = np.random.default_rng(42)

n = 150
X = rng.uniform(-3, 3, size=(n, 2))

y = (
    2
    + 2.5*(X[:, 0] > 0)
    - 1.5*(X[:, 1] > 1)
    + rng.normal(0, 0.4, n)
)

def tree_cost(tree, alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if tree["is_leaf"]:
        return tree["rss"]+alpha
    return tree_cost(tree["left"],alpha)+tree_cost(tree["right"],alpha)


def prune_tree(tree, alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be nonnegative")
    pruned_tree = copy.deepcopy(tree)

    def prune(node):
        current_cost = node["rss"]+alpha
        if node["is_leaf"]:
            return current_cost
        left_cost = prune(node["left"])
        right_cost = prune(node["right"])
        sum_cost = left_cost+right_cost

        if sum_cost>=current_cost:
            node["is_leaf"] = True
            for key in ("feature","threshold","left","right"):
                node.pop(key,None)
            return current_cost
        else:
            return sum_cost

    prune(pruned_tree)
    return pruned_tree

alphas = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
def pruning_cv(X, y, alphas, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(X))
    
    n_train = int(np.ceil(len(X) * 0.8))
    X_train, X_test = X[index[:n_train]], X[index[n_train:]]
    Y_train, Y_test = y[index[:n_train]], y[index[n_train:]]
    X_1 = np.array_split(X_train, k)
    Y_1 = np.array_split(Y_train, k)
    CV_MSE = []
    for alpha in alphas:
        MSE = []
        for i in range(k):
            X_train_1 = np.concatenate(X_1[:i]+X_1[i+1:])
            Y_train_1 = np.concatenate(Y_1[:i]+Y_1[i+1:])
            X_val = X_1[i]
            Y_val = Y_1[i]
            tree = build_tree(X_train_1,
                              Y_train_1,
                              max_depth=8,
                              min_samples_split=6,
                              min_samples_leaf=3)
            pruned_tree = prune_tree(tree,alpha)
            y_pred = tree_predict(X_val,pruned_tree)
            mse = np.mean((Y_val-y_pred)**2)
            MSE.append(mse)
        CV_MSE.append(np.mean(MSE))
    index = np.argmin(CV_MSE)
    best_alpha = alphas[index]
    tree = build_tree(X_train,
                    Y_train,
                    max_depth=8,
                    min_samples_split=6,
                    min_samples_leaf=3)
    pruned_tree = prune_tree(tree,best_alpha)
    y_pred = tree_predict(X_test,pruned_tree)
    test_mse = np.mean((Y_test-y_pred)**2)
    df = pd.DataFrame({"alpha":alphas,"CV MSE":CV_MSE}).sort_values("CV MSE")
    return df,best_alpha,test_mse,X_train,Y_train


if __name__ == "__main__":
    df,best_alpha,test_mse,X_train,Y_train = pruning_cv(X,y,alphas)
    print(df)
    print(f"best_alpha is {best_alpha}")
    print(f"test_mse is {test_mse}")

    df1 = df.sort_values("alpha")
    fig,ax = plt.subplots(1,2)
    ax[0].plot(df1["alpha"],df1["CV MSE"],label = "CV MSE")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("CV MSE")
    ax[0].legend()
    Number = []
    for alpha in alphas:
        tree = build_tree(X_train,Y_train,max_depth=8,min_samples_split=6,min_samples_leaf=3)
        pruned_tree = prune_tree(tree,alpha)
        number = count_leaves(pruned_tree)
        Number.append(number)
    ax[1].plot(alphas,Number,label = "The number of leaf")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("Number")
    ax[1].legend()
    fig.tight_layout()
    plt.show()    


