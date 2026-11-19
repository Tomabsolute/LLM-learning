import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression_tree import best_split, node_rss

rng = np.random.default_rng(42)

n = 120
X = rng.uniform(-3, 3, size=(n, 2))

y = (
    2
    + 2.5 * (X[:, 0] > 0)
    - 1.5 * (X[:, 1] > 1)
    + rng.normal(0, 0.3, n)
)

def build_tree(
    X,
    y,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    depth=0
):
    node = {
        "is_leaf": True,
        "prediction": np.mean(y),
        "n_samples": len(y),
        "rss": node_rss(y),
    }
    if len(y) < min_samples_split or depth >= max_depth or np.all(y == y[0]):
        return node

    best = best_split(X, y, min_samples_leaf)
    if best is None or best["rss"] >= node["rss"]:
        return node

    node.update({
        "is_leaf": False,
        "feature": best["feature"],
        "threshold": best["threshold"],
        "left": build_tree(
            X[best["left_indices"]], y[best["left_indices"]],
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
        ),
        "right": build_tree(
            X[best["right_indices"]], y[best["right_indices"]],
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
        ),
    })
    return node

def tree_predict(X_new, tree):
    predictions = []
    for i in range(len(X_new)):
        flag = tree["is_leaf"]
        tree1 = tree
        while not flag:
            feature = tree1["feature"]
            threshold = tree1["threshold"]
            tree1 = tree1["left"] if X_new[i,feature] <= threshold else tree1["right"]
            flag = tree1["is_leaf"]
        predictions.append(tree1["prediction"])
    return np.array(predictions)

def count_leaves(tree):
    if tree["is_leaf"]:
        return 1
    return count_leaves(tree["left"]) + count_leaves(tree["right"])

if __name__ == "__main__":
    max_depth_values = [0, 1, 2, 3, 4, 5, 6, 8, 10]
    index = rng.permutation(len(X))
    n_train = int(np.ceil(len(X) * 0.8))
    X_train = X[index[:n_train]]
    X_test = X[index[n_train:]]
    y_train = y[index[:n_train]]
    y_test = y[index[n_train:]]

    Train_MSE = []
    Test_MSE = []
    Number = []
    for max_depth in max_depth_values:
        tree = build_tree(X_train,y_train,max_depth)
        y_train_pred = tree_predict(X_train,tree)
        train_mse = np.mean((y_train-y_train_pred)**2)
        y_test_pred = tree_predict(X_test,tree)
        test_mse = np.mean((y_test-y_test_pred)**2)
        number = count_leaves(tree)
        Train_MSE.append(train_mse)
        Test_MSE.append(test_mse)
        Number.append(number)
    df = pd.DataFrame({"Max depth":max_depth_values,"Train MSE":Train_MSE,"Test MSE":Test_MSE,"Number":Number})
    print(df)

    fig,ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(max_depth_values,Train_MSE,label = "Train MSE")
    ax[0].plot(max_depth_values,Test_MSE,label = "Test MSE")
    ax[0].set_xlabel("max depth")
    ax[0].set_ylabel("MSE")
    ax[0].legend()
    x1_grid = np.linspace(-3, 3, 150)
    x2_grid = np.linspace(-3, 3, 150)

    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

    X_grid = np.column_stack([
        xx1.ravel(),
        xx2.ravel()
    ])

    max_depth = 3
    tree = build_tree(X_train,y_train,max_depth)
    # 将一维预测结果恢复为与 xx1、xx2 相同的二维网格。
    grid_pred = tree_predict(X_grid, tree).reshape(xx1.shape)
    regions = ax[1].contourf(xx1, xx2, grid_pred, levels=20, cmap="OrRd")
    fig.colorbar(regions, ax=ax[1], label="Predicted y")
    ax[1].set_xlabel("Feature 0")
    ax[1].set_ylabel("Feature 1")
    ax[1].set_title("Prediction regions (max_depth=3)")

    fig.tight_layout()
    plt.show()
