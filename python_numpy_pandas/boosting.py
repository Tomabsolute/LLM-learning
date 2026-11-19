import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recursive_binary_splitting import (
    build_tree,
    tree_predict
)

rng = np.random.default_rng(42)

n = 240
X = rng.uniform(-3, 3, size=(n, 2))

y = (
    2
    + 2 * np.sin(X[:, 0])
    + 1.5 * (X[:, 1] > 0)
    + rng.normal(0, 0.4, n)
)

idx = rng.permutation(n)

X_train = X[idx[:192]]
X_test = X[idx[192:]]

y_train = y[idx[:192]]
y_test = y[idx[192:]]

def boosting_fit(
    X, y, M=100,
    learning_rate=0.1,
    max_depth=2
):
    if not isinstance(M,(int,np.integer)) or M<0 or learning_rate <= 0:
        raise ValueError
    f = np.zeros(len(y))
    Tree = []
    for m in range(M):
        r = y-f
        tree = build_tree(X,r,max_depth=max_depth,min_samples_split=6,min_samples_leaf=3)
        Tree.append(tree)
        h = tree_predict(X,tree)
        f += learning_rate*h
    return Tree,learning_rate

def boosting_predict(X_new, model, M=None):
    Tree,learning_rate = model
    if M is None:
        M = len(Tree)
    if not isinstance(M, (int, np.integer)) or M < 0 or M > len(Tree):
        raise ValueError("M 必须是 0 到已训练树数量之间的整数")
    predictions = np.zeros(len(X_new))
    for tree in Tree[:M]:
        predictions += learning_rate*tree_predict(X_new,tree)
    return predictions

if __name__ == "__main__":
    learning_rates = [0.01, 0.05, 0.1, 0.3]
    M_values = [0, 1, 5, 10, 20, 50, 100, 200]
    Learning_rates = []
    Ms = []
    Train_MSE = []
    Test_MSE = []
    for learning_rate in learning_rates:
        model = boosting_fit(X_train,y_train,200,learning_rate,2)
        for M in M_values:
            y_train_pred = boosting_predict(X_train,model,M)
            y_test_pred = boosting_predict(X_test,model,M)
            train_mse = np.mean((y_train-y_train_pred)**2)
            test_mse = np.mean((y_test-y_test_pred)**2)
            Learning_rates.append(learning_rate)
            Ms.append(M)
            Train_MSE.append(train_mse)
            Test_MSE.append(test_mse)
    df = pd.DataFrame({"learning_rate":Learning_rates,
                       "M":Ms,
                       "Train MSE":Train_MSE,
                       "Test MSE":Test_MSE})
    print(df)
    fig,ax = plt.subplots(1,2)
    for learning_rate, group in df.groupby("learning_rate"):
        ax[0].plot(group["M"], group["Train MSE"], label=f"learning_rate = {learning_rate}")
        ax[1].plot(group["M"], group["Test MSE"], label=f"learning_rate = {learning_rate}")
    for axis, title in zip(ax, ("Train MSE", "Test MSE")):
        axis.set_xlabel("M")
        axis.set_ylabel("MSE")
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    plt.show()
