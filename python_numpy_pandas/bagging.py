import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recursive_binary_splitting import (
    build_tree,
    tree_predict
)

rng = np.random.default_rng(42)

n = 240
X = rng.uniform(-3, 3, size=(n, 3))

y = (
    2
    + 2 * (X[:, 0] > 0)
    - 1.5 * (X[:, 1] > 1)
    + np.sin(2 * X[:, 2])
    + rng.normal(0, 0.5, n)
)

idx = rng.permutation(n)

X_train = X[idx[:192]]
X_test = X[idx[192:]]

y_train = y[idx[:192]]
y_test = y[idx[192:]]

def bootstrap_sample(X, y, rng):
    sample_indices = rng.choice(np.arange(len(X)),len(X),replace = True)
    oob_indices = np.setdiff1d(np.arange(len(X)), sample_indices)
    X_boot = X[sample_indices]
    y_boot = y[sample_indices]
    return X_boot,y_boot,sample_indices,oob_indices

def bagging_fit(X, y, B=50, seed=42):
    rng = np.random.default_rng(seed)
    Tree = []
    Oob_indices = []
    for b in range(B):
        X_boot,y_boot,sample_indices,oob_indices = bootstrap_sample(X,y,rng)
        tree = build_tree(
                            X_boot,
                            y_boot,
                            max_depth=8,
                            min_samples_split=6,
                            min_samples_leaf=3
                        )
        Tree.append(tree)
        Oob_indices.append(oob_indices)
    return Tree,Oob_indices

def bagging_predict(X_new, model):
    trees,oob_indices = model
    predictions = []
    for tree in trees:
        y_pred = tree_predict(X_new,tree)
        predictions.append(y_pred)
    return np.mean(predictions,axis = 0)

def bagging_oob_mse(X_train, y_train, model):
    trees,oob_indices = model
    n = len(trees)
    pred_sum = np.zeros(len(X_train))           
    pred_count = np.zeros(len(X_train)) 
    for i in range(n):
        tree = trees[i]
        indices = oob_indices[i]
        if len(indices) == 0:
            continue
        pred = tree_predict(X_train[indices],tree)
        pred_sum[indices] += pred
        pred_count[indices] += 1
    
    valid = pred_count>0
    coverage = np.mean(valid)
    if not np.any(valid):
        return np.nan,coverage

    oob_pred = pred_sum[valid]/pred_count[valid]
    oob_mse = np.mean((oob_pred-y_train[valid])**2)
    return oob_mse,coverage


if __name__ == "__main__":
    B_values = [1, 5, 20, 50, 100]
    Train_MSE = []
    Test_MSE = []
    OOB_MSE = []
    OOB_Coverage = []
    for B in B_values:
        model = bagging_fit(X_train,y_train,B)
        oob_mse,oob_coverage = bagging_oob_mse(X_train,y_train,model)
        train_mse = np.mean((bagging_predict(X_train,model)-y_train)**2)
        test_mse = np.mean((bagging_predict(X_test,model)-y_test)**2)
        Train_MSE.append(train_mse)
        Test_MSE.append(test_mse)
        OOB_MSE.append(oob_mse)
        OOB_Coverage.append(oob_coverage)
    df = pd.DataFrame({"B":B_values,"Train MSE":Train_MSE,"Test MSE":Test_MSE,"OOB MSE":OOB_MSE,"OOB Coverage":OOB_Coverage})

    plt.figure()
    plt.plot(B_values,Test_MSE,label = "Test MSE")
    plt.plot(B_values,OOB_MSE,label = "OOB MSE")
    plt.xlabel("B")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()