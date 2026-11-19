import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_tree import node_rss
from recursive_binary_splitting import tree_predict

rng = np.random.default_rng(42)

n = 300
p = 8

X = rng.normal(size=(n, p))

y = (
    3
    + 2 * (X[:, 0] > 0)
    - 1.5 * (X[:, 1] > 0.5)
    + np.sin(2 * X[:, 2])
    + rng.normal(0, 0.5, n)
)

idx = rng.permutation(n)

X_train = X[idx[:240]]
X_test = X[idx[240:]]

y_train = y[idx[:240]]
y_test = y[idx[240:]]

def random_best_split(
    X, y, max_features, rng, min_samples_leaf=3
):
    n,p = X.shape
    best_rss = np.inf
    best_threshold = np.nan
    best_feature = np.nan
    features = rng.choice(np.arange(p),max_features,replace = False)
    for feature in features:
        values = np.unique(X[:,feature])
        thresholds = (values[:-1]+values[1:])/2
        for threshold in thresholds:
            left_mask = X[:,feature]<=threshold
            right_mask = X[:,feature]>threshold
            if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
                continue
            if node_rss(y[left_mask])+node_rss(y[right_mask]) < best_rss:
                best_feature = feature
                best_threshold = threshold
                best_rss = node_rss(y[left_mask])+node_rss(y[right_mask])
                left_indices = left_mask
                right_indices = right_mask
    if best_rss == np.inf:
        return None
    best = {"feature": best_feature, "threshold": best_threshold, "rss": best_rss, "left_indices": left_indices, "right_indices": right_indices}
    return best

def build_random_tree(
    X, y, max_features, rng,
    max_depth=8,
    min_samples_split=6,
    min_samples_leaf=3,
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

    best = random_best_split(X, y, max_features,rng,min_samples_leaf)
    if best is None or best["rss"] >= node["rss"]:
        return node

    node.update({
        "is_leaf": False,
        "feature": best["feature"],
        "threshold": best["threshold"],
        "left": build_random_tree(
            X[best["left_indices"]], y[best["left_indices"]],max_features,rng,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
        ),
        "right": build_random_tree(
            X[best["right_indices"]], y[best["right_indices"]],max_features,rng,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
        ),
    })
    return node

def random_forest_fit(
    X, y, B=50, max_features=3, seed=42
):
    rng = np.random.default_rng(seed)
    OOB_indices = []
    Tree = []
    for b in range(B):
        sample_indices = rng.choice(np.arange(len(X)),len(X),replace = True)
        oob_indices = np.setdiff1d(np.arange(len(X)),sample_indices)
        tree = build_random_tree(X[sample_indices],y[sample_indices],max_features,rng)
        OOB_indices.append(oob_indices)
        Tree.append(tree)
    return Tree,OOB_indices

def random_forest_predict(X_new, model):
    Tree,OOB_indices = model
    predictions = []
    for tree in Tree:
        pred = tree_predict(X_new,tree)
        predictions.append(pred)
    return np.mean(predictions,axis = 0)

def bagging_oob_mse(X_train, y_train, model):
    Tree,OOB_indices = model
    pred_sum = np.zeros(len(X_train))
    pred_count = np.zeros(len(X_train))

    for tree,oob_indices in zip(Tree,OOB_indices):
        if len(oob_indices) == 0:
            continue
        pred = tree_predict(X_train[oob_indices],tree)
        pred_sum[oob_indices]+=pred
        pred_count[oob_indices]+=1

    valid = pred_count!=0
    coverage = np.mean(valid)
    if not np.any(valid):
        return np.nan,coverage
    y_pred = pred_sum[valid]/pred_count[valid]
    return np.mean((y_train[valid]-y_pred)**2),coverage

if __name__ == "__main__":
    max_features_values = [1, 2, 3, 4, 8]
    Train_MSE = []
    Test_MSE = []
    OOB_MSE = []
    for max_features in max_features_values:
        model = random_forest_fit(X_train,y_train,max_features = max_features)
        train_mse = np.mean((random_forest_predict(X_train,model)-y_train)**2)
        test_mse = np.mean((random_forest_predict(X_test,model)-y_test)**2)
        oob_mse,oob_coverage = bagging_oob_mse(X_train,y_train,model)
        Train_MSE.append(train_mse)
        Test_MSE.append(test_mse)
        OOB_MSE.append(oob_mse)
    df = pd.DataFrame({"max_features":max_features_values,
                       "Train MSE":Train_MSE,
                       "Test MSE":Test_MSE,
                       "OOB MSE":OOB_MSE})
    print(df)

    plt.figure()
    plt.plot(max_features_values,Test_MSE,label = "Test MSE")
    plt.plot(max_features_values,OOB_MSE,label = "OOB MSE")
    plt.xlabel("max_features")
    plt.ylabel("mse")
    plt.legend()
    plt.show()