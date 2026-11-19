import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recursive_binary_splitting import count_leaves

rng = np.random.default_rng(42)

n = 240

X = rng.uniform(-3, 3, size=(n, 2))

score = (
    1.5 * (X[:, 0] > 0)
    + 1.2 * (X[:, 1] > 1)
    - 0.8 * (X[:, 0] < -1.5)
)

prob = 1 / (1 + np.exp(-(score - 1.0)))

y = rng.binomial(1, prob, size=n)

def gini_impurity(y):
    if y.size == 0:
        raise ValueError
    classes,counts = np.unique(y,return_counts = True)
    if len(classes) == 1:
        return 0
    p = counts/np.sum(counts)
    return np.sum(p*(1-p))

def entropy_impurity(y):
    if y.size == 0:
        raise ValueError
    classes,counts = np.unique(y,return_counts = True)
    if len(classes) == 1:
        return 0
    p = counts/np.sum(counts)
    p = p[p > 0]  
    entropy = -p*np.log(p)
    return np.sum(entropy)  

def best_classification_split(
    X, y, criterion="gini", min_samples_leaf=5
):
    if X.ndim !=2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError

    if criterion == "gini":
        impurity_fn = gini_impurity
    elif criterion == "entropy":
        impurity_fn = entropy_impurity
    else:
        raise ValueError("criterion 必须为 gini 或 entropy")

    best = None
    best_impurity = np.inf
    for i in range(X.shape[1]):
        values = np.unique(X[:,i])
        thresholds = (values[1:]+values[:-1])/2
        for threshold in thresholds:
            left_mask = X[:,i]<=threshold
            right_mask = X[:,i]>threshold

            if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
                continue

            weighted_impurity = (
                np.sum(left_mask) / len(y) * impurity_fn(y[left_mask])
                + np.sum(right_mask) / len(y) * impurity_fn(y[right_mask])
            )
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best = {"feature":i,"threshold":threshold,
                        "impurity":weighted_impurity,
                        "left_indices":left_mask,"right_indices":right_mask}
    return best

def build_classification_tree(
    X,
    y,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion="gini",
    depth=0
):
    if criterion == "gini":
        impurity = gini_impurity(y)
    if criterion == "entropy":
        impurity = entropy_impurity(y)
    classes,counts = np.unique(y,return_counts = True)
    index = np.argmax(counts)
    prediction = classes[index]
    p = counts/np.sum(counts)
    node = {"is_leaf":True,
            "n_samples":len(y),
            "impurity":impurity,
            "prediction":prediction,
            "classes":classes,
            "probabilities":p
            }

    if depth >= max_depth or len(y) < min_samples_split or np.all(y==y[0]):
        return node

    best = best_classification_split(X,y,criterion,min_samples_leaf)
    if best is None or best["impurity"] >= impurity:
        return node
    else:
        left_mask = best["left_indices"]
        right_mask = best["right_indices"]
        node.update({"feature":best["feature"],
                     "threshold":best["threshold"],
                     "is_leaf":False,
                     "left":build_classification_tree(X[left_mask],y[left_mask],max_depth,min_samples_split,min_samples_leaf,criterion,depth = depth+1),
                     "right":build_classification_tree(X[right_mask],y[right_mask],max_depth,min_samples_split,min_samples_leaf,criterion,depth = depth+1)
                     })
    return node

def classification_tree_predict(X_new, tree):
    predictions = []
    for x in X_new:
        tree1 = tree
        flag = tree1["is_leaf"]

        while not flag:
            feature = tree1["feature"]
            threshold = tree1["threshold"]
            if x[feature] <= threshold:
                tree1 = tree1["left"]
            else:
                tree1 = tree1["right"]
            flag = tree1["is_leaf"]
        predictions.append(tree1["prediction"])
    return np.array(predictions)

if __name__ == "__main__":
    criterion_values = ["gini", "entropy"]
    max_depth_values = [1, 2, 3, 4, 5, 6]
    index = rng.permutation(len(X))

    n_train = int(np.ceil(len(X))*0.8)
    X_train = X[index][:n_train]
    Y_train = y[index][:n_train]
    X_test = X[index][n_train:]
    Y_test = y[index][n_train:]

    Criterion = []
    Max_depth = []
    Train_accuracy = []
    Test_accuracy = []
    N_leaves = []
    for criterion in criterion_values:
        for max_depth in max_depth_values:
            tree = build_classification_tree(X_train,Y_train,max_depth,criterion=criterion)
            y_train_pred = classification_tree_predict(X_train,tree)
            train_accuracy = np.sum(y_train_pred == Y_train)/len(Y_train)
            y_test_pred = classification_tree_predict(X_test,tree)
            test_accuracy = np.sum(y_test_pred == Y_test)/len(Y_test)
            n_leaves = count_leaves(tree)
            Criterion.append(criterion)
            Max_depth.append(max_depth)
            Train_accuracy.append(train_accuracy)
            Test_accuracy.append(test_accuracy)
            N_leaves.append(n_leaves)
    df = pd.DataFrame(dict(criterion = Criterion,
                           max_depth = Max_depth,
                           train_accuracy = Train_accuracy,
                           test_accuracy = Test_accuracy,
                           n_leaves = N_leaves))
    print(df)
