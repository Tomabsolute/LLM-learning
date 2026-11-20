import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recursive_binary_splitting import build_tree,tree_predict,count_leaves
from cost_complexity_pruning import prune_tree
from bagging import bagging_oob_mse,bagging_predict,bootstrap_sample
from random_forest import build_random_tree
from boosting import boosting_fit,boosting_predict
from itertools import product

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

def bagging_fit(X,y,B=50,max_depth = 8,min_samples_split = 6,min_samples_leaf = 3,seed=42):
    rng = np.random.default_rng(seed)
    Tree = []
    Oob_indices = []
    for b in range(B):
        X_boot,y_boot,sample_indices,oob_indices = bootstrap_sample(X,y,rng)
        tree = build_tree(
                            X_boot,
                            y_boot,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf
                        )
        Tree.append(tree)
        Oob_indices.append(oob_indices)
    return Tree,Oob_indices

def random_forest_fit(
    X, y, B=50, max_features=3, max_depth = 8,min_samples_split = 6,min_samples_leaf = 3,seed=42
):
    rng = np.random.default_rng(seed)
    OOB_indices = []
    Tree = []
    for b in range(B):
        sample_indices = rng.choice(np.arange(len(X)),len(X),replace = True)
        oob_indices = np.setdiff1d(np.arange(len(X)),sample_indices)
        tree = build_random_tree(X[sample_indices],y[sample_indices],max_features,rng,max_depth,min_samples_split,min_samples_leaf)
        OOB_indices.append(oob_indices)
        Tree.append(tree)
    return Tree,OOB_indices

if __name__ == "__main__":
    ALL_Train_MSE = []
    ALL_Test_MSE = []
    # Part A：单棵 CART 与剪枝
    print("======= Part A：单棵 CART 与剪枝 =======")
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
    y_train_pred = tree_predict(X_train,pruned_tree)
    y_test_pred = tree_predict(X_test,pruned_tree)
    test_mse = np.mean((Y_test-y_test_pred)**2)
    number_leaves = count_leaves(tree)
    ALL_Train_MSE.append(y_train_pred)
    ALL_Test_MSE.append(y_test_pred)

    plt.figure()
    positions = np.arange(len(alphas))
    plt.plot(positions,CV_MSE,label = "CV MSE")
    plt.xticks(positions,labels = [str(a) for a in alphas])
    plt.legend()
    plt.show()

    # Part B：Bagging 与 Random Forest
    print("======= Part B：Bagging 与 Random Forest =======")
    Train_MSE = []
    Test_MSE = []
    OOB_MSE = []
    OOB_Coverage = []
    model_bagging = bagging_fit(X_train,Y_train,100,8,6,3)
    model_random_2 = random_forest_fit(X_train,Y_train,100,2,8,6,3)
    model_random_4 = random_forest_fit(X_train,Y_train,100,4,8,6,3)
    models = [model_bagging,model_random_2,model_random_4]

    for model in models:
            y_train_pred = bagging_predict(X_train,model)
            train_mse = np.mean((Y_train-y_train_pred)**2)
            y_test_pred = bagging_predict(X_test,model)
            test_mse = np.mean((Y_test-y_test_pred)**2)
            oob_mse,oob_coverage = bagging_oob_mse(X_train,Y_train,model)
            Train_MSE.append(train_mse)
            Test_MSE.append(test_mse)
            OOB_MSE.append(oob_mse)
            OOB_Coverage.append(oob_coverage)
    ALL_Train_MSE += Train_MSE
    ALL_Test_MSE += Test_MSE
    models = ["model_bagging","model_random_2","model_random_4"]
    df = pd.DataFrame({"model":models,"Train MSE":Train_MSE,"Test MSE":Test_MSE,"OOB MSE":OOB_MSE,"OOB Coverage":OOB_Coverage})
    print(df)

    # Part C：Boosting
    print("====== Part C：Boosting ======")
    learning_rates = [0.05, 0.1, 0.2]
    M_values = [20, 50, 100, 200]
    CV_MSE = []
    elements = list(product(learning_rates,M_values))
    for learning_rate,M in elements:
        MSE = []
        for i in range(5):
            X_train_1 = np.concatenate(X_1[:i]+X_1[i+1:])
            X_val = X_1[i]
            Y_train_1 = np.concatenate(Y_1[:i]+Y_1[i+1:])
            Y_val = Y_1[i]
            model = boosting_fit(X_train_1,Y_train_1,200,learning_rate,2)
            y_pred = boosting_predict(X_val,model,M)
            mse = np.mean((Y_val-y_pred)**2)
            MSE.append(mse)
        CV_MSE.append(np.mean(MSE))
    idx = np.argmin(CV_MSE)
    best_learning_rate,best_M = elements[idx]
    model = boosting_fit(X_train,Y_train,200,best_learning_rate,2)
    y_train_pred = boosting_predict(X_train,model,best_M)
    train_mse = np.mean((Y_train-y_train_pred)**2)
    y_test_pred = boosting_predict(X_test,model,best_M)
    test_mse = np.mean((Y_test-y_test_pred)**2)
    ALL_Train_MSE.append(train_mse)
    ALL_Test_MSE.append(test_mse)
    print(f"best_learing_rate is {best_learning_rate}")
    print(f"best_M is {best_M}")
    print(f"test mse is {test_mse}")

    fig,ax = plt.subplots(1,len(learning_rates))
    for i in range(len(learning_rates)):
        length = len(M_values)*i
        ax[i].plot(M_values,CV_MSE[length:length+len(M)],label = f"learning_rate = {learning_rates[i]}")
        ax[i].set_xlabel("M")
        ax[i].set_ylabel("CV MSE")
        ax[i].legend()
    fig.tight_layout()
    plt.show()

    # Part D：模型综合比较
    print("====== Part D：模型综合比较 ======")
    Models = ["Pruned CART","Bagging","Random Forest(m=2)","Random Forest(m=4)","Boosting"]
    df = pd.DataFrame({"model":Models,"Train MSE":ALL_Train_MSE,"Test MSE":ALL_Test_MSE})
    print(df)

    plt.figure()
    plt.bar(Models,Train_MSE)
    plt.show()