import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

x = np.linspace(-3, 3, 60)
y = 2 + 1.5*x - 0.8*x**2 + rng.normal(0, 0.8, 60)
def evaluate_models(x, y, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(np.arange(len(x)))
    X = x.copy()[index]
    y_1 = y.copy()[index]
    X_train = X[:int(0.7*len(X))]
    y_train = y_1[:int(0.7*len(X))]
    X_val = X[int(0.7*len(X)):]
    y_val = y_1[int(0.7*len(X)):]

    train_data = [np.ones(len(X_train))]
    val_data = [np.ones(len(X_val))]
    dic = dict(model = [],train_mse = [],val_mse = [],train_r2 = [],val_r2 = [])
    TSS_train = np.sum((y_train-np.mean(y_train))**2)
    TSS_val = np.sum((y_val-np.mean(y_val))**2)

    plt.figure()
    plt.scatter(X_train,y_train,color = "red",label = "train")
    plt.scatter(X_val,y_val,color = "blue",label = "val")    
    for i in range(1,4):
        dic["model"].append(f"degree_{i}")
        train_data.append(X_train**i)
        val_data.append(X_val**i)
        X_train_i = np.column_stack(train_data)
        X_val_i = np.column_stack(val_data)

        ols_train = np.linalg.lstsq(X_train_i,y_train,rcond = None)
        beta_hat = ols_train[0]
        y_train_hat = X_train_i@beta_hat
        train_mse = np.mean((y_train-y_train_hat)**2)
        RSS_train = np.sum((y_train-y_train_hat)**2)
        train_r2 = 1-RSS_train/TSS_train

        y_val_hat = X_val_i@beta_hat
        val_mse = np.mean((y_val-y_val_hat)**2)
        RSS_val = np.sum((y_val-y_val_hat)**2)
        val_r2 = 1-RSS_val/TSS_val

        dic["train_mse"].append(train_mse)
        dic["train_r2"].append(train_r2)
        dic["val_mse"].append(val_mse)
        dic["val_r2"].append(val_r2)

        plt_index = np.argsort(X)
        y_plt = np.concatenate([y_train_hat,y_val_hat],axis = 0)[plt_index]
        plt.plot(x,y_plt,label = f"degree_{i}")

    plt.legend()
    plt.show()
    df = pd.DataFrame(dic).sort_values("val_mse")
    return df

result = evaluate_models(x, y)

print(result)

## 可视化部分可以修改成以下内容，避免其初传入的X就是乱序的
# plt_index = np.argsort(X)
# y_plt = np.concatenate(
#     [y_train_hat, y_val_hat], axis=0
# )[plt_index]

# plt.plot(x, y_plt, label=f"degree_{i}")