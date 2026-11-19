import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

x = np.linspace(-3, 3, 60)
y = 2 + 1.5*x - 0.8*x**2 + rng.normal(0, 0.8, 60)


def kfold_cv_polynomial(x, y, k=5, max_degree=6, seed=42):
    if (not 2<=k<=len(x)) or (not max_degree>=1) or (x.ndim !=1) or (y.ndim != 1) or (len(x) != len(y)):
        raise ValueError

    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))
    x_1 = x[index]
    y_1 = y[index]

    X = np.ones(len(x_1))
    MSE = []
    STD = []
    for j in range(1,max_degree+1):
        X = np.column_stack([X,x_1**j])
        X_1 = np.array_split(X,k)
        Y = np.array_split(y_1,k)
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            X_val = X_1[i]
            y_train = np.concatenate(Y[:i]+Y[i+1:],axis = 0)
            y_val = Y[i]

            beta = np.linalg.lstsq(X_train,y_train)[0]
            y_pred = X_val@beta
            MSE_1.append(np.mean((y_val-y_pred)**2))
        print(f"degree{j}:",MSE_1)
        mse = np.mean(MSE_1)
        std = np.std(MSE_1,ddof = 0)
        MSE.append(mse)
        STD.append(std)

    df = pd.DataFrame({"degree":list(range(1,max_degree+1)),"mean_cv_mse":MSE,"std_cv_mse":STD})
    df = df.sort_values(by = "mean_cv_mse")
    return df
    
result = kfold_cv_polynomial(x, y)

print(result)