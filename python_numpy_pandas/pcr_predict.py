import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 120

z1 = rng.normal(size=n)
z2 = rng.normal(size=n)

X = np.column_stack([
    z1 + rng.normal(0, 0.1, n),
    z1 + rng.normal(0, 0.1, n),
    z2 + rng.normal(0, 0.1, n),
    z2 + rng.normal(0, 0.1, n),
    rng.normal(size=n)
])

y = (
    3
    + 2*z1
    - 1.5*z2
    + rng.normal(0, 0.8, n)
)

def pcr_fit(X, y, M):
    if (X.ndim != 2) or (y.ndim != 1) or (len(X) != len(y)):
        raise ValueError
    n,p = X.shape
    if M<1 or M>min(n-1,p):
        raise ValueError
    
    X_c = X - np.mean(X,axis = 0)
    y_c = y-np.mean(y)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    T_M = X_c@Vt.T[:,:M]
    gamma = np.linalg.lstsq(T_M,y_c,rcond = None)[0]
    beta = Vt.T[:,:M]@gamma
    b = np.mean(y)-np.mean(X,axis = 0)@beta
    return dict(beta = beta,intercept = b,V_M = Vt.T[:,:M],gamma = gamma,x_mean = np.mean(X,axis = 0))

def pcr_predict(X_new, model):
    beta = model["beta"]
    intercept = model["intercept"]
    y_pred = X_new@beta+intercept
    return y_pred

def pcr_cross_validation(
    X, y, k=5, seed=42
):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    X_1 = np.array_split(X[idx], k)
    y_1 = np.array_split(y[idx], k)
    MSE = []
    min_train_size = len(X) - max(len(fold) for fold in X_1)
    max_M = min(X.shape[1], min_train_size - 1)
    for M in range(1,max_M+1):
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            model = pcr_fit(X_train,y_train,M)
            y_pred = pcr_predict(X_val,model)
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))

    df = pd.DataFrame(dict(M = list(range(1,6)),mean_cv_mse = MSE)).sort_values(by = "mean_cv_mse")
    best_M = int(df["M"].iloc[0])
    return df,best_M

df,best_M = pcr_cross_validation(X,y)
print(df)
print("best_M:",best_M)
