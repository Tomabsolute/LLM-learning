import numpy as np
import pandas as pd

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

y = 3 + 2*z1 - 1.5*z2 + rng.normal(0, 0.8, n)

def pls_fit(X, y, M):
    if (X.ndim!=2) or (y.ndim!=1) or (len(X)!=len(y)):
        raise ValueError
    n,p = X.shape
    if (M<1 or M>min(n-1,p)):
        raise ValueError
    
    X_c = X - np.mean(X,axis = 0)
    y_c = y - np.mean(y)
    E = X_c.copy()
    f = y_c.copy()
    W = []
    P = []
    Q = []
    for i in range(M):
        w = E.T@f/np.sqrt(np.sum((E.T@f)**2))
        t = E@w
        p = E.T@t/(t.T@t)
        q = f.T@t/(t.T@t)
        E = E-np.outer(t,p)
        f = f-q*t
        W.append(w)
        P.append(p)
        Q.append(q)
    W = np.column_stack(W)
    P = np.column_stack(P)
    Q = np.array(Q)
    beta = W@np.linalg.solve(P.T@W,np.array(Q))
    b = np.mean(y)-np.mean(X,axis = 0)@beta
    return dict(beta = beta,intercept = b,W = W,P = P,q = np.array(Q),x_mean = np.mean(X,axis = 0))

def pls_predict(X_new, model):
    beta = model["beta"]
    b = model["intercept"]
    y_pred = X_new@beta+b
    return y_pred

def pls_cross_validation(X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)

    max_M = min(X.shape[1],len(X)-max([len(fold) for fold in X_1]))
    MSE = []
    for M in range(1,max_M+1):
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            model = pls_fit(X_train,y_train,M)
            y_pred = pls_predict(X_val,model)
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    df = pd.DataFrame(dict(M = list(range(1,max_M+1)),mean_cv_mse = MSE))
    best_idx = df["mean_cv_mse"].idxmin()
    best_M = df.loc[best_idx, "M"]
    return df,best_M

df,best_M = pls_cross_validation(X,y)
print(df)
print("best_M:",best_M)