import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

rng = np.random.default_rng(42)

n = 180

z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
z3 = rng.normal(size=n)

X = np.column_stack([
    z1 + rng.normal(0, 0.1, n),
    z1 + rng.normal(0, 0.1, n),
    z2 + rng.normal(0, 0.1, n),
    z2 + rng.normal(0, 0.1, n),
    z3 + rng.normal(0, 0.1, n),
    rng.normal(size=n)
])

y = (
    3
    + 2.5*z1
    - 1.8*z2
    + 0.5*z3
    + rng.normal(0, 1, n)
)

idx = rng.permutation(len(X))
X_1 = X[idx]
X_train = X_1[:int(len(X_1)*0.8)]
X_test = X_1[int(len(X_1)*0.8):]
y_1 = y[idx]
y_train = y_1[:int(len(y_1)*0.8)]
y_test = y_1[int(len(y_1)*0.8):]

def best_subset_selection(X,y):
    n,p = X.shape
    features = []
    BIC = []
    Beta = []
    for i in range(1,p+1):
        for cols in combinations(range(p),i):
            X_subset = np.column_stack([np.ones(len(X)),X[:,cols]])
            beta = np.linalg.lstsq(X_subset,y,rcond = None)[0]
            y_pred = X_subset@beta
            rss = np.sum((y-y_pred)**2)
            d = X_subset.shape[1]
            bic = n*np.log(rss/n)+d*np.log(n) if rss != 0 else -np.inf
            features.append(cols)
            BIC.append(bic)
            Beta.append(beta)
    index = np.argmin(BIC)
    feature = features[index]
    beta = Beta[index]
    return feature,beta

def bss_predict(X,y,k = 5,seed = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)
    MSE = []
    for i in range(k):
        X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
        y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
        X_val = np.array(X_1[i])
        y_val = np.array(y_1[i])
        feature,beta = best_subset_selection(X_train,y_train)
        X_subset = np.column_stack([np.ones(len(X_val)),X_val[:,feature]])
        y_pred = X_subset@beta
        mse = np.mean((y_pred-y_val)**2)
        MSE.append(mse)
    cv_mse = np.mean(MSE)
    feature,beta = best_subset_selection(X,y)
    return feature,cv_mse,beta

def ridge_predict(X,y,lambdas,k=5,seed = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)
    MSE = []
    for lam in lambdas:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            X_c = (X_train - np.mean(X_train,axis = 0))/np.std(X_train,ddof = 1,axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            y_c = y_train - np.mean(y_train)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            beta = np.linalg.solve(X_c.T@X_c+lam*np.identity(X_c.shape[1]),X_c.T@y_c)/np.std(X_train,ddof = 1,axis = 0)
            intercept = np.mean(y_train) - np.mean(X_train,axis = 0)@beta
            y_pred = X_val@beta+intercept
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    index = np.argmin(MSE)
    best_lam = lambdas[index]
    cv_mse = MSE[index]
    X_c = (X.copy() - np.mean(X,axis = 0))/np.std(X,ddof = 1,axis = 0)
    y_c = y.copy() - np.mean(y)
    beta = np.linalg.solve(X_c.T@X_c+best_lam*np.identity(X_c.shape[1]),X_c.T@y_c)/np.std(X,ddof = 1,axis = 0)
    intercept = np.mean(y) - np.mean(X,axis = 0)@beta
    return best_lam,cv_mse,beta,intercept

def soft_threshold(z, t):
    return np.select([z>t,np.abs(z)<=t],[z-t,0],z+t)

def lasso_fit(X_c,y_c,lam,max_iter,tol):
    beta = np.zeros(X_c.shape[1])
    for i in range(max_iter):
        difference = []
        for j in range(X_c.shape[1]):
            r_j = y_c - X_c@beta + X_c[:,j]*beta[j]
            rho_j = X_c[:,j].T@r_j
            a_j = X_c[:,j].T@X_c[:,j]
            difference.append(np.abs(beta[j] - (soft_threshold(rho_j,lam/2))/a_j))
            beta[j] = (soft_threshold(rho_j,lam/2))/a_j
        if np.all(np.array(difference) < tol):
            return beta
    return beta

def lasso_coordinate_descent(X, y,lambdas,k=5,max_iter=10000,tol=1e-8,seed = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)
    MSE = []    
    for lam in lambdas:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            X_c = (X_train - np.mean(X_train,axis = 0))/np.std(X_train,ddof = 1,axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            y_c = y_train - np.mean(y_train)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            beta = lasso_fit(X_c,y_c,lam,max_iter,tol)/np.std(X_train,ddof = 1,axis = 0)
            intercept = np.mean(y_train)-np.mean(X_train,axis = 0)@beta
            y_pred = X_val@beta+intercept
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    index= np.argmin(MSE)
    best_lam = lambdas[index]
    cv_mse = MSE[index]
    X_c = (X.copy() - np.mean(X,axis = 0))/np.std(X,ddof = 1,axis = 0)
    y_c = y.copy() - np.mean(y)
    beta = lasso_fit(X_c,y_c,best_lam,max_iter,tol)/np.std(X,ddof = 1,axis = 0)
    intercept = np.mean(y) - np.mean(X,axis = 0)@beta
    return best_lam,cv_mse,beta,intercept

def PCR_predict(X,y,M,k = 5,seed = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)
    MSE = []    
    for m in M:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            X_c = X_train - np.mean(X_train,axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            y_c = y_train - np.mean(y_train)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            U,s,Vt = np.linalg.svd(X_c,full_matrices = False)
            T_m = X_c@Vt.T[:,:m]
            beta = Vt.T[:,:m]@np.linalg.lstsq(T_m,y_c,rcond = None)[0]
            intercept = np.mean(y_train) - np.mean(X_train,axis = 0)@beta
            y_pred = X_val@beta + intercept
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    index = np.argmin(MSE)
    cv_mse = MSE[index]
    best_M = M[index]
    X_c = X - np.mean(X,axis = 0)
    y_c = y - np.mean(y)
    U,s,Vt = np.linalg.svd(X_c,full_matrices = False)
    T_m = X_c@Vt.T[:,:best_M]
    beta = Vt.T[:,:best_M]@np.linalg.lstsq(T_m,y_c,rcond = None)[0]
    intercept = np.mean(y) - np.mean(X,axis = 0)@beta
    return best_M,cv_mse,beta,intercept

# PLS
def pls_fit(X, y, M):
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

def pls_cross_validation(X,y,M,k=5,seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    X_1 = np.array_split(X[idx],k)
    y_1 = np.array_split(y[idx],k)
    MSE = []
    for m in M:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:],axis = 0)
            y_train = np.concatenate(y_1[:i]+y_1[i+1:],axis = 0)
            X_val = np.array(X_1[i])
            y_val = np.array(y_1[i])
            model = pls_fit(X_train,y_train,m)
            y_pred = pls_predict(X_val,model)
            mse = np.mean((y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    index = np.argmin(MSE)
    cv_mse = MSE[index]
    best_M = M[index]
    model = pls_fit(X,y,best_M)
    beta = model["beta"]
    intercept = model["intercept"]
    return best_M,cv_mse,beta,intercept
    
lambdas = [0.1,1,10,100]
M = [1,2,3,4,5,6]
Model = ["Best Subset","Ridge","Lasso","PCR","PLS"]
Selected_config = []
Cv_mse = []
Test_mse = []
Test_mae = []
Test_r2 = []

# best_subset_selection
feature,cv_mse,beta = bss_predict(X_train,y_train)
Selected_config.append(feature)
Cv_mse.append(cv_mse)
y_pred = np.column_stack([np.ones(len(X_test)),X_test[:,feature]])@beta
Test_mse.append(np.mean((y_test-y_pred)**2))
Test_mae.append(np.mean(np.abs(y_test-y_pred)))
rss = np.sum((y_test-y_pred)**2)
tss = np.sum((y_test-np.mean(y_test))**2)
Test_r2.append(1-rss/tss)

# Ridge
best_lam,cv_mse,beta,intercept = ridge_predict(X_train,y_train,lambdas)
Selected_config.append(best_lam)
Cv_mse.append(cv_mse)
y_pred = X_test@beta+intercept
Test_mse.append(np.mean((y_test-y_pred)**2))
Test_mae.append(np.mean(np.abs(y_test-y_pred)))
rss = np.sum((y_test-y_pred)**2)
tss = np.sum((y_test-np.mean(y_test))**2)
Test_r2.append(1-rss/tss)

# Lasso
best_lam,cv_mse,beta,intercept = lasso_coordinate_descent(X_train, y_train,lambdas)
Selected_config.append(best_lam)
Cv_mse.append(cv_mse)
y_pred = X_test@beta+intercept
Test_mse.append(np.mean((y_test-y_pred)**2))
Test_mae.append(np.mean(np.abs(y_test-y_pred)))
rss = np.sum((y_test-y_pred)**2)
tss = np.sum((y_test-np.mean(y_test))**2)
Test_r2.append(1-rss/tss)

# PCR
best_M,cv_mse,beta,intercept = PCR_predict(X_train,y_train,M)
Selected_config.append(best_M)
Cv_mse.append(cv_mse)
y_pred = X_test@beta+intercept
Test_mse.append(np.mean((y_test-y_pred)**2))
Test_mae.append(np.mean(np.abs(y_test-y_pred)))
rss = np.sum((y_test-y_pred)**2)
tss = np.sum((y_test-np.mean(y_test))**2)
Test_r2.append(1-rss/tss)

# PLS
best_M,cv_mse,beta,intercept = pls_cross_validation(X_train,y_train,M)
Selected_config.append(best_M)
Cv_mse.append(cv_mse)
y_pred = X_test@beta+intercept
Test_mse.append(np.mean((y_test-y_pred)**2))
Test_mae.append(np.mean(np.abs(y_test-y_pred)))
rss = np.sum((y_test-y_pred)**2)
tss = np.sum((y_test-np.mean(y_test))**2)
Test_r2.append(1-rss/tss)

df = pd.DataFrame(dict(model = Model,selected_config = Selected_config,cv_mse = Cv_mse,test_mse = Test_mse,test_mae = Test_mae,test_r2 = Test_r2)).sort_values(by = "test_mse")
print(df)

colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple"]

fig, ax = plt.subplots()

# 使用 df，柱子的顺序与排序后的表格一致
bars = ax.bar(df["model"], df["test_mse"], color=colors)

# 在柱顶显示数值，保留三位小数
ax.bar_label(bars, fmt="%.3f", padding=3)

ax.set_ylabel("Test MSE")
ax.set_title("Model Comparison")
ax.margins(y=0.15)  # 顶部留空间，避免数值贴边

fig.tight_layout()
plt.show()