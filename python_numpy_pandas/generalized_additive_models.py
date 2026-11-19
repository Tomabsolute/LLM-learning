import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_nature_splines import natural_spline_features

rng = np.random.default_rng(42)

n = 240

x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-3, 3, n)

y = (
    2
    + 2 * np.sin(x1)
    + 0.5 * x2**2
    + rng.normal(0, 0.4, n)
)

X = np.column_stack([x1, x2])

def gam_features(X, knots1, knots2, a=-3, b=3):
    if X.ndim != 2:
        raise ValueError
    
    B_1 = natural_spline_features(X[:,0],knots1,a,b)
    B_2 = natural_spline_features(X[:,1],knots2,a,b)
    B = np.column_stack([B_1,B_2[:,1:]])

    is_intercept = np.all(np.isclose(B, 1.0), axis=0)
    if np.count_nonzero(is_intercept) != 1:
        raise ValueError
    return B

def gam_fit(X, y, knots1, knots2):
    B = gam_features(X,knots1,knots2)
    beta = np.linalg.lstsq(B,y,rcond = None)[0]
    return beta

def gam_predict(X_new, beta, knots1, knots2):
    X_1 = gam_features(X_new,knots1,knots2)
    y_pred = X_1@beta
    return y_pred

def gam_cv(X, y, K_values, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(X))

    X_1 = np.array_split(X[index],k)
    Y_1 = np.array_split(y[index],k)

    MSE = []
    for K in K_values:
        knots = np.linspace(-3,3,K+2)[1:-1]
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X_1[:i]+X_1[i+1:])
            X_val = X_1[i]
            Y_train = np.concatenate(Y_1[:i]+Y_1[i+1:])
            Y_val = Y_1[i]

            beta = gam_fit(X_train,Y_train,knots,knots)
            y_pred = gam_predict(X_val,beta,knots,knots)
            mse = np.mean((Y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    idx = np.argmin(MSE)
    best_K = K_values[idx]
    df = pd.DataFrame(dict(K = K_values,cv_mse = MSE)).sort_values("cv_mse")
    return best_K,df

if __name__ == "__main__":
    K_values = [1, 2, 3, 4, 5, 6]
    best_K,df = gam_cv(X,y,K_values)
    print(df)
    print("best_K:",best_K)
    knots = np.linspace(-3,3,best_K+2)[1:-1]
    beta = gam_fit(X,y,knots,knots)

    x_grid = np.linspace(-3, 3, 300)
    beta_1 = beta[1:best_K+2]
    beta_2 = beta[best_K+2:]
    X_1 = natural_spline_features(x_grid,knots)[:,1:]

    y_pred1 = X_1@beta_1
    y_pred2 = X_1@beta_2

    plt.figure()
    plt.plot(x_grid,y_pred1,label = "f_1(x)")
    plt.plot(x_grid,y_pred2,label = "f_2(x)")
    plt.plot(x_grid,2*np.sin(x_grid),label = "2sin(x)")
    plt.plot(x_grid,0.5*x_grid**2,label = "0.5x**2")
    plt.legend()
    plt.show()