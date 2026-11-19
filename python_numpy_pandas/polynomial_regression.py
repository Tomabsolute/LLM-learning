import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 120

x = rng.uniform(-2.5, 2.5, n)

y = (
    2
    + 1.5*x
    - 0.8*x**2
    + 0.3*x**3
    + rng.normal(0, 1, n)
)

def polynomial_features(x, degree):
    if x.ndim != 1 or degree < 1:
        raise ValueError(x.ndim,degree)

    return np.column_stack([np.ones(len(x))] + [x**i for i in range(1,degree+1)])

def polynomial_fit(x, y, degree):
    X = polynomial_features(x,degree)
    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    return beta

def polynomial_predict(x_new, beta):
    X_new = polynomial_features(x_new, len(beta) - 1)
    y_pred = X_new@beta
    return y_pred

def polynomial_cv(x, y, degrees, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))
    x_1 = x[index]
    y_1 = y[index]
    X = np.array_split(x_1,k)
    Y = np.array_split(y_1,k)
    MSE = []

    for degree in degrees:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X[:i] + X[i+1:])
            Y_train = np.concatenate(Y[:i] + Y[i+1:])
            X_val = X[i]
            Y_val = Y[i]
            beta = polynomial_fit(X_train,Y_train,degree)
            y_pred = polynomial_predict(X_val,beta)
            mse = np.mean((Y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))

    df = pd.DataFrame(dict(degree = degrees,cv_mse = MSE)).sort_values(by = "cv_mse")
    best_degree = int(df["degree"].iloc[0])
    return df,best_degree
if __name__ == "__main__":
    df,best_degree = polynomial_cv(x,y,list(range(1,11)))
    print(df)
    print("best_degree:",best_degree)

    index = rng.permutation(len(x))
    X = x[index]
    Y = y[index]

    idx = np.argsort(X)
    X_1 = X[idx]
    # dgree = 1
    beta_1 = polynomial_fit(X, Y, 1)
    y_pred_1 = polynomial_predict(X_1, beta_1)

    #degree = 3
    beta_3 = polynomial_fit(X, Y, 3)
    y_pred_3 = polynomial_predict(X_1, beta_3)

    #best_degree
    beta = polynomial_fit(X, Y, best_degree)
    y_pred = polynomial_predict(X_1, beta)

    plt.figure()
    plt.scatter(x,y)
    plt.plot(X_1,y_pred_1,label = "linear polynomial")
    plt.plot(X_1,y_pred_3,label = "cubic polynomial")
    plt.plot(X_1,y_pred,label = "optimum polynomial")
    plt.legend()
    plt.show()