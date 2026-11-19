import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 150
x = rng.uniform(-3, 3, n)

y = (
    2
    + np.sin(2*x)
    + 0.3*x
    + rng.normal(0, 0.3, n)
)

def cubic_spline_features(x, knots):
    if x.ndim != 1 or knots.ndim != 1 or not np.all(np.diff(knots)>0):
        raise ValueError
    
    polynomial = x[:,None]**np.arange(4)
    truncated = np.maximum(x[:,None]-knots[None,:],0)**3

    return np.column_stack([polynomial,truncated])

def cubic_spline_fit(x, y, knots):
    X = cubic_spline_features(x,knots)
    beta = np.linalg.lstsq(X,y,rcond = None)[0]
    return beta

def cubic_spline_predict(x_new, beta, knots):
    X = cubic_spline_features(x_new,knots)
    y_pred = X@beta
    return y_pred

def spline_cv(x, y, K_values, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))

    X = np.array_split(x[index],k)
    Y = np.array_split(y[index],k)
    MSE = []
    for K in K_values:
        knots = np.array([-3+6*j/(K+1) for j in range(1,K+1)])
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X[:i]+X[i+1:])
            Y_train = np.concatenate(Y[:i]+Y[i+1:])
            X_val = X[i]
            Y_val = Y[i]
            beta = cubic_spline_fit(X_train,Y_train,knots)
            y_pred = cubic_spline_predict(X_val,beta,knots)
            mse = np.mean((Y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    idx = np.argmin(MSE)

    best_K = K_values[idx]
    df = pd.DataFrame(dict(K = K_values,cv_mse = MSE)).sort_values("cv_mse")
    return best_K,df

if __name__ == "__main__":
    K_values = [1, 2, 3, 4, 5, 6]
    best_K,df = spline_cv(x,y,K_values)
    print(df)
    print("best_K:",best_K)

    knots = np.array([-3+6*j/(best_K+1) for j in range(1,best_K+1)])
    beta = cubic_spline_fit(x,y,knots)
    index = np.argsort(x)
    X = x[index]
    y_pred = cubic_spline_predict(X,beta,knots)


    plt.figure()
    plt.scatter(x,y)
    plt.plot(X,y_pred,label = "best_K_cubic_splines",color = "red")
    for knot in knots:
        plt.axvline(
            x=knot,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    plt.legend()
    plt.show()



