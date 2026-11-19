import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 120
x = rng.uniform(-3, 3, n)

y = (
    2
    + np.sin(2*x)
    + 0.5*x
    + rng.normal(0, 0.4, n)
)

def step_features(x, K, x_min=-3, x_max=3):
    if K < 2 or x_min >= x_max or x.ndim != 1:
        raise ValueError

    Step = []
    for i in x:
        for k in range(K):
            if (x_min+k*(x_max-x_min)/K <= i < x_min+(k+1)*(x_max-x_min)/K) or (k == K-1 and i == x_max):
                Step.append([0]*k+[1]+[0]*(K-k-1))
                break
    return np.array(Step)

def step_fit(x, y, K):
    step = step_features(x,K)
    beta = np.linalg.lstsq(step,y,rcond = None)[0]
    return beta

def step_predict(x_new, beta, K):
    X = step_features(x_new,K)
    y_pred = X@beta
    return y_pred

def step_cv(x, y, K_values, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))

    X = np.array_split(x[index],k)
    Y = np.array_split(y[index],k)
    MSE = []
    for K in K_values:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X[:i]+X[i+1:])
            Y_train = np.concatenate(Y[:i]+Y[i+1:])
            X_val = X[i]
            Y_val = Y[i]

            beta = step_fit(X_train,Y_train,K)
            y_pred = step_predict(X_val,beta,K)
            mse = np.mean((Y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))

    best_K = K_values[np.argmin(MSE)]
    df = pd.DataFrame(dict(K = K_values,cv_mse = MSE)).sort_values(by = "cv_mse")
    return df, best_K

if __name__ == "__main__":
    df,best_K = step_cv(x,y,list(range(2,13)))
    print(df)
    print("best_K:",best_K)

    plt.figure()
    plt.scatter(x,y)
    beta = step_fit(x,y,best_K)
    plt.stairs(beta,np.linspace(-3,3,best_K+1),label=f"K={best_K}",color = "red")
    plt.legend()
    plt.show()


    