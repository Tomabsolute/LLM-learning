import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

x = rng.uniform(-3, 3, 120)
y = (
    2
    + 1.5*x
    - 0.8*x**2
    + rng.normal(0, 0.8, 120)
)

def select_polynomial_degree(
    x, y, k=5, max_degree=6, seed=42
):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))
    x_1 = x[index]
    y_1 = y[index]

    X = np.ones(len(x_1))
    mean_cv_mse = []
    for i in range(1,max_degree+1):
        X = np.column_stack([X,x_1**i])
        X_1 = np.array_split(X,k)
        Y = np.array_split(y_1,k)
        MSE = []
        for j in range(k):
            x_train = np.concatenate(X_1[:j]+X_1[j+1:],axis = 0)
            x_val = X_1[j]
            y_train = np.concatenate(Y[:j]+Y[j+1:])
            y_val = Y[j]
            beta = np.linalg.lstsq(x_train,y_train)[0]
            y_hat = x_val@beta
            mse = np.mean((y_val-y_hat)**2)
            MSE.append(mse)
        mse_1 = np.mean(MSE)
        mean_cv_mse.append(mse_1)

    df = pd.DataFrame({"degree":list(range(1,max_degree+1)),"mean_cv_mse":mean_cv_mse})
    best_degree = np.argmin(mean_cv_mse)+1
    return best_degree,df

def fit_final_model(x, y, degree):
    x_1 = np.column_stack([x**i for i in range(degree+1)])
    beta = np.linalg.lstsq(x_1,y)[0]
    y_pred = x_1@beta
    return beta,y_pred

def bootstrap_coefficients(
    x, y, degree, B=1000, seed=42
):
    rng = np.random.default_rng(seed)
    Beta = []
    for i in range(B):
        index = rng.choice(len(x),size = len(x),replace = True)
        sample_x = x[index]
        sample_y = y[index]
        beta,y_pred = fit_final_model(sample_x,sample_y,degree)
        Beta.append(beta)
    bootstrap_betas = np.array(Beta)
    bootstrap_se = np.std(bootstrap_betas,ddof = 1,axis = 0)
    ci_lower,ci_upper = np.percentile(bootstrap_betas,[2.5,97.5],axis = 0)
    return dict(bootstrap_betas = bootstrap_betas,bootstrap_se = bootstrap_se,ci_lower = ci_lower,ci_upper = ci_upper)

def resampling_experiment(x, y, seed=42):
    best_degree,df = select_polynomial_degree(x, y, seed=seed)
    beta,y_pred = fit_final_model(x, y, best_degree)
    result = bootstrap_coefficients(x, y, best_degree)
    bootstrap_se = result["bootstrap_se"]
    ci = np.column_stack([result["ci_lower"],result["ci_upper"]])
    print(df)
    print("best_degree:",best_degree)
    print("final_beta:",beta)
    print("bootstrap_se:",bootstrap_se)
    print("95%CI:",ci)

    x_1 = x.copy()
    y_1 = y.copy()
    index = np.argsort(x_1)
    x_2 = x_1[index]
    y_2 = y_1[index]
    y_pred_1 = y_pred[index]

    plt.figure()
    plt.scatter(x_2,y_2,label = "original")
    plt.plot(x_2,y_pred_1,label = "fit_linear",color = "red")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    resampling_experiment(x, y, seed=42)

