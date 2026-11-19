import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 150
x = rng.uniform(-3, 3, n)

y = (
    np.sin(2*x)
    + 0.3*x
    + rng.normal(0, 0.3, n)
)

def tricube_weights(x_train, x0, frac):
    if not 0<frac<=1:
        raise ValueError
    m = max(3,int(np.ceil(len(x_train) * frac)))
    assert m<=len(x_train),ValueError
    weights = np.zeros(len(x_train))

    d = np.abs(x_train-x0)
    index = np.argsort(d)
    d_1 = d[index[:m]]
    h = d_1[-1]

    if h == 0:
        weights[d==0] = 1
    else:
        weights[index[:m]] = (1-(d_1/h)**3)**3
    return weights

def local_linear_predict(x_train, y_train, x_new, frac):
    predictions = []
    for x_0 in x_new:
        weights = tricube_weights(x_train,x_0,frac)
        X = np.column_stack([np.ones(len(x_train)),x_train-x_0])
        sqrt_w = np.sqrt(weights)
        X_w = X*sqrt_w[:,None]
        y_w = y_train*sqrt_w

        beta = np.linalg.lstsq(X_w,y_w,rcond = None)[0]
        predictions.append(beta[0])
    return np.array(predictions)

def local_regression_cv(x, y, fracs, k=5, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))

    X = np.array_split(x[index],k)
    Y = np.array_split(y[index],k)
    MSE = []
    for frac in fracs:
        MSE_1 = []
        for i in range(k):
            X_train = np.concatenate(X[:i]+X[i+1:])
            X_val = X[i]
            Y_train = np.concatenate(Y[:i]+Y[i+1:])
            Y_val = Y[i]
            y_pred = local_linear_predict(X_train,Y_train,X_val,frac)
            mse = np.mean((Y_val-y_pred)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    idx = np.argmin(MSE)
    best_frac = fracs[idx]
    df = pd.DataFrame(dict(frac = fracs,cv_mse = MSE))
    return best_frac,df

if __name__ == "__main__":
    fracs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    best_frac,df = local_regression_cv(x,y,fracs)
    print(df)
    print("best_frac:", best_frac)

    x_grid = np.linspace(x.min(), x.max(), 300)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=20, alpha=0.5, label="Data")

    for frac, linestyle, label in [
        (0.1, ":", "frac=0.1"),
        (0.4, "--", "frac=0.4"),
        (best_frac, "-", f"CV best frac={best_frac:g}"),
    ]:
        y_pred = local_linear_predict(x, y, x_grid, frac)
        ax.plot(x_grid, y_pred, linestyle=linestyle, linewidth=2, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Local linear regression")
    ax.legend()
    fig.tight_layout()
    plt.show()

