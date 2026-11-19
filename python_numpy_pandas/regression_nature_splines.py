import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_splines import cubic_spline_fit, cubic_spline_predict

def natural_spline_features(x, knots, a=-3, b=3):
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if x.ndim != 1 or knots.ndim != 1:
        raise ValueError("x 和 knots 必须是一维数组")
    if not np.all(np.diff(knots)>0):
        raise ValueError("knots 必须严格递增")
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("边界必须是有限数，且 a < b")
    if not np.all((a < knots) & (knots < b)):
        raise ValueError("内部节点必须严格位于边界 (a, b) 内")
    
    knots_1 = np.insert(knots,0,a)

    polynomial = x[:,None]**np.arange(2)
    d = (
        np.maximum(x[:,None]-knots_1[None,:],0)**3
        - np.maximum(x[:,None]-b,0)**3
    ) / (b-knots_1)
    truncated = d[:,:-1]-d[:,-1:]
    return np.column_stack([polynomial,truncated])

def natural_spline_fit(x, y, knots, a=-3, b=3):
    X = natural_spline_features(x,knots,a,b)
    beta = np.linalg.lstsq(X,y,rcond = None)[0]
    return beta

def natural_spline_predict(x_new, beta, knots, a=-3, b=3):
    X = natural_spline_features(x_new,knots,a,b)
    y_pred = X@beta
    return y_pred

def natural_spline_cv(x, y, K_values, k=5, seed=42, a=-3, b=3):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(x))

    X = np.array_split(x[index],k)
    Y = np.array_split(y[index],k)
    MSE = []
    for K in K_values:
        MSE_1 = []
        # K 个内部节点；生成时排除两端的边界节点。
        knots = np.linspace(a,b,K+2)[1:-1]
        for i in range(k):
            X_train = np.concatenate(X[:i]+X[i+1:])
            X_val = np.array(X[i])
            Y_train = np.concatenate(Y[:i]+Y[i+1:])
            Y_val = np.array(Y[i])

            beta = natural_spline_fit(X_train,Y_train,knots,a,b)
            y_pred = natural_spline_predict(X_val,beta,knots,a,b)
            mse = np.mean((y_pred-Y_val)**2)
            MSE_1.append(mse)
        MSE.append(np.mean(MSE_1))
    idx = np.argmin(MSE)
    best_K = K_values[idx]
    df = pd.DataFrame(dict(K = K_values,cv_mse = MSE)).sort_values("cv_mse")
    return best_K,df

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 150
    x = rng.uniform(-3, 3, n)
    y = (
        2
        + np.sin(2*x)
        + 0.3*x
        + rng.normal(0, 0.3, n)
    )

    a, b = -3, 3
    K_values = [1, 2, 3, 4, 5, 6]
    best_K_natural,df = natural_spline_cv(x,y,K_values,a=a,b=b)
    print("Natural spline CV:")
    print(df)
    print("Natural spline best_K:", best_K_natural)

    best_K = best_K_natural
    print("Shared best_K for comparison:", best_K)

    knots = np.linspace(a, b, best_K + 2)[1:-1]
    print("Shared internal knots:", knots)
    beta_cubic = cubic_spline_fit(x, y, knots)
    beta_natural = natural_spline_fit(x, y, knots, a, b)

    def predict_natural(x_new):
        return natural_spline_predict(
            x_new, beta_natural, knots, a, b
        )

    h = 1e-3
    x_check = np.array([-4.0, 4.0])

    second_derivative = (
        predict_natural(x_check + h)
        - 2 * predict_natural(x_check)
        + predict_natural(x_check - h)
    ) / h**2

    for point, value in zip(x_check, second_derivative):
        print(f"f''({point:g}) ≈ {value:.3e}")

    assert np.all(np.abs(second_derivative) < 1e-5), "自然样条在边界外的二阶导数未通过检查"

    x_grid = np.linspace(-5, 5, 1000)
    y_cubic = cubic_spline_predict(
        x_grid, beta_cubic, knots
    )
    y_natural = predict_natural(x_grid)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=15, alpha=0.5, label="Training data")

    plt.plot(
        x_grid, y_cubic,
        color="red", linestyle="--", label="Cubic spline"
    )
    plt.plot(
        x_grid, y_natural,
        color="blue", label="Natural cubic spline"
    )

    for knot in knots:
        plt.axvline(knot, color="gray", linestyle=":", alpha=0.5)

    # 两个边界节点
    plt.axvline(a, color="black", linestyle="--", label="Boundaries")
    plt.axvline(b, color="black", linestyle="--")

    plt.xlim(-5, 5)
    plt.xlabel("x")
    plt.ylabel("Predicted y")
    plt.title(f"Spline comparison: {best_K} internal knots")
    plt.legend()
    plt.show()
