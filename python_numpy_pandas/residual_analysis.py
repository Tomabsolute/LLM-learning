import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 11, dtype=float)

y = np.array([
    5.2, 6.8, 8.1, 9.3, 10.0,
    10.2, 10.1, 9.5, 8.7, 7.4
])

def residual_analysis(x, y):
    X = np.concatenate([np.ones((len(x),1)),x.reshape(-1,1)],axis = 1)
    lstsq = np.linalg.lstsq(X,y,rcond = None)
    beta_hat = lstsq[0]
    RSS = lstsq[1]
    y_hat = X@beta_hat
    e = y-y_hat
    residual_mean = np.mean(e)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：原始散点与回归直线
    axes[0].scatter(x, y, label="Observed")
    axes[0].plot(x, y_hat, color="red", label="Regression line")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Linear Regression")
    axes[0].legend()

    # 右图：预测值与残差
    axes[1].scatter(y_hat, e)
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted value")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()  # 自动调整间距
    plt.show()
    return dict(beta = beta_hat,residual_mean = residual_mean,rss = RSS[0])

result = residual_analysis(x, y)

print("Coefficients:", result["beta"])
print("Residual mean:", result["residual_mean"])
print("RSS:", result["rss"])