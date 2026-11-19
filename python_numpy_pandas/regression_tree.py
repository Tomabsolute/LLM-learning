import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 120

X = rng.uniform(-3, 3, size=(n, 2))

y = (
    2
    + 2.5 * (X[:, 0] > 0)
    - 1.5 * (X[:, 1] > 1)
    + rng.normal(0, 0.3, n)
)

def node_rss(y):
    if y.size == 0:
        raise ValueError("y.size = 0")
    elif y.size == 1:
        return 0
    else:
        return np.sum((y-np.mean(y))**2)

def best_split(X, y, min_samples_leaf=5):
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X 必须是二维数组，y 必须是一维数组")
    if X.shape[0] != y.size:
        raise ValueError("X 和 y 的样本数必须一致")
    if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf 必须是正整数")

    n, p = X.shape
    best = None
    best_rss = np.inf

    for j in range(p):
        values = np.unique(X[:, j])
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, j] <= threshold
            right_mask = X[:, j] > threshold

            if (left_mask.sum() < min_samples_leaf
                    or right_mask.sum() < min_samples_leaf):
                continue

            rss = node_rss(y[left_mask]) + node_rss(y[right_mask])
            if rss < best_rss:
                best_rss = rss
                best = {"feature": j, "threshold": threshold, "rss": rss, "left_indices": left_mask, "right_indices": right_mask}
    return best
if __name__ == "__main__":
    best = best_split(X,y)
    delta_rss = node_rss(y)-best["rss"]

    print(f"best feature is {best["feature"]}")
    print(f"threshold is {best["threshold"]}")
    print(f"the sample number of the left tree is {np.sum(best["left_indices"])}")
    print(f"the sample number of the right tree is {np.sum(best["right_indices"])}")
    print(f"the reduction of rss is {delta_rss}")

    fig,ax = plt.subplots(1,2)
    ax[0].scatter(X[:,best["feature"]],y,s = 20,alpha = 0.5)
    ax[0].axvline(x=best["threshold"], color="red", linestyle="--",
                label=f"threshold = {best['threshold']:.4f}")
    ax[0].set_xlabel(f"X[:, {best['feature']}]")
    ax[0].set_ylabel("y")
    ax[0].legend()
    bars = ax[1].bar(["rss_before", "rss_after"], [node_rss(y), best["rss"]])
    ax[1].bar_label(bars, fmt="%.2f", padding=3)
    ax[1].margins(y=0.15)
    ax[1].set_ylabel("RSS")
    fig.tight_layout()
    plt.show()
