import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(42)

n = 100

X = rng.normal(size=(n, 2))

y = np.where(
    X[:, 0] + X[:, 1] > 0.5,
    1,
    -1
)

def maximal_margin_fit(X, y):
    """
    输入：
        X: (n, p)
        y: (n,), 取值为 -1 或 +1

    输出：
        beta_0
        beta
        support_vectors: 支持向量的坐标
        margin: 分类超平面到单侧间隔边界的距离
    """
    if X.ndim != 2:
        raise ValueError
    if X.shape[1] < 1:
        raise ValueError
    if y.shape != (X.shape[0],):
        raise ValueError

    def objective(params):
        w = params[1:]
        return np.dot(w,w)

    def objective_gradient(params):
        return np.concatenate(([0.0], 2 * params[1:]))

    def constraints(params):
        return y*(params[0]+X@params[1:])-1

    constraints_gradient = np.column_stack([y,y[:,None]*X])
    
    result = minimize(
        fun = objective,
        x0 = np.zeros(X.shape[1]+1),
        method = "SLSQP",
        jac = objective_gradient,
        constraints = {
            "type":"ineq",
            "fun":constraints,
            "jac":lambda params:constraints_gradient
        },
        options = {
            "maxiter":1000,
            "ftol":1e-9
        }
    )

    if not result.success:
        raise RuntimeError(f"最大间隔分类器拟合失败：{result.message}")

    beta_0 = result.x[0]
    beta = result.x[1:]
    functional_margins = y * (beta_0 + X @ beta)
    support_vectors = X[np.isclose(functional_margins, 1.0, rtol=0, atol=1e-5)]
    margin = 1.0 / np.linalg.norm(beta)

    return beta_0,beta,support_vectors,margin

def maximal_margin_predict(X_new, model):
    beta_0,beta,support_vectors,margin = model
    y_pred = np.where(beta_0+X_new@beta>=0,1,-1)
    return y_pred

if __name__ == "__main__":
    # 留出少量间隔，确保数据线性可分
    mask = np.abs(X[:, 0] + X[:, 1] - 0.5) > 0.25
    X = X[mask]
    y = y[mask]

    model = maximal_margin_fit(X,y)
    beta_0,beta,support_vectors,margin = model

    # Part A
    fig, ax = plt.subplots(1,3, figsize=(15, 7))
    ax[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class +1", color="tab:blue")
    ax[0].scatter(X[y == -1, 0], X[y == -1, 1], label="Class -1", color="tab:orange")

    # 用等高线绘制 beta_0 + beta @ x = -1, 0, 1，也适用于竖直分类直线。
    padding = 0.5
    x_limits = (X[:, 0].min() - padding, X[:, 0].max() + padding)
    y_limits = (X[:, 1].min() - padding, X[:, 1].max() + padding)
    xx, yy = np.meshgrid(
        np.linspace(*x_limits, 400),
        np.linspace(*y_limits, 400),
    )
    scores = beta_0 + beta[0] * xx + beta[1] * yy
    ax[0].contour(
        xx, yy, scores,
        levels=[-1, 0, 1],
        colors="black",
        linestyles=["--", "-", "--"],
        linewidths=[1.3, 2.0, 1.3],
    )
    # 空线条仅用于为等高线添加图例。
    ax[0].plot([], [], color="black", linewidth=2, label="Hyperplane: f(x) = 0")
    ax[0].plot([], [], color="black", linestyle="--", label="Margins: f(x) = +/-1")
    ax[0].scatter(
        support_vectors[:, 0], support_vectors[:, 1],
        s=180, marker="s", facecolors="none", edgecolors="crimson",
        linewidths=1.8, label="Support vectors", zorder=3,
    )

    ax[0].set(xlim=x_limits, ylim=y_limits, xlabel="X1", ylabel="X2")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_title(f"Maximal margin classifier (one-sided margin = {margin:.3f})")
    ax[0].legend()
    ax[0].grid(alpha=0.2)

    # Part B
    scores = beta_0+X@beta
    max_idx = np.argmax(scores)
    X_1 = np.delete(X,max_idx,axis=0)
    y_1 = np.delete(y,max_idx)
    model_1 = maximal_margin_fit(X_1,y_1)
    beta_0_1,beta_1,_,margin_1 = model_1
    print(f"实验一的数据：beta_0 = {beta_0_1},beta = {beta_1},Margin = {margin_1}")

    # 随机选择一整行支持向量，再找到它在原始样本中的行索引。
    support_vector = support_vectors[rng.integers(len(support_vectors))]
    indices = np.flatnonzero(
        np.all(np.isclose(X, support_vector, rtol=0, atol=1e-12), axis=1)
    )
    if indices.size == 0:
        raise ValueError("在 X 中未找到选中的支持向量")
    X_2 = np.delete(X,indices[0],axis=0)
    y_2 = np.delete(y,indices[0])
    model_2 = maximal_margin_fit(X_2,y_2)
    beta_0_2,beta_2,_,margin_2 = model_2
    print(f"实验二的数据：beta_0 = {beta_0_2},beta = {beta_2},Margin = {margin_2}")

    scores_1 = beta_0_1 + beta_1[0] * xx + beta_1[1] * yy
    ax[1].contour(
        xx, yy, scores_1,
        levels=[0],
        colors="black",
        linestyles=["-"],
        linewidths=[2],
    )
    scores_2 = beta_0_2 + beta_2[0] * xx + beta_2[1] * yy
    ax[1].contour(
        xx, yy, scores_2,
        levels=[0],
        colors="red",
        linestyles=["-"],
        linewidths=[2],
    )
    # contour 不直接使用 label，通过空线条添加图例。
    ax[1].plot([], [], color="black", label="Remove largest-score sample")
    ax[1].plot([], [], color="red", label="Remove one support vector")
    ax[1].set(xlim=x_limits, ylim=y_limits, xlabel="X1", ylabel="X2")
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].grid(alpha=0.2)
    ax[1].legend()
    fig.tight_layout()

    rng = np.random.default_rng(2026)

    n = 150

    X = rng.normal(size=(n, 2))

    # 生成带有重叠区域的两类数据
    score = X[:, 0] + X[:, 1] + rng.normal(0, 1.2, n)

    y = np.where(score > 0, 1, -1)

    model = maximal_margin_fit(X,y)
    beta_0,beta,support_vectors,margin = model

    ax[2].scatter(X[y == 1, 0], X[y == 1, 1], label="Class +1", color="tab:blue")
    ax[2].scatter(X[y == -1, 0], X[y == -1, 1], label="Class -1", color="tab:orange")

    # 用等高线绘制 beta_0 + beta @ x = -1, 0, 1，也适用于竖直分类直线。
    padding = 0.5
    x_limits = (X[:, 0].min() - padding, X[:, 0].max() + padding)
    y_limits = (X[:, 1].min() - padding, X[:, 1].max() + padding)
    xx, yy = np.meshgrid(
        np.linspace(*x_limits, 400),
        np.linspace(*y_limits, 400),
    )
    scores = beta_0 + beta[0] * xx + beta[1] * yy
    ax[3].contour(
        xx, yy, scores,
        levels=[-1, 0, 1],
        colors="black",
        linestyles=["--", "-", "--"],
        linewidths=[1.3, 2.0, 1.3],
    )
    # 空线条仅用于为等高线添加图例。
    ax[2].plot([], [], color="black", linewidth=2, label="Hyperplane: f(x) = 0")
    ax[2].plot([], [], color="black", linestyle="--", label="Margins: f(x) = +/-1")
    ax[2].scatter(
        support_vectors[:, 0], support_vectors[:, 1],
        s=180, marker="s", facecolors="none", edgecolors="crimson",
        linewidths=1.8, label="Support vectors", zorder=3,
    )

    ax[2].set(xlim=x_limits, ylim=y_limits, xlabel="X1", ylabel="X2")
    ax[2].set_aspect("equal", adjustable="box")
    ax[2].set_title("Linearly Non-separable Data")
    ax[2].legend()
    ax[2].grid(alpha=0.2)
    plt.show()