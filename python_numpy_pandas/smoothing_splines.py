import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_nature_splines import natural_spline_features

rng = np.random.default_rng(42)

n = 100
x = np.sort(rng.uniform(-3, 3, n))

y = (
    np.sin(2*x)
    + 0.3*x
    + rng.normal(0, 0.35, n)
)

a, b = -3, 3

knots = np.linspace(a, b, 14)[1:-1]

def spline_penalty_matrix(knots, a=-3, b=3, grid_size=2001):
    knots_1 = np.insert(np.array(knots),0,a)
    grid = np.linspace(a,b,grid_size)

    d2 = 6*(np.maximum(grid[:,None]-knots_1[None,:],0)-np.maximum(grid[:,None]-b,0))/(b-knots_1)

    D = np.column_stack([np.zeros((grid_size, 2)),d2[:, :-1] - d2[:, -1:]])
    dx = (b-a)/(grid_size-1)
    weights = np.full(grid_size, dx)
    weights[[0, -1]] *= 0.5    
    Omega = D.T @ (weights[:, None] * D)

    if np.any(Omega[:2,:] != 0) or np.any(Omega[:,:2] != 0):
        raise ValueError

    return (Omega+Omega.T)/2

def smoothing_spline_fit(x, y, knots, lam, a=-3, b=3):
    Omega = spline_penalty_matrix(knots)
    B = natural_spline_features(x,knots)
    beta = np.linalg.solve(B.T@B+lam*Omega,B.T@y)
    S_1 = np.linalg.solve(B.T@B+lam*Omega,np.identity(len(Omega)))
    S = B@S_1@B.T
    return beta,Omega,S

if __name__ == "__main__":
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    index = np.argsort(x)
    x_1 = x[index]
    y_1 = y[index]
    X = natural_spline_features(x_1,knots)
    RSS = []
    DF = []
    fig,ax = plt.subplots(1,2)
    for lam in lambdas:
        beta,Omega,S = smoothing_spline_fit(x,y,knots,lam)
        df_lam = np.trace(S)
        y_pred = X@beta
        rss = np.sum((y-y_pred)**2)
        RSS.append(rss)
        DF.append(df_lam)
        ax[0].plot(x_1,y_pred,label = f"lam = {lam}")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Fitted curves")
    ax[0].legend()
    ax[1].plot(lambdas, DF, marker="o")
    ax[1].set_xscale("symlog", linthresh=0.01)
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("Effective df")
    ax[1].set_title("Model complexity")
    df = pd.DataFrame({"lambda":lambdas,"rss":RSS,"effective_df":DF})
    print(df)

    fig.tight_layout()
    plt.show()




