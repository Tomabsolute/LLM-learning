import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 100

x1 = rng.normal(size=n)
x2 = x1 + rng.normal(0, 0.1, n)
x3 = rng.normal(size=n)

X = np.column_stack([x1, x2, x3])

y = (
    3
    + 2*x1
    + 1.5*x2
    + rng.normal(0, 1, n)
)


def ridge_regression(X, y, lam=1.0):
    if (X.ndim != 2) or (y.ndim != 1) or (not lam >= 0) or (len(X) != len(y)):
        raise ValueError
    X_c = X - np.mean(X,axis = 0)
    y_c = y - np.mean(y)
    if lam == 0:
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(X)),X]),y,rcond = None)[0]
        intercept = coefficients[0]
        beta = coefficients[1:]
    else:
        beta = np.linalg.solve(X_c.T@X_c+lam*np.identity(X_c.shape[1]),X_c.T@y_c)
        intercept = np.mean(y) - np.mean(X, axis=0) @ beta
    return beta,intercept


def ridge_path(X, y, lambdas):
    Beta = []
    Intercept = []
    for lam in lambdas:
        beta,intercept = ridge_regression(X,y,lam)
        Beta.append(beta)
        Intercept.append(intercept)
    Beta = np.array(Beta)
    L2_norm = np.sqrt(np.sum(Beta**2,axis = 1))
    df = pd.DataFrame({"lambda":lambdas,"intercept":Intercept,"beta_1":Beta[:,0],"beta_2":Beta[:,1],"beta_3":Beta[:,2],"L2_norm":L2_norm})
    return df

lambdas = [0, 0.1, 1, 10, 100, 1000]

result = ridge_path(X, y, lambdas)
beta_1 = result["beta_1"]
beta_2 = result["beta_2"]
beta_3 = result["beta_3"]
print(result)

plt.figure()
plt.plot(lambdas,beta_1,label = "beta_1")
plt.plot(lambdas,beta_2,label = "beta_2")
plt.plot(lambdas,beta_3,label = "beta_3")
plt.legend()
plt.show()