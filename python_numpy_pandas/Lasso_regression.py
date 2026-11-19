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

def soft_threshold(z, t):
    return np.select([z>t,np.abs(z)<=t],[z-t,0],z+t)

def lasso_coordinate_descent(
    X, y, lam=1.0, max_iter=10000, tol=1e-8
):
    if (lam<0) or (max_iter<1) or (tol<=0) or (X.ndim!=2) or (y.ndim!=1) or (len(X)!=len(y)):
        raise ValueError
    X_c = X-np.mean(X,axis = 0)
    y_c = y-np.mean(y)
    beta = np.zeros(X_c.shape[1])
    for i in range(max_iter):
        difference = []
        for j in range(X_c.shape[1]):
            r_j = y_c - X_c@beta + X_c[:,j]*beta[j]
            rho_j = X_c[:,j].T@r_j
            a_j = X_c[:,j].T@X_c[:,j]
            difference.append(np.abs(beta[j] - (soft_threshold(rho_j,lam/2))/a_j))
            beta[j] = (soft_threshold(rho_j,lam/2))/a_j
        if np.all(np.array(difference) < tol):
            break
    b = np.mean(y)-np.mean(X,axis = 0)@beta
    return (beta,b,i+1)

lambdas = [0, 0.1, 1, 10, 100, 1000]
beta_1 = []
beta_2 = []
beta_3 = []
Intercept = []
n_zero = []
for lam in lambdas:
    beta,intercept,n_iter = lasso_coordinate_descent(X,y,lam)
    beta_1.append(beta[0])
    beta_2.append(beta[1])
    beta_3.append(beta[2])
    Intercept.append(intercept)
    n_zero.append(np.sum(np.abs(beta)<10e-6))
df = pd.DataFrame({"lambda":lambdas,"intercept":Intercept,"beta_1":beta_1,"beta_2":beta_2,"beta_3":beta_3,"n_zero":n_zero})
print(df)

plt.figure()
plt.plot(df["lambda"],df["beta_1"],label = "beta_1")
plt.plot(df["lambda"],df["beta_2"],label = "beta_2")
plt.plot(df["lambda"],df["beta_3"],label = "beta_3")
plt.legend()
plt.show()