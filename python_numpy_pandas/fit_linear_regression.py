import numpy as np
import pandas as pd

df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [52, 55, 61, 66, 68, 74, 79, 83]
})

def fit_linear_regression(df):
    x = np.array(df["hours"]).reshape(-1,1)
    y = np.array(df["score"]).reshape(-1,1)
    X = np.concatenate([np.array([1]*len(x)).reshape(-1,1),x],axis = 1)
    beta_hat = np.linalg.inv(X.T@X)@X.T@y
    y_hat = X@beta_hat
    MSE = np.mean((y-y_hat)**2)
    y_mean = np.mean(y)
    R_2 = 1 - np.sum((y-y_hat)**2)/np.sum((y-y_mean)**2)
    return dict(intercept = beta_hat[0,0],coefficient = beta_hat[1,0],mse = MSE,r2 = R_2)


result = fit_linear_regression(df)

print("Intercept:", result["intercept"])
print("Coefficient:", result["coefficient"])
print("MSE:", result["mse"])
print("R²:", result["r2"])