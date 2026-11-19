import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(1, 11, dtype=float)

y = np.array([
    5.2, 6.8, 8.1, 9.3, 10.0,
    10.2, 10.1, 9.5, 8.7, 7.4
])

def compare_models(x, y):
    TSS = np.sum((y-np.mean(y))**2)

    X_1 = np.concatenate([np.ones((len(x),1)),x.reshape(-1,1)],axis = 1)
    ols_1 = np.linalg.lstsq(X_1,y,rcond = 0)
    beta_hat_1 = ols_1[0]
    y_hat_1 = X_1@beta_hat_1
    RSS_1 = np.sum((y-y_hat_1)**2)
    MSE_1 = RSS_1/len(y)
    R_1 = 1-RSS_1/TSS

    X_2 = np.concatenate([np.ones((len(x),1)),x.reshape(-1,1),(x**2).reshape(-1,1)],axis = 1)
    ols_2 = np.linalg.lstsq(X_2,y,rcond = 0)
    beta_hat_2 = ols_2[0]
    y_hat_2 = X_2@beta_hat_2
    RSS_2 = np.sum((y-y_hat_2)**2)
    MSE_2 = RSS_2/len(y)
    R_2 = 1-RSS_2/TSS

    plt.figure()
    plt.scatter(x,y,label = "origin")
    plt.plot(x,y_hat_1,label = r"moel_1",color = "red")
    plt.plot(x,y_hat_2,label = r"model_2",color = "green")
    plt.legend()
    plt.show()
    return pd.DataFrame({"model":["model_1","model_2"],"mse":[MSE_1,MSE_2],"r2":[R_1,R_2]})

result = compare_models(x, y)

print(result)