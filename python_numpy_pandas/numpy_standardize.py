import numpy as np

X = np.array([
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [4.0, 40.0, 400.0]
])

def standardize(X):
    mu = X.mean(axis = 0,keepdims = True)
    std = X.std(axis = 0,ddof = 0,keepdims = True)
    std_safe = np.where(std == 0,1,std)
    Z = (X-mu)/std_safe
    return Z

print(standardize(X))