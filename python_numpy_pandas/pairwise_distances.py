import numpy as np

X_train = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

X_test = np.array([
    [2.0, 3.0],
    [4.0, 5.0]
])

def pairwise_distances(X_test, X_train):
    return np.sqrt(np.sum((X_test[:,None,:]-X_train[None,:,:])**2,axis = 2))

D = pairwise_distances(X_test, X_train)

print(D)
print(D.shape)