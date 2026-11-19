import numpy as np

X_train = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [2.0, 1.5],
    [4.0, 4.0],
    [4.5, 5.0],
    [5.0, 4.5],
    [7.0, 1.0],
    [7.5, 1.5]
])

y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2])

X_test = np.array([
    [1.8, 1.8],
    [4.6, 4.4],
    [7.2, 1.2],
    [3.0, 3.0]
])

def predict_knn(X_train, y_train, X_test, k=3):
    if not 1<=k<=len(X_train):
        raise ValueError
    if len(X_train.shape) != len(X_test.shape):
        raise ValueError
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError
    
    diff = X_test[:, None, :] - X_train[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    index = np.argsort(d,axis = 1)[:,:k]

    neighbor_labels = y_train[index]
    predictions = []

    for labels in neighbor_labels:
        classes, counts = np.unique(labels, return_counts=True)
        winner = classes[np.argmax(counts)]
        predictions.append(winner)

    return np.array(predictions)


predictions = predict_knn(
    X_train, y_train, X_test, k=3
)

print("Predictions:", predictions)