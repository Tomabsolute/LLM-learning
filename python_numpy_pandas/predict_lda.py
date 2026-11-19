import numpy as np

X_train = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 2.5],
    [2.5, 2.2],
    [4.0, 4.5],
    [4.5, 4.0],
    [5.0, 5.2],
    [5.5, 4.8]
])

y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_test = np.array([
    [1.8, 2.0],
    [4.8, 4.6],
    [3.0, 3.0]
])


def fit_lda(X, y):
    classes,counts = np.unique(y,return_counts = True)
    means = np.array([X[y==c].mean(axis = 0) for c in classes])
    p = counts/np.sum(counts)
    S = np.sum(np.array([(X[y==c]-means[classes==c]).T@(X[y==c]-means[classes==c]) for c in classes]),axis = 0)/(len(y)-len(classes))
    return {"classes":classes,"means":means,"covariance":S,"priors":p}

def predict_lda(X, model):
    means = model["means"]
    covariance = model["covariance"]

    weights = np.linalg.solve(covariance, means.T)
    quadratic = np.sum(means.T * weights, axis=0)

    delta = (
        X @ weights
        - 0.5 * quadratic
        + np.log(model["priors"])
    )

    return model["classes"][np.argmax(delta, axis=1)]


model = fit_lda(X_train, y_train)
predictions = predict_lda(X_test, model)

print("Class means:", model["means"])
print("Shared covariance:", model["covariance"])
print("Class priors:", model["priors"])
print("Predictions:", predictions)