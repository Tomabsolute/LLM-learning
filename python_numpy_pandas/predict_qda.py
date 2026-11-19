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


def fit_qda(X, y):
    classes,counts = np.unique(y,return_counts = True)
    means = np.array([np.mean(X[y == c],axis = 0) for c in classes])
    p = counts/np.sum(counts)
    denominator = 1/(counts-1)
    S = np.array([(X[y==c]-means[classes==c]).T@(X[y==c]-means[classes==c])*denominator[classes==c] for c in classes])
    return {"classes":classes,"means":means,"covariances":S,"priors":p}
    
def predict_qda(X, model):
    means = model["means"]
    covariances = model["covariances"]
    classes = model["classes"]
    priors = model["priors"]

    result = []
    for i in range(len(classes)):
        means_k = means[i]
        covariances_k = covariances[i]
        X_new = X-means_k
        p_k = priors[i]

        weights = np.linalg.solve(covariances_k,X_new.T)
        quadratic = np.sum(X_new.T*weights,axis = 0)

        delta = -np.linalg.slogdet(covariances_k)[1]/2-quadratic/2+np.log(p_k)
        result.append(delta)
    result = np.column_stack(result)
    index = np.argmax(result,axis = 1)
    return classes[index]

model = fit_qda(X_train, y_train)
predictions = predict_qda(X_test, model)

print("Class means:", model["means"])
print("Class covariances:", model["covariances"])
print("Class priors:", model["priors"])
print("Predictions:", predictions)