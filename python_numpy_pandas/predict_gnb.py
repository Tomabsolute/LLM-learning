import numpy as np

X_train = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 2.5],
    [2.5, 2.2],
    [4.0, 4.5],
    [4.5, 4.0],
    [5.0, 5.2],
    [5.5, 4.8],
    [6.0, 5.5],
    [6.5, 5.0]
])

y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_test = np.array([
    [1.8, 2.0],
    [4.8, 4.6],
    [3.0, 3.0]
])


def fit_gnb(X, y, var_smoothing=1e-9):
    classes,counts = np.unique(y,return_counts = True)
    p = counts/np.sum(counts)
    means = np.array([np.mean(X[y==c],axis = 0) for c in classes])
    Var = np.array([
    np.var(X[y == c], axis=0)
    for c in classes
]) + var_smoothing
    return dict(classes = classes,means=means,variances = Var,priors = p)


def predict_gnb(X, model):
    classes = model["classes"]
    means = model["means"]
    variances = model["variances"]
    priors = model["priors"]

    results = []
    for i in range(len(classes)):
        means_k = means[i]
        variances_k = variances[i]
        priors_k = priors[i]

        delta = np.log(priors_k)-np.sum(np.log(2*np.pi*variances_k)+(X-means_k)**2/variances_k,axis = 1)/2
        results.append(delta)
    results = np.column_stack(results)
    index = np.argmax(results,axis = 1)
    return classes[index]


model = fit_gnb(X_train, y_train)
predictions = predict_gnb(X_test, model)

print("Means:", model["means"])
print("Variances:", model["variances"])
print("Priors:", model["priors"])
print("Predictions:", predictions)