import numpy as np
import pandas as pd
from itertools import combinations

rng = np.random.default_rng(42)

n = 100
p = 5

X = rng.normal(size=(n, p))

y = (
    3
    + 2 * X[:, 0]
    - 1.5 * X[:, 2]
    + rng.normal(0, 1, n)
)


def best_subset_selection(X, y):
    if (X.ndim != 2) or (y.ndim != 1) or (len(X) != len(y)):
        raise ValueError
    n, p = X.shape
    features = []
    n_features = []
    RSS = []
    BIC = []
    for i in range(1,p+1):
        for cols in combinations(range(p),i):
            X_subset = np.column_stack([np.ones(len(X)),X[:,cols]])
            beta = np.linalg.lstsq(X_subset,y,rcond = None)[0]
            y_pred = X_subset@beta
            rss = np.sum((y-y_pred)**2)
            d = X_subset.shape[1]
            bic = n*np.log(rss/n)+d*np.log(n) if rss != 0 else -np.inf
            features.append(cols)
            n_features.append(i)
            RSS.append(rss)
            BIC.append(bic)
    dic = dict(features = features,n_features = n_features,RSS = RSS,BIC = BIC)
    df = pd.DataFrame(dic).sort_values(by = "BIC")
    return df

result = best_subset_selection(X, y)

print(result)
