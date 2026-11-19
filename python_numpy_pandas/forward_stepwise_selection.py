import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n, p = 100, 5
X = rng.normal(size=(n, p))

y = (
    3 + 2*X[:, 0]
    - 1.5*X[:, 2]
    + rng.normal(0, 1, n)
)

def forward_stepwise_selection(X, y):
    if (X.ndim != 2) or (y.ndim != 1) or (len(X) != len(y)):
        raise ValueError
    n,p = X.shape
    features = []
    cols = []
    cols_choose = list(range(p))
    k = len(cols_choose)
    RSS = []
    features.append(tuple(cols))
    rss = np.linalg.lstsq(np.ones(n).reshape(-1,1),y,rcond = None)[1][0]
    RSS.append(rss)
    for i in range(k):
        RSS_1 = []
        k_1 = len(cols_choose)
        for j in range(k_1):
            cols.append(cols_choose[j])
            X_subset = np.column_stack([np.ones(n),X[:,cols]])
            rss = np.linalg.lstsq(X_subset,y,rcond = None)[1][0]
            RSS_1.append(rss)
            cols.pop()
        idx = np.argmin(RSS_1)
        RSS.append(RSS_1[idx])
        cols.append(cols_choose.pop(idx))
        features.append(tuple(cols))
    RSS = np.array(RSS)
    BIC = n*np.log(RSS/n)+np.log(n)*np.arange(1,p+2)
    df = pd.DataFrame(dict(step = np.arange(0,p+1),features = features,RSS = RSS,BIC = BIC))
    return df

result = forward_stepwise_selection(X, y)
print(result)