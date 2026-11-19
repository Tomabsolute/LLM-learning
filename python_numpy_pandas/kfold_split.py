import numpy as np

X = np.arange(30).reshape(10, 3)
y = np.arange(10)

def kfold_split(X, y, k=3, seed=42):
    if (k<2 or k>len(y)) or (len(X)!=len(y)):
        raise ValueError
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(y))
    parts = np.array_split(index,k)
    kfold = []

    for i in range(k):
        train_index = np.concatenate(parts[:i]+parts[i+1:])
        val_index = parts[i]
        kfold.append(tuple([train_index,val_index]))
    return kfold

folds = kfold_split(X, y, k=3)

for i, (train_idx, val_idx) in enumerate(folds):
    print(f"Fold {i+1}")
    print("Train:", train_idx)
    print("Val:", val_idx)

