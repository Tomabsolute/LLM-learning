import numpy as np

X = np.arange(30).reshape(10, 3)
y = np.arange(10)

def train_val_split(X, y, val_ratio=0.2, seed=42):
    if not(0<val_ratio<1) or (X.shape[0]!=y.shape[0]):
        raise ValueError
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(y))
    X_1 = X[index,:]
    y_1 = y[index]
    X_train = X_1[0:-int(len(y)*val_ratio)]
    y_train = y_1[0:-int(len(y)*val_ratio)]
    X_val = X_1[-int(len(y)*val_ratio):]
    y_val = y_1[-int(len(y)*val_ratio):]
    return X_train,X_val,y_train,y_val


X_train, X_val, y_train, y_val = train_val_split(X, y)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)