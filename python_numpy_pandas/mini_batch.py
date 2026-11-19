import numpy as np

X = np.arange(30).reshape(10, 3)
y = np.arange(10)

def create_batches(X, y, batch_size, shuffle=True):
    if shuffle:
        index = np.random.permutation(len(y))
    else:
        index = np.arange(len(y))
    X_1 = X[index,:]
    y_1 = y[index]
    l = []
    n = len(y)//batch_size
    m = len(y)%batch_size
    for i in range(n):
        l.append(tuple([X_1[i*batch_size:(i+1)*batch_size],y_1[i*batch_size:(i+1)*batch_size]]))
    if m:
        l.append(tuple([X_1[n*batch_size:],y_1[n*batch_size:]]))
    return l

batches = create_batches(X, y, batch_size=4)
for X_batch, y_batch in batches:
    print(X_batch.shape, y_batch.shape)
