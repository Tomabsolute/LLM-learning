import numpy as np

X = np.arange(24).reshape(8, 3)
y = np.arange(8)

def batch_generator(X, y, batch_size):
    if batch_size <= 0:
        raise ValueError
    n = len(y)//batch_size
    m = len(y)%batch_size
    for i in range(n):
        yield(tuple([X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]]))
    if m:
        yield(tuple([X[n*batch_size:],y[n*batch_size:]]))

for X_batch, y_batch in batch_generator(X, y, 3):
    print(X_batch.shape, y_batch.shape)