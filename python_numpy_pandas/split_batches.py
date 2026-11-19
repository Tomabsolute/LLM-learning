import numpy as np

X = np.arange(21).reshape(7, 3)
y = np.arange(7)

def split_batches(X, y, batch_size):
    batch = []
    start = 0
    for i in range(0,len(X),batch_size):
        end = start+batch_size
        batch.append((X[start:end,:],y[start:end]))
        start = end
    return batch

batches = split_batches(X, y, batch_size=3)

for X_batch, y_batch in batches:
    print("X:", X_batch.shape, "y:", y_batch.shape)
    print("labels:", y_batch)