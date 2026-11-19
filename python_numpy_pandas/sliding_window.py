import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8])

def sliding_window(data, window_size, step=1):
    if window_size <= 0 or step <= 0:
        raise ValueError
    if window_size > len(data):
        return np.empty((0,window_size))
    l = []
    for i in range(0,len(data),step):
        if i+window_size > len(data):
            break
        l.append(data[i:i+window_size])
    return np.array(l)
    

print(sliding_window(data, window_size=3, step=2))