import numpy as np

sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [10]
]

def pad_sequences(sequences, max_length, padding_value=0):
    if max_length <= 0:
        raise ValueError
    if not sequences:
        return np.empty((0,max_length))
    l = []
    for i in sequences:
        if len(i) < max_length:
            i = i+[padding_value for _ in range(max_length-len(i))]
            l.append(i)
        elif len(i) >max_length:
            l.append(i[0:max_length])
        else:
            l.append(i)
    return np.array(l)


print(pad_sequences(sequences, max_length=3))