sequences = [
    [1, 2, 3, 2],
    [2, 3, 4],
    [1, 2, 5, 5],
    [3, 4, 4]
]

def token_frequency(sequences, top_k=3):
    if top_k <= 0:
        raise ValueError
    if not sequences:
        return []
    dic = dict()
    for i in sequences:
        for j in i:
            if j in dic:
                dic[j] += 1 # dic[j] = dic.get(j, 0) + 1
            else:
                dic[j] = 1
    items = sorted(dic.items(),key = lambda x: (-x[1],x[0]))
    if top_k >= len(items):
        return items
    else:
        return items[:top_k]

print(token_frequency(sequences, top_k=3))