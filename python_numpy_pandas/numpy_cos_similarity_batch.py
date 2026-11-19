import numpy as np

query = np.array([1.0, 2.0, 3.0])

documents = np.array([
    [1.0, 2.0, 3.0],
    [3.0, 2.0, 1.0],
    [0.0, 1.0, 0.0],
    [2.0, 4.0, 6.0]
])

def batch_cosine_similarity(query, documents):
    query_norm = np.linalg.norm(query)
    documents_norm = np.linalg.norm(documents,axis = 1)
    if query_norm == 0 or np.any(documents_norm == 0):
        raise ValueError("余弦相似度不支持零向量")
    else:
        return query@documents.T/(query_norm*documents_norm)

print(batch_cosine_similarity(query, documents))