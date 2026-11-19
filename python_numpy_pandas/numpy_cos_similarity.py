import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 1.0, 4.0])

def cosine_similarity(a, b):
    return a@b/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))

print(cosine_similarity(a, b))

# gpt's answer
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a) #np.sqrt(np.sum(a**2))
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量的余弦相似度未定义")

    return (a @ b) / (norm_a * norm_b)