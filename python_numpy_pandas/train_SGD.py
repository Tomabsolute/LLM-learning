# import numpy as np

# X = np.array([
#     [1.0],
#     [2.0],
#     [3.0],
#     [4.0],
#     [5.0],
#     [6.0],
#     [7.0]
# ])

# y = 2 * X[:, 0] + 1

# w = np.array([0.0])
# b = 0.0


# def train_sgd(
#     X, y, w, b,
#     learning_rate=0.01,
#     epochs=500,
#     batch_size=3,
#     seed=42
# ):
#     pass


# w_new, b_new, losses = train_sgd(X, y, w, b)

# print("Weights:", w_new)
# print("Bias:", b_new)
# print("First loss:", losses[0])
# print("Last loss:", losses[-1])
# print("Loss count:", len(losses))
import numpy as np
a = np.array([1,2,3])
print(a.var())
print(a.var(ddof = 1))