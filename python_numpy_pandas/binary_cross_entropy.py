import numpy as np

z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
y = np.array([0, 0, 1, 1, 1])

def binary_cross_entropy(z, y):
    return np.mean(-y*z+np.logaddexp(0,z))

print(binary_cross_entropy(z, y))

z2 = np.array([-1000.0, 1000.0])
y2 = np.array([1, 0])

print(binary_cross_entropy(z2, y2))