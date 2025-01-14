"""
A variety of general-use functions.
"""

import numpy as np

"""
Inverts a dictionary (assumes injectivity).
"""
def invert[K,V](inDict: dict[K,V]) -> dict[V,K]:
    return {v: k for k,v in inDict.items()}


# Calculate Euclidean distance
def eucDist(p1, p2):
    return np.sqrt(sum((p1[i]-p2[i])**2 for i in range(len(p1))))

