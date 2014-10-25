__author__ = 'punki'
import time, random, numpy as np

n, N = 2, 100000
allW = []


def target_function(x):
    return np.sin(np.pi * x)


for step in range(N):
    samples = np.random.uniform(-1.0, 1.0, (n, 1))
    y = []
    for s in samples[:,0]:
        y.append(target_function(s))
    w = np.linalg.pinv(samples).dot(y)
    allW.append(w)

print 'avg w={}'.format(np.average(allW))