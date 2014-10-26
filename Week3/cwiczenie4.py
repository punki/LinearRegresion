__author__ = 'punki'
import time, random, numpy as np



def target_function(x):
    return np.sin(np.pi * x)
    # return 2*x

def h_function(samples, w):
    return samples * w

def generate_samples():
    samples = np.random.uniform(-1.0, 1.0, (n, 1))
    return samples

n, N = 2000, 10
allW = []
avgW = 0

# approximate target function

for step in range(N):
    samples = generate_samples()
    y = []
    for s in samples[:, 0]:
        y.append(target_function(s))
    samples = np.power(samples, 2)
    w = np.linalg.pinv(samples).dot(y)
    allW.append(w)

avgW = np.average(allW)
print 'avg w={}'.format(avgW)

# bias
bias_errors =[]

for step in range(N):
    samples = generate_samples()
    y = []
    for s in samples[:, 0]:
        y.append([target_function(s)])
    samples = np.power(samples, 2)
    gy = h_function(samples, avgW)
    bias_errors.append(np.power(gy - y, 2))
print 'bias={}'.format(np.average(bias_errors))

# variance
variance_errors = []
for step in range(N):
    samples = generate_samples()
    y = []
    for s in samples[:, 0]:
        y.append(target_function(s))
    samples = np.power(samples, 2)
    w = np.linalg.pinv(samples).dot(y)
    gy = h_function(samples, avgW)
    hy = h_function(samples, w)
    variance_errors.append(np.power(gy - hy, 2))
print 'variance={}'.format(np.average(variance_errors))



