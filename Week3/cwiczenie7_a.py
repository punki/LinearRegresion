__author__ = 'punki'
import time, random, numpy as np


def h_function(samples, w):
    values = []
    for s in samples:
        values.append(s.dot(w))
    return values


def target_function_all(samples):
    y = []
    for s in samples[:, 1]:
        y.append([target_function(s)])
    return y


def target_function(x):
    return np.sin(np.pi * x)
    # return 2


def generate_samples():
    samples = np.ones((n, 2), float)
    samples[:, 1:2] = np.random.uniform(-1.0, 1.0, (n, 1))
    return samples

def generate_samples_e(n):
    samples = np.ones((n, 6), float)
    samples[:, 1:6] = np.random.uniform(-1.0, 1.0, (n, 1))
    return samples


def regresion_samples_a():
    r = np.ones((n, 2), float)
    r[:, 1:2] = np.zeros((n, 1))
    return r


n, N = 200, 1000
allW = []

# approximate target function
def regresion_a(y):
    return np.linalg.pinv(regresion_samples_a()).dot(y)

def regresion(samples, y):
    return np.linalg.pinv(samples).dot(y)


def transformation(samples):
    return [np.array([x0, x1, np.power(x2, 2), np.power(x3, 3)]) for
            x0, x1, x2, x3, x4, x5 in samples]


for step in range(N):
    samples = generate_samples_e(n)
    y = target_function_all(samples)
    samples = transformation(samples)
    w = regresion(samples, y)
    allW.append(w)

avgW = np.average(allW,axis=0)
print 'avg w={}'.format(avgW)

# bias
bias_errors = []
for step in range(N):
    samples = generate_samples_e(n)
    y = target_function_all(samples)
    samples = transformation(samples)
    gy = h_function(samples, avgW)
    bias_errors.append(np.power(np.subtract(gy, y), 2))
avgBias = np.average(bias_errors)
print 'bias={}'.format(avgBias)

# variance
variance_errors = []
for step in range(N):
    samples = generate_samples_e(n)
    y = target_function_all(samples)
    samples = transformation(samples)
    w = regresion(samples, y)
    gy = h_function(samples, avgW)
    hy = h_function(samples, w)
    variance_errors.append(np.power(np.subtract(gy, hy), 2))
avgVariance = np.average(variance_errors)
print 'variance={}'.format(avgVariance)

print 'Eout={}'.format(avgBias+avgVariance)

