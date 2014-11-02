__author__ = 'punki'
import time, random, numpy as np


def line(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y2 - a * x2
    return lambda x: a * x + b

def generate_samples(number_of_samples, d):
    samples = np.ones((number_of_samples, d+1), float)
    samples[:, 1:d+1] = np.random.uniform(-1.0, 1.0, (number_of_samples, d))
    return samples

