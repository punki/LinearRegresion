__author__ = 'punki'
import time, random, numpy as np


def line_classifier(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y2 - a * x2
    return lambda x, y: 1 if y >= a * x + b else -1


def generate_samples(number_of_samples, d):
    samples = np.ones((number_of_samples, d + 1), float)
    samples[:, 1:d + 1] = np.random.uniform(-1.0, 1.0, (number_of_samples, d))
    return samples


def new_classifier_line():
    pionts = np.random.uniform(-1, 1, (1, 4))
    return line_classifier(pionts[0][0], pionts[0][1], pionts[0][2], pionts[0][3])


def classified(samples, function):
    classes = []
    for x, y in samples[:, 1:3]:
        classes.append(function(x, y))
    return np.array(classes)


def d_cross_entropy_error(point, w, target_class):
    exp_power = target_class * (w.transpose()).dot(point)
    return (target_class * point) / (1 + np.exp(exp_power))


def gradient(point, w, target_class):
    return -1 * d_cross_entropy_error(point, w, target_class)


def compute_error(out_size, dimension, w, target_function):
    out_samples = generate_samples(out_size, dimension)
    out_target_classes = classified(out_samples, target_function)
    classes = map(lambda s: sigmoid(s), out_samples.dot(w))
    diff = out_target_classes - classes
    return len(filter(lambda x: x != 0, diff))


def sigmoid(s):
    e = np.exp(s)
    return 1 if e / (1 + e) >= 0.5 else -1


def logistic_regression(number_of_samples, repeat=100, learning_rate=0.01):

    errors = []
    iters=0

    for i in range(repeat):
        target_function = new_classifier_line()
        dimension = 2
        samples = generate_samples(number_of_samples, dimension)
        samples_target_classes = classified(samples, target_function)
        w = np.zeros(dimension + 1)
        iter = 0
        while iter <= 1000000:
            s_idx = range(number_of_samples)
            np.random.shuffle(s_idx)
            new_w = np.copy(w)
            for idx in s_idx:
                new_w -= learning_rate * gradient(samples[idx], new_w, samples_target_classes[idx])
            if np.linalg.norm(w - new_w) < 0.01:
                break
            w = new_w
            iter += 1
            out_sample_size = number_of_samples * 10.0
        iters+=iter
        errors.append(compute_error(out_sample_size, dimension, w, target_function)/out_sample_size)

    print 'avg #iter={}, avg error={}'.format(iters/float(repeat), np.average(errors))

logistic_regression(100)