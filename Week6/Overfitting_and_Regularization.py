__author__ = 'punki'
import time, random, numpy as np


def load_data_from_file():
    sample_in = np.fromfile('in.dta', float, -1, ' ').reshape(-1, 3)
    sample_out = np.fromfile('out.dta', float, -1, ' ').reshape(-1, 3)
    return sample_in, sample_out


def countDiffElements(a, b):
    diffElements = 0
    for idx, he in enumerate(a):
        if b[idx] != he:
            diffElements += 1
    return diffElements


def transformation(data):
    newData = []
    for val in data:
        x1 = val[0]
        x2 = val[1]
        row = [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
        newData.append(row)
    return np.array(newData)


def linearRegresion(s_in, s_out, learn_w_function):
    s_in_t = transformation(s_in)
    s_in_y = s_in[:, 2:3]

    s_out_t = transformation(s_out)
    s_out_y = s_out[:, 2:3]

    w = learn_w_function(s_in_t, s_in_y)
    h_in_class = map(lambda e: 1 if e >= 0 else -1, s_in_t.dot(w))
    h_out_class = map(lambda e: 1 if e >= 0 else -1, (s_out_t).dot(w))
    e_in = countDiffElements(h_in_class, s_in_y) / float(len(s_in))
    e_out = countDiffElements(h_out_class, s_out_y) / float(len(s_out))
    return e_in, e_out


def with_regularization_function(x, y):
    lambda_value = 10 ** k
    a1 = x.T.dot(x)
    a2 = lambda_value * np.identity(len(a1))
    a3 = a1 + a2
    b1 = x.T.dot(y)
    return np.linalg.inv(a3).dot(b1)


s_in, s_out = load_data_from_file()
# ex 2
without_regularization = (lambda x, y: np.linalg.pinv(x).dot(y))
print('without_regularization (e_in, e_out): {}'.format(linearRegresion(s_in, s_out, without_regularization)))

# with regularization
for k in range(-2, 3):
    print('with_regularization k={} (e_in, e_out): {}'
          .format(k, linearRegresion(s_in, s_out, with_regularization_function)))

#  ex 6
all_e_out = []
for k in range(-7, 8):
    e_out = linearRegresion(s_in, s_out, with_regularization_function)[1]
    print('k={} eout={}'.format(k,e_out))
    all_e_out.append(e_out)
print 'min eout={}'.format(min(all_e_out))
