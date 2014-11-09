__author__ = 'punki'
import time, random, numpy as np


def load_data_from_file():
    sample_in = (np.fromfile('in.dta', float, -1, '   ')).reshape(-1, 3)
    sample_out = (np.fromfile('out.dta', float, -1, '   ')).reshape(-1, 3)
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


def linearRegresion(s_in, s_out):
    s_in_t = transformation(s_in)
    s_in_y = s_in[:, 2:3]

    s_out_t = transformation(s_out)
    s_out_y = s_out[:, 2:3]

    w = np.linalg.pinv(s_in_t).dot(s_in_y)
    h_in_class = map(lambda e: 1 if e >= 0 else -1, s_in_t.dot(w))
    h_out_class = map(lambda e: 1 if e >= 0 else -1, (s_out_t).dot(w))
    e_in = countDiffElements(h_in_class, s_in_y) / float(len(s_in))
    e_out = countDiffElements(h_out_class, s_out_y) / float(len(s_out))
    return e_in, e_out


s_in, s_out = load_data_from_file()
print(linearRegresion(s_in, s_out))
