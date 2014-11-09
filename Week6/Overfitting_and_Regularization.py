__author__ = 'punki'
import time, random, numpy as np


def load_data_from_file():
    in_data = np.fromfile('/home/punki/PycharmProjects/LearningFromData/LinearRegresion/Week6/in.dta', float, -1, '   ')
    out_data = np.fromfile('/home/punki/PycharmProjects/LearningFromData/LinearRegresion/Week6/out.dta', float, -1,
                           '   ')
    sample_in = in_data.reshape(-1, 3)
    sample_out = out_data.reshape(-1, 3)
    return sample_in, sample_out


def transformation(data):
    newData = []
    for val in data:
        x1 = val[0]
        x2 = val[1]
        row = [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
        newData.append(row)
    return np.array(newData)

s_in, s_out = load_data_from_file()

print(transformation(s_in))