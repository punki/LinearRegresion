__author__ = 'punki'

from Final.LinearRegresion import LinearRegresion
from common.DataSet import DataSet
import time, random, numpy as np
from sklearn import svm

training_data_set = DataSet('features.train.txt')
test_data_set = DataSet('features.test.txt')
classes = np.unique(training_data_set.get_y())

fake_transofrmation = (lambda x1, x2: (1, x1, x2))
reg_lambda = 1

all_e_in = []
for one in range(0, 10):
    lr = LinearRegresion(reg_lambda, fake_transofrmation)
    lr.fit(training_data_set.one_versus_all(one))
    e_in = lr.error(training_data_set.one_versus_all(one))
    e_out = lr.error(test_data_set.one_versus_all(one))
    all_e_in.append((one, e_in))
    print('one={} vs all e_in={} e_out={}'.format(one, e_in, e_out))

print('min={}'.format(min(all_e_in, key=lambda x: x[1])))
