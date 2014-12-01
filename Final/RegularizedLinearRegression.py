
__author__ = 'punki'

from common.DataSet import DataSet
from common.LinearRegresion import LinearRegresion
import time, random, numpy as np
from sklearn import svm

training_data_set = DataSet('features.train.txt')
test_data_set = DataSet('features.test.txt')
classes = np.unique(training_data_set.get_y())

fake_transofrmation = (lambda x1, x2: (1, x1, x2))
ex8_transofrmation = (lambda x1, x2: (1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2))


def experiment1(training, test, transformation, reg_lambda, exp_id):
    lr = LinearRegresion(reg_lambda, transformation)
    lr.fit(training)
    e_in = lr.error(training)
    e_out = lr.error(test)
    all_e_in.append((exp_id, e_in))
    all_e_out.append((exp_id, e_out))
    print('lambda={} exp_id={} e_in={} e_out={}'.format(reg_lambda, exp_id, e_in, e_out))


print 'trans non'
all_e_in = []
all_e_out = []
for one in range(0, 10):
    experiment1(training_data_set.one_versus_all(one), test_data_set.one_versus_all(one), fake_transofrmation, 1, one)
print('min e_in={}'.format(min(all_e_in, key=lambda x: x[1])))
print('min e_out={}'.format(min(all_e_out, key=lambda x: x[1])))

print 'trans quadrat'
all_e_in = []
all_e_out = []
for one in range(0, 10):
    experiment1(training_data_set.one_versus_all(one), test_data_set.one_versus_all(one), ex8_transofrmation, 1, one)
print('min e_in={}'.format(min(all_e_in, key=lambda x: x[1])))
print('min e_out={}'.format(min(all_e_out, key=lambda x: x[1])))

print 'ove vs one'
all_e_in = []
all_e_out = []
for reg_lambda in {0.01, 1}:
    experiment1(training_data_set.one_versus_one(1, 5), test_data_set.one_versus_one(1, 5), ex8_transofrmation,
                reg_lambda, reg_lambda)
print('min e_in={}'.format(min(all_e_in, key=lambda x: x[1])))
print('min e_out={}'.format(min(all_e_out, key=lambda x: x[1])))




