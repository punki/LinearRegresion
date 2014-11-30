from Final.LinearRegresion import LinearRegresion

__author__ = 'punki'
from common.DataSet import DataSet
import time, random, numpy as np
from sklearn import svm

training_data_set = DataSet('features.train.txt')
test_data_set = DataSet('features.test.txt')
classes = np.unique(training_data_set.get_y())

fake_transofrmation = (lambda x1, x2: (1, x1, x2))
vlambda = 1



lr = LinearRegresion(training_data_set, test_data_set, vlambda, fake_transofrmation)

