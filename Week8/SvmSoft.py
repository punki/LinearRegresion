__author__ = 'punki'
from common.DataSet import DataSet
import time, random, numpy as np
from sklearn import svm

training_data_set = DataSet('features.train.txt')
test_data_set = DataSet('features.test.txt')

clf = svm.SVC(kernel='poly', C=0.01, degree=2)
clf.fit(training_data_set.get_x(), training_data_set.get_target())

test_predict = clf.predict(test_data_set.get_x())
test_target = test_data_set.get_target()
print 'Error: {}'.format(len(test_target[test_target != test_predict])/float(len(test_target)))