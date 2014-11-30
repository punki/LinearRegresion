__author__ = 'punki'
from common.DataSet import DataSet
import time, random, numpy as np
from sklearn import svm


def compute_error(clf, x, target):
    predict = clf.predict(x)
    return len(target[target != predict]) / float(len(target))


training_data_set = DataSet('features.train.txt')
test_data_set = DataSet('features.test.txt')
classes = np.unique(training_data_set.get_y())

e_in_for_n = []
e_out_for_n = []

for one in {1, 5}:
    e_in_all = []
    e_out_all = []
    for C in {0.0001,0.001, 0.01, 0.1, 1}:
        q=5
        clf = svm.SVC(kernel='poly', C=C, degree=q)
        training_one_versus = training_data_set.new_one_versus_all(one)
        test_one_versus = test_data_set.new_one_versus_all(one)
        clf.fit(training_one_versus.get_x(), training_one_versus.get_y())

        e_in = compute_error(clf, training_one_versus.get_x(), training_one_versus.get_y())
        e_out = compute_error(clf, test_one_versus.get_x(), test_one_versus.get_y())
        e_in_all.append(e_in)
        e_out_all.append(e_out)

        print 'Q={}, n={} C={} svm={} ein={} eout={}'.format(q, one, C, clf.n_support_, e_in, e_out)

    e_in_mean = np.mean(e_in_all)
    e_out_mean = np.mean(e_out_all)
    e_in_for_n.append((one, e_in_mean))
    e_out_for_n.append((one, e_out_mean))
    print 'n={} Ein: {}'.format(one, e_in_mean)
    print 'n={} Eout: {}'.format(one, e_out_mean)

print 'Min ein: {}'.format(min(e_in_for_n, key=lambda i: i[1]))