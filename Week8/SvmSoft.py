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

for n in range(1, len(classes)):
    # classes_split = np.array_split(classes, len(classes)-n)
    e_in_all = []
    e_out_all = []
    for idx in range(0, len(classes) - (n - 1)):
        for C in { 0.001, 0.01, 0.1, 1}:
            cs = classes[idx:idx + n]
            clf = svm.SVC(kernel='poly', C=C, degree=2)
            training_n_versus = training_data_set.new_n_versus_all(cs)
            test_n_versus = test_data_set.new_n_versus_all(cs)
            clf.fit(training_n_versus.get_x(), training_n_versus.get_y())

            print 'C={} n={} svm={}'.format(C, n, clf.n_support_)

            e_in = compute_error(clf, training_n_versus.get_x(), training_n_versus.get_y())
            e_out = compute_error(clf, test_n_versus.get_x(), test_n_versus.get_y())
            e_in_all.append(e_in)
            e_out_all.append(e_out)

    e_in_mean = np.mean(e_in_all)
    e_out_mean = np.mean(e_out_all)
    e_in_for_n.append((n, e_in_mean))
    e_out_for_n.append((n, e_out_mean))
    print 'n={} Ein: {}'.format(n, e_in_mean)
    print 'n={} Eout: {}'.format(n, e_out_mean)

print 'Min ein: {}'.format(min(e_in_for_n, key=lambda i: i[1]))