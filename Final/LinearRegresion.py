import time, random, numpy as np


class LinearRegresion:
    def __init__(self, reg_lambda, transofrmation):
        self.transofrmation = transofrmation
        self.reg_lambda = reg_lambda
        self.w = []

    def fit(self, training_data_set):
        x = training_data_set.get_x()
        y = training_data_set.get_y()
        a1 = x.T.dot(x)
        a2 = self.reg_lambda * np.identity(len(a1))
        a3 = a1 + a2
        b1 = x.T.dot(y)
        self.w = np.linalg.inv(a3).dot(b1)

    def error(self, data_set):
        predicted = [1 if x>=0 else -1 for x in data_set.get_x().dot(self.w)]
        correct = data_set.get_y()
        return len(correct[correct != predicted])/float(len(correct))