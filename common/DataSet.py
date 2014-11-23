__author__ = 'punki'
import numpy as np


class DataSet:
    def __init__(self, fileName):
        self.targetColumn = 0
        self.numberOfFeatures = 2
        self.data = np.fromfile(fileName, float, -1, ' ').reshape(-1, self.numberOfFeatures + 1)

    def get_x(self):
        return np.delete(self.data, self.targetColumn, self.targetColumn + 1)

    def get_target(self):
        return self.data[:, self.targetColumn]