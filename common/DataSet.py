__author__ = 'punki'
import numpy as np


class DataSet:
    def __init__(self, fileName = None, data=None):
        self.classColumn = 0
        self.numberOfFeatures = 2
        if fileName is not None:
            self.data = np.fromfile(fileName, float, -1, ' ').reshape(-1, self.numberOfFeatures + 1)
        else:
            self.data = data
        self.x = self.only_x(self.data)
        self.y = self.only_y(self.data)

    def only_x(self, data):
        return np.delete(data, self.classColumn, self.classColumn + 1)

    def only_y(self,data):
        return data[:, self.classColumn]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def one_versus_all(self, one):
        new_data = np.copy(self.data)
        for d in new_data:
            d[self.classColumn] = 1 if d[self.classColumn] == one else -1
        return DataSet(data=new_data)

    def one_versus_one(self,one,versus):
        new_data = []
        for z in self.data:
            if z[self.classColumn] in {one, versus}:
                new_z = np.copy(z)
                new_z[self.classColumn] = 1 if z[self.classColumn] == one else -1
                new_data.append(new_z)
        return DataSet(data=np.array(new_data))

