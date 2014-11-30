class LinearRegresion:
    def __init__(self, training_data_set, test_data_set, vlambda, transofrmation):
        self.transofrmation = transofrmation
        self.vlambda = vlambda
        self.test_data_set = test_data_set
        self.training_data_set = training_data_set

    def