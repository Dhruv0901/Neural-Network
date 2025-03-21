import numpy as np

from Accuracy import Accuracy

class Accuracy_Regression(Accuracy):#accuracy for regression model

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions-y) < self.precision
