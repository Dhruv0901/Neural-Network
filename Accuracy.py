import numpy as np


class Accuracy:# common accuracy class

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy
