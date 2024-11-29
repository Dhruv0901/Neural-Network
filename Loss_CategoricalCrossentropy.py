import numpy as np
from Loss import Loss

class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)# this is to prevent division by 0
        # so values less than 1e-7 becomes 1e-7 and values more than 1 - 1e-7 becomes 1 - 1e-7

        if len(y_true.shape) == 1:# when the desired output are in the form of one dimension array
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_pred_clipped) == 2:# when we are dealing with two dimension array
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        neg_log = -np.log(correct_confidences)
        return neg_log