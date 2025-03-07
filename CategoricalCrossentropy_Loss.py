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

    def backward(self, dvalues, y_true):# vector of predicted values(dvalues) and ground-truth vector(y_true)
        samples = len(dvalues)
        labels = len(dvalues[0])# 3 in this case
        if len(y_true.shape) == 1:# if ground vector is in form for eg)[1, 2, 0]
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues# derivative of loss function is -(ground-truth vector/predicted values)
        self.dinputs = self.dinputs / samples# normalisation for scalling during optimisation