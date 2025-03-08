import numpy as np

import Loss

class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_true, y_pred):
        sample_loss = np.mean(np.abs(y_true-y_pred),axis=-1)
        return sample_loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)# no of observations
        outputs = len(dvalues[0])# no of predictions in each observation

        self.dinputs = np.sign(y_true-dvalues) / outputs # normalising the loss among each prediction
        self.dinputs = self.dinputs / samples # normalising the total loss among each observation