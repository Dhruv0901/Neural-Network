import numpy as np

from Loss import Loss

class Loss_MeanSquaredError(Loss):

    def forward(self, y_true, y_pred):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)# average of square of difference
                                                    # between the actual value and the predicted value
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs # derivative of mean square error loss
        self.dinputs = self.dinputs / samples # this is done for the normalisation