import numpy as np

from CategoricalCrossentropy_Loss import Loss_CategoricalCrossentropy
from Softmax_Activation import Activation_Softmax


class Activation_Softmax_Loss_CategoricalCrossEntropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:# if no of dimensions are 2 aka if y_true is in one-hot encode form
            y_true = np.argmax(y_true, axis=1) # we want it to be in a single array
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1 # derivative of categorical loss with respect to softmax function is
        # predicted values - ground-truth values
        self.dinputs = self.dinputs / samples
        # Normalisation
        # essentially by calculating partial derivative of categorical cross-entropy loss
        # with respect to softmax function can be reduced to predicted values - true values
        # this can save a lot of computing time
