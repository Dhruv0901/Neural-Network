import numpy as np


class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)# creates an array with the same structure as dvalues

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)# creates an array of 1 column and -1(entire) rows
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)# results a 3 X 3 matrix if single_output has three elements for eg
            # jacobian matrix is a matrix of all the vector of the multi-variable function
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            # dinputs is the array of gradients for each sample