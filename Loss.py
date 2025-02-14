import numpy as np


class Loss:

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularisation(self, layer):

        regularisation_loss = 0

        if layer.weight_regulariser_l1 > 0:
            regularisation_loss += layer.weight_regulariser_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regulariser_l2 > 0:
            regularisation_loss += layer.weight_regulariser_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regulariser_l1 > 0:
            regularisation_loss += layer.bias_regulariser_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regulariser_l2 > 0:
            regularisation_loss += layer.bias_regulariser_l2 * np.sum(layer.bias * layer.bias)

        return regularisation_loss
