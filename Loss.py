import numpy as np




class Loss:

    def remember_trainable_layers(self, trainable_Layers):
        self.trainable_layers = trainable_Layers

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss, self.regularisation_trainLayer()

    def regularisation_trainLayer(self):

        regularisation_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regulariser_l1 > 0:
                regularisation_loss += layer.weight_regulariser_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regulariser_l2 > 0:
                regularisation_loss += layer.weight_regulariser_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regulariser_l1 > 0:
                regularisation_loss += layer.bias_regulariser_l1 * np.sum(np.abs(layer.weights))
            if layer.bias_regulariser_l2 > 0:
                regularisation_loss += layer.bias_regulariser_l2 * np.sum(layer.biases * layer.biases)

        return regularisation_loss

    def regularisation(self, layer):

        regularisation_loss = 0

        if layer.weight_regulariser_l1 > 0:
            regularisation_loss += layer.weight_regulariser_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regulariser_l2 > 0:
            regularisation_loss += layer.weight_regulariser_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regulariser_l1 > 0:
            regularisation_loss += layer.bias_regulariser_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regulariser_l2 > 0:
            regularisation_loss += layer.bias_regulariser_l2 * np.sum(layer.biases * layer.biases)

        return regularisation_loss
