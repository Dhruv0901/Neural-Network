from CategoricalCrossentropy_Loss import Loss_CategoricalCrossentropy
from CrossEntropyLoss_Softmax_Activtion import Activation_Softmax_Loss_CategoricalCrossEntropy
from Input_Layer import Layer_Input
from Softmax_Activation import Activation_Softmax


class Model:
    def __init__(self):
        self.layers = []# this is our array of layers
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)# appends layers in our main array

    def set(self,*,loss,optimiser,accuracy):
        self.loss = loss# initialises loss
        self.optimiser = optimiser# initialises optimiser
        self.accuracy = accuracy# initialises accuracy

    def finalise(self):# joins each layer amongst themselves like a linked list

        self.input_layer = Layer_Input()# initial layer points to itself thanks to input layer class
        self.trainable_layers = []

        layer_count = len(self.layers)

        for i in range(layer_count):

            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i<layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss# outermost layer points to loss
                self.output_layer_activation = self.layers[i]# pointing to output layer

            if hasattr(self.layers[i], "weights"):# pointing towards dense layer since dropout layer cant be trained
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)#initialises trainable layers

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):

            output = self.forward(X, training=True)# if def train then it means we are training
            data_loss, regularisation_loss = self.loss.calculate(output, y, include_regularisation=True)
            loss = data_loss + regularisation_loss
            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)
            self.backward(output, y)
            self.optimiser.pre_update_params()
            for layer in self.trainable_layers:
                self.optimiser.update_params(layer)
            self.optimiser.post_update()

            if not epoch % print_every:
                print(f'epoch: {epoch}' +
                      f'acc: {accuracy}' +
                      f'loss: {loss}' +
                      f'lr: {self.optimiser.current_learning_rate}')
        if validation_data is not None:

            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)# we are not training since this is validation data
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            print(f'validation: '+
                  f'accuracy: {accuracy}'+
                  f'loss: {loss}')

    def forward(self, X, training):

        self.input_layer.forward(X, training)# initialises the input aka the first layer
        # training sets the way for dropout layer

        for layer in self.layers:# uses the layer's own method to forward input
            layer.forward(layer.prev.output, training)

        return layer.output# returns the final output of the entire forward pass

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)

        return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):# reverses our array of layers and reverts to gradients of previous layers
            layer.backward(layer.next.dinputs)


