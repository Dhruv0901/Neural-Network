from caffe2.python.helpers.train import accuracy

from Input_Layer import Layer_Input
from Simulation import regularisation_loss


class Model:
    def __init__(self):
        self.layers = []# this is our array of layers

    def __add__(self, layer):
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

    def train(self, X, y, *, epochs=1, print_every=1):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):

            output = self.forward(X)
            data_loss, regularisation_loss = self.loss.calculate(output, y)
            loss = data_loss + regularisation_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

    def forward(self, X):

        self.input_layer.forward(X)# initialises the input aka the first layer

        for layer in self.layers:# uses the layer's own method to forward input
            layer.forward(layer.prev.output)

        return layer.output# returns the final output of the entire forward pass

