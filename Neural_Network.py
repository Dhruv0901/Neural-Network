from nnfs.datasets import spiral_data
import numpy as np
from CrossEntropyLoss_Softmax_Activtion import Activation_Softmax_Loss_CategoricalCrossEntropy
from Dense_Layer import Layer_Dense
from ReLU_Activation import Activation_ReLU
from Softmax_Activation import Activation_Softmax
from CategoricalCrossentropy_Loss import Loss_CategoricalCrossentropy



X,y = spiral_data(100, 3)# spiral data of two points x and y stored as X and three classes 0, 1 and 2
dense1 = Layer_Dense(2,3)# two inputs(x and y of the points) and three neurons
dense1.forward(X)# values stored in X passed so a list with two elements are multiplied with the 3 neurons
# print(dense1.output[:7])
activation1 = Activation_ReLU()# used in hidden layers
activation1.forward(dense1.output)# all the negative values gets replaced by 0
# print(activation1.output[:7])
dense2 = Layer_Dense(3,3)# neuron for outer layer 3 inputs for three neurons
dense2.forward(activation1.output)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
loss = loss_activation.forward(dense2.output, y)
print(loss)
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print('acc:', accuracy)

#backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

