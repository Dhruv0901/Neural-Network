from nnfs.datasets import spiral_data

from Adam_Optimiser import Optimiser_Adam
from Categorical_Accuracy import Accuracy_Categorical
from Dense_Layer import Layer_Dense
from Model import Model
from ReLU_Activation import Activation_ReLU
from Sigmoid_Activation import Activation_Sigmoid
from BinaryCrossentropy_Loss import Loss_BinaryCrossentropy

X, y = spiral_data(samples=100, classes=2)
X_Test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)# hot encode
y_test = y_test.reshape(-1, 1)

model1 = Model()

model1.add(Layer_Dense(2, 64, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))
model1.add(Activation_ReLU())
model1.add(Layer_Dense(64, 1))
model1.add(Activation_Sigmoid())

model1.set(
    loss=Loss_BinaryCrossentropy(),
    optimiser=Optimiser_Adam(decay=5e-7),
    accuracy=Accuracy_Categorical(binary=True)
)

model1.finalise()
model1.train(X, y, validation_data=(X_Test, y_test),epochs=10000, print_every=100)
