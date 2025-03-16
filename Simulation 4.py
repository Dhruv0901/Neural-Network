from Adam_Optimiser import Optimiser_Adam
from Dense_Layer import Layer_Dense
from Linear_Activation import Activation_Linear
from MeanSquaredError_Loss import Loss_MeanSquaredError
from Model import Model

from nnfs.datasets import sine_data
from ReLU_Activation import Activation_ReLU
from Regression_Accuracy import Accuracy_Regression

X,y = sine_data()
model1 = Model()

model1.add(Layer_Dense(1,64))
model1.add(Activation_ReLU())
model1.add(Layer_Dense(64, 64))
model1.add(Activation_ReLU())
model1.add(Layer_Dense(64, 1))
model1.add(Activation_Linear())

model1.set(
    loss=Loss_MeanSquaredError(),
    optimiser=Optimiser_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Regression()
)

model1.finalise()
model1.train(X, y, epochs=10000, print_every=100)

