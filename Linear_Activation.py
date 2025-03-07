class Activation_Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs

    def backward(self, dvalues):
        self.dinput = dvalues.copy()
