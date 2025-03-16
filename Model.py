class Model:
    def __init__(self):
        self.layers = []

    def __add__(self, layer):
        self.layers.append(layer)