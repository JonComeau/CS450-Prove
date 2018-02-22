from random import uniform
from math import exp


class Neuron:
    def __init__(self, threshold=0):
        self.weights = []
        self.threshold = threshold

    def output(self, inputs, bias):
        comp_weights = bias[0] * bias[1]

        if len(self.weights) == 0:
            for ind in range(len(inputs)):
                round(uniform(-1, 1), 2)

        for index in range(len(inputs)):
            comp_weights += inputs[index] * self.weights[index]

        value = 1 / (1 + exp(-1 * comp_weights))

        if value < self.threshold:
            return 0
        else:
            return 1


class NetLayer:
    def __init__(self, node_count, bias=-1):
        self.nodes = [Neuron() for i in range(node_count)]
        self.bias_weights = [round(uniform(-1, 1), 2) for i in range(node_count)]
        self.bias = bias

    def calculate(self, inputs):
        return [self.nodes[index].output(inputs, (self.bias, self.bias_weights[index])) for index in range(len(self.nodes))]


class NNetClassifier:
    def __init__(self, layers):
        self.layers = []

        for layer in layers:
            self.layers.append(NetLayer(layer))

    def calculate(self, inputs):
        return [self.layers[index].calculate(inputs) for index in range(len(self.layers))]
