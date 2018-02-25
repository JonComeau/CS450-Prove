from random import uniform
from math import exp

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, threshold=0):
        self.comp = 0
        self.act = 0
        self.weights = []
        self.threshold = threshold
        self.bias_weight = round(uniform(-1, 1), 2)

    def output(self, inputs, bias):
        print("\t\tNode Start")
        print("\t\t\tInputs: {}".format(inputs))
        comp_weights = bias * self.bias_weight

        # Setting up weights if not exists
        if len(self.weights) == 0:
            for ind in range(len(inputs)):
                self.weights.append(round(uniform(-1, 1), 2))
            print("\t\t\tInitialize weights")

        for index in range(len(inputs)):
            comp_weights += inputs[index] * self.weights[index]

        self.comp = comp_weights
        self.act = sigmoid(comp_weights)

        print("\t\t\tOutput value: {}".format(comp_weights))
        print("\t\t\tActivation value: {}".format(self.act))

        return self.act


class NetLayer:
    def __init__(self, node_count, bias=-1):
        self.nodes = [Neuron() for i in range(node_count)]
        self.bias = bias

    def calculate(self, inputs):
        print("\tLayer Start")
        print("\t\tNumber of nodes: {}".format(len(self.nodes)))
        return [self.nodes[index].output(inputs, self.bias) for index in range(len(self.nodes))]


class NNetClassifier:
    def __init__(self, layers):
        self.layers = []

        for layer in layers:
            self.layers.append(NetLayer(layer))

    def calculate(self, inputs):
        print("Calculate Start")

        output = inputs

        for layer in self.layers:
            output = layer.calculate(output)

        return output

    def calculate_all(self, inputs):
        outputs = [self.calculate(inputs[index]) for index in range(len(inputs))]

        return outputs
