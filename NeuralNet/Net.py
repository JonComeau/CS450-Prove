from random import uniform
from math import exp


class Neuron:
    def __init__(self, threshold=0):
        self.weights = []
        self.threshold = threshold

    def output(self, inputs, bias):
        print("\t\tNode Start")
        print("\t\t\tBias: {}".format(bias))
        comp_weights = bias[0] * bias[1]

        # Setting up weights if not exists
        if len(self.weights) == 0:
            for ind in range(len(inputs)):
                self.weights.append(round(uniform(-1, 1), 2))
            print("\t\t\tInitialize weights")

        for index in range(len(inputs)):
            comp_weights += inputs[index] * self.weights[index]

        print("\t\t\tOutput value: {}".format(comp_weights))

        if comp_weights < self.threshold:
            return 0
        else:
            return 1


class NetLayer:
    def __init__(self, node_count, bias=-1):
        self.nodes = [Neuron() for i in range(node_count)]
        self.bias_weights = [round(uniform(-1, 1), 2) for i in range(node_count)]
        self.bias = bias

    def calculate(self, inputs):
        print("\tLayer Start")
        print("\t\tNumber of nodes: {}".format(len(self.nodes)))
        return [self.nodes[index].output(inputs, (self.bias, self.bias_weights[index])) for index in range(len(self.nodes))]


class NNetClassifier:
    def __init__(self, layer_count):
        self.layers = []

        for layer in range(layer_count):
            self.layers.append(NetLayer(3))

    def calculate(self, inputs):
        print("Calculate Start")
        return [self.layers[index].calculate(inputs) for index in range(len(self.layers))]
