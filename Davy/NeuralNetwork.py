import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def nn_and(inputs):
    nn = NeuralNetwork([2, 1])
    nn.fit(inputs, [[[-30], [20], [20]]])
    return nn


def nn_xor(inputs):
    nn = NeuralNetwork([2, 2, 1])
    nn.fit(inputs, [[[-10, 30], [20, -20], [20, -20]], [[-30], [20], [20]]])
    return nn


def nn_test():
    nn = NeuralNetwork([3, 1])
    tr = pd.DataFrame({'one': [1, 2, 3], 'two': [4, 5, 6], 'three': [7, 8, 9]})
    te = pd.DataFrame({'one': [4, 2, 6], 'two': [3, 8, 4], 'three': [3, 5, 6]})
    y = [[1], [0], [1]]
    w = [[[1], [1], [1], [1]]]
    nn.fit(tr, te, y, w)
    return nn


class NeuralNetwork():

    def __init__(self, shape):
        self.layers = []
        self.shape = shape
        self.df = pd.DataFrame()
        for i in range(len(shape)):
            print('new layer of ' + str(len(shape)) + ' size')
            bias = 1
            if(i == len(shape) - 1):
                bias = 0
            layer = Layer(shape[i], self, bias)
            self.layers.append(layer)

    def fit(self, train, test, labels, weights):
        self.input_nodes = self.layers[0].nodes
        self.train = train
        self.test = test
        self.labels = labels
        self.feature_count = len(train.iloc[0, :])

        for l in range(len(self.layers) - 1):
            for n in range(len(self.layers[l].nodes)):
                self.layers[l].nodes[n].create_connections(self.layers[l + 1])
                self.layers[l].nodes[n].set_weights(weights[l][n])

        if(self.feature_count + 1 != len(self.input_nodes)):
            print('inputs mismatched to input layer shape')

    def predict(self, data):
        predictions = []
        for i in range(0, data.shape[0]):
            predictions.append(self.predict_row(data.iloc[i, :]))
        return predictions

    def predict_row(self, inputs):
        for i in range(len(self.input_nodes)):
            if(i == 0):
                self.input_nodes[i].activation = 1.0
            else:
                print(type(self.input_nodes[i]))
                print(self.input_nodes[i].activation)
                self.input_nodes[i].activation = inputs[i - 1]

        for i in range(1, len(self.layers)):
            # print('layer ' + str(i))
            l = self.layers[i]
            # skip bias if predicting hidden layer outputs
            adj = 0 if i == len(self.layers) - 1 else 1
            for n in range(len(l.nodes) - adj):
                for x in self.layers[i - 1].nodes:
                    # print(str(i) + ' : ' + str(x.weights[n]))
                    l.nodes[n + adj].activation += x.weights[n] * x.activation
                l.nodes[n + adj].activation = NeuralNetwork.sigmoid(l.nodes[n + adj].activation)
                print('activation = ' + str(l.nodes[n + adj].activation))
            if(i == len(self.layers) - 1):
                a = self.layers[i].activations()[0]
                self.clear_activations()
                return a

    def cost(self, hx, y):
        m = len(hx)
        L = len(self.layers)
        s = []
        for l in self.layers:
            s.append(l.size)
        K = s[-1]

        cost = 0.0

        for i in range(m):
            for k in range(K):
                cost += ((y[i][k] * math.log(hx[i])) + ((1 - y[i][k]) * math.log(1 - hx[i]))) / -m

        thetasq = 0.0

        '''
        for l in range(L - 1):
            sl = self.layers[l].size
            for i in range(sl):
                for j in range(sl + 1):
                    thetasq += self.layers[l].nodes[sl] ** 2
        '''

        return cost

    def backpropogation():
        return 'todo'

    def gradient_approximation(epsilon=.01):
        print('!!! Gradiant Approximation Time: epsilon = ' + str(epsilon))

    def clear_activations(self):
        for l in self.layers:
            l.clear_activations()

    def sigmoid(input):
        return 1 / (1 + math.e**-input)


class Node():

    def __init__(self):
        self.weights = []
        self.forward_connections = []
        self.activation = 1.0

    def set_weights(self, weights):
        self.weights = weights

    def create_connections(self, layer):

        for i in range(layer.bias, len(layer.nodes)):
            self.forward_connections.append(layer.nodes[i])


class Layer():
    count = 0

    def __init__(self, size, network, bias=1):
        Layer.count += 1
        self.output = not bool(bias)
        self.network = network
        self.size = size
        self.nodes = []
        self.bias = bias
        for i in range(size + bias):
            self.create_node()

    def create_connections(self, layer):
        for n in self.nodes:
            if(self.nodes.index(n) - self.bias >= 0):
                n.create_connections(layer)

    def create_node(self):
        print('new node')
        n = Node()
        # set bias activation to 1
        # n.activation = 1 if len(self.nodes) == 0 & self.bias == 1 else 0
        self.nodes.append(n)
        return n

    def clear_activations(self):
        for i in range(self.bias, len(self.nodes)):
            self.nodes[i].activation = 0.0

    def activations(self):
        activations = []
        for n in self.nodes:
            activations.append(n.activation)
        return activations
