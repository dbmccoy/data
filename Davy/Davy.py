
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

'''
class Table(object):

    cols = []
    rows = []

    def __init__(self, data, to_float=True):

        [self.rows.append(r) for r in data]
        [self.cols.append
        ([row[i] for row in self.rows]) for i in range(len(self.rows[0]))]

        if to_float:
            self.cols = [[float(i) for i in c] for c in self.cols]

    def shape(self):
        return (len(self.cols),len(self.rows))

    def pop(self, i):
        self.cols.pop(i)
        [r.pop(i) for r in self.rows]
        print(self.shape())
'''


class LinearRegression(object):

    def gradient_descent(self, rate):
        errors = []

        for i in range(100):
            preds = pd.Series(np.dot(self.features, self.weights))

            error = (2/len(preds) * sum(preds - self.labels)**2)
            d_error = sum(preds - self.labels)/len(preds)
            errors.append(error)
            new_weights = []
            for w in range(len(self.weights)):
                # print((self.weights[w] - (rate * d_error)/len(preds)))
                new_weights.append
                (self.weights[w] - (rate * d_error) / len(preds) / len(preds))
                # w * rate * (pred - label)

            self.weights = new_weights
            # print(self.weights)

        print(self.weights)
        plt.plot(errors)
        plt.show()

    def fit(self, data, w=0):
        self.labels = data.iloc[:, -1]
        self.features = data.iloc[:, :-1]
        self.features['bias'] = 1
        cols = self.features.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.features = self.features[cols]
        self.m = len(self.labels)

        # self.weights = [w for i in range(len(self.features.columns))]

        self.weights = self.normal()
        print(self.normal())

    def normal(self):
        x = self.features
        xT = self.features.transpose()
        y = self.labels

        theta = np.linalg.inv(xT.dot(x))
        theta = theta.dot(xT)
        theta = theta.dot(y)

        return theta

    def predict(self):
        preds = pd.Series(np.dot(self.features, self.weights))
        print(self.rmse(preds, self.labels))
        plt.scatter(self.features.iloc[:, -1], preds)
        plt.scatter(self.features.iloc[:, -1], self.labels)
        plt.show()

    def rmse(self, preds, labels):
        error = 0.0
        for i in range(len(preds)):
            error += (preds[i] - labels[i])**2
        error = error / len(labels)
        return error

    def j(self, h, y):
        return .5 * (h - y)**2

    def d_j(self, h, y):
        return (h - y)


class LogisticRegression(object):

    def gradient_descent(self, rate):
        errors = []

        for i in range(100):
            preds = pd.Series(np.dot(self.features, self.weights))

            error = (2/len(preds) * sum(preds - self.labels)**2)
            d_error = sum(preds - self.labels)/len(preds)
            errors.append(error)
            new_weights = []
            for w in range(len(self.weights)):
                # print((self.weights[w] - (rate * d_error)/len(preds)))
                new_weights.append
                (self.weights[w] - (rate * d_error) / len(preds) / len(preds))
                # w * rate * (pred - label)

            self.weights = new_weights
            # print(self.weights)

        print(self.weights)
        plt.plot(errors)
        plt.show()

    def fit(self, data, w=0):
        self.labels = data.iloc[:, -1]
        self.features = data.iloc[:, :-1]
        self.features['bias'] = 1
        cols = self.features.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        self.features = self.features[cols]
        self.m = len(self.labels)

        # self.weights = [w for i in range(len(self.features.columns))]

        self.weights = self.normal()
        print(self.normal())

    def normal(self):
        x = self.features
        xT = self.features.transpose()
        y = self.labels

        theta = np.linalg.inv(xT.dot(x))
        theta = theta.dot(xT)
        theta = theta.dot(y)

        return theta

    def predict(self):
        preds = pd.Series(np.dot(self.features, self.weights))
        for i in range(len(preds)):
            preds[i] = 1 / (1 + math.e ** -preds[i])

        print(self.rmse(preds, self.labels))
        print(pd.DataFrame({'pred': preds, 'label': self.labels}))
        plt.scatter(self.features.iloc[:, -1], preds)
        plt.scatter(self.features.iloc[:, -1], self.labels)
        plt.show()

    def rmse(self, preds, labels):
        error = 0.0
        for i in range(len(preds)):
            error += (labels[i] * math.log(preds[i]) + (1 - labels[i])
                      * math.log(1 - preds[i]))

        error = error / len(labels)
        return error

    def j(self, h, y):
        return .5 * (h - y)**2

    def d_j(self, h, y):
        return (h - y)
