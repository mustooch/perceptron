import random
import math

import matplotlib.pyplot as plt
import numpy as np

def heaviside_nz(x):
    if x < 0: return 0
    else: return 1

def heaviside_z(x):
    if x <= 0: return 0
    else: return 1

def relu(x):
    if x < 0: return 0
    else: return x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Perceptron:
    def __init__(self, W, bias, activation):
        self.W = W  # weight vector without the bias
        self.n = len(self.W)    # length of the vector
        self.bias = bias
        self.activation = activation    # the activation function
        self.learning_rate = 0.1 
        self.epochs = 50    # number of times the whole training is done

    def wheighted_sum(self, X):
        return sum(X[i] * self.W[i] for i in range(self.n)) + self.bias
    
    def output(self, X):
        # returns the binary output given the input vector X
        z = self.wheighted_sum(X)
        return self.activation(z)

    def learn(self, points):
        # simple learning algorithm taken from https://en.wikipedia.org/wiki/Perceptron#Learning_algorithm_for_a_single-layer_perceptron

        data_length = len(points)

        # repeat across multiple epochs
        for epoch in range(self.epochs):

            # cycle through the entire data set
            for i in range(data_length):
                point = points[i]
                y_hat = self.output(point.X)
                error = point.y - y_hat

                # update each weight and the bias
                for j in range(self.n):
                    self.W[j] = self.W[j] + self.learning_rate * error * point.X[j]

                self.bias = self.bias + self.learning_rate * error

    def get_equation(self):
        # returns the equation defined by the weights and bias of the perceptron
        # the equation is given by w0*x + w1*y + bias = 0
        # putting y on the other side we get y = -(w0*x + bias) / w1
        y = lambda x: -(self.W[0] * x + self.bias) / self.W[1]
        return y
    
    def test_points(self, points):
        for p in points:
            print(f"{p.X} -> {p.y}; got {self.output(p.X)}")

    def __str__(self):
        return f"Weights: {' '.join(str(w) for w in self.W)}\nbias: {self.bias}\nactivation function: {self.activation.__name__}"

# simple Point class, the extra value is used to classify the point
class Point:
    def __init__(self, X, y):
        self.X = X
        self.y = y

# Below are helper functions for plotting data

def plot_points(points):
    # scatter plot of all the points (red = 0, blue = 1)
    plt.scatter([p.X[0] for p in points], [p.X[1] for p in points],
               c = ["blue" if p.y == 1 else "red" for p in points])

def plot_perceptron_classifier(perceptron, x_range):
    # plot the line given by the perceptron's wheights and bias
    eq = perceptron.get_equation()
    x = np.array(x_range)
    y = eq(x)
    plt.plot(x, y)

def classify_point(X, perceptron):
    value = perceptron.output(X)
    return Point(X, value)

def generate_random_points(n, x_range, y_range, perceptron):
    new_points = []
    for i in range(n):
        X = [
            random.randint(x_range.start, x_range.stop),
            random.randint(y_range.start, y_range.stop)
        ]
        point = classify_point(X, perceptron)
        new_points.append(point)
    
    return new_points

if __name__ == "__main__":
    points = [
        Point([1, 2], 0),
        Point([2, 1], 0),
        Point([2, 2], 0),
        Point([4, 5], 1),
        Point([5, 4], 1),
        Point([3, 6], 1),
        Point([5, 2], 1),
        Point([1, 6], 1),
    ]

    p1 = Perceptron([0, 0], 0, heaviside_nz)
    p1.test_points(points)

    p1.learn(points)

    p1.test_points(points)
    print(p1)


    """
    plot_points(points)
    plot_perceptron_classifier(p1, range(-10, 10))

    plt.figure()

    new_points = generate_random_points(100, range(-10, 10), range(-10, 10), p1)
    plot_points(new_points)
    plot_perceptron_classifier(p1, range(-10, 10))

    plt.show()

    """

