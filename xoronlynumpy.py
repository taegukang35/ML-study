import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Dense:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, dLdY, learning_rate):
        dLdW = np.dot(dLdY, self.input.T)
        dLdX = np.dot(self.weights.T, dLdY)
        self.weights -= learning_rate * dLdW
        self.bias -= learning_rate * dLdY
        return dLdX


class Activation:
    def __init__(self, activation, derivative):
        self.input = None
        self.activation = activation
        self.derivative = derivative

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, dLdX, learning_rate):
        return np.multiply(dLdX, self.derivative(self.input))


class Tanh(Activation):
    def __init__(self):
        def tanh(x): return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x): return np.tanh(x)

        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


# xor example
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Y = np.array([[0], [1], [1], [0]])
X = np.reshape(X, (4, 2, 1))
Y = np.reshape(Y, (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1

# train model
for i in range(epochs):
    for x, y in zip(X, Y):
        # forward pass
        output = x
        for layer in network:
            output = layer.forward(output)
        error = mse(y, output)

        # backward pass
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        error /= len(X)
        #print(f'{i}/{epochs}, error= {error}')


# (0,0)~(1,1) test

def xor(x,y):
    X = np.reshape([x,y],(2,1))
    output = X
    for layer in network:
        output = layer.forward(output)
    return output[0]

x = np.linspace(0,1,20)
y = np.linspace(0,1,20)
points = [(i,j,xor(i,j)) for i in x for j in y]
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
for point in points:
    ax.scatter3D(*point)
plt.show()
