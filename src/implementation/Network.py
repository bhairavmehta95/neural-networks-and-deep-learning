import random

import numpy as np

# sizes = number of neurons in respective layers
# i.e 2 neurons in 1st layer, 3 in second, 1 in last:
# net = Network([2, 3, 1])

# initialize biases, weights randomly w mean 0, SD 1
# Assumes 1st layer = input, omits biases to any neurons there
# weights[1] = Numpy matrix connecting second, third layers of neurons
# weights[1][j][k] = weight bw kth neuron in 2nd layer and jth in 3rd 
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)  for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Return output of network if a is input
        # a' = sigmoid(w . a + b)
        # applies to each layer
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a 

    def stoichastic_grad_descent(self, training_data, epochs,
        mini_batch_size, eta, test_data=None):

        ''' Train NN with mini-batch SGD. Training data = tuples "(x,y)" 
        representing the training inputs + desired outputs. '''

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in xrange(0, n, mini_batch_size)
                ]

            for mb in mini_batches:
                self.update_mini_batch(mb, data)

            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))

            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # delta_nabla_b, delta_nabla_w = self.backprop(x, y) figures out
        # partial deriv. dCx / dbj^l and dCx, dwjk^l
        # eta = learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        # Return (nabla_b, nabla_w) representing cost function C_x's gradient
        # They are both layer by layer lists of numpy arrays (same as self.b and self.w)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


# Numpy applies sigmoid fn element wise when z = vector
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)
)


