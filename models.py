import numpy as np
from matplotlib import pyplot as plt

class NN:

    def __init__(self, h1, seed):
        np.random.seed(seed)
        self.h1 = h1   # number of hidden units in the hidden layer

    def init_params(self, X, Y):
        W1 = np.random.randn(self.h1, X.shape[0]) * 0.01
        b1 = np.zeros((self.h1, 1))
        W2 = np.random.randn(Y.shape[0], self.h1) * 0.01
        b2 = np.zeros((Y.shape[0],1))

        params = {
            'W1':W1,
            'b1':b1,
            'W2':W2,
            'b2':b2
        }
        return params

    def forward_pass(self, params, X):
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        Z1 = np.dot(W1, X) + b1  # Forward Propagation
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = 1/(1+(np.exp(-Z2)))

        cache = {
            'Z1':Z1,
            'A1':A1,
            'Z2':Z2,
            'A2':A2
        }
        return cache
    
    def compute_cost(self, cache, Y):

        A2 = cache['A2']
        m = Y.shape[1]

        cost = (-1/m) * np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) # Computing Cost

        return cost

    def back_prop(self, cache, params, X, Y):

        W2 = params['W2']

        m = Y.shape[1]

        A1 = cache['A1']
        A2 = cache['A2']

        # Updating the gradient

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.transpose())
        db2 = (1/m) * (np.sum(dZ2, axis=1, keepdims=True))
        dZ1 = np.dot(W2.transpose(),dZ2) * (1-(A1**2))
        dW1 = (1/m) * np.dot(dZ1, X.transpose())
        db1 = (1/m) * (np.sum(dZ1, axis=1, keepdims=True))

        grads = {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2}
        return grads

    def optimize(self, X, Y, grads, params, alpha):

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 = W1 - alpha*dW1 # Gradient Descent
        b1 = b1 - alpha*db1
        W2 = W2 - alpha*dW2
        b2 = b2 - alpha*db2

        params = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        return params

    def fit(self, X, Y, alpha=0.01, epochs=10000, plot_cost=False, verbose=True):

        params = self.init_params(X, Y)
        cost = []
        iterations = []
        c = 0

        for i in range (epochs):

            cache = self.forward_pass(params, X)

            if plot_cost and ((i+1)%1000 == 0):
                cost.append(self.compute_cost(cache, Y))
                iterations.append(i+1)
            
            if verbose and ((i+1)%1000 == 0):
                c = self.compute_cost(cache, Y)
                print("Cost after {} iterations is: {}".format((i+1), c))
            
            grads = self.back_prop(cache, params, X, Y)
            params = self.optimize(X, Y, grads, params, alpha)

        if plot_cost:
            plt.plot(iterations, cost)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title("Cost vs. Iterations")
            plt.show()
            return params, cost, iterations
        else:
            return params
    
    def predict(self, X, params):

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = 1/(1+(np.exp(-Z2)))

        A2[A2 > .5] = 1
        A2[A2 < .5] = 0
        A2 = A2.astype(int)

        return A2

    def accuracy(self, A2, Y):
        c=0
        for i in range(Y.shape[1]):
            if A2[0,i] == Y[0,i]:
                c+=1
        return (c/Y.shape[1])*100