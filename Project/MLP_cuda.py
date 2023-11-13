#!pip install numpy
#!pip install sklearn

# Imports of useful packages
import matplotlib # For the plots
import matplotlib.pyplot as plt 
import numpy as np # To perform operations on matrices efficiently


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(Z):
    exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y


class MLP(object):
    def __init__(self, activation_function_l1, forward_layer, learning_rate, lambd, num_epochs, d_input, d_hidden, d_output):
        self.lr = learning_rate
        self.num_epochs = num_epochs
        np.random.seed(0)    
        self.activation_function_l1 = activation_function_l1
        self.forward_layer = forward_layer
        self.W1 = np.random.rand(d_input, d_hidden)-0.5 
        self.b1 = np.random.rand(d_hidden)-0.5 
        self.W2 = np.random.rand(d_hidden, d_output)-0.5 
        self.b2 = np.random.rand(d_output)-0.5 
        self.best_model_iteration = 0
        self.lambd = lambd # Weight decay
        
    def update_params(self, W1, b1, W2, b2, best_model_iteration):
        self.W1 = W1
        self.b1 = b1 
        self.W2 = W2
        self.b2 = b2
        self.best_model_iteration = best_model_iteration

        # redifine all the following three functions in GP-GPU code
    def forward_function(self, X):
        z1 = forward_layer(X, self.W1, self.b1) # Output of the first layer
        a1 = self.activation_function_l1(z1) # Sigmoid activation of the first layer
        z2 = a1.dot(self.W2) + self.b2.T # Output of the second layer
        probs = softmax(z2) #Apply softmax activation function on z2
        return z1, a1, z2, probs
    
    def backpropagation(self, z1, a1, z2, probs, X, y):
        delta2 = probs - one_hot(y)
        dW2 = a1.T.dot(delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = sigmoid_prime(z1) * delta2.dot(self.W2.T)
        dW1 = X.T.dot(delta1)
        db1 = np.sum(delta1, axis=0)
        return dW1, db1, dW2, db2

    def gradient_descent(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.lr*self.lambd*self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr*self.lambd*self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr*self.lambd*self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr*self.lambd*self.b2 - self.lr * db2
    
    def predict(self, X):
        _, _, _, probs = self.forward_function(X)
        return np.argmax(probs, axis=1)
    
    def get_predictions(self, probs):
        return np.argmax(probs, axis=1)
    
    def accuracy(self, y, y_pred):
        return np.average(y == y_pred)                        
    
    def estimate_loss(self, y, probs):
        correct_logprobs = -np.log(probs[np.arange(len(probs)), y]) # Calculation of cross entropy for each example
        return np.average(correct_logprobs) # Total loss
          
    def fit(self, X, y, print_loss=False, return_best_model=True):
        z1, a1, z2, probs = self.forward_function(X) # this will help us not compute the accuracy two times in the loop.
        # best model
        bm = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 
                      'accuracy': self.accuracy(y, self.get_predictions(probs)), 'iteration': -1}
        
        for i in range(0, self.num_epochs):
            dW1, db1, dW2, db2 = self.backpropagation(z1, a1, z2, probs, X, y)
            self.gradient_descent(dW1, db1, dW2, db2) # update parameters
            z1, a1, z2, probs = self.forward_function(X)
    
            current_accuracy = self.accuracy(y, self.get_predictions(probs)) # predictions of epoch

            if print_loss and i % 100 == 0:
                print("Loss at epoch %i: %f" %(i, self.estimate_loss(y, probs)), "Accuracy :", current_accuracy)
                
            if(return_best_model and current_accuracy > bm['accuracy']):
                bm = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 
                              'accuracy': self.accuracy(y, self.get_predictions(probs)), 'iteration': i}
        
        if(print_loss and return_best_model):
            self.update_params(bm['W1'], bm['b1'], bm['W2'], bm['b2'], bm['iteration'])
            print("The best accuracy obtained is :", self.accuracy(y, self.predict(X)), ", at this iteration :", self.best_model_iteration)
        elif(print_loss):
            print("The model accuracy obtained is :", self.accuracy(y, self.predict(X)))
            
                   
def shuffle_and_split_data(X, y, train_percentage):
    N = len(X)
    train_size = int(N * train_percentage) # 80% of samples for training
    test_size = N - train_size # 20% remaining for test
    p = np.random.permutation(N) # shuffle X and y with the same random permutation. 
    X = X[p]
    y = y[p]

    # split data
    X_train = X[0:train_size]
    y_train = y[0:train_size]
    
    X_test = X[train_size:N]
    y_test = y[train_size:N]
    return X_train, y_train, X_test, y_test