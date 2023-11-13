from MLP_cuda import MLP, shuffle_and_split_data, sigmoid

import sklearn.datasets
import numpy as np

# INIT data
np.random.seed(1)
X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.20)  # We create a dataset with 300 elements

X_train, y_train, X_test, y_test = shuffle_and_split_data(X, y, 0.8)

activation_function_l1=sigmoid
learning_rate = 3e-2
lambd=0.0
num_epochs=200
d_input=2 
d_hidden=10 
d_output=2

def forward_layer(X, W, b):
    return X.dot(W) + b.T #[add_bias(col, b) for col in matrix_multiplication(X, W)]

model = MLP(activation_function_l1, forward_layer, learning_rate, lambd, num_epochs, d_input, d_hidden, d_output)
model.fit(X_train, y_train, print_loss=True, return_best_model=False)
print("The test accuracy obtained is :", model.accuracy(y_test, model.predict(X_test)))

