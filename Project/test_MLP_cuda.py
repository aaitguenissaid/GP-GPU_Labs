import ctypes
from MLP_cuda import MLP, shuffle_and_split_data

import sklearn.datasets
import numpy as np

# INIT data
np.random.seed(1)
X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.20)  # We create a dataset with 300 elements

X_train, y_train, X_test, y_test = shuffle_and_split_data(X, y, 0.8)


def cuda_sigmoid(x):
    # Load the shared library
    mylibrary = ctypes.CDLL('./sigmoid.so')

    # Define the function signature

    my_library.sigmoid_of_matrix.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    mylibrary.sigmoid_of_matrix.restype = ctypes.c_float

    # Get a pointer to the data
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the CUDA function with the pointer
    result = mylibrary.sigmoid_of_matrix(x_ptr, len(x), len(x[0]))

    return result


activation_function_l1=cuda_sigmoid
learning_rate = 3e-2
lambd=0.0
num_epochs=200
d_input=2 
d_hidden=10 
d_output=2

model = MLP(activation_function_l1, learning_rate, lambd, num_epochs, d_input, d_hidden, d_output)
model.fit(X_train, y_train, print_loss=True, return_best_model=False)
print("The test accuracy obtained is :", model.accuracy(y_test, model.predict(X_test)))

