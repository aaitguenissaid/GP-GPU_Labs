import ctypes
from MLP_cuda import MLP, shuffle_and_split_data, forward_layer, sigmoid

import sklearn.datasets
import numpy as np

# INIT data
np.random.seed(1)
X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.20)  # We create a dataset with 300 elements

X_train, y_train, X_test, y_test = shuffle_and_split_data(X, y, 0.8)

# Load the shared library
mylibrary = ctypes.CDLL('./library.so')

# Define the function signature
mylibrary.sigmoid_of_matrix.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
mylibrary.sigmoid_of_matrix.restype = ctypes.POINTER(ctypes.c_float)  # The function doesn't return anything directly

def cuda_sigmoid(matrix):
    # Flatten the matrix to a 1D array
    flattened_matrix = matrix.flatten().astype(np.float32)

    # Get a pointer to the flattened data
    matrix_ptr = flattened_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the CUDA function
    result_matrix_ptr = mylibrary.sigmoid_of_matrix(matrix_ptr, len(matrix), len(matrix[0]))

    # Convert the result pointer to a NumPy array
    result_matrix = np.fromiter(result_matrix_ptr, dtype=np.float32, count=len(matrix) * len(matrix[0]))

    # Reshape the flattened result back to a matrix
    result_matrix = np.reshape(result_matrix, matrix.shape)

    return result_matrix


d_input=2 
d_hidden=3 
d_output=2

np.random.seed(0)    

W1 = np.random.rand(d_input, d_hidden).astype(np.float32)-0.5
b1 = np.random.rand(d_hidden).astype(np.float32)-0.5

#z1 = forward_layer(X_train, W1, b1) # Output of the first layer


a1 = sigmoid(W1) # Sigmoid activation of the first layer
a1_cuda = cuda_sigmoid(W1) # Sigmoid activation of the first layer

print("equal ? \n", a1==a1_cuda)
print("sum equal ? \n", np.sum(a1), " =?", np.sum(a1_cuda))

print("W1 :\n", W1)

print("\na1 :\n", a1)
print("\n a1_cuda :\n", a1_cuda)

