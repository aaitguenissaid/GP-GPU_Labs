import ctypes
from MLP_cuda import MLP, shuffle_and_split_data

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
mylibrary.sigmoid_of_matrix.restype = None  # The function doesn't return anything directly

def cuda_sigmoid(matrix):
    # Flatten the matrix to a 1D array
    flattened_matrix = matrix.flatten().astype(np.float32)

    # Get a pointer to the flattened data
    matrix_ptr = flattened_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the CUDA function
    mylibrary.sigmoid_of_matrix(matrix_ptr, len(matrix), len(matrix[0]))

    # Reshape the flattened result back to a matrix
    result_matrix = np.reshape(flattened_matrix, matrix.shape)

    return result_matrix

mylibrary.forward_layer.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.c_int,                     # wA
    ctypes.c_int,                     # hA
    ctypes.POINTER(ctypes.c_float),  # W
    ctypes.c_int,                     # wB
    ctypes.c_int,                     # hB
    ctypes.POINTER(ctypes.c_float),  # b
    ctypes.c_int                      # hb
]
mylibrary.forward_layer.restype = ctypes.POINTER(ctypes.c_float)  # The function doesn't return anything directly

def forward_layer(X, W, b):
    # float *A, int wA, int hA,  float *B, nt wB, int hB, float *b, int hb):

    # Flatten the matrix to a 1D array
    flattened_X = X.flatten().astype(np.float32)
    flattened_W = W.flatten().astype(np.float32)
    flattened_b = b.flatten().astype(np.float32)

    # Get a pointer to the flattened data
    X_ptr = flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    W_ptr = flattened_W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = flattened_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the CUDA function
    res_ptr = mylibrary.sigmoid_of_matrix(X_ptr, len(X), len(X[0]), W_ptr, len(W), len(W[0]), b_ptr, len(b))

    # Convert the result pointer to a NumPy array
    result_matrix = np.fromiter(res_ptr, dtype=np.float32, count=len(X) * len(W[0]))

    # Reshape the flattened result back to a matrix
    result_matrix = result_matrix.reshape(len(X), len(W[0]))

    return result_matrix #[add_bias(col, b) for col in matrix_multiplication(X, W)]

activation_function_l1=forward_layer
learning_rate = 3e-2
lambd=0.0
num_epochs=200
d_input=2 
d_hidden=10 
d_output=2

model = MLP(activation_function_l1, learning_rate, lambd, num_epochs, d_input, d_hidden, d_output)
model.fit(X_train, y_train, print_loss=True, return_best_model=False)
print("The test accuracy obtained is :", model.accuracy(y_test, model.predict(X_test)))

