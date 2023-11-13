//
// Created by agsa on 13/11/23.
//

#include <iostream>

__global__ void sigmoid_of_matrix(float *a, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if(i < rows && j < cols) {
        int index = i * cols + j;
        a[index] = 1/(1 + np.exp(-a[index]))
    }
}
