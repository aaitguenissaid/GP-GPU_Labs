//
// Created by agsa on 13/11/23.
//

#include <iostream>

__global__ void sigmoid_kernel(float *input, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if(i < rows && j < cols) {
        int index = i * cols + j;
<<<<<<< HEAD
        input[index] = 1/(1 + np.exp(-input[index]))
=======
        a[index] = 1.0 /(1.0 + expf(-a[index]));
>>>>>>> 73dce5baab5ffad71c84736c72ac14a437c4804b
    }
}

extern "C"
void sigmoid_of_matrix(float *input, int rows, int cols) {
    const int blocksize = 256;
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid((rows-1)/dimBlock.x + 1, ceil(float(cols)/dimBlock.y));
    sigmoid_kernel<<<dimGrid, dimBlock>>>(input, rows, cols);
    cudaDeviceSynchronize();  // Wait for the kernel to finish
}