//
// Created by agsa on 13/11/23.
//

#include <iostream>

/*** Sigmoid function ***/
__global__ void sigmoid_kernel(float *input, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if(i < rows && j < cols) {
        int index = i * cols + j;
        input[index] = 1/(1 + expf(-input[index]));
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

/*** Matrix multiplication ***/
__global__ void matrix_mul_kernel(float *C, float *A, float *B, int wA, int hB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Accumulate row i of A and column j of B
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    float accu = 0.0;

    for (int k = 0; k < wA; k++) {
        accu = accu + A[i * wA + k] * B[k * hB + j];
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[i * hB + j] = accu;
}

extern "C"
float * matrix_multiplication(float *A, int wA, int hA,  float *B, nt wB, int hB) {
    const int blocksize = 256;
    if(hA == wB) {
        int size = wA * hB;
        float* C;
        cudaMalloc((void **) &C, size);
        dim3 dimBlock(blocksize, blocksize);
        dim3 dimGrid((wA-1)/dimBlock.x + 1, ceil(float(hB)/dimBlock.y));
        matrix_mul_kernel<<<dimGrid, dimBlock>>>(A, B, C, wA, hB);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        return C;
    }
}



/*** Forward layer ***/
__global__ void forward_layer_kernel(float *C, float *A, float *B, int wA, int hB, b) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Accumulate row i of A and column j of B
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    float accu = 0.0;

    for (int k = 0; k < wA; k++) {
        accu = accu + A[i * wA + k] * B[k * hB + j] + b[j];
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[i * hB + j] = accu;
}

extern "C"
float * forward_layer(float *A, int wA, int hA,  float *B, nt wB, int hB, float *b, int hb) {
    const int blocksize = 256;
    if(hA == wB && hB == hb) {
        int size = wA * hB;
        float *C;
        cudaMalloc((void **) &C, size);
        dim3 dimBlock(blocksize, blocksize);
        dim3 dimGrid((wA-1)/dimBlock.x + 1, ceil(float(hB)/dimBlock.y));
        forward_layer_kernel<<<dimGrid, dimBlock>>>(A, B, C, wA, hB, b);
        cudaDeviceSynchronize();  // Wait for the kernel to finish
        return C;
    }
}
