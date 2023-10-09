// Assigns every element in an array with its index.
// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <iostream>
#include <cmath>

__global__ void simple(float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = pow(threadIdx.x, 0.5);
}

int main() {
    // Define problem size
    const int N = 1024;

    // Define number of blocks
    const int blocksize = 16;

    // Create host and device data strutures
    float *c_h = new float[N];
    float *c_d;

    // Give size of array to allocate on GPU
    const int size = N * sizeof(float);

    //	Allocate array on GP GPU
    cudaMalloc((void **) &c_d, size);

    // Define workspace topology
    dim3 dimBlock(blocksize, 1);
    dim3 dimGrid(N / dimBlock.x, 1);

    // Execute kernel
    simple<<<dimGrid, dimBlock>>>(c_d);

    // Wait for kernel completion
    cudaDeviceSynchronize();

    // Copy result of computation back on host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << c_h[i] << " ";

    std::cout << std::endl;

    // Free memory
    cudaFree(c_d);
    delete[] c_h;

    std::cout << "done" << std::endl;

    return EXIT_SUCCESS;
}
