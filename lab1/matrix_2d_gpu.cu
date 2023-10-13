// Matrix addition, GPU version

#include <iostream>

__global__ void add_matrix(float *a, float *b, float *c, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    
    if(i < rows && j < cols) {
        int index = i * cols + j;
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int rows = 8;
    const int cols = 5;
    const int blocksize = 16;

    float *a_h = new float[rows * cols];
    float *b_h = new float[rows * cols];
    float *c_h = new float[rows * cols];
    float *a_d;
    float *b_d;
    float *c_d;

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            a_h[i * cols + j] = i;
            b_h[i * cols + j] = j;
        }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << a_h[i * cols + j ] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << b_h[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;

    const int size = rows * cols * sizeof(float);
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid((rows-1)/dimBlock.x + 1, ceil(float(cols)/dimBlock.y));

    add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, rows, cols);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << c_h[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
