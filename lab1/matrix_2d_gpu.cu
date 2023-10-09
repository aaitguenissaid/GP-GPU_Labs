// Matrix addition, GPU version

#include <iostream>

__global__ void add_matrix(float **a, float **b, float **c) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    c[idx_x][idx_y] = a[idx_x][idx_y] + b[idx_x][idx_y];
}

int main() {
    const int N = 16;
    const int rows = N;
    const int cols = N;
    const int blocksize = 4;

    float **a_h = new float[rows][cols];
    float **b_h = new float[rows][cols];
    float **c_h = new float[rows][cols];
    float **a_d;
    float **b_d;
    float **c_d;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a_h[i][j] = 10 + i;
            b_h[i][j] = float(j) / N;
        }
    }

    const int rows_size = N * sizeof(float);
    const int cols_size = N * sizeof(float);
    cudaMalloc((void **) &a_d, rows_size);
    cudaMalloc((void **) &b_d, rows_size);
    cudaMalloc((void **) &c_d, rows_size);

    for (int i = 0; i < N; i++) {
        cudaMalloc((void **) &a_d[i], cols_size);
        cudaMalloc((void **) &b_d[i], cols_size);
        cudaMalloc((void **) &c_d[i], cols_size);
    }

    cudaMemcpy2D(a_d[0], cols_size, a_h[0], cols_size, cols_size, rows, cudaMemcpyHostToDevice);
    cudaMemcpy2D(b_d[0], cols_size, b_h[0], cols_size, cols_size, rows, cudaMemcpyHostToDevice);

    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(cols / dimBlock.x, cols / dimBlock.x);

    add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

    cudaMemcpy2D(c_d[0], cols_size, c_h[0], cols_size, cols_size, rows, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c_h[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
