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

    float **a_h = new float *[rows];
    float **b_h = new float *[rows];
    float **c_h = new float *[rows];

    for (int i = 0; i < cols; i++) {
        a_h[i] = new float[rows];
        b_h[i] = new float[rows];
        b_h[i] = new float[rows];
    }

    float **a_d;
    float **b_d;
    float **c_d;

    const int rows_size = rows * sizeof(float);
    const int cols_size = cols * sizeof(float);

    cudaMalloc((void **) &a_d, rows_size);
    cudaMalloc((void **) &b_d, rows_size);
    cudaMalloc((void **) &c_d, rows_size);

    for (int i = 0; i < rows; i++) {
        cudaMalloc((void **) &a_d[i], cols_size);
        cudaMalloc((void **) &b_d[i], cols_size);
        cudaMalloc((void **) &c_d[i], cols_size);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a_h[i][j] = 10 + i;
            b_h[i][j] = float(j) / N;
        }
    }

    size_t **a_dev_pitch;
    size_t **b_dev_pitch;
    size_t **c_dev_pitch;

    cudaMallocPitch(&a_d[0], &a_dev_pitch, cols_size * sizeof(float), rows);
    cudaMallocPitch(&b_d[0], &b_dev_pitch, cols_size * sizeof(float), rows);

    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(rows / dimBlock.x, cols / dimBlock.y);

    add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

    cudaMallocPitch(&c_h[0], &c_dev_pitch, cols_size * sizeof(float), rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << c_h[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
