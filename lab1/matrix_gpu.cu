// Matrix addition, GPU version

#include <iostream>

__global__ void add_matrix(float *a, float *b, float *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    const int N = 16;
    const int blocksize = 16;

    float *a = new float[N * N];
    float *b = new float[N * N];
    float *c;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i + j * N] = 10 + i;
            b[i + j * N] = float(j) / N;
        }

    const int size = N * N * sizeof(float);
    cudaMalloc((void **) &c_d, size);

    dim3 dimBlock(blocksize, 1);
    dim3 dimGrid(N / dimBlock.x, 1);

    add_matrix<<<dimGrid, dimBlock>>>(a, b, c);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}
