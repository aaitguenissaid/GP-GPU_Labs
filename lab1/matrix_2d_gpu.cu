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
    const int rows = 1024;
    const int cols = 1024;

    std::cout <<"rows : "<< rows <<", cols : "<< cols << std::endl;    
    std::cout <<"block size\tdimGrid\t\tElapsed time" << std::endl;

    for(int k=2; k<=1024; k=k*2) {

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

        const int size = rows * cols * sizeof(float);
        cudaMalloc((void **) &a_d, size);
        cudaMalloc((void **) &b_d, size);
        cudaMalloc((void **) &c_d, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

        const int blocksize = k;
        dim3 dimBlock(blocksize, blocksize);
        dim3 dimGrid((rows-1)/dimBlock.x + 1, ceil(float(cols)/dimBlock.y));

        cudaEventRecord(start, 0);
        add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, rows, cols);
        cudaEventRecord(stop, 0);
        
        cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        std::cout << blocksize <<"\t\t"<< dimGrid.x <<"\t\t";
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cout << cudaGetErrorString(err)<<std::endl;
        else {
            std::cout << elapsedTime << std::endl;
        }

        // Free device memory
        cudaFree(a_d);
        cudaFree(a_b);
        cudaFree(a_c);

        // Free host memory
        free(a_h);
        free(b_h);
        free(c_h);
    }
}
