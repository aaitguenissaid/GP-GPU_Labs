__global__ void gemm_naive(float *C, float *A, float *B, int wA, int hB) {
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

