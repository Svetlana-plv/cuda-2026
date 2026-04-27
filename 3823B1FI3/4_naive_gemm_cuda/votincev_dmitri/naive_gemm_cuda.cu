#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void NaiveGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < n / TILE_SIZE; ++t) {
        tileA[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        tileB[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> res(n * n);
    size_t bytes = n * n * sizeof(float);

    float* d_a, * d_b, * d_res;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_res, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(n / TILE_SIZE, n / TILE_SIZE);

    NaiveGemmKernel << <grid, block >> > (d_a, d_b, d_res, n);

    cudaMemcpy(res.data(), d_res, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return res;
}
