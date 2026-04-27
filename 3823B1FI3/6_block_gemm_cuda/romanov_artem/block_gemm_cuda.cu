#include "block_gemm_cuda.h"

constexpr int BS = 16;

__global__ void BlockGemmKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BS + ty;
    int col = blockIdx.x * BS + tx;

    float sum = 0.0f;

    int num_tiles = n / BS;

    for (int t = 0; t < num_tiles; ++t) {
        As[ty][tx] = A[row * n + (t * BS + tx)];
        Bs[ty][tx] = B[(t * BS + ty) * n + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BS; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = size_t(n) * n * sizeof(float);

    float* dA = nullptr;
    cudaMalloc(&dA, bytes);

    float* dB = nullptr;
    cudaMalloc(&dB, bytes);

    float* dC = nullptr;
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BS, BS);
    dim3 grid(n / BS, n / BS);

    BlockGemmKernel << <grid, block >> > (dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> result(size_t(n) * n);
    cudaMemcpy(result.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return result;
}