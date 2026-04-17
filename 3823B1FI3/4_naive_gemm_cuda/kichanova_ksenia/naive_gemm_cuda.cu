#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < n; k += TILE_SIZE) {
        if (row < n && (k + threadIdx.x) < n) {
            a_tile[threadIdx.y][threadIdx.x] = a[row * n + (k + threadIdx.x)];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && (k + threadIdx.y) < n) {
            b_tile[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            sum += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static size_t allocated_size = 0;

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = n * n * sizeof(float);
    
    if (allocated_size < bytes) {
        if (d_a) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated_size = bytes;
    }
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    return c;
}