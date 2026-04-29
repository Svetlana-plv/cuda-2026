#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstring>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_blocks = n / BLOCK_SIZE;
    
    for (int block_k = 0; block_k < num_blocks; ++block_k) {
        // Загрузка блока A
        shared_a[threadIdx.y][threadIdx.x] = a[row * n + block_k * BLOCK_SIZE + threadIdx.x];
        
        // Загрузка блока B
        shared_b[threadIdx.y][threadIdx.x] = b[(block_k * BLOCK_SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        
        // Умножение блоков с разверткой цикла
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    c[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    
    static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    static size_t prev_bytes = 0;
    static cudaStream_t stream = nullptr;
    
    // Создаем stream один раз
    if (!stream) {
        cudaStreamCreate(&stream);
    }
    
    // Переиспользование памяти
    if (bytes != prev_bytes) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        
        prev_bytes = bytes;
    }
    
    // Асинхронное копирование в stream
    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(n / BLOCK_SIZE, n / BLOCK_SIZE);
    
    // Запуск ядра в том же stream
    blockGemmKernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(size);
    // Асинхронное копирование результата
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    
    // Синхронизация stream
    cudaStreamSynchronize(stream);
    
    return c;
}