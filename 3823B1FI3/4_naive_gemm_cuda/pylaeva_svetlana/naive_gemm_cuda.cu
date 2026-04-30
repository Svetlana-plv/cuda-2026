#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void naiveGemmKernel(const float* d_a, const float* d_b, float* d_c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += d_a[row * n + k] * d_b[k * n + col];
        }
        d_c[row * n + col] = sum;
    }
}

// Глобальные переменные для переиспользования
static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static int last_n = 0;

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    int size = n * n;
    
    // Переиспользуем память если размер не изменился
    if (last_n != n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));
        last_n = n;
    }
    
    // Создаем stream для асинхронных операций
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Асинхронное копирование
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Выбор размера блока для максимальной загрузки GPU
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Запуск ядра
    naiveGemmKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);
    
    // Асинхронное копирование результата
    std::vector<float> c(size);
    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Синхронизация
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return c;
}