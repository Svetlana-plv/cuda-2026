#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// ядро для вычисления умножения блоков матриц A и B на каждом ядре GPU
__global__ void BlockGemmKernel(const float *A, const float *B, float *C, int n) {
    // выделение разделяемой памяти для блоков матриц A и B
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE];

    int threadX = threadIdx.x; // локальный индекс потока в блоке по оси X
    int threadY = threadIdx.y; // локальный индекс потока в блоке по оси Y
    int row = blockIdx.y * BLOCK_SIZE + threadY; // глобальный индекс потока по оси Y
    int col = blockIdx.x * BLOCK_SIZE + threadX; // глобальный индекс потока по оси X

    float sum = 0.0f; // сумма произведений для элемента C[row][col]

    for (int m = 0; m < (n / BLOCK_SIZE); ++m) {
        // каждый поток загружает элемент блока A и B в разделяемую память
        blockA[threadY][threadX] = A[row * n + (m * BLOCK_SIZE + threadX)]; 
        blockB[threadY][threadX] = B[(m * BLOCK_SIZE + threadY) * n + col];

        // барьерная синхронизация на блок потоков 
        __syncthreads(); // ожидание загрузки блоков A и B в разделяемую память

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[threadY][k] * blockB[k][threadX];
        } 

        // барьерная синхронизация на блок потоков 
        __syncthreads(); // ожидание завершения вычислений без следующей загрузкой блоков A и B
    }

    // запись результата в глобальную память
    if (row < n && col < n) {
        C[row *  n + col] = sum;
    }
}

// основная функция (выполняется на CPU)
std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = (size_t)n * n;

    // статическая указатели, чтобы не делать аллокацию и деаллокацию памяти на каждом вызове 
    static float *d_a = nullptr;
    static float *d_b = nullptr;
    static float *d_c = nullptr;
    static int allocated_size = 0;

    // поток для асинхронности
    static cudaStream_t stream = nullptr;

    // выделение памяти на GPU
    if (allocated_size < n) {
        if (d_a)
            cudaFree(d_a); 
        if (d_b)
            cudaFree(d_b);
        if (d_c)
            cudaFree(d_c);
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));
        if (!stream) {
            cudaStreamCreate(&stream); // создание потока для асинхронных операций
        }
        allocated_size = n;
    }

    // асинхронное копирование входных матриц с CPU на GPU в потоке stream
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // настройки сетки
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // BLOCK_SIZE x BLOCK_SIZE потоков на блок
    dim3 blocks(n / BLOCK_SIZE, n / BLOCK_SIZE); // количество блоков для покрытия всей матрицы

    BlockGemmKernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n); // запуск ядра на GPU с конфигурацией запуска асинхронно в потоке stream

    std::vector<float> c(size); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, stream); // асинхронное копирование результата с GPU на CPU в потоке stream

    cudaStreamSynchronize(stream); // // синхронизация всех операций в потоке stream

    return c;
}