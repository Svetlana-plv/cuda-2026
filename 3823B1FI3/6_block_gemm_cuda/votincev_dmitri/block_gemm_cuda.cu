#include "block_gemm_cuda.h"
#include <cuda_runtime.h>



#define BLOCK_SIZE 32

__global__ void BlockGemmCUDA_kernel(const float* a, const float* b, float* res, int n) {
    // shared memory под тайлы (блоки) матриц A и B
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum_res = 0.0f;

    // по "тайлам" вдоль строки A и столбца B
    for (int tile_idx = 0; tile_idx < (n / BLOCK_SIZE); ++tile_idx) {

       
        a_shared[ty][tx] = a[row * n + (tile_idx * BLOCK_SIZE + tx)];
        b_shared[ty][tx] = b[(tile_idx * BLOCK_SIZE + ty) * n + col];

        // ждем, пока весь блок заполнит shared memory
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum_res += a_shared[ty][k] * b_shared[k][tx];
        }

        
        __syncthreads();
    }

   
    if (row < n && col < n) {
        res[row * n + col] = sum_res;
    }
}



std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> answer(n * n, 0.0f);



    float* a_gpu;
    cudaMalloc((void**)&a_gpu, n * n * sizeof(float));
    cudaMemcpy(a_gpu, &a[0], n * n * sizeof(float), cudaMemcpyHostToDevice);

    float* b_gpu;
    cudaMalloc((void**)&b_gpu, n * n * sizeof(float));
    cudaMemcpy(b_gpu, &b[0], n * n * sizeof(float), cudaMemcpyHostToDevice);

    float* answer_gpu;
    cudaMalloc((void**)&answer_gpu, n * n * sizeof(float));



    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    dim3 grid(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    BlockGemmCUDA_kernel << < grid, block >> > (a_gpu, b_gpu, answer_gpu, n);



    cudaDeviceSynchronize();
    cudaMemcpy(&answer[0], answer_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(answer_gpu);

    return answer;
}