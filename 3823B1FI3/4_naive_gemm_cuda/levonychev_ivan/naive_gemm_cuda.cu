#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>



__global__ void kernel(float* A, float* B, float* C, int n) {


    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }

}




std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {

    int bytes = n * n * sizeof(float);                              
    float* gpu_A;
    float* gpu_B;
    float* gpu_C;

    cudaMalloc(&gpu_A, bytes);
    cudaMalloc(&gpu_B, bytes);
    cudaMalloc(&gpu_C, bytes);

    cudaMemcpy(gpu_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b.data(), bytes, cudaMemcpyHostToDevice);


    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    


    kernel<<<grid_size, block_size>>>(gpu_A, gpu_B, gpu_C, n);


    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), gpu_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    return C;
}