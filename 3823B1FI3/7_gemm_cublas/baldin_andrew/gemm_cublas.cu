#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
                                
    size_t bytes = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                d_B, n,
                d_A, n,
                &beta, 
                d_C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
