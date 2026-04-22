#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static size_t allocated = 0;
static cublasHandle_t handle = nullptr;
static bool initialized = false;

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = n * n * sizeof(float);
    
    if (!initialized) {
        cublasCreate(&handle);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated = bytes;
        initialized = true;
    }
    
    if (allocated < bytes) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated = bytes;
    }
    
    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_a, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_b, n);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n);
    
    std::vector<float> c(n * n);
    cublasGetMatrix(n, n, sizeof(float), d_c, n, c.data(), n);
    
    return c;
}
