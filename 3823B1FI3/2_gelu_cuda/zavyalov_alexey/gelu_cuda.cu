#include "gelu_cuda.h"
#include <iostream>
#pragma GCC optimize("O3")
#define sqrt_2_div_pi 0.7978845608028653558f
#define two_mult_sqrt_2_div_pi 1.5957691216057307116f

__global__ void gelu_kernel(const float* input, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = input[i];
    if (i < n)
        result[i] = x / (1.0f + (__expf(-(two_mult_sqrt_2_div_pi * (x + 0.044715f * x * x * x)))));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    const int block_size = 256;
    int num_blocks_first = (n / 2 + block_size - 1) / block_size;
    int num_blocks_second = ((n + 1) / 2 + block_size - 1) / block_size;

    static float *a_gpu_first_half = nullptr, *result_gpu_first_half = nullptr;
    static float *a_gpu_second_half = nullptr, *result_gpu_second_half = nullptr;
    static int device_capacity = 0;
    if (n > device_capacity) {
        if (device_capacity > 0) {
            cudaFree(a_gpu_first_half);
            cudaFree(result_gpu_first_half);
            cudaFree(a_gpu_second_half);
            cudaFree(result_gpu_second_half);
        }
        cudaMalloc(&a_gpu_first_half,     n / 2 * sizeof(float));
        cudaMalloc(&result_gpu_first_half, n / 2 * sizeof(float));
        cudaMalloc(&a_gpu_second_half,     (n+1) / 2 * sizeof(float));
        cudaMalloc(&result_gpu_second_half,(n+1) / 2 * sizeof(float));
        device_capacity = n;
    }

    cudaMemcpy(a_gpu_first_half, input.data(), n / 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t strm;
    cudaStreamCreate(&strm);
    cudaMemcpyAsync(a_gpu_second_half, input.data() + n / 2, (n + 1) / 2 * sizeof(float), cudaMemcpyHostToDevice, strm); // надо сделать асинк

    gelu_kernel<<<num_blocks_first, block_size>>>(a_gpu_first_half, result_gpu_first_half, n / 2);
    static float* res = nullptr;
    static int pinned_capacity = 0;
    static int call_number = 0;
    if (n > pinned_capacity) {
        if (res) cudaFreeHost(res);
        cudaMallocHost(&res, n * sizeof(float));
        pinned_capacity = n;
    }
    call_number++;

    cudaMemcpy(res, result_gpu_first_half, n / 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(strm);

    gelu_kernel<<<num_blocks_second, block_size>>>(a_gpu_second_half, result_gpu_second_half, (n + 1) / 2);

    cudaMemcpy(res + n / 2, result_gpu_second_half, (n + 1) / 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_gpu_first_half);
    cudaFree(a_gpu_second_half);
    cudaFree(result_gpu_second_half);
    cudaFree(result_gpu_first_half);

    std::vector<float> res_vec(res, res + n);
    if (call_number == 5) {
        cudaFreeHost(res);
        cudaFree(a_gpu_first_half);
        cudaFree(result_gpu_first_half);
        cudaFree(a_gpu_second_half);
        cudaFree(result_gpu_second_half);
    }
    return res_vec;
}
