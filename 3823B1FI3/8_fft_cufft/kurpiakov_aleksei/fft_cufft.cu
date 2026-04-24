#include "fft_cufft.h"

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void NormalizeKernel(cufftComplex* data, int total_complex, float inv_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_complex) {
        data[idx].x *= inv_n;
        data[idx].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int total_floats = static_cast<int>(input.size());
    const int total_complex = total_floats >> 1;
    const int n = total_complex / batch;
    const size_t bytes = input.size() * sizeof(float);

    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    const float inv_n = 1.0f / static_cast<float>(n);
    const int threads = 256;
    const int blocks = (total_complex + threads - 1) / threads;
    NormalizeKernel<<<blocks, threads>>>(d_data, total_complex, inv_n);

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return result;
}