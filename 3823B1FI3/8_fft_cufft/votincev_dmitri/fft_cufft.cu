#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void NormalizeKernel(cufftComplex* data, int total_elements, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int total_complex_elements = input.size() / 2;
    int n = total_complex_elements / batch;
    size_t bytes = input.size() * sizeof(float);

    cufftComplex *d_data;
    cudaStream_t stream;
    cufftHandle plan;

    cudaStreamCreate(&stream);
    cudaMalloc(&d_data, bytes);
    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, stream);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (total_complex_elements + threads - 1) / threads;
    NormalizeKernel<<<blocks, threads, 0, stream>>>(d_data, total_complex_elements, 1.0f / n);

    std::vector<float> output(input.size());
    cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cufftDestroy(plan);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    return output;
}