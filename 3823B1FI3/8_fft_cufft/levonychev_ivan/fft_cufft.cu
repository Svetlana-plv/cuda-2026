#include <cufft.h>
#include <cuda_runtime.h>
#include "fft_cufft.h"

__global__ void kernel(float* vec, int size, float scale) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < size)
        vec[threadID] *= scale;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

    int input_size = input.size();
    int n = input_size / (2 * batch);
    float scale = 1.0f / static_cast<float>(n);
    int bytes = input_size * sizeof(float);

    float* gpu_input;
    cudaMalloc(&gpu_input, bytes);
    cudaMemcpy(gpu_input, input.data(), bytes, cudaMemcpyHostToDevice);

    // ---------------------
    cufftHandle plan;

    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftExecC2C(plan, (cufftComplex*)gpu_input, (cufftComplex*)gpu_input, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex*)gpu_input, (cufftComplex*)gpu_input, CUFFT_INVERSE);


    cufftDestroy(plan);
    // ---------------------
    int block_size = 256;
    int grid_size = (block_size + input_size - 1) / block_size;
    kernel<<<grid_size, block_size>>>(gpu_input, input_size, scale);

    std::vector<float> output(input_size);

    cudaMemcpy(output.data(), gpu_input, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    return output;
}