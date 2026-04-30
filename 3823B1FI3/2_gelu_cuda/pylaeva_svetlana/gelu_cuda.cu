#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>

static const float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static const float COEFF = 0.044715f;

__global__ void geluKernel(const float* d_input, float* d_output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = d_input[idx];
        float x3 = x * x * x;
        float z = SQRT_2_OVER_PI * (x + COEFF * x3);
        
        // Заменяем tanh на exp: tanh(z) = (1 - exp(-2z)) / (1 + exp(-2z))
        // Итоговая формула: GELU(x) = x / (1 + exp(-2z))
        float exp_neg_2z = __expf(-2.0f * z);
        float result = x / (1.0f + exp_neg_2z);
        
        d_output[idx] = result;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    int N = input.size();
    
    std::vector<float> output(N);
    
    // Статические переменные для переиспользования памяти
    static float* d_input = nullptr;
    static float* d_output = nullptr;
    static int prev_N = 0;
    
    if (prev_N != N)
    {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        prev_N = N;
    }
    
    cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    geluKernel<<<blocks, threads>>>(d_input, d_output, N);
    
    cudaMemcpy(output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    return output;
}