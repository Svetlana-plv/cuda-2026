#include "gelu_cuda.h"

#include <cuda_runtime.h>

__constant__ float PI_coeff = 0.797884F;
__constant__ float coeff = 0.044715F;

const int block_size = 256;

//__restrict__ - указание nvcc о том, что данные доступны только через этот указатель в данной области видимости. Тем самым мы уменьшаем количество проверок с его стороны
__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, const int n)
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x; //получаем позицию текущего потока
    if(id >= n) return; //если вдруг потоков будет больше, чем входных данных
    
    float x = input[id];
    float exp_value = __expf(2.0F * PI_coeff * (x + coeff * x * x * x));
    float tanh_value = 1.0F - 2.0F / (exp_value + 1.0F);
    output[id] = 0.5F * x * (1.0F + tanh_value);
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    static int run = 0;
    run++;

    const int size = input.size();
    const int num_blocks = (size + block_size - 1) / block_size;

    static float *input_gpu = nullptr;
    static float *output_gpu = nullptr;

    const int bytes = size * sizeof(float);

    if(run == 1)
    {
        cudaMalloc(&input_gpu, bytes);
        cudaMalloc(&output_gpu, bytes);
    }

    cudaMemcpy(input_gpu, input.data(), bytes, cudaMemcpyHostToDevice);

    gelu_kernel <<<num_blocks, block_size>>> (input_gpu, output_gpu, size);

    static std::vector<float> output(size);
    cudaMemcpy(output.data(), output_gpu, bytes, cudaMemcpyDeviceToHost);

    if(run == 5)
    {
        cudaFree(input_gpu);
        cudaFree(output_gpu);
    }

    return output;
}