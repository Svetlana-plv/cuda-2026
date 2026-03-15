#include "gelu_cuda.h"

#include <cuda_runtime.h>

#pragma GCC optimize("O3")

__constant__ float PI_coeff = 0.797884F;
__constant__ float coeff = 0.044715F;

const int block_size = 256;
const int count_of_streams = 16;

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
    const int size = input.size();
    const int bytes = size * sizeof(float);
    const int stream_size = size / count_of_streams;
    const int stream_bytes = stream_size * sizeof(float);

    std::vector<float> output(size);

    float *device_in, *device_out;

    cudaMalloc((void**)&device_in, bytes);
    cudaMalloc((void**)&device_out, bytes);

    cudaStream_t streams[count_of_streams];
    for(int i = 0;i < count_of_streams;i++)
        cudaStreamCreate(&streams[i]);
    
    const int num_blocks = (stream_size + block_size - 1) / block_size;
    for(int i = 0;i < count_of_streams;i++)
    {
        const int offset = i * stream_size;
        cudaMemcpyAsync(device_in + offset, input.data() + offset, stream_bytes, cudaMemcpyHostToDevice, streams[i]);
        gelu_kernel <<<num_blocks, block_size, 0, streams[i]>>>(device_in + offset, device_out + offset, stream_size);
        cudaMemcpyAsync(output.data() + offset, device_out + offset, stream_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    cudaDeviceSynchronize();

    for(int i = 0;i < count_of_streams;i++)
        cudaStreamDestroy(streams[i]);

    cudaFree(device_in);
    cudaFree(device_out);

    return output;
}