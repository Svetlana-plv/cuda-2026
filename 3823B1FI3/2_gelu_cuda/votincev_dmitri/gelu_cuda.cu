#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>







// это функция GPU
__global__ void Gelu_kernel(
    const float* input,
    float* result, size_t* n) {

    // получаю индекс потока в данном блоке
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= *n) {
        return;
    }

    const float c1 = 0.79788456f;
    const float k = 0.044715f;

    float x = input[i];
    float x3 = x * x * x;
    float tanh_arg = c1 * (x + k * x3);
    float fast_tanh_res = 1.0f - 2.0f / (expf(2.0f * tanh_arg) + 1.0f);
    result[i] = 0.5f * x * (1.0f + fast_tanh_res);


}




// это функция CPU
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();

    std::vector<float> answer(n);



    // мы считаем для n = 2^27
    // выделение памяти - долго
    // считать на GPU - быстро
    // можно не выделять разом всю память, а делать это частями

    int part_size = 2*1024*1024; 
    int num_part = (n+part_size-1) / part_size;


    

    size_t* n_gpu;
    cudaMalloc((void**)&n_gpu, 1 * sizeof(size_t));

    
    float* buffer_gpu;
    cudaMalloc((void**)&buffer_gpu, part_size * sizeof(float));

    for (int i = 0; i < num_part; i++) {
        
        size_t curr_part_size = part_size;
        if (curr_part_size > (n-i*part_size)) {
            curr_part_size = n-i*part_size;
        }
        

        cudaMemcpy(n_gpu, &curr_part_size, sizeof(size_t), cudaMemcpyHostToDevice);

        cudaMemcpy(buffer_gpu, &input[0] + i*part_size, curr_part_size * sizeof(float), cudaMemcpyHostToDevice);

        const int block_size = 1024; // размер блока потоков (сколько потоков?)
        int num_blocks = (curr_part_size + block_size - 1) / block_size; // кол-во блоков для задачи

        //               параметры кол-ва блока 
        //                 и размера блока
        //                       |                      параметры функции
        Gelu_kernel << < num_blocks, block_size >> > (buffer_gpu, buffer_gpu, n_gpu); // n_gpu исключительно для проверки выхода за границу

        cudaDeviceSynchronize();
        cudaMemcpy(&answer[i*part_size], buffer_gpu, curr_part_size * sizeof(float), cudaMemcpyDeviceToHost);

        
    }
    

    cudaFree(buffer_gpu);
    cudaFree(n_gpu);
    
    


    return answer;
}





