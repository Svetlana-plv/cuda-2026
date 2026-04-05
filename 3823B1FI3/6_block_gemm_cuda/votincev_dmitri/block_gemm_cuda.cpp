#include "Block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>




#define BLOCK_SIZE 16

#define N 4096

__global__ void BlockGemmCUDA_kernel(
    float* a,
    float* b,
    float* res,
    const int n) {

    // шарная память в рамках блока, блок размера BLOCK_SIZE^2
    // значит столько и шарная, чтобы каждый поток взял свою "ячейку"
    __shared__ float a_shared[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE*BLOCK_SIZE];

    // каждый поток отвечает за свою ячейку в res: (row,col)
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;


    /*if (row >= n || col >= n) {
        return;
    }*/


    // шарная память -  линейный массив
    // надо "спроецировать" чтобы был блок (матрица)
    // для этого индекс делаю 2-мерным
    int shared_idx = threadIdx.y*BLOCK_SIZE+threadIdx.x;
    
    // в это же время - a - это тоже не матрица, а линейный массив
    // поэтому проецируем тоже его как бы в 2-мерный
    int a_glob_idx = row * n + threadIdx.x;
    // b - то же самое но по ней будем ходить по столбцам
    int b_glob_idx = threadIdx.y * n + col;

    int tile_count = n / BLOCK_SIZE;

    float sum_res = 0.0f;
    
    for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) {

        // тайл (окно) движется "слева-направо"
        int offset = tile_idx * BLOCK_SIZE;

        a_shared[shared_idx] = a[a_glob_idx+offset];
        b_shared[shared_idx] = b[b_glob_idx + offset*n];

        __syncthreads();

        // теперь нужно умножить часть строки матрицы A на часть столбца B
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum_res += a_shared[threadIdx.y*BLOCK_SIZE+k] * b_shared[BLOCK_SIZE*k+threadIdx.x];
        }

        __syncthreads();
    }

    // теперь все потоки содержат в sum_res свою ячейку 
    // результата умножения матрицы A на B
    // надо только собрать в res
    
    // res - тоже линейный массив, его проекцируем на матрицу

    res[row * n + col] = sum_res;
}






// это функция CPU
std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> answer(n * n, 0.0f);

    

    float* a_gpu;
    cudaMalloc((void**)&a_gpu, n * n * sizeof(float));
    cudaMemcpy(a_gpu, &a[0], n * n * sizeof(float), cudaMemcpyHostToDevice);

    float* b_gpu;
    cudaMalloc((void**)&b_gpu, n * n * sizeof(float));
    cudaMemcpy(b_gpu, &b[0], n * n * sizeof(float), cudaMemcpyHostToDevice);

    float* answer_gpu;
    cudaMalloc((void**)&answer_gpu, n*n * sizeof(float));

    // каждый поток будет умножать одну строку матрицы a на один столбец матрицы b
    // я уже выделил память на a_gpu и b_gpu
    // буду просто передавать сами матрицы и в качестве результата - указатель float* на памяти GPU
    // это будет один вычисленный элемент, который потом просто добавляю к результирующей




    // это 1D сетка и 1D блоки
    //const int block_size = SH_BL_SZ; // максимум 1024!!!
    // int num_blocks = (n*n + block_size - 1) / block_size; // кол-во блоков для задачи

    // в данной задаче - матрицы, 2D
    // поэтому лучше и сетку сделать 2D и сами блоки 2D
    // тогда пространство потоков будет как бы ложиться на матрицу
    // и каждый поток будет лежать на элементе, который считает
   
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // округление вверх для некратных размеров
    dim3 grid(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // каждый поток вычисляет ровно 1 элемент матрицы
    // размер матрицы nxn , ровно столько надо вычислить
    // ровно столько потоков и создастся (num_blocks*block_size == n*n)
    BlockGemmCUDA_kernel << < grid, block >> > (a_gpu, b_gpu, answer_gpu, n);
    // поэтому надо просто сделать само умножение внутри _kernel


    cudaDeviceSynchronize();
    cudaMemcpy(&answer[0], answer_gpu, n*n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(answer_gpu);

    return answer;
}




