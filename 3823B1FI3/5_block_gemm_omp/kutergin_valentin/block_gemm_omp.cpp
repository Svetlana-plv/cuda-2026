#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2,fma")


#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>


std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    // оптимальный размер блока для современных процессоров (позволяет уместиться в кэш L1/L2)
    const int BLOCK_SIZE = 64;

#pragma omp parallel for schedule(static) // распараллеливание внешнего цикла по блокам строк матрицы C
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {

                // внутреннее умножение внутри блоков (используется i-k-j порядок)
                for (int i = i0; i < std::min(i0 + BLOCK_SIZE, n); ++i) {
                    for (int k = k0; k < std::min(k0 + BLOCK_SIZE, n); ++k) {
                        float a_ik = a[i * n + k]; // кэширование элемента a[i][k]

#pragma omp simd // применение векторизации при последовательном обходе
                        for (int j = j0; j < std::min(j0 + BLOCK_SIZE, n); ++j) {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}