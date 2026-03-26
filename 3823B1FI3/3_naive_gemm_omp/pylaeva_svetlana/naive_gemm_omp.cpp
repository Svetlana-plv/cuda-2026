#include <vector>
#include <omp.h>

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float temp_a = a[i * n + k];
            
            // Развертка на 4 итерации
#pragma omp simd
            for (int j = 0; j < n; j += 4) {
                c[i * n + j]     += temp_a * b[k * n + j];
                c[i * n + j + 1] += temp_a * b[k * n + j + 1];
                c[i * n + j + 2] += temp_a * b[k * n + j + 2];
                c[i * n + j + 3] += temp_a * b[k * n + j + 3];
            }
        }
    }
    return c;
}