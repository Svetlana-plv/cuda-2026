#pragma GCC optimize("Ofast")
#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>



std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> C(n * n, 0.0f);
    // i -> k -> j (for simd)
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_i_k = a[i * n + k];
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a_i_k * b[k * n + j];
            }
        }
    }
    return C;
}