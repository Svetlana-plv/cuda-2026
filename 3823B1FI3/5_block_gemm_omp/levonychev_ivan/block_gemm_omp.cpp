#pragma GCC optimize("O3,unroll-loops")
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include "block_gemm_omp.h"




std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> output(n * n, 0.0f);

    const int BLOCK_SIZE = 128;

#pragma omp parallel for schedule(static)
    for (int I = 0; I < n; I += BLOCK_SIZE) {
        for (int J = 0; J < n; J += BLOCK_SIZE) {

            for (int K = 0; K < n; K += BLOCK_SIZE) {
                
                for (int i = I; i < std::min(I + BLOCK_SIZE, n); ++i) {

                    for (int k = K; k < std::min(K + BLOCK_SIZE, n); ++k) {

                        float tmp = a[i * n + k];
#pragma omp simd
                        for (int j = J; j < std::min(J + BLOCK_SIZE, n); ++j) {
                            output[i * n + j] += tmp * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return output;
}