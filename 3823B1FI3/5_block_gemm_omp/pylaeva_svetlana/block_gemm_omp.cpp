#include <algorithm>
#include <omp.h>

#include "block_gemm_omp.h"

const int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for
    for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {
            for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
                
                int end_i = std::min(i_block + BLOCK_SIZE, n);
                int end_j = std::min(j_block + BLOCK_SIZE, n);
                int end_k = std::min(k_block + BLOCK_SIZE, n);

                for (int i = i_block; i < end_i; i++) {
                    for (int k = k_block; k < end_k; k++) {
                        float temp_a = a[i * n + k];
                        #pragma omp simd
                        for (int j = j_block; j < end_j; j+=4) {
                            c[i * n + j]     += temp_a * b[k * n + j];
                            c[i * n + j + 1] += temp_a * b[k * n + j + 1];
                            c[i * n + j + 2] += temp_a * b[k * n + j + 2];
                            c[i * n + j + 3] += temp_a * b[k * n + j + 3];
                        }
                    }
                }
            }
        }
    }

    return c;
}