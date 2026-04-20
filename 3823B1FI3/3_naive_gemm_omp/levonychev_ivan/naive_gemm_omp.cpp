#include "naive_gemm_omp.h"
#include <vector>




std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    std::vector<float> C(n * n);
    
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    return C;
}