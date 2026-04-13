#pragma GCC optimize("O3,fast-math")
#include "naive_gemm_omp.h"


std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n){
    
    std::vector<float> res(a.size(), 0.0f);

#pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){

            #pragma omp simd
            for (int k = 0; k < n; ++k){
                res[i * n + k] += a[i * n + j] * b[j * n + k];
            }
        
        }
    }

    return res;
}