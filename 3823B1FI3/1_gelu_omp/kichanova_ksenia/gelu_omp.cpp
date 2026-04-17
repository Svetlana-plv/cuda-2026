#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

#pragma GCC target("avx2")

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    
    #pragma omp parallel
    {
        const int thread_count = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int chunk = (n + thread_count - 1) / thread_count;
        const size_t start = tid * chunk;
        const size_t end = std::min(start + chunk, n);
        
        #pragma omp simd
        for (size_t i = start; i < end; ++i) {
            const float x = input[i];
            
            const float x2 = x * x;
            const float x3 = x2 * x;
            const float p = SQRT_2_OVER_PI * (x + COEFF * x3);
            
            const float exp_2p = expf(2.0f * p);
            const float tanh_approx = 1.0f - 2.0f / (exp_2p + 1.0f);
            
            result[i] = 0.5f * x * (1.0f + tanh_approx);
        }
    }
    
    return result;
}