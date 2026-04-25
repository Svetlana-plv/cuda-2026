#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")


#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

static constexpr float C1 = 1.59576912f; 
static constexpr float C2 = 0.044715f;

#pragma GCC target("avx", "avx2", "fma")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);


#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < (int)n; ++i) {
        float x = input[i];
        float x2 = x * x;
        float x3 = x * x2;
        float arg = C1 * (x + C2 * x3);

        output[i] = x / (1.0f + std::exp(-arg)); 
    }

    return output;
}
