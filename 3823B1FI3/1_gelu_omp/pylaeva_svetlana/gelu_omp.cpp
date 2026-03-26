#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

const float PI = 3.14159265358979323846f;
const float GELU_2_SQRT_2_DIV_PI = 2.0f * std::sqrt(2.0f / PI);
const float GELU_APPROX_COEFF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> output(size);

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        output[i] = x / (1.0f + std::exp(-(GELU_2_SQRT_2_DIV_PI * (x + GELU_APPROX_COEFF * x3))));
    }
    return output;
}