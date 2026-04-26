#include "gelu_ocl.h"
#include <CL/cl.h>

const char* source = 
"__kernel void gelu_kernel(__global const float* in, __global float* out, int n) {"
"    int i = get_global_id(0);"
"    if (i < n) {"
"        float x = in[i];"
"        float x3 = x * x * x;"
"        float arg = 0.7978845608f * (x + 0.044715f * x3);"
"        out[i] = x / (1.0f + exp(-2.0f * arg));"
"    }"
"}";

struct OCLResources {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    bool initialized = false;

    void init(int platform_idx) {
        if (initialized) return;

        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id platform = platforms[platform_idx];

        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

        kernel = clCreateKernel(program, "gelu_kernel", nullptr);
        initialized = true;
    }
};

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    static OCLResources ocl;
    ocl.init(platform);

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    cl_mem d_in = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), nullptr);
    cl_mem d_out = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(ocl.kernel, 2, sizeof(int), &n);

    size_t local_size = 256;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    std::vector<float> output(n);

    clEnqueueReadBuffer(ocl.queue, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    return output;
}