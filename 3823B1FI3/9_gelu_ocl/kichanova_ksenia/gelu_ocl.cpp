#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cstdio>

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float p = 0.7978845608028654f * (x + 0.044715f * x3);
    
    float exp_2p = exp(2.0f * p);
    float tanh_approx = (exp_2p - 1.0f) / (exp_2p + 1.0f);
    
    output[idx] = 0.5f * x * (1.0f + tanh_approx);
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_kernel kernel = nullptr;
static cl_mem d_input = nullptr;
static cl_mem d_output = nullptr;
static size_t allocated = 0;
static cl_device_id device = nullptr;
static bool initialized = false;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    cl_int err;
    
    if (!initialized) {
        cl_platform_id platforms[32];
        cl_uint num_platforms;
        err = clGetPlatformIDs(32, platforms, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            return std::vector<float>();
        }
        
        if (platform < 0 || platform >= (int)num_platforms) {
            platform = 0;
        }
        
        cl_platform_id platform_id = platforms[platform];
        
        cl_device_id devices[16];
        cl_uint num_devices;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 16, devices, &num_devices);
        
        if (num_devices == 0) {
            err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, devices, &num_devices);
        }
        
        if (err != CL_SUCCESS || num_devices == 0) {
            return std::vector<float>();
        }
        
        device = devices[0];
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || context == nullptr) {
            return std::vector<float>();
        }
        
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS || queue == nullptr) {
            return std::vector<float>();
        }
        
        cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
        if (err != CL_SUCCESS || program == nullptr) {
            return std::vector<float>();
        }
        
        err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char build_log[65536];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
            fprintf(stderr, "OpenCL build error:\n%s\n", build_log);
            clReleaseProgram(program);
            return std::vector<float>();
        }
        
        kernel = clCreateKernel(program, "gelu_kernel", &err);
        if (err != CL_SUCCESS || kernel == nullptr) {
            clReleaseProgram(program);
            return std::vector<float>();
        }
        
        clReleaseProgram(program);
        initialized = true;
    }
    
    if (allocated < bytes) {
        if (d_input != nullptr) clReleaseMemObject(d_input);
        if (d_output != nullptr) clReleaseMemObject(d_output);
        
        d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || d_input == nullptr) {
            return std::vector<float>();
        }
        
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || d_output == nullptr) {
            return std::vector<float>();
        }
        
        allocated = bytes;
    }
    
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return std::vector<float>();
    }
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    if (err != CL_SUCCESS) return std::vector<float>();
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    if (err != CL_SUCCESS) return std::vector<float>();
    
    err = clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) return std::vector<float>();
    
    size_t global_size = n;
    size_t local_size = 256;
    
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return std::vector<float>();
    }
    
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return std::vector<float>();
    }
    
    std::vector<float> output(n);
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return std::vector<float>();
    }
    
    return output;
}