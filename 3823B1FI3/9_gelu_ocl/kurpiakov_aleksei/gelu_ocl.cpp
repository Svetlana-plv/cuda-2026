#define CL_TARGET_OPENCL_VERSION 300
#include <vector>
#include <CL/cl.h>
#include "gelu_ocl.h"

const char* kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float argument = 1.595769f * (x + 0.044715f * x * x * x);
        output[i] = x - x / (exp(argument) + 1.0f);
    }
}
)";

static cl_device_id g_device;
static cl_context g_context;
static cl_command_queue g_queue;
static cl_program g_program;
static cl_kernel g_kernel;
static bool g_is_init = false;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (!g_is_init) {
        cl_platform_id pid = platforms[platform];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);

        g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, nullptr);

        cl_queue_properties props[] = {0};
        g_queue = clCreateCommandQueueWithProperties(g_context, g_device, props, nullptr);

        g_program = clCreateProgramWithSource(g_context, 1, &kernel_source, nullptr, nullptr);
        clBuildProgram(g_program, 1, &g_device, nullptr, nullptr, nullptr);
        g_kernel = clCreateKernel(g_program, "gelu_kernel", nullptr);

        g_is_init = true;
    }

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    cl_mem gpu_input = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      bytes, const_cast<float*>(input.data()), nullptr);
    cl_mem gpu_output = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    int n_for_kernel = static_cast<int>(n);
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &gpu_input);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &gpu_output);
    clSetKernelArg(g_kernel, 2, sizeof(int), &n_for_kernel);

    size_t work_size = n;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &work_size, nullptr, 0, nullptr, nullptr);

    std::vector<float> result(n);
    clEnqueueReadBuffer(g_queue, gpu_output, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr);

    clReleaseMemObject(gpu_input);
    clReleaseMemObject(gpu_output);

    return result;
}