#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

__global__ void profiling_capability_probe(float* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = values[index] * 2.0f + 1.0f;
}

int main() {
    constexpr int count = 4096;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("GPU=%s SM=%d%d SMS=%d MEMORY=%zu\n", properties.name,
                properties.major, properties.minor,
                properties.multiProcessorCount, properties.totalGlobalMem);
    std::vector<float> host(count, 1.0f);
    float* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    profiling_capability_probe<<<16, 256>>>(device, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    for (const float value : host) {
        if (value != 3.0f) return 3;
    }
    std::puts("PROBE_CORRECTNESS_PASS");
    return 0;
}
