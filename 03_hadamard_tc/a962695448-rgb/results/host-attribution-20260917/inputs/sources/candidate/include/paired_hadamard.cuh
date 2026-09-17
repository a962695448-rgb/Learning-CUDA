#pragma once

#include "kernels.cuh"

namespace hadamard {

// Own the first butterfly pair in registers, then exchange pairs between lanes.
// Scalar loads/stores keep legal two-byte storage offsets in the public contract.
template <class T, int N, bool Quantize>
__global__ void paired_hadamard_kernel(const T* input, T* output,
                                       std::uint8_t* packed, float* scales,
                                       std::size_t rows, float transform_scale) {
    static_assert(N >= 2 && N <= 16 && (N & (N - 1)) == 0);
    constexpr int Lanes = N / 2;
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = pair / Lanes;
    const int lane = threadIdx.x % Lanes;
    const bool valid = row < rows;
    const float first = valid ? as_float(input[pair * 2]) : 0.0f;
    const float second = valid ? as_float(input[pair * 2 + 1]) : 0.0f;
    float low = first + second;
    float high = first - second;
#pragma unroll
    for (int stride = 1; stride < Lanes; stride *= 2) {
        const float peer_low = __shfl_xor_sync(0xffffffff, low, stride, Lanes);
        const float peer_high = __shfl_xor_sync(0xffffffff, high, stride, Lanes);
        low = (lane & stride) ? peer_low - low : low + peer_low;
        high = (lane & stride) ? peer_high - high : high + peer_high;
    }
    low = as_float(as_storage<T>(low * transform_scale));
    high = as_float(as_storage<T>(high * transform_scale));
    if constexpr (!Quantize) {
        if (valid) {
            output[pair * 2] = as_storage<T>(low);
            output[pair * 2 + 1] = as_storage<T>(high);
        }
    } else {
        float magnitude = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
        for (int stride = Lanes / 2; stride > 0; stride /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride, Lanes));
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        const int q_low = max(-7, min(7, __float2int_rn(low / scale)));
        const int q_high = max(-7, min(7, __float2int_rn(high / scale)));
        if (valid) {
            packed[pair] = static_cast<std::uint8_t>((q_low & 15) | ((q_high & 15) << 4));
            if (lane == 0) scales[row] = scale;
        }
    }
}

}  // namespace hadamard
