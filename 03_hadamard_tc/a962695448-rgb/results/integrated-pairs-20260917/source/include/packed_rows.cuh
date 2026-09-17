#pragma once

#include "kernels.cuh"

namespace hadamard {

// Keep all callers consistent with the device row mapping.
template <int N, bool Transform>
inline constexpr int packed_row_lanes = Transform && N >= 2 ? N / 2 : N;

template <class T, int N, bool Transform, bool Quantize>
__global__ void packed_rows_kernel(const T* input, T* output, std::uint8_t* packed,
                                  float* scales, std::size_t rows, float transform_scale) {
    static_assert(N >= 1 && N <= 16 && (N & (N - 1)) == 0);
    if constexpr (Transform && N >= 2) {
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
    } else {
        const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t row = element / N;
        const int lane = threadIdx.x % N;
        const bool valid = row < rows;
        // Every lane reaches all shuffles, including complete masked subgroups at the tail.
        float value = valid ? as_float(input[element]) : 0.0f;
        if constexpr (Transform) {
#pragma unroll
            for (int stride = 1; stride < N; stride *= 2) {
                const float peer = __shfl_xor_sync(0xffffffff, value, stride, N);
                value = (lane & stride) ? peer - value : value + peer;
            }
            value = as_float(as_storage<T>(value * transform_scale));
        }
        if constexpr (!Quantize) {
            if (valid) output[element] = as_storage<T>(value);
        } else {
            float magnitude = fabsf(value);
#pragma unroll
            for (int stride = N / 2; stride > 0; stride /= 2)
                magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride, N));
            const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
            if (valid && lane == 0) scales[row] = scale;
            const int q = max(-7, min(7, __float2int_rn(value / scale)));
            int next = 0;
            if constexpr (N > 1) next = __shfl_down_sync(0xffffffff, q, 1, N);
            if (valid && !(lane & 1)) {
                const int high = N > 1 ? (next & 15) : 0;
                packed[row * ((N + 1) / 2) + lane / 2] = static_cast<std::uint8_t>((q & 15) | (high << 4));
            }
        }
    }
}

}  // namespace hadamard
