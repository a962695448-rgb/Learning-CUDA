#pragma once
#include "baseline/kernels.cuh"

// Experiment only: same mathematical tile and k-loop as the byte-exact
// 9f5fdc3 snapshot. Four independent warps reuse one 16*N input tile.
namespace wmma_reuse {
template<class T, int N>
__global__ void four_warp_kernel(const T* input, const T* matrix, T* output,
                                std::size_t rows, float scale) {
    static_assert(N >= 16 && N <= 256 && (N & (N - 1)) == 0);
    __shared__ __align__(32) T a[16 * N];
    __shared__ __align__(32) float c[4][16 * 16];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t row_start = std::size_t(blockIdx.x) * 16;
    const int column_start = blockIdx.y * 64 + warp * 16;
    for (int i = threadIdx.x; i < 16 * N; i += blockDim.x) {
        const auto row = row_start + i / N;
        a[i] = row < rows ? input[row * N + i % N] : hadamard::as_storage<T>(0);
    }
    __syncthreads();
    // Whole-warp condition. Inactive N16/N32 warps still reach both barriers.
    if (column_start < N) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T,
                               nvcuda::wmma::row_major> af;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T,
                               nvcuda::wmma::row_major> bf;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
        nvcuda::wmma::fill_fragment(cf, 0.0f);
        #pragma unroll
        for (int k = 0; k < N; k += 16) {
            nvcuda::wmma::load_matrix_sync(af, a + k, N);
            nvcuda::wmma::load_matrix_sync(bf, matrix + k * N + column_start, N);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        nvcuda::wmma::store_matrix_sync(c[warp], cf, 16, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    if (column_start < N) {
        for (int i = lane; i < 16 * 16; i += 32) {
            const auto row = row_start + i / 16;
            if (row < rows)
                output[row * N + column_start + i % 16] =
                    hadamard::as_storage<T>(c[warp][i] * scale);
        }
    }
}
}  // namespace wmma_reuse
