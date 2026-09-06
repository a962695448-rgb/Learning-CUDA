#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "kernels.cuh"
#include "contiguous256.cuh"

#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace {

void validate(const at::Tensor& input, double scale, int block_threads) {
    TORCH_CHECK(block_threads == 128 || block_threads == 256, "block_threads must be 128 or 256");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "input dtype must be float16 or bfloat16");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 4, "input must have 2 or 4 dimensions");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(!input.is_neg(), "input must not have a lazy negative bit; call resolve_neg() first");
    TORCH_CHECK(input.numel() > 0, "input must be nonempty");
    const auto dim = input.size(-1);
    TORCH_CHECK(dim >= 1 && dim <= 256 && !(dim & (dim - 1)),
                "last dimension must be a power of two in [1,256]");
    TORCH_CHECK(!input.requires_grad(), "this forward-only extension does not support requires_grad=True");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                    std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale) > 0,
                "scale must be finite, positive, and representable as float32");
    const auto rows = input.numel() / dim;
    TORCH_CHECK((rows - 1) / (block_threads / 32) + 1 <= std::numeric_limits<int>::max(),
                "input has too many rows for the CUDA launch grid");
}

template <class T, int N, bool Transform, bool Quantize>
void launch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
            at::Tensor& scales, float scale, cudaStream_t stream, int block_threads,
            bool contiguous256_fused) {
    const auto rows = static_cast<std::size_t>(input.numel() / N);
    const auto blocks = static_cast<unsigned int>((rows - 1) / (block_threads / 32) + 1);
    const auto* source = reinterpret_cast<const T*>(input.data_ptr());
    auto* destination = output.defined() ? reinterpret_cast<T*>(output.data_ptr()) : nullptr;
    auto* bytes = packed.defined() ? packed.data_ptr<std::uint8_t>() : nullptr;
    auto* row_scales = scales.defined() ? scales.data_ptr<float>() : nullptr;
    if constexpr (N == 256 && Transform && Quantize) {
        if (contiguous256_fused) {
            hadamard::contiguous256_kernel<T, true, true><<<blocks, block_threads, 0, stream>>>(
                source, destination, bytes, row_scales, rows, scale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    hadamard::warp_kernel<T, N, Transform, Quantize><<<blocks, block_threads, 0, stream>>>(
        source, destination, bytes, row_scales, rows, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, bool Transform, bool Quantize>
void dispatch_dim(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
                  at::Tensor& scales, float scale, cudaStream_t stream, int block_threads,
                  bool contiguous256_fused) {
    switch (input.size(-1)) {
#define DIM_CASE(N) case N: launch<T, N, Transform, Quantize>(input, output, packed, scales, scale, stream, block_threads, contiguous256_fused); break
        DIM_CASE(1); DIM_CASE(2); DIM_CASE(4); DIM_CASE(8); DIM_CASE(16);
        DIM_CASE(32); DIM_CASE(64); DIM_CASE(128); DIM_CASE(256);
#undef DIM_CASE
        default: TORCH_CHECK(false, "unsupported dimension");
    }
}

template <bool Transform, bool Quantize>
void dispatch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
              at::Tensor& scales, double scale, int block_threads, bool contiguous256_fused = false) {
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    if (input.scalar_type() == at::kHalf)
        dispatch_dim<__half, Transform, Quantize>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused);
    else
        dispatch_dim<__nv_bfloat16, Transform, Quantize>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused);
}

at::Tensor transform(const at::Tensor& input, double scale, int block_threads) {
    validate(input, scale, block_threads);
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = at::empty_like(input);
    at::Tensor packed, scales;
    dispatch<true, false>(input, output, packed, scales, scale, block_threads);
    return output;
}

template <bool Transform>
std::tuple<at::Tensor, at::Tensor> quantized(const at::Tensor& input, double scale, int block_threads,
                                         const std::string& fused_layout = "original") {
    validate(input, scale, block_threads);
    TORCH_CHECK(fused_layout == "original" || fused_layout == "contiguous256",
                "fused_layout must be 'original' or 'contiguous256'");
    const bool contiguous256_fused = fused_layout == "contiguous256";
    if (contiguous256_fused) {
        TORCH_CHECK(Transform, "contiguous256 is only supported for fused Hadamard INT4");
        TORCH_CHECK(input.size(-1) == 256, "fused_layout='contiguous256' requires last dimension 256");
        TORCH_CHECK(block_threads == 128, "fused_layout='contiguous256' requires block_threads=128");
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto packed_shape = input.sizes().vec();
    packed_shape.back() = (packed_shape.back() + 1) / 2;
    auto scale_shape = input.sizes().vec();
    scale_shape.pop_back();
    auto packed = at::empty(packed_shape, input.options().dtype(at::kByte));
    auto scales = at::empty(scale_shape, input.options().dtype(at::kFloat));
    at::Tensor output;
    dispatch<Transform, true>(input, output, packed, scales, scale, block_threads, contiguous256_fused);
    return {packed, scales};
}

std::tuple<at::Tensor, at::Tensor> quantize_only(const at::Tensor& input, int block_threads) {
    return quantized<false>(input, 1.0, block_threads);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("hadamard", &transform, pybind11::arg("input"), pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128,
               "Forward-only last-axis Hadamard; finite CUDA FP16/BF16 input, 2D or 4D contiguous. block_threads=128 (default) or 256; explicit, no auto-tuning.");
    module.def("hadamard_int4", &quantized<true>, pybind11::arg("input"), pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128,
               pybind11::arg("fused_layout") = "original",
               "Fused transform and rowwise symmetric INT4; returns (uint8 packed, float32 scales). fused_layout='original' (default) supports block_threads=128 or 256; explicit 'contiguous256' requires N256 and block_threads=128.");
    module.def("quantize_int4", &quantize_only, pybind11::arg("input"), pybind11::arg("block_threads") = 128,
               "Quantize an already-rounded FP16/BF16 tensor; even values occupy the low nibble. block_threads=128 (default) or 256.");
}
