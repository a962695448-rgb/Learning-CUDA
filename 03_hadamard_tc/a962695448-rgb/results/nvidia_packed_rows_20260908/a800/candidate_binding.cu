#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include "packed_rows.cuh"

namespace {
void validate(const at::Tensor& input, double scale, int threads) {
    TORCH_CHECK(threads == 128 || threads == 256, "block_threads must be 128 or 256");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16, "input must be FP16 or BF16");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 4, "input must be 2D or 4D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(!input.is_neg(), "input must not have a lazy negative bit; call resolve_neg() first");
    TORCH_CHECK(input.numel() > 0, "input must be nonempty");
    const auto n = input.size(-1);
    TORCH_CHECK(n >= 1 && n <= 16 && !(n & (n - 1)), "last dimension must be a power of two in [1,16]");
    TORCH_CHECK(!input.requires_grad(), "this forward-only extension does not support requires_grad=True");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale) > 0, "scale must be finite, positive, and representable as float32");
    const auto rows = input.numel() / n;
    TORCH_CHECK((rows - 1) / (threads / n) + 1 <= std::numeric_limits<int>::max(), "input has too many rows for the CUDA launch grid");
}

template <class T, int N, bool Transform, bool Quantize>
void launch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed, at::Tensor& scales,
            float scale, int threads, cudaStream_t stream) {
    const std::size_t rows = input.numel() / N;
    const unsigned blocks = static_cast<unsigned>((rows - 1) / (threads / N) + 1);
    hadamard::packed_rows_kernel<T, N, Transform, Quantize><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const T*>(input.data_ptr()), output.defined() ? reinterpret_cast<T*>(output.data_ptr()) : nullptr,
        packed.defined() ? packed.data_ptr<std::uint8_t>() : nullptr, scales.defined() ? scales.data_ptr<float>() : nullptr, rows, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, bool Transform, bool Quantize>
void dispatch_dim(const at::Tensor& input, at::Tensor& output, at::Tensor& packed, at::Tensor& scales,
                  float scale, int threads, cudaStream_t stream) {
    switch (input.size(-1)) {
#define ROW_CASE(N) case N: launch<T,N,Transform,Quantize>(input,output,packed,scales,scale,threads,stream); break
        ROW_CASE(1); ROW_CASE(2); ROW_CASE(4); ROW_CASE(8); ROW_CASE(16);
#undef ROW_CASE
    }
}

template <bool Transform, bool Quantize>
void dispatch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed, at::Tensor& scales,
              double scale, int threads) {
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    if (input.scalar_type() == at::kHalf) dispatch_dim<__half,Transform,Quantize>(input,output,packed,scales,static_cast<float>(scale),threads,stream);
    else dispatch_dim<__nv_bfloat16,Transform,Quantize>(input,output,packed,scales,static_cast<float>(scale),threads,stream);
}

at::Tensor transform(const at::Tensor& input, double scale, int threads) {
    validate(input,scale,threads); const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty_like(input); at::Tensor packed, scales;
    dispatch<true,false>(input,output,packed,scales,scale,threads); return output;
}

template <bool Transform>
std::tuple<at::Tensor,at::Tensor> quantized(const at::Tensor& input, double scale, int threads) {
    validate(input,scale,threads); const c10::cuda::CUDAGuard guard(input.device());
    auto shape = input.sizes().vec(); shape.back() = (shape.back()+1)/2;
    auto scale_shape = input.sizes().vec(); scale_shape.pop_back();
    auto packed = at::empty(shape,input.options().dtype(at::kByte));
    auto scales = at::empty(scale_shape,input.options().dtype(at::kFloat)); at::Tensor output;
    dispatch<Transform,true>(input,output,packed,scales,scale,threads); return {packed,scales};
}
std::tuple<at::Tensor,at::Tensor> quantize_only(const at::Tensor& input, int threads) { return quantized<false>(input,1.0,threads); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard",&transform,pybind11::arg("input"),pybind11::arg("scale")=1.0,pybind11::arg("block_threads")=128);
    m.def("hadamard_int4",&quantized<true>,pybind11::arg("input"),pybind11::arg("scale")=1.0,pybind11::arg("block_threads")=128);
    m.def("quantize_int4",&quantize_only,pybind11::arg("input"),pybind11::arg("block_threads")=128);
}
