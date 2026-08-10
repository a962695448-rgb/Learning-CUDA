#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
template <typename T>
__device__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ half fromFloat<half>(float value) {
  return __float2half(value);
}
template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= rows) {
    return;
  }

  size_t row_offset = row * hidden_dim;
  float sum_square = 0.0f;

  for (size_t col = 0; col < hidden_dim; ++col) {
    float value = toFloat(input[row_offset + col]);
    sum_square += value * value;
  }

  float mean_square =
      sum_square / static_cast<float>(hidden_dim);

  float inverse_rms = rsqrtf(mean_square + eps);

  for (size_t col = 0; col < hidden_dim; ++col) {
    float value = toFloat(input[row_offset + col]);
    float scale = toFloat(weight[col]);

    output[row_offset + col] =
        fromFloat<T>(value * inverse_rms * scale);
  }
}
/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function
  if (rows == 0 || hidden_dim == 0) {
    return;
  }

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;

  size_t input_bytes = rows * hidden_dim * sizeof(T);
  size_t weight_bytes = hidden_dim * sizeof(T);

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_input), input_bytes));

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_weight), weight_bytes));

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_output), input_bytes));

  RUNTIME_CHECK(cudaMemcpy(
      d_input, h_input.data(), input_bytes,
      cudaMemcpyHostToDevice));

  RUNTIME_CHECK(cudaMemcpy(
      d_weight, h_weight.data(), weight_bytes,
      cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks =
      static_cast<int>((rows + threads - 1) / threads);

  rmsNormKernel<T><<<blocks, threads>>>(
      d_input, d_weight, d_output,
      rows, hidden_dim, eps);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(
      h_output.data(), d_output, input_bytes,
      cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
