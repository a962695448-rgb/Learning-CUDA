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
template <typename T>
__global__ void attentionScoreKernel(
    const T* q, const T* k, float* scores,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal) {
  size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  size_t total_scores =
      static_cast<size_t>(batch_size) *
      target_seq_len *
      query_heads *
      src_seq_len;

  if (index >= total_scores) {
    return;
  }

  int src_pos = static_cast<int>(index % src_seq_len);

  size_t temp = index / src_seq_len;
  int query_head = static_cast<int>(temp % query_heads);

  temp /= query_heads;
  int target_pos = static_cast<int>(temp % target_seq_len);

  int batch = static_cast<int>(temp / target_seq_len);

  if (is_causal && src_pos > target_pos) {
    scores[index] = -1.0e20f;
    return;
  }

  int group_size = query_heads / kv_heads;
  int kv_head = query_head / group_size;

  size_t q_offset =
      (((static_cast<size_t>(batch) * target_seq_len + target_pos)
          * query_heads + query_head)
          * head_dim);

  size_t k_offset =
      (((static_cast<size_t>(batch) * src_seq_len + src_pos)
          * kv_heads + kv_head)
          * head_dim);

    float dot = 0.0f;

  for (int d = 0; d < head_dim; ++d) {
    float q_value = toFloat(q[q_offset + d]);
    float k_value = toFloat(k[k_offset + d]);
    dot += q_value * k_value;
  }

  float scale =
      1.0f / sqrtf(static_cast<float>(head_dim));

  scores[index] = dot * scale;
}
__global__ void softmaxKernel(
    float* scores,
    int batch_size,
    int target_seq_len,
    int query_heads,
    int src_seq_len) {
  size_t row =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  size_t total_rows =
      static_cast<size_t>(batch_size) *
      target_seq_len *
      query_heads;

  if (row >= total_rows) {
    return;
  }

  size_t row_offset =
      row * static_cast<size_t>(src_seq_len);

  float max_score = -1.0e20f;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    max_score = fmaxf(
        max_score,
        scores[row_offset + src_pos]);
  }

  float sum_exp = 0.0f;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    sum_exp += expf(
        scores[row_offset + src_pos] - max_score);
  }

  float inverse_sum = 1.0f / sum_exp;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    float value = expf(
        scores[row_offset + src_pos] - max_score);

    scores[row_offset + src_pos] =
        value * inverse_sum;
  }
}

template <typename T>
__global__ void attentionOutputKernel(
    const float* scores,
    const T* v,
    T* output,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim) {
  size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  size_t total_outputs =
      static_cast<size_t>(batch_size) *
      target_seq_len *
      query_heads *
      head_dim;

  if (index >= total_outputs) {
    return;
  }

  int dim = static_cast<int>(index % head_dim);

  size_t temp = index / head_dim;
  int query_head = static_cast<int>(temp % query_heads);

  temp /= query_heads;
  int target_pos = static_cast<int>(temp % target_seq_len);

  int batch = static_cast<int>(temp / target_seq_len);

  int group_size = query_heads / kv_heads;
  int kv_head = query_head / group_size;

  float result = 0.0f;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    size_t score_offset =
        (((static_cast<size_t>(batch) * target_seq_len + target_pos)
            * query_heads + query_head)
            * src_seq_len + src_pos);

    size_t v_offset =
        (((static_cast<size_t>(batch) * src_seq_len + src_pos)
            * kv_heads + kv_head)
            * head_dim + dim);

    float probability = scores[score_offset];
    float value = toFloat(v[v_offset]);

    result = __fmaf_rn(probability, value, result);
  }

  output[index] = fromFloat<T>(result);
}

template <typename T>
__global__ void fusedAttentionKernel(
    const T* q,
    const T* k,
    const T* v,
    T* output,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal) {
  int row =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;

  int total_rows =
      batch_size * target_seq_len * query_heads;

  if (row >= total_rows) {
    return;
  }

  int rows_per_batch =
      target_seq_len * query_heads;

  int batch = row / rows_per_batch;
  int row_in_batch = row % rows_per_batch;
  int target_pos = row_in_batch / query_heads;
  int query_head = row_in_batch % query_heads;

  int group_size = query_heads / kv_heads;
  int kv_head = query_head / group_size;

  int q_offset =
      (((batch * target_seq_len + target_pos)
          * query_heads + query_head)
          * head_dim);

  float q_cache[256];
  float output_acc[256] = {0.0f};

  for (int d = 0; d < head_dim; ++d) {
    q_cache[d] = toFloat(q[q_offset + d]);
  }

  float scale =
      1.0f / sqrtf(static_cast<float>(head_dim));

  float max_score = -INFINITY;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    if (is_causal && src_pos > target_pos) {
      continue;
    }

    int k_offset =
        (((batch * src_seq_len + src_pos)
            * kv_heads + kv_head)
            * head_dim);

    float score = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
      score = __fmaf_rn(
          q_cache[d],
          toFloat(k[k_offset + d]),
          score);
    }

    score *= scale;
    max_score = fmaxf(max_score, score);
  }

  float sum_exp = 0.0f;

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    if (is_causal && src_pos > target_pos) {
      continue;
    }

    int k_offset =
        (((batch * src_seq_len + src_pos)
            * kv_heads + kv_head)
            * head_dim);

    float score = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
      score = __fmaf_rn(
          q_cache[d],
          toFloat(k[k_offset + d]),
          score);
    }

    score *= scale;
    sum_exp += expf(score - max_score);
  }

  float inverse_sum = 0.0f;

  if (sum_exp != 0.0f) {
    inverse_sum = 1.0f / sum_exp;
  }

  for (int src_pos = 0; src_pos < src_seq_len; ++src_pos) {
    if (is_causal && src_pos > target_pos) {
      continue;
    }

    int k_offset =
        (((batch * src_seq_len + src_pos)
            * kv_heads + kv_head)
            * head_dim);

    float score = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
      score = __fmaf_rn(
          q_cache[d],
          toFloat(k[k_offset + d]),
          score);
    }

    score *= scale;

    float probability =
        expf(score - max_score) * inverse_sum;

    int v_offset =
        (((batch * src_seq_len + src_pos)
            * kv_heads + kv_head)
            * head_dim);

    for (int d = 0; d < head_dim; ++d) {
      output_acc[d] = __fmaf_rn(
          probability,
          toFloat(v[v_offset + d]),
          output_acc[d]);
    }
  }

  for (int d = 0; d < head_dim; ++d) {
    output[q_offset + d] =
        fromFloat<T>(output_acc[d]);
  }
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
  size_t q_elements =
      static_cast<size_t>(batch_size) *
      target_seq_len *
      query_heads *
      head_dim;

  size_t k_elements =
      static_cast<size_t>(batch_size) *
      src_seq_len *
      kv_heads *
      head_dim;

  size_t v_elements = k_elements;

  size_t output_elements = q_elements;

  h_o.resize(output_elements);

  size_t q_bytes = q_elements * sizeof(T);
  size_t k_bytes = k_elements * sizeof(T);
  size_t v_bytes = v_elements * sizeof(T);
  size_t output_bytes = output_elements * sizeof(T);

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_output = nullptr;

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_q), q_bytes));

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_k), k_bytes));

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_v), v_bytes));

  RUNTIME_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_output), output_bytes));

  RUNTIME_CHECK(cudaMemcpy(
      d_q,
      h_q.data(),
      q_bytes,
      cudaMemcpyHostToDevice));

  RUNTIME_CHECK(cudaMemcpy(
      d_k,
      h_k.data(),
      k_bytes,
      cudaMemcpyHostToDevice));

  RUNTIME_CHECK(cudaMemcpy(
      d_v,
      h_v.data(),
      v_bytes,
      cudaMemcpyHostToDevice));

  const int threads = 256;

  size_t attention_rows =
      static_cast<size_t>(batch_size) *
      target_seq_len *
      query_heads;

  int attention_blocks = static_cast<int>(
      (attention_rows + threads - 1) / threads);

  fusedAttentionKernel<T><<<attention_blocks, threads>>>(
      d_q,
      d_k,
      d_v,
      d_output,
      batch_size,
      target_seq_len,
      src_seq_len,
      query_heads,
      kv_heads,
      head_dim,
      is_causal);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(
      h_o.data(),
      d_output,
      output_bytes,
      cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_output));
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
