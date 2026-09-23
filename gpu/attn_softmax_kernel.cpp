// gpu/attn_softmax_kernel.cpp -- softmax causal, com as threads a trabalhar.
//
// A PRIMEIRA VERSAO NAO USAVA threadIdx.x. Cada uma das 64 threads executava a
// linha inteira, escrevia o mesmo valor (por isso o resultado estava certo), e
// fazia 64 vezes o trabalho necessario. Medido: 19,136 ms contra 2,800 ms da
// CPU, ou 6,8 vezes MAIS LENTA. Um resultado correcto nao e' um kernel correcto.
//
// Agora: um bloco por linha (batch, head, ii), as threads dividem o trabalho ao
// longo de j, e o maximo e a soma passam por reducao em memoria partilhada.
// A matematica e' a mesma do causal_attn: cap antes da mascara, so' jj <= ii,
// maximo por linha, exp, soma, divisao, e zero acima da diagonal.
#include <hip/hip_runtime.h>
#include <math.h>

#define WARP 32

__device__ __forceinline__ float capf(float x, float cap) {
  return (cap > 0.0f) ? cap * tanhf(x / cap) : x;
}

// Reducao de bloco por soma, em memoria partilhada.
__device__ float block_sum(float v, float* sh) {
  int t = threadIdx.x, n = blockDim.x;
  sh[t] = v;
  __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  return sh[0];
}

__device__ float block_max(float v, float* sh) {
  int t = threadIdx.x, n = blockDim.x;
  sh[t] = v;
  __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) {
    if (t < s && sh[t + s] > sh[t]) sh[t] = sh[t + s];
    __syncthreads();
  }
  return sh[0];
}

__global__ void attn_softmax_causal_kernel(const float* __restrict__ S,
                                           float* __restrict__ P,
                                           int T, float cap) {
  extern __shared__ float sh[];
  int row = blockIdx.x;      // ii
  int bh  = blockIdx.y;      // (batch, head)
  int t   = threadIdx.x, n = blockDim.x;
  if (row >= T) return;
  const float* srow = S + ((size_t)bh * T + row) * T;
  float* prow = P + ((size_t)bh * T + row) * T;
  // maximo sobre jj <= row, com as threads a cobrir a linha
  float m = -INFINITY;
  for (int j = t; j <= row; j += n) {
    float x = capf(srow[j], cap);
    if (x > m) m = x;
  }
  m = block_max(m, sh);
  // soma dos exp
  float s = 0.0f;
  for (int j = t; j <= row; j += n) s += __expf(capf(srow[j], cap) - m);
  s = block_sum(s, sh);
  float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
  // escrita
  for (int j = t; j <= row; j += n)
    prow[j] = __expf(capf(srow[j], cap) - m) * inv;
  for (int j = row + 1 + t; j < T; j += n) prow[j] = 0.0f;
}

extern "C" int attn_softmax_causal(const float* S, float* P, int T, int BH, float cap) {
  if (T <= 0 || BH <= 0) return -1;
  int threads = 128;
  dim3 grid((unsigned)T, (unsigned)BH);
  hipLaunchKernelGGL(attn_softmax_causal_kernel, grid, dim3(threads),
                     threads * sizeof(float), 0, S, P, T, cap);
  return (int)hipDeviceSynchronize();
}
