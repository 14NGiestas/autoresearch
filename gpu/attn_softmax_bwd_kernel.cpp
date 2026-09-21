// gpu/attn_softmax_bwd_kernel.cpp -- o backward do softmax causal.
//
// A formula e' a padrao do softmax, e a derivacao foi verificada por diferencas
// finitas no trabalho do relu_l1 (exata a 1e-17):
//   dS_k = P_k * ( dP_k - sum_i P_i * dP_i )     com mascara causal
//
// Um bloco por linha, as threads dividem j, a soma passa por reducao em memoria
// partilhada. A licao do primeiro kernel esta' aplicada: threadIdx.x existe.
#include <hip/hip_runtime.h>

__global__ void attn_softmax_bwd_kernel(const float* __restrict__ P,
                                        const float* __restrict__ dP,
                                        float* __restrict__ dS,
                                        int T) {
  extern __shared__ float sh[];
  int row = blockIdx.x;
  int bh  = blockIdx.y;
  int tx  = threadIdx.x, n = blockDim.x;
  if (row >= T) return;
  const float* prow  = P  + ((size_t)bh * T + row) * T;
  const float* dprow = dP + ((size_t)bh * T + row) * T;
  float* dsrow = dS + ((size_t)bh * T + row) * T;
  // rowsum = sum_i P_i * dP_i, sobre a parte causal
  float acc = 0.0f;
  for (int j = tx; j <= row; j += n) acc += prow[j] * dprow[j];
  sh[tx] = acc;
  __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) {
    if (tx < s) sh[tx] += sh[tx + s];
    __syncthreads();
  }
  float rowsum = sh[0];
  for (int j = tx; j <= row; j += n) dsrow[j] = prow[j] * (dprow[j] - rowsum);
  for (int j = row + 1 + tx; j < T; j += n) dsrow[j] = 0.0f;
}

extern "C" int attn_softmax_bwd(const float* P, const float* dP, float* dS, int T, int BH) {
  if (T <= 0 || BH <= 0) return -1;
  int threads = 128;
  hipLaunchKernelGGL(attn_softmax_bwd_kernel, dim3((unsigned)T, (unsigned)BH),
                     dim3(threads), threads * sizeof(float), 0, P, dP, dS, T);
  return (int)hipDeviceSynchronize();
}
