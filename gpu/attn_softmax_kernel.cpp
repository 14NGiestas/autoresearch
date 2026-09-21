// gpu/attn_softmax_kernel.cpp -- o primeiro kernel a serio: softmax causal.
//
// Substitui a fase que, medida, e' 24 por cento da atencao no head_dim 128 (e
// 71,6 por cento no head_dim 16). A matematica e' a do causal_attn, e tem de
// casar com ela:
//   cap antes da mascara; causal: so' jj <= ii; maximo por linha para
//   estabilidade; exp; soma; divisao; e zero acima da diagonal.
//
// Uma thread block por linha (batch, head, ii). O launch vem de um wrapper
// extern "C", que e' o desenho documentado do hipfort: o kernel em HIP C++, o
// wrapper em C, e a interface declarada em Fortran com iso_c_binding.
#include <hip/hip_runtime.h>
#include <math.h>

// P[ii][jj] = softmax sobre jj<=ii de cap(S[ii][jj]), zero acima.
__global__ void attn_softmax_causal_kernel(const float* __restrict__ S,
                                           float* __restrict__ P,
                                           int T, float cap) {
  int row = blockIdx.x;          // ii
  int bh  = blockIdx.y;          // (batch, head) index
  if (row >= T) return;
  const float* srow = S + ((size_t)bh * T + row) * T;
  float* prow = P + ((size_t)bh * T + row) * T;
  int j;
  float x, mx = -INFINITY;
  for (j = 0; j <= row; j++) {
    x = srow[j];
    if (cap > 0.0f) x = cap * tanhf(x / cap);
    if (x > mx) mx = x;
  }
  float sm = 0.0f;
  for (j = 0; j <= row; j++) {
    x = srow[j];
    if (cap > 0.0f) x = cap * tanhf(x / cap);
    x = __expf(x - mx);
    prow[j] = x;
    sm += x;
  }
  float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
  for (j = 0; j <= row; j++) prow[j] *= inv;
  for (j = row + 1; j < T; j++) prow[j] = 0.0f;
}

extern "C" int attn_softmax_causal(const float* S, float* P, int T, int BH, float cap) {
  if (T <= 0 || BH <= 0) return -1;
  dim3 grid((unsigned)T, (unsigned)BH);
  hipLaunchKernelGGL(attn_softmax_causal_kernel, grid, dim3(64), 0, 0, S, P, T, cap);
  return (int)hipDeviceSynchronize();
}
