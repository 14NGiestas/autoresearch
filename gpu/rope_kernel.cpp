// gpu/rope_kernel.cpp -- a RoPE, o terceiro kernel.
//
// A convencao e' a do rope_4d do modelo: meia-rotacao. As duas metades do
// head_dim rodam juntas, com as tabelas cos/sin indexadas pela POSICAO t:
//   y[..., d]      =  x1*c + x2*s
//   y[..., d+d2]   = -x1*s + x2*c        com d2 = D/2
//
// E' embaraçosamente paralela: um thread por (t, h, d), sem reducao nenhuma.
// A propriedade que a verifica e' a rotacao preservar a norma de cada par.
#include <hip/hip_runtime.h>

__global__ void rope_kernel(const float* __restrict__ x,
                            const float* __restrict__ cosb,
                            const float* __restrict__ sinb,
                            float* __restrict__ y,
                            int T, int H, int D) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int d2 = D / 2;
  long total = (long)T * H * d2;
  if (i >= total) return;
  int dd = (int)(i % d2);
  long th = i / d2;
  int hh = (int)(th % H);
  int tt = (int)(th / H);
  long base = ((long)tt * H + hh) * D;
  float x1 = x[base + dd], x2 = x[base + dd + d2];
  float c = cosb[(long)tt * d2 + dd], s = sinb[(long)tt * d2 + dd];
  y[base + dd]      = x1 * c + x2 * s;
  y[base + dd + d2] = -x1 * s + x2 * c;
}

extern "C" int rope_fwd(const float* x, const float* cosb, const float* sinb,
                        float* y, int T, int H, int D) {
  if (T <= 0 || H <= 0 || D <= 0 || (D % 2)) return -1;
  long total = (long)T * H * (D / 2);
  int threads = 256;
  long blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(rope_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0,
                     x, cosb, sinb, y, T, H, D);
  return (int)hipDeviceSynchronize();
}
