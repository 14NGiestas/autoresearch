// gpu/rope_bwd_kernel.cpp -- o backward da RoPE: a rotacao inversa.
// A inversa de uma rotacao e' a sua transposta, e a formula esta' no
// rope_4d_bwd do modelo:
//   dx1 = e1*c - e2*s      dx2 = e1*s + e2*c
// O porteiro e' o round-trip: forward e depois inversa tem de dar a identidade.
#include <hip/hip_runtime.h>

__global__ void rope_bwd_kernel(const float* __restrict__ dy,
                                const float* __restrict__ cosb,
                                const float* __restrict__ sinb,
                                float* __restrict__ dx,
                                int T, int H, int D) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d2 = D / 2;
  int64_t total = (int64_t)T * H * d2;
  if (i >= total) return;
  int dd = (int)(i % d2);
  int64_t th = i / d2;
  int hh = (int)(th % H);
  int tt = (int)(th / H);
  int64_t base = ((int64_t)tt * H + hh) * D;
  float e1 = dy[base + dd], e2 = dy[base + dd + d2];
  float c = cosb[(int64_t)tt * d2 + dd], s = sinb[(int64_t)tt * d2 + dd];
  dx[base + dd]      = e1 * c - e2 * s;
  dx[base + dd + d2] = e1 * s + e2 * c;
}

extern "C" int rope_bwd(const float* dy, const float* cosb, const float* sinb,
                        float* dx, int T, int H, int D) {
  if (T <= 0 || H <= 0 || D <= 0 || (D % 2)) return -1;
  int64_t total = (int64_t)T * H * (D / 2);
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(rope_bwd_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0,
                     dy, cosb, sinb, dx, T, H, D);
  return (int)hipDeviceSynchronize();
}
