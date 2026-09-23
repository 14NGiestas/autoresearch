// gpu/rmsnorm_bwd_kernel.cpp -- o backward da RMSNorm.
//
// A formula e' a do rmsnorm0_bwd do modelo, lida de la' e nao derivada de cabeca:
//   ss   = media(x^2) + eps
//   inv  = 1/sqrt(ss)
//   dot  = soma(dy*x)
//   coef = dot / (D * ss * sqrt(ss))
//   dx   = dy*inv - x*coef
//
// Duas reducoes por linha (ss e dot), ambas em memoria partilhada. Um bloco por
// linha, as threads dividem as dimensoes.
#include <hip/hip_runtime.h>

__global__ void rmsnorm_bwd_kernel(const float* __restrict__ dy,
                                   const float* __restrict__ x,
                                   float* __restrict__ dx,
                                   int D, float eps) {
  extern __shared__ float sh[];
  int tx = threadIdx.x, n = blockDim.x;
  long row = blockIdx.x;
  const float* xr  = x  + row * D;
  const float* dyr = dy + row * D;
  float* dxr = dx + row * D;
  float ss = 0.0f, dot = 0.0f;
  for (int i = tx; i < D; i += n) { ss += xr[i] * xr[i]; dot += dyr[i] * xr[i]; }
  sh[tx] = ss;                 __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) { if (tx < s) sh[tx] += sh[tx + s]; __syncthreads(); }
  ss = sh[0];
  __syncthreads();
  sh[tx] = dot;                __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) { if (tx < s) sh[tx] += sh[tx + s]; __syncthreads(); }
  dot = sh[0];
  ss = ss / (float)D + eps;
  float inv = rsqrtf(ss);
  float coef = dot / ((float)D * ss * sqrtf(ss));
  for (int i = tx; i < D; i += n) dxr[i] = dyr[i] * inv - xr[i] * coef;
}

extern "C" int rmsnorm_bwd(const float* dy, const float* x, float* dx, int rows, int D, float eps) {
  if (rows <= 0 || D <= 0) return -1;
  int threads = 128;
  hipLaunchKernelGGL(rmsnorm_bwd_kernel, dim3((unsigned)rows), dim3(threads),
                     threads * sizeof(float), 0, dy, x, dx, D, eps);
  return (int)hipDeviceSynchronize();
}
