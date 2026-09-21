// gpu/rmsnorm_kernel.cpp -- RMSNorm, o segundo kernel.
//
// A matematica e' a do rmsnorm0 do modelo, que e' pre-norm e nao tem peso nem
// viés: y_i = x_i / sqrt(media(x^2) + eps). Aplica-se antes da atencao e antes
// do MLP, 12 vezes por passo.
//
// A licao do softmax esta' aplicada desde a primeira linha: threadIdx.x existe e
// faz trabalho. O kernel anterior era correcto por acidente (todas as threads
// escreviam o mesmo valor) e 22 vezes mais lento.
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int D, float eps) {
  extern __shared__ float sh[];
  int t = threadIdx.x, n = blockDim.x;
  long row = blockIdx.x;
  const float* xr = x + row * D;
  float* yr = y + row * D;
  float acc = 0.0f;
  for (int i = t; i < D; i += n) acc += xr[i] * xr[i];
  sh[t] = acc;
  __syncthreads();
  for (int s = n / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  float inv = rsqrtf(sh[0] / (float)D + eps);
  for (int i = t; i < D; i += n) yr[i] = xr[i] * inv;
}

extern "C" int rmsnorm_fwd(const float* x, float* y, int rows, int D, float eps) {
  if (rows <= 0 || D <= 0) return -1;
  int threads = 128;
  hipLaunchKernelGGL(rmsnorm_kernel, dim3((unsigned)rows), dim3(threads),
                     threads * sizeof(float), 0, x, y, D, eps);
  return (int)hipDeviceSynchronize();
}
