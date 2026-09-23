// gpu/relu2_bwd_kernel.cpp -- o backward do relu2: 2*max(0,x)*dy, elementwise.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void relu2_bwd_kernel(const float* __restrict__ dy,
                                 const float* __restrict__ x,
                                 float* __restrict__ dx, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  dx[i] = (v > 0.0f) ? 2.0f * v * dy[i] : 0.0f;
}

extern "C" int relu2_bwd(const float* dy, const float* x, float* dx, int64_t n) {
  if (n <= 0) return -1;
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(relu2_bwd_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0, dy, x, dx, n);
  return (int)hipDeviceSynchronize();
}
