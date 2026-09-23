// gpu/relu2_kernel.cpp -- o quarto kernel: a ativacao do MLP.
//
// O modelo usa relu2, e nao gelu: y = max(0, x)^2. O backward e' 2*max(0,x)*dy.
// Fecha o forward, junto com as GEMMs, o softmax, a norma e a RoPE.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void relu2_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = (v > 0.0f) ? v * v : 0.0f;
}

extern "C" int relu2_fwd(const float* x, float* y, int64_t n) {
  if (n <= 0) return -1;
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(relu2_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0, x, y, n);
  return (int)hipDeviceSynchronize();
}
