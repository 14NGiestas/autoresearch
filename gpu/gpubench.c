// gpu/gpubench.c -- rocBLAS sgemm nas formas do nosso passo, contra o CPU.
//
// Por que existe: a analise estatica (scripts/flop_breakdown.py) mostrou que
// 72,7% dos nossos MACs sao atencao e cabeca de vocabulario, e que head_dim 16
// e' o gargalo (9,8 GFLOP/s contra 74,3 com head_dim 128). A GPU e' forte
// exatamente ali.
//
// Roda no shell rocm (que ja' e' torch-free):
//   nix develop .#rocm --command bash -c '
//     RB=$(echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep rocblas | head -1)
//     hipcc -O2 -D__HIP_PLATFORM_AMD__ -I$HIP_PATH/include -I$(dirname $RB)/include \
//       -o /tmp/gpubench gpu/gpubench.c -lrocblas -L$RB && /tmp/gpubench'
//
// Medido em 2026-09-20, Radeon 780M (gfx1100, APU Ryzen 7 8745HS):
//   atencao k=16    499,1 GFLOP/s   <- o nosso pior caso na CPU: 9,8. Fator 51x
//   atencao k=96   1059,6
//   MLP up         1263,9
//   vocab head     2348,1           <- 2,3 TFLOP/s, o maior ganho
//   k=128          2137,4           <- CPU: 74,3. Fator 29x
//
// RESSALVA: isto mede GEMM. O passo tem softmax entre os GEMMs da atencao, e
// otimizador e norms elementwise, que ficariam na CPU sem fusao. Entao o ganho
// do PASSO sera menor que 27x -- quanto menor e' a proxima medida.
// Microbenchmark: rocBLAS sgemm nas formas do nosso passo, contra o CPU.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void run(int m, int n, int k, int iters, const char *tag) {
    float *A, *B, *C; rocblas_handle h;
    float alpha = 1.0f, beta = 0.0f;
    rocblas_create_handle(&h);
    hipMalloc((void**)&A, (size_t)m*k*4); hipMalloc((void**)&B, (size_t)k*n*4); hipMalloc((void**)&C, (size_t)m*n*4);
    hipMemset(A, 0, (size_t)m*k*4); hipMemset(B, 0, (size_t)k*n*4);
    // aquecimento (compila o kernel, aloca scratch)
    for (int i = 0; i < 3; i++)
        rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, m, n, k,
                      &alpha, A, m, B, k, &beta, C, m);
    hipDeviceSynchronize();
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, m, n, k,
                      &alpha, A, m, B, k, &beta, C, m);
    hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double gflops = 2.0 * m * n * k * iters / dt / 1e9;
    printf("  %-22s m=%4d n=%4d k=%4d  %8.1f GFLOP/s  (%.2f ms/iter)\n",
           tag, m, n, k, gflops, 1000.0*dt/iters);
    hipFree(A); hipFree(B); hipFree(C); rocblas_destroy_handle(h);
}

int main(void) {
    int dev; hipGetDevice(&dev);
    hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
    printf("GPU: %s  (%d CUs)\n", p.name, p.multiProcessorCount);
    run(1024, 1024,  16, 200, "atencao (k=head_dim 16)");
    run(1024, 1024,  96, 200, "atencao (k=d_model 96)");
    run(1024,  384,  96, 200, "MLP up");
    run(1024,   96,  96, 200, "linear q/k/v/o");
    run(1024, 8192,  96, 100, "cabeca de vocab");
    run(2048, 2048, 128, 100, "escala maior (k=128)");
    return 0;
}
