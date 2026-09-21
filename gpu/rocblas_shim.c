// gpu/rocblas_shim.c -- uma ABI em C sobre o rocBLAS, para o Fortran chamar.
//
// Por que existe um shim em C, e nao hipfort: o hipfort esta' AUSENTE neste
// nix, e o gfortran nao compila HIP. O shim resolve as duas coisas de uma vez,
// porque expoe uma ABI de C explicita. Com bind(C) no lado Fortran, os tipos sao
// declarados, e a armadilha registrada em fortran_blas.f90 (OpenBLAS ILP64 le'
// lixo com interface LP64) deixa de existir: aqui nao ha' interface implicita.
//
// O desenho que importa: os PESOS FICAM RESIDENTES no device. O 1B tem 352 MB de
// pesos; copiar isso a cada passo e' 352 MB por passo, o que mata o ganho. Entao
// o shim registra o peso uma vez (gpublas_register) e guarda o ponteiro do device
// num cache por endereco de host. As ativacoes sao pequenas (BT*IF floats, 3 MB
// no formato d768), e essas sim trafegam a cada chamada.
//
// Compila com hipcc, que traz o rocblas:
//   hipcc -O3 -fPIC -shared -o librocblas_shim.so gpu/rocblas_shim.c -lrocblas
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXW 512

typedef struct { const float* host; float* dev; int64_t IF, OF; } WEntry;
static WEntry W[MAXW];
static int NW = 0;
static rocblas_handle H = NULL;

static int ensure(void) {
  if (H) return 0;
  if (rocblas_create_handle(&H) != rocblas_status_success) { H = NULL; return -1; }
  return 0;
}

// Registra um peso (uma vez). Devolve 0 em sucesso.
int32_t gpublas_register(const float* w, int64_t IF, int64_t OF) {
  for (int i = 0; i < NW; i++)
    if (W[i].host == w && W[i].IF == IF && W[i].OF == OF) return 0;
  if (NW >= MAXW) return -2;
  if (ensure() != 0) return -1;
  size_t n = (size_t)IF * (size_t)OF * sizeof(float);
  float* d = NULL;
  if (hipMalloc((void**)&d, n) != hipSuccess) return -3;
  if (hipMemcpy(d, w, n, hipMemcpyHostToDevice) != hipSuccess) return -4;
  W[NW].host = w; W[NW].dev = d; W[NW].IF = IF; W[NW].OF = OF; NW++;
  return 0;
}

int32_t gpublas_count(void) { return NW; }

// y(BT,OF) = x(BT,IF) . w(OF,IF)^T, flats row-major, a MESMA matematica e o
// MESMO truque de layout do linear3d_sgemm do lado CPU:
//   row-major Y(BT,OF) e' column-major Yf(OF,BT), entao Yf = Wf^T . Xf
//   -> sgemm('T','N', OF, BT, IF, 1, w, IF, x, IF, 0, y, OF)
int32_t gpublas_sgemm_fwd(const float* x, int64_t BTw, float* y, int64_t IF, int64_t OF) {
  const float* dw = NULL;
  for (int i = 0; i < NW; i++) if (W[i].IF == IF && W[i].OF == OF) { dw = W[i].dev; break; }
  if (!dw) return -1;
  if (ensure() != 0) return -1;
  int64_t BT = BTw;
  size_t nx = (size_t)BT * (size_t)IF, ny = (size_t)BT * (size_t)OF;
  float *dx = NULL, *dy = NULL;
  if (hipMalloc((void**)&dx, nx * 4) != hipSuccess) return -3;
  if (hipMalloc((void**)&dy, ny * 4) != hipSuccess) { hipFree(dx); return -3; }
  hipMemcpy(dx, x, nx * 4, hipMemcpyHostToDevice);
  const float alpha = 1.0f, beta = 0.0f;
  rocblas_status s = rocblas_sgemm(H, rocblas_operation_transpose, rocblas_operation_none,
      (rocblas_int)OF, (rocblas_int)BT, (rocblas_int)IF,
      &alpha, dw, (rocblas_int)IF, dx, (rocblas_int)IF, &beta, dy, (rocblas_int)OF);
  if (s == rocblas_status_success) hipMemcpy(y, dy, ny * 4, hipMemcpyDeviceToHost);
  hipFree(dx); hipFree(dy);
  return s == rocblas_status_success ? 0 : -5;
}
