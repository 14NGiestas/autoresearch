// gpu/rocblas_shim.c -- uma ABI em C sobre o rocBLAS, para o Fortran chamar.
//
// Por que um shim em C: o hipfort esta' AUSENTE neste nix, e o gfortran nao
// compila HIP. O shim resolve os dois, porque expoe uma ABI de C explicita. Com
// bind(C) no lado Fortran os tipos sao declarados, e a armadilha registrada em
// fortran_blas.f90 (OpenBLAS ILP64 le' lixo com interface LP64) deixa de existir.
//
// O shim compila como C++, porque o hip_runtime.h e' C++ apenas. A API sai com
// ligacao de C (extern "C"), entao o Fortran e um teste em C chamam sem mangling.
//
// DECISOES DE DESENHO.
//
// 1. Os pesos ficam RESIDENTES no device, com o endereco de host como chave. O 1B
//    tem 352 MB de pesos e copiar isso por passo mata o ganho.
// 2. Os buffers usam SLOTS explicitos. Um pool "primeiro que serve" entrega o
//    MESMO buffer para entrada e saida, e o resultado fica errado em silencio.
// 3. Todo retorno da HIP e' conferido com CHK. Um hipMemcpy que falha calado
//    entrega lixo ao kernel.
//
// A receita de compilacao esta' no topo de gpu/gpubench.c. O shim vai como C++ e
// o teste como C:
//   hipcc -O3 -x c++ gpu/rocblas_shim.c -x c gpu/shim_test.c \
//     -o /tmp/shim_test -lrocblas -L$RB
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAXW 1024
#define MAXB 16

#define CHK(call) do { if ((call) != hipSuccess) return -5; } while (0)

typedef struct { const float* host; float* dev; int64_t IF, OF; } WEntry;
typedef struct { float* d[4]; size_t n[4]; int have[4]; } Slot;

static WEntry W[MAXW];
static int NW = 0;
static Slot S[MAXB];
static int NS = 0;
static rocblas_handle H = NULL;

static int ensure(void) {
  if (H) return 0;
  if (rocblas_create_handle(&H) != rocblas_status_success) { H = NULL; return -1; }
  return 0;
}

// Um buffer de device para (conjunto, posicao), criado na primeira vez e reusado.
static float* buff(int set, int pos, size_t n) {
  if (set < 0 || set >= MAXB || pos < 0 || pos > 3) return NULL;
  if (S[set].have[pos] && S[set].n[pos] >= n) return S[set].d[pos];
  if (S[set].have[pos]) { (void)hipFree(S[set].d[pos]); S[set].have[pos] = 0; }
  float* d = NULL;
  if (hipMalloc((void**)&d, n * 4) != hipSuccess) return NULL;
  S[set].d[pos] = d; S[set].n[pos] = n; S[set].have[pos] = 1;
  if (set + 1 > NS) NS = set + 1;
  return d;
}

// Acha o peso, e registra na primeira vez que o ve'. Sem chamada do chamador.
static float* weight(const float* w, int64_t IF, int64_t OF) {
  if (!w || IF <= 0 || OF <= 0) return NULL;
  for (int i = 0; i < NW; i++)
    if (W[i].host == w && W[i].IF == IF && W[i].OF == OF) return W[i].dev;
  if (NW >= MAXW) return NULL;
  if (ensure() != 0) return NULL;
  size_t n = (size_t)IF * (size_t)OF;
  float* d = buff(0, 0, n);
  if (!d) return NULL;
  // Aqui nao se usa CHK, porque CHK devolve -5 e esta funcao devolve float*.
  if (hipMemcpy(d, w, n * 4, hipMemcpyHostToDevice) != hipSuccess) return NULL;
  W[NW].host = w; W[NW].dev = d; W[NW].IF = IF; W[NW].OF = OF; NW++;
  return d;
}

int32_t gpublas_count(void) { return NW; }
int32_t gpublas_bufs(void)  { return NS; }

// y(BT,OF) = x(BT,IF) . w(OF,IF)^T, flats row-major.
//   Yf(OF,BT) = Wf(IF,OF)^T . Xf(IF,BT) -> sgemm('T','N', OF,BT,IF, w,IF, x,IF, y,OF)
int32_t gpublas_sgemm_fwd(const float* x, const float* w, int64_t BT,
                          float* y, int64_t IF, int64_t OF) {
  if (ensure() != 0) return -1;
  float* dw = weight(w, IF, OF);
  if (!dw) return -2;
  size_t nx = (size_t)BT * (size_t)IF, ny = (size_t)BT * (size_t)OF;
  float* dx = buff(1, 0, nx); float* dy = buff(1, 1, ny);
  if (!dx || !dy) return -3;
  CHK(hipMemcpy(dx, x, nx * 4, hipMemcpyHostToDevice));
  const float a = 1.0f, b = 0.0f;
  rocblas_status s = rocblas_sgemm(H, rocblas_operation_transpose, rocblas_operation_none,
      (rocblas_int)OF, (rocblas_int)BT, (rocblas_int)IF, &a, dw, (rocblas_int)IF,
      dx, (rocblas_int)IF, &b, dy, (rocblas_int)OF);
  if (s != rocblas_status_success) return -4;
  CHK(hipMemcpy(y, dy, ny * 4, hipMemcpyDeviceToHost));
  return 0;
}

// dx(BT,IF) = dy(BT,OF) . w(OF,IF)
//   DXf(IF,BT) = Wf(IF,OF) . DYf(OF,BT) -> sgemm('N','N', IF,BT,OF, w,IF, dy,OF, dx,IF)
int32_t gpublas_sgemm_bwd_dx(const float* dy, const float* w, int64_t BT,
                             float* dxo, int64_t IF, int64_t OF) {
  if (ensure() != 0) return -1;
  float* dw = weight(w, IF, OF);
  if (!dw) return -2;
  size_t ndy = (size_t)BT * (size_t)OF, ndx = (size_t)BT * (size_t)IF;
  float* ddy = buff(2, 0, ndy); float* ddx = buff(2, 1, ndx);
  if (!ddy || !ddx) return -3;
  CHK(hipMemcpy(ddy, dy, ndy * 4, hipMemcpyHostToDevice));
  const float a = 1.0f, b = 0.0f;
  rocblas_status s = rocblas_sgemm(H, rocblas_operation_none, rocblas_operation_none,
      (rocblas_int)IF, (rocblas_int)BT, (rocblas_int)OF, &a, dw, (rocblas_int)IF,
      ddy, (rocblas_int)OF, &b, ddx, (rocblas_int)IF);
  if (s != rocblas_status_success) return -4;
  CHK(hipMemcpy(dxo, ddx, ndx * 4, hipMemcpyDeviceToHost));
  return 0;
}

// dw(OF,IF) = dy(BT,OF)^T . x(BT,IF), beta 0. O acumulo e' do chamador, como no CPU.
//   DWf(IF,OF) = Xf(IF,BT) . DYf(OF,BT)^T -> sgemm('N','T', IF,OF,BT, x,IF, dy,OF, dw,IF)
int32_t gpublas_sgemm_bwd_dw(const float* dy, const float* x, int64_t BT,
                             float* dwo, int64_t IF, int64_t OF) {
  if (ensure() != 0) return -1;
  size_t nx = (size_t)BT * (size_t)IF, ndy = (size_t)BT * (size_t)OF, ndw = (size_t)IF * (size_t)OF;
  float* dx = buff(3, 0, nx); float* ddy = buff(3, 1, ndy); float* ddw = buff(3, 2, ndw);
  if (!dx || !ddy || !ddw) return -3;
  CHK(hipMemcpy(dx, x, nx * 4, hipMemcpyHostToDevice));
  CHK(hipMemcpy(ddy, dy, ndy * 4, hipMemcpyHostToDevice));
  const float a = 1.0f, b = 0.0f;
  rocblas_status s = rocblas_sgemm(H, rocblas_operation_none, rocblas_operation_transpose,
      (rocblas_int)IF, (rocblas_int)OF, (rocblas_int)BT, &a, dx, (rocblas_int)IF,
      ddy, (rocblas_int)OF, &b, ddw, (rocblas_int)IF);
  if (s != rocblas_status_success) return -4;
  CHK(hipMemcpy(dwo, ddw, ndw * 4, hipMemcpyDeviceToHost));
  return 0;
}

#ifdef __cplusplus
}
#endif
