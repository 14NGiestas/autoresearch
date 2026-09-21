// lib/gpu_shim.cpp -- as tres chamadas de BLAS, com o caminho GPU sob a macro.
//
// UM ARQUIVO, DOIS BUILDS. Este e' o ponto do desenho: com ARCH_GPU o fpm chama
// o hipcc e este arquivo vira o caminho rocBLAS; sem a macro, o fpm chama o
// compilador C++ padrao (g++) e este arquivo vira um stub que devolve erro. Nos
// dois casos ele COMPILA, entao ele pode viver dentro do pacote, em src/lib/, e o
// build CPU-only continua verde.
//
// Sem isso a alternativa era um artefato externo, como o OpenBLAS. Com isso o
// pacote carrega o kernel.
//
// A ABI e' de C (extern "C"), entao o Fortran chama com bind(C) sem mangling, e
// nenhum tipo fica implicito. A armadilha registrada em fortran_blas.f90
// (OpenBLAS ILP64 le' lixo com interface LP64) nao se aplica.
#include <stdint.h>
#include <stddef.h>

#ifdef ARCH_GPU
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
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
// Um buffer de device para (conjunto, posicao). Um pool "primeiro que serve"
// entrega o MESMO buffer para entrada e saida, e o resultado fica errado calado.
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
// Acha o peso e registra na primeira vez. Chave: o endereco de host. O 1B tem
// 352 MB de pesos, e copiar isso por passo mata o ganho.
static float* weight(const float* w, int64_t IF, int64_t OF) {
  if (!w || IF <= 0 || OF <= 0) return NULL;
  for (int i = 0; i < NW; i++)
    if (W[i].host == w && W[i].IF == IF && W[i].OF == OF) return W[i].dev;
  if (NW >= MAXW) return NULL;
  if (ensure() != 0) return NULL;
  size_t n = (size_t)IF * (size_t)OF;
  // O PESO TEM BUFFER PROPRIO, e nunca passa pelo pool de slots. Usar o pool aqui
  // foi um use-after-free: registrar um segundo peso LIBERTAVA o buffer do
  // primeiro, e o cache continuava a apontar para memoria libertada. O resultado
  // foi adam_v em 5,3e+21. O pool serve para ativacao transitoria, nao para o que
  // vive a corrida inteira.
  float* d = NULL;
  if (hipMalloc((void**)&d, n * 4) != hipSuccess) return NULL;
  // Sem CHK aqui: CHK devolve -5, e esta funcao devolve float*.
  if (hipMemcpy(d, w, n * 4, hipMemcpyHostToDevice) != hipSuccess) { (void)hipFree(d); return NULL; }
  W[NW].host = w; W[NW].dev = d; W[NW].IF = IF; W[NW].OF = OF; NW++;
  return d;
}
#endif

extern "C" {

// y(BT,OF) = x(BT,IF) . w(OF,IF)^T
int32_t gpublas_sgemm_fwd(const float* x, const float* w, int64_t BT,
                          float* y, int64_t IF, int64_t OF) {
#ifdef ARCH_GPU
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
#else
  (void)x; (void)w; (void)BT; (void)y; (void)IF; (void)OF;
  return -100;   // sem GPU neste binario
#endif
}

// dx(BT,IF) = dy(BT,OF) . w(OF,IF)
int32_t gpublas_sgemm_bwd_dx(const float* dy, const float* w, int64_t BT,
                             float* dxo, int64_t IF, int64_t OF) {
#ifdef ARCH_GPU
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
#else
  (void)dy; (void)w; (void)BT; (void)dxo; (void)IF; (void)OF;
  return -100;
#endif
}

// dw(OF,IF) = dy(BT,OF)^T . x(BT,IF), beta 0. O acumulo e' do chamador.
int32_t gpublas_sgemm_bwd_dw(const float* dy, const float* x, int64_t BT,
                             float* dwo, int64_t IF, int64_t OF) {
#ifdef ARCH_GPU
  if (ensure() != 0) return -1;
  size_t nx = (size_t)BT * (size_t)IF, ndy = (size_t)BT * (size_t)OF;
  size_t ndw = (size_t)IF * (size_t)OF;
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
#else
  (void)dy; (void)x; (void)BT; (void)dwo; (void)IF; (void)OF;
  return -100;
#endif
}

int32_t gpublas_count(void) {
#ifdef ARCH_GPU
  return NW;
#else
  return 0;
#endif
}

int32_t gpublas_bufs(void) {
#ifdef ARCH_GPU
  return NS;
#else
  return 0;
#endif
}

}  // extern "C"
