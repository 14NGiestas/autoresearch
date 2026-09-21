// gpu/shim_test.c -- o shim esta' certo, e quanto ele anda?
// Mesma matematica do linear3d_sgemm, conferida contra uma referencia serial.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int32_t gpublas_register(const float*, int64_t, int64_t);
int32_t gpublas_sgemm_fwd(const float*, int64_t, float*, int64_t, int64_t);
int32_t gpublas_count(void);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char** argv) {
  int64_t BT = 1024, IF = 768, OF = 3072;   // MLP up no formato 88M
  if (argc > 3) { BT = atoll(argv[1]); IF = atoll(argv[2]); OF = atoll(argv[3]); }
  size_t nx = (size_t)BT * IF, nw = (size_t)OF * IF, ny = (size_t)BT * OF;
  float *x = malloc(nx * 4), *w = malloc(nw * 4), *y = malloc(ny * 4), *r = malloc(ny * 4);
  for (size_t i = 0; i < nx; i++) x[i] = (float)((i % 7) - 3) * 0.01f;
  for (size_t i = 0; i < nw; i++) w[i] = (float)((i % 5) - 2) * 0.02f;
  if (gpublas_register(w, IF, OF) != 0) { printf("register FALHOU\n"); return 1; }
  printf("pesos registrados: %d\n", gpublas_count());

  // referencia serial, so' algumas linhas (BT*OF e' grande demais para tudo)
  int64_t chk = BT < 4 ? BT : 4;
  for (int64_t b = 0; b < chk; b++)
    for (int64_t o = 0; o < OF; o++) {
      float acc = 0;
      for (int64_t i = 0; i < IF; i++) acc += x[b * IF + i] * w[o * IF + i];
      r[b * OF + o] = acc;
    }
  // AQUECIMENTO. A primeira chamada paga a criacao do contexto HIP e a
  // alocacao. Cronometrar a primeira e' medir o contexto, nao o kernel. Esta e'
  // a licao do dia: um numero sem as suas condicoes nao e' um numero.
  int32_t rc = gpublas_sgemm_fwd(x, BT, y, IF, OF);
  if (rc != 0) { printf("sgemm FALHOU rc=%d\n", rc); return 1; }
  int reps = 20;
  double t0 = now();
  for (int k = 0; k < reps; k++) gpublas_sgemm_fwd(x, BT, y, IF, OF);
  double t1 = now();
  double maxe = 0;
  for (int64_t b = 0; b < chk; b++)
    for (int64_t o = 0; o < OF; o++) {
      double e = fabs((double)y[b * OF + o] - (double)r[b * OF + o]);
      if (e > maxe) maxe = e;
    }
  double fl = 2.0 * (double)BT * (double)OF * (double)IF;
  printf("  BT=%lld IF=%lld OF=%lld\n", (long long)BT, (long long)IF, (long long)OF);
  printf("  erro maximo contra a referencia serial: %.3e\n", maxe);
  printf("  %d reps aquecidas: %.3f ms por chamada -> %.1f GFLOP/s\n",
         reps, 1e3 * (t1 - t0) / reps, fl * reps / (t1 - t0) / 1e9);
  return maxe < 1e-3 ? 0 : 2;
}
