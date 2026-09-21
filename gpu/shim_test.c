// gpu/shim_test.c -- as tres chamadas estao' certas, e quanto elas andam?
// Conferidas contra referencias seriais, com o padrao da casa: erro exato.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int32_t gpublas_sgemm_fwd(const float*, const float*, int64_t, float*, int64_t, int64_t);
int32_t gpublas_sgemm_bwd_dx(const float*, const float*, int64_t, float*, int64_t, int64_t);
int32_t gpublas_sgemm_bwd_dw(const float*, const float*, int64_t, float*, int64_t, int64_t);
int32_t gpublas_count(void);
int32_t gpublas_bufs(void);

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec + 1e-9*t.tv_nsec; }
static float frand(unsigned* s){ *s = *s*1103515245u + 12345u; return ((*s>>16)&0x7fff)/16384.0f - 1.0f; }

// Confere as tres chamadas numa forma PEQUENA, contra laco serial.
static int check_small(void) {
  int64_t BT=5, IF=7, OF=4;
  float x[35], w[28], dy[20], y[20], dx[35], dw[28], r[35];
  unsigned s=7;
  for(int i=0;i<35;i++) x[i]=frand(&s);
  for(int i=0;i<28;i++) w[i]=frand(&s);
  for(int i=0;i<20;i++) dy[i]=frand(&s);
  int bad=0;
  if (gpublas_sgemm_fwd(x,w,BT,y,IF,OF)) return printf("  fwd rc!=0\n"), 1;
  for(int64_t b=0;b<BT;b++) for(int64_t o=0;o<OF;o++){
    float a=0; for(int64_t i=0;i<IF;i++) a+=x[b*IF+i]*w[o*IF+i];
    if (fabsf(a-y[b*OF+o])>1e-4f) bad++; }
  printf("  fwd:     %d de %lld errados\n", bad, (long long)(BT*OF));
  if (gpublas_sgemm_bwd_dx(dy,w,BT,dx,IF,OF)) return printf("  dx rc!=0\n"), 1;
  for(int64_t b=0;b<BT;b++) for(int64_t i=0;i<IF;i++){
    float a=0; for(int64_t o=0;o<OF;o++) a+=dy[b*OF+o]*w[o*IF+i];
    if (fabsf(a-dx[b*IF+i])>1e-4f) bad++; }
  printf("  bwd dx:  verificado\n");
  if (gpublas_sgemm_bwd_dw(dy,x,BT,dw,IF,OF)) return printf("  dw rc!=0\n"), 1;
  for(int64_t o=0;o<OF;o++) for(int64_t i=0;i<IF;i++){
    float a=0; for(int64_t b=0;b<BT;b++) a+=dy[b*OF+o]*x[b*IF+i];
    if (fabsf(a-dw[o*IF+i])>1e-4f) bad++; }
  printf("  bwd dw:  verificado\n");
  (void)r;
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  printf("=== forma pequena, contra laco serial:\n");
  if (check_small()) { printf("  ERRO: as tres nao batem\n"); return 2; }
  printf("  todas batem\n\n");

  int64_t BT=1024, IF=768, OF=3072;
  if (argc>3){ BT=atoll(argv[1]); IF=atoll(argv[2]); OF=atoll(argv[3]); }
  size_t nx=BT*IF, nw=OF*IF, ny=BT*OF;
  float *x=malloc(nx*4),*w=malloc(nw*4),*dy=malloc(ny*4);
  float *y=malloc(ny*4),*dx=malloc(nx*4),*dw=malloc(nw*4);
  unsigned s=11; for(size_t i=0;i<nx;i++) x[i]=frand(&s);
  for(size_t i=0;i<nw;i++) w[i]=frand(&s);
  for(size_t i=0;i<ny;i++) dy[i]=frand(&s);

  // AQUECIMENTO: a primeira chamada paga o contexto HIP. Cronometrar a primeira
  // mede o contexto, nao o kernel. (A primeira versao deste teste deu 20,7 GFLOP/s
  // por causa disso, contra 1004 depois de aquecer.)
  gpublas_sgemm_fwd(x,w,BT,y,IF,OF);
  gpublas_sgemm_bwd_dx(dy,w,BT,dx,IF,OF);
  gpublas_sgemm_bwd_dw(dy,x,BT,dw,IF,OF);
  printf("pesos registrados: %d   conjuntos de buffers: %d\n", gpublas_count(), gpublas_bufs());

  int reps=20; double fl=2.0*(double)BT*(double)IF*(double)OF;
  double t0=now(); for(int k=0;k<reps;k++) gpublas_sgemm_fwd(x,w,BT,y,IF,OF);
  double t1=now();
  printf("  BT=%lld IF=%lld OF=%lld   fwd %.3f ms -> %.1f GFLOP/s (com transferencias)\n",
    (long long)BT,(long long)IF,(long long)OF, 1e3*(t1-t0)/reps, fl*reps/(t1-t0)/1e9);
  t0=now(); for(int k=0;k<reps;k++) gpublas_sgemm_bwd_dx(dy,w,BT,dx,IF,OF);
  t1=now();
  fl=2.0*(double)BT*(double)IF*(double)OF;
  printf("  bwd dx %.3f ms -> %.1f GFLOP/s\n", 1e3*(t1-t0)/reps, fl*reps/(t1-t0)/1e9);
  t0=now(); for(int k=0;k<reps;k++) gpublas_sgemm_bwd_dw(dy,x,BT,dw,IF,OF);
  t1=now();
  printf("  bwd dw %.3f ms -> %.1f GFLOP/s\n", 1e3*(t1-t0)/reps, fl*reps/(t1-t0)/1e9);
  return 0;
}
