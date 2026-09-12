! lib/fortran_arch.f90 — A ARQUITETURA em um lugar só, parametrizável no build.
!
! Antes, os mesmos números (N_LAYER = 12, VV = 8192, N_HEAD = 6, HD = 128) estavam
! duplicados em 10 arquivos, e mudar o tamanho do modelo exigia editar todos. Agora
! basta compilar com -cpp e defines:
!
!   fortran-fpm build --flag "-cpp -DARQ_VOCAB=8189 -DARQ_D_MODEL=96 -DARQ_N_LAYER=12 ..."
!
! Os defaults abaixo são EXATAMENTE os valores históricos, para que todo número já
! medido (bpb 2,40; 2,27/2,29 dos ramos; o passo de 5,77 s) continue válido.
!
! Restrição: D_MODEL tem de ser divisível por N_HEAD (HD = D_MODEL/N_HEAD).
! Para o modelo de varredura de 3M (d=98) usar N_HEAD=7 (HD=14) ou d=96 com 6.
module fortran_arch_mod
  implicit none

#ifndef ARQ_D_MODEL
#define ARQ_D_MODEL 768
#endif
#ifndef ARQ_N_HEAD
#define ARQ_N_HEAD 6
#endif
#ifndef ARQ_N_KV
#define ARQ_N_KV 6
#endif
#ifndef ARQ_N_LAYER
#define ARQ_N_LAYER 12
#endif
#ifndef ARQ_VOCAB
#define ARQ_VOCAB 8192
#endif
#ifndef ARQ_CTX
#define ARQ_CTX 2048
#endif
#ifndef ARQ_BOS
#define ARQ_BOS 8188
#endif

  integer, parameter :: D_MODEL = ARQ_D_MODEL
  integer, parameter :: N_HEAD = ARQ_N_HEAD
  integer, parameter :: N_KV = ARQ_N_KV
  integer, parameter :: HD = ARQ_D_MODEL / ARQ_N_HEAD
  integer, parameter :: N_LAYER = ARQ_N_LAYER
  integer, parameter :: VV = ARQ_VOCAB
  integer, parameter :: TT = ARQ_CTX
  integer, parameter :: BOS = ARQ_BOS
end module fortran_arch_mod
