! app/bench_attn_split.f90 — onde o TEMPO da atencao vai, por fase.
!
! Por que existe: a analise estatica (scripts/flop_breakdown.py) mediu MACs, e a
! medicao do cap mediu que UMA transcendental por score custa ~45% do passo. Mas
! MAC nao e' tempo em regime de forma ruim, e o softmax esta' DENTRO da atencao.
!
! O que decide o desenho do swap para GPU: se o softmax ficar na CPU, o teto do
! ganho e' ~1,8x, nao 51x. Entao a unidade do swap tem que ser o BLOCO inteiro
! (QK^T + softmax + AV), e para saber isso precisa separar as fases.
!
! Este app roda as tres fases do attn_sgemm explicitamente -- sgemm1, o laco de
! mascara+softmax, sgemm2 -- e cronometra cada uma. Nao instrumenta a biblioteca,
! entao nao muda o caminho de treino nem o custo dos jobs que estao rodando.
!
! Uso: bench_attn_split [T] [H] [K_H] [D] [REPS]
program bench_attn_split
  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only: real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: sgemm
  use M_CLI2, only: set_args, iget
  implicit none
  integer :: T, H, K_H, D, REPS
  integer :: jj, ii, rep
  real(wp), allocatable :: q(:), k(:), v(:), y(:), S(:)
  real(wp) :: scale, mx, sm, inv, tmp
  real(real64) :: tg1, tg2, ts, ttot
  integer(c_int64_t) :: m, n, kk, lda, ldb, ldc
  integer :: c0, c1, crate

  call set_args('--T 1024 --H 6 --K_H 2 --D 16 --reps 20', help_text=[character(len=90) :: &
      'NAME', '  bench_attn_split - fases da atencao em separado', &
      '', 'SYNOPSIS', '  bench_attn_split [--T N] [--H N] [--K_H N] [--D N] [--reps N]'])
  T = iget('T'); H = iget('H'); K_H = iget('K_H'); D = iget('D'); REPS = iget('reps')

  allocate (q(T*H*D), k(T*K_H*D), v(T*K_H*D), y(T*H*D), S(T*T))
  call random_number(q); call random_number(k); call random_number(v)
  q = q - 0.5_wp; k = k - 0.5_wp; v = v - 0.5_wp
  scale = 1.0_wp / sqrt(real(D, wp))
  m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
  lda = int(K_H*D, c_int64_t); ldb = int(H*D, c_int64_t); ldc = int(T, c_int64_t)

  print '(A,I0,A,I0,A,I0,A,I0,A,I0)', "shape: T=", T, " H=", H, " K_H=", K_H, &
      " head_dim=", D, " reps=", REPS

  tg1 = 0.0_real64; ts = 0.0_real64; tg2 = 0.0_real64
  do rep = 1, REPS + 3
    ! ---- fase 1: S = Q K^T escalado (um sgemm por batch/cabeca) ----
    call system_clock(c0, crate)
    do ii = 1, H
      call sgemm('T', 'N', m, n, kk, scale, k, lda, q, ldb, 0.0_wp, S, ldc)
    end do
    call system_clock(c1, crate)
    if (rep > 3) tg1 = tg1 + real(c1 - c0, real64)/real(crate, real64)

    ! ---- fase 2: mascara causal + softmax (a parte elementwise, com exp) ----
    call system_clock(c0, crate)
    do ii = 1, T
      mx = -huge(1.0_wp)
      do jj = 1, ii
        if (S((ii-1)*T + jj) > mx) mx = S((ii-1)*T + jj)
      end do
      sm = 0.0_wp
      do jj = 1, ii
        S((ii-1)*T + jj) = exp(S((ii-1)*T + jj) - mx)
        sm = sm + S((ii-1)*T + jj)
      end do
      inv = 1.0_wp / sm
      do jj = 1, ii
        S((ii-1)*T + jj) = S((ii-1)*T + jj)*inv
      end do
      do jj = ii + 1, T
        S((ii-1)*T + jj) = 0.0_wp
      end do
    end do
    call system_clock(c1, crate)
    if (rep > 3) ts = ts + real(c1 - c0, real64)/real(crate, real64)

    ! ---- fase 3: Y = P V ----
    call system_clock(c0, crate)
    do ii = 1, H
      call sgemm('N', 'N', int(D, c_int64_t), n, m, 1.0_wp, v, lda, S, ldc, &
          0.0_wp, y, int(H*D, c_int64_t))
    end do
    call system_clock(c1, crate)
    if (rep > 3) tg2 = tg2 + real(c1 - c0, real64)/real(crate, real64)
  end do

  tg1 = 1000.0_real64*tg1/REPS
  ts = 1000.0_real64*ts/REPS
  tg2 = 1000.0_real64*tg2/REPS
  ttot = tg1 + ts + tg2
  print '(A,F9.3,A,F7.2,A)', "  sgemm  QK^T   ", tg1, " ms  ", 100.0_real64*tg1/ttot, "%"
  print '(A,F9.3,A,F7.2,A)', "  softmax+mask  ", ts, " ms  ", 100.0_real64*ts/ttot, "%"
  print '(A,F9.3,A,F7.2,A)', "  sgemm  PV     ", tg2, " ms  ", 100.0_real64*tg2/ttot, "%"
  print '(A,F9.3,A)', "  TOTAL         ", ttot, " ms  (das 3 fases, 1 batch)"
  print '(A)', "note: o treino faz isto por 12 camadas, e o passo real e' ~1211 ms"
  print '(A,F6.2,A)', "      entao as 3 fases x12 dao ~", 12.0_real64*ttot, &
      " ms; se for perto de 1211, a atencao explica o passo quase todo."
  print '(A,F6.2)', "      razao (sgemm total)/(softmax) = ", (tg1 + tg2)/max(1e-9_real64, ts)
  ! consome o resultado para o compilador nao remover nada
  tmp = sum(y) + sum(S)
  if (ieee_is_nan(tmp)) print '(A)', "  (aviso: NaN)"
end program bench_attn_split
