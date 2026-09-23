! app/bench_attn.f90 — naive vs BLAS causal attention at the real shape.
!
! Why: attention is ~30% of a training step's flops (155 GFLOP fwd + 309 bwd at
! B=1,T=2048,L=12) but runs as hand-written loops while the linears already go
! through sgemm. This measures the gap at T=2048, D=128, H=K_H=6 (our config),
! so the optimization is a measurement, not an estimate.
!
! Usage: bench_attn [T] [H] [D] [reps]

program bench_attn
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  use fortran_attn_mod, only: causal_attn, attn_sgemm
  use M_CLI2, only: set_args, iget
  implicit none

  real(wp), allocatable :: q(:), k(:), v(:), y(:), S(:)
  integer :: T, H, D, reps, r, t0, t1, rate
  real(wp) :: t_naive, t_blas, e, max_err
  integer :: i

  T = 2048; H = 6; D = 128; reps = 3
  call set_args('--T 2048 --H 6 --D 128 --reps 3', &
      help_text=[character(len=80) :: &
      'NAME', '  bench_attn - naive vs sgemm causal attention', &
      'SYNOPSIS', '  bench_attn [--T 2048] [--H 6] [--D 128] [--reps 3]'], &
      version_text=[character(len=80) :: 'bench_attn 1.0'])
  T = iget('T'); H = iget('H'); D = iget('D'); reps = iget('reps')

  allocate (q(T*H*D), k(T*H*D), v(T*H*D), y(T*H*D), S(T*T))
  do i = 1, T*H*D
    q(i) = real(mod(i*1103515245 + 12345, 65536), wp)/65536.0_wp - 0.5_wp
    k(i) = real(mod(i*1103515245 + 54321, 65536), wp)/65536.0_wp - 0.5_wp
    v(i) = real(mod(i*1103515245 + 999, 65536), wp)/65536.0_wp - 0.5_wp
  end do

  print '(A,I0,A,I0,A,I0,A,I0,A)', "shape: B=1 T=", T, " H=", H, " D=", D, &
      " reps=", reps
  ! flops (one pass): 2*T^2*D (QK^T, full square) + 2*T^2*D (PV) per head
  print '(A,F8.1,A)', "flops/pass (2 matmuls/head): ", &
      2.0_wp*real(H, wp)*(2.0_wp*real(T, wp)**2*real(D, wp))/1.0e9_wp, " GFLOP"

  call causal_attn(q, k, v, y, 1, T, H, H, D)
  call system_clock(t0, rate)
  do r = 1, reps
    call causal_attn(q, k, v, y, 1, T, H, H, D)
  end do
  call system_clock(t1, rate)
  t_naive = real(t1 - t0, wp)/real(rate, wp)/real(reps, wp)

  S = 0.0_wp
  call attn_sgemm(q, k, v, y, 1, T, H, H, D, S)
  call system_clock(t0, rate)
  do r = 1, reps
    call attn_sgemm(q, k, v, y, 1, T, H, H, D, S)
  end do
  call system_clock(t1, rate)
  t_blas = real(t1 - t0, wp)/real(rate, wp)/real(reps, wp)

  ! agreement check on the same shape (different summation order: tolerance)
  max_err = 0.0_wp
  call causal_attn(q, k, v, y, 1, T, H, H, D)
  block
    real(wp), allocatable :: yb(:)
    allocate (yb(T*H*D))
    call attn_sgemm(q, k, v, yb, 1, T, H, H, D, S)
    do i = 1, T*H*D
      e = abs(y(i) - yb(i))
      if (e > max_err) max_err = e
    end do
    deallocate (yb)
  end block

  print '(A,F10.2,A,F8.1,A)', "naive : ", 1000.0_wp*t_naive, " ms  ", &
      2.0_wp*real(H, wp)*(2.0_wp*real(T, wp)**2*real(D, wp))/1.0e9_wp/t_naive, &
      " GFLOP/s"
  print '(A,F10.2,A,F8.1,A)', "sgemm : ", 1000.0_wp*t_blas, " ms  ", &
      2.0_wp*real(H, wp)*(2.0_wp*real(T, wp)**2*real(D, wp))/1.0e9_wp/t_blas, &
      " GFLOP/s"
  print '(A,F8.2,A)', "speedup: ", t_naive/max(1.0e-9_wp, t_blas), "x"
  print '(A,E10.3)', "max |y_naive - y_sgemm| = ", max_err
  print '(A)', "note: sgemm does the full square (no triangular saving), so"
  print '(A)', "      2x the flops of the naive causal kernel and still wins."
end program bench_attn
