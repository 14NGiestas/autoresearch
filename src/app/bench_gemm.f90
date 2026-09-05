! app/bench_gemm.f90 — naive triple-loop vs OpenBLAS sgemm, depth-12 shape.
! Usage: fortran-fpm run bench_gemm
! Prints GFLOPS (2*BT*IF*OF/wall) for each + speedup. Set
! OMP_NUM_THREADS for the naive path, OPENBLAS_NUM_THREADS for sgemm.
program bench_gemm
  use iso_c_binding
  implicit none
  integer, parameter :: sp = c_float
  integer, parameter :: BT = 2048, IF = 768, OF = 768, REPS = 3
  real(sp), allocatable :: x(:), w(:), y(:)
  integer :: i, r, t0, t1, rate
  real(sp) :: gflops, t_naive, t_blas
  interface
    subroutine linear3d(x, w, y, B, T, IF, OF) bind(c, name='linear3d')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine linear3d_sgemm(x, w, y, B, T, IF, OF) &
        bind(c, name='linear3d_sgemm')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
  end interface

  allocate(x(BT*IF), w(OF*IF), y(BT*OF))
  do i = 1, BT*IF
    x(i) = real(mod(i * 1103515245 + 12345, 65536), sp) / 65536.0_sp - 0.5_sp
  end do
  do i = 1, OF*IF
    w(i) = real(mod(i * 1103515245 + 12345, 65536), sp) / 65536.0_sp - 0.5_sp
  end do

  call system_clock(t0, rate)
  do r = 1, REPS
    call linear3d(x, w, y, 1, BT, IF, OF)
  end do
  call system_clock(t1)
  t_naive = real(t1 - t0, sp) / real(rate, sp) / real(REPS, sp)
  gflops = 2.0_sp * real(BT, sp) * real(IF, sp) * real(OF, sp) / 1.0e9_sp
  print '(A,F10.4,A,F10.2)', "naive  s/token-ish: ", t_naive, &
      "  GFLOPS: ", gflops / max(t_naive, 1.0e-9_sp)

  call system_clock(t0, rate)
  do r = 1, REPS
    call linear3d_sgemm(x, w, y, 1, BT, IF, OF)
  end do
  call system_clock(t1)
  t_blas = real(t1 - t0, sp) / real(rate, sp) / real(REPS, sp)
  print '(A,F10.4,A,F10.2)', "sgemm  s/token-ish: ", t_blas, &
      "  GFLOPS: ", gflops / max(t_blas, 1.0e-9_sp)
  print '(A,F10.2)', "speedup: ", t_naive / max(t_blas, 1.0e-9_sp)
  print '(A,F10.6)', "checksum: ", sum(y) / real(BT * OF, sp)
end program bench_gemm
