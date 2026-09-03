! RMSNorm module — pure Fortran implementation
!
! Implements:
!   rmsnorm:  y = norm(x) * w  (with learned weight)
!   rmsnorm0: y = norm(x)       (no weight, used for pre-norms)
!
! All arrays are row-major (C order) flat float32 buffers.
! Parallelized with OpenMP.

module fortran_rmsnorm_mod
  use iso_c_binding
  implicit none
contains

  ! y = x / sqrt(mean(x^2) + eps) * w
  subroutine rmsnorm(x, w, y, NN, CC, eps_val) bind(c, name='rmsnorm')
    integer(c_int), intent(in) :: NN, CC
    real(c_float), intent(in)  :: x(NN*CC), w(CC)
    real(c_float), intent(out) :: y(NN*CC)
    real(c_float), value :: eps_val
    integer :: ii, jj
    real(c_float) :: ss, inv

    !$omp parallel do private(ss, inv)
    do ii = 1, NN
      ss = 0.0_c_float
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      inv = 1.0_c_float / sqrt(ss / real(CC) + eps_val)
      do jj = 1, CC
        y((ii-1)*CC + jj) = x((ii-1)*CC + jj) * inv * w(jj)
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm

  ! y = x / sqrt(mean(x^2) + eps)  (no weight)
  subroutine rmsnorm0(x, y, NN, CC, eps_val) bind(c, name='rmsnorm0')
    integer(c_int), intent(in) :: NN, CC
    real(c_float), intent(in)  :: x(NN*CC)
    real(c_float), intent(out) :: y(NN*CC)
    real(c_float), value :: eps_val
    integer :: ii, jj
    real(c_float) :: ss, inv

    !$omp parallel do private(ss, inv)
    do ii = 1, NN
      ss = 0.0_c_float
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      inv = 1.0_c_float / sqrt(ss / real(CC) + eps_val)
      do jj = 1, CC
        y((ii-1)*CC + jj) = x((ii-1)*CC + jj) * inv
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm0

end module fortran_rmsnorm_mod