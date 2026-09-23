! RMSNorm module — pure Fortran implementation
!
! Implements:
!   rmsnorm:  y = norm(x) * w  (with learned weight)
!   rmsnorm0: y = norm(x)       (no weight, used for pre-norms)
!
! All arrays are row-major flat real(wp) buffers.
! Parallelized with OpenMP.

module fortran_rmsnorm_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  implicit none
contains

  ! y = x / sqrt(mean(x^2) + eps) * w
  subroutine rmsnorm(x, w, y, NN, CC, eps_val)
    integer(c_int), intent(in) :: NN, CC
    real(wp), intent(in)  :: x(:), w(:)
    real(wp), intent(out) :: y(:)
    real(wp), value :: eps_val
    integer :: ii, jj
    real(wp) :: ss, inv

    !$omp parallel do private(ss, inv)
    do ii = 1, NN
      ss = 0.0_wp
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      inv = 1.0_wp / sqrt(ss / real(CC, wp) + eps_val)
      do jj = 1, CC
        y((ii-1)*CC + jj) = x((ii-1)*CC + jj) * inv * w(jj)
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm

  ! y = x / sqrt(mean(x^2) + eps)  (no weight)
  subroutine rmsnorm0(x, y, NN, CC, eps_val)
    integer(c_int), intent(in) :: NN, CC
    real(wp), intent(in)  :: x(:)
    real(wp), intent(out) :: y(:)
    real(wp), value :: eps_val
    integer :: ii, jj
    real(wp) :: ss, inv

    !$omp parallel do private(ss, inv)
    do ii = 1, NN
      ss = 0.0_wp
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      inv = 1.0_wp / sqrt(ss / real(CC, wp) + eps_val)
      do jj = 1, CC
        y((ii-1)*CC + jj) = x((ii-1)*CC + jj) * inv
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm0

end module fortran_rmsnorm_mod