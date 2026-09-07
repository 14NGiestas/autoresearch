! lib/fortran_adamw.f90 — AdamW optimizer step (hyp_34ea7c training loop).
!
! Standard AdamW with bias correction, decoupled weight decay:
!   m = b1*m + (1-b1)*g
!   v = b2*v + (1-b2)*g^2
!   mhat = m / (1 - b1^t),  vhat = v / (1 - b2^t)
!   p -= lr * (mhat / (sqrt(vhat) + eps) + wd * p)
! State (m, v) is caller-owned, same shape as p (train.py keeps fp32
! master states; our params are fp32 throughout, so no copy needed).

module fortran_adamw_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  implicit none
contains

  subroutine adamw_step(p, g, m, v, N, lr, b1, b2, eps, wd, t)
    integer(c_int), intent(in) :: N, t
    real(wp), intent(inout) :: p(:)
    real(wp), intent(in)  :: g(:)
    real(wp), intent(inout) :: m(:), v(:)
    real(wp), value :: lr, b1, b2, eps, wd
    integer :: i
    real(wp) :: mi, vi, mhat, vhat, bc1, bct1, bc2, bct2

    bc1 = 1.0_wp - b1
    bc2 = 1.0_wp - b2
    bct1 = 1.0_wp - b1**t
    bct2 = 1.0_wp - b2**t

    !$omp parallel do private(mi, vi, mhat, vhat)
    do i = 1, N
      mi = b1 * m(i) + bc1 * g(i)
      vi = b2 * v(i) + bc2 * g(i) * g(i)
      m(i) = mi
      v(i) = vi
      mhat = mi / bct1
      vhat = vi / bct2
      p(i) = p(i) - lr * (mhat / (sqrt(vhat) + eps) + wd * p(i))
    end do
    !$omp end parallel do
  end subroutine adamw_step

end module fortran_adamw_mod
