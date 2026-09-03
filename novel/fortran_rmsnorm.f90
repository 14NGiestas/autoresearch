! Fortran RMSNorm kernel — called from Python via ctypes (novel custom numeric kernel).
! y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
subroutine rmsnorm_f(x, w, y, n, c, eps) bind(c, name="rmsnorm_f")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n, c
  real(c_float), value :: eps
  real(c_float), intent(in)  :: x(n * c)
  real(c_float), intent(in)  :: w(c)
  real(c_float), intent(out) :: y(n * c)
  integer :: i, j
  real(c_float) :: ss, inv
  do i = 1, n
    ss = 0.0_c_float
    do j = 1, c
      ss = ss + x((i - 1) * c + j) * x((i - 1) * c + j)
    end do
    inv = 1.0_c_float / sqrt(ss / real(c, kind=c_float) + eps)
    do j = 1, c
      y((i - 1) * c + j) = x((i - 1) * c + j) * inv * w(j)
    end do
  end do
end subroutine rmsnorm_f
