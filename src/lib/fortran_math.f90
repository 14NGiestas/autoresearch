! lib/fortran_math.f90 — one fast tanh, with a measured error.
!
! The attention soft cap uses tanh: s = cap * tanh(s / cap). A call to the exact
! tanh costs time in the hot loop, so this module holds a fast one.
!
! The coefficients come from fastGPT (certik/fastGPT, gpt2.f90). The polynomial
! is odd in x, of degree 17, and it is valid on [-5, 5]. Outside that range the
! function returns +1 or -1, because tanh saturates there.
!
! The approximation changes the result. That is the point of an approximation.
! Two rules follow:
!   1. The forward and the backward must call the SAME function. A test of the
!      gradient against finite differences then proves that the pair agrees. It
!      does not prove that the value equals the exact tanh.
!   2. A separate test measures the error against the exact tanh. The test
!      reports the number, so the error is a fact and not a hope.
!
! A cheaper alternative is the Pade [2/2] form x*(27 + x*x)/(27 + 9*x*x). Its
! worst case error is about 0.025, which is 1000 times worse. We use the
! polynomial.
module fortran_math_mod
  use fortran_kinds_mod, only: wp
  implicit none
  private

  public :: fast_tanh, fast_softcap

  real(wp), parameter :: SAT = 5.0_wp

contains

  ! tanh(x), approximately, for x in [-5, 5]. Returns +-1 outside that range.
  elemental function fast_tanh(x) result(y)
    real(wp), intent(in) :: x
    real(wp) :: y, x2
    if (x > SAT) then
      y = 1.0_wp
    else if (x < -SAT) then
      y = -1.0_wp
    else
      x2 = x*x
      y = x*(0.98569772605911309407_wp + x2*(-0.2794500993392901382_wp &
          + x2*(6.8280504526399188164e-2_wp + x2*(-1.0972014877337651823e-2_wp &
          + x2*(1.1132367134444316902e-3_wp + x2*(-7.018851897305717565e-5_wp &
          + x2*(2.656616768082727089e-6_wp + x2*(-5.5138381821615909058e-8_wp &
          + x2*4.8162484477588665996e-10_wp))))))))
    end if
  end function fast_tanh

  ! The attention soft cap: cap * tanh(x / cap). A cap of 0 or less means no cap.
  elemental function fast_softcap(x, cap) result(y)
    real(wp), intent(in) :: x, cap
    real(wp) :: y
    if (cap <= 0.0_wp) then
      y = x
    else
      y = cap*fast_tanh(x/cap)
    end if
  end function fast_softcap

end module fortran_math_mod
