! test_math.f90 — the fast tanh keeps its promise.
!
! Two different claims, tested separately:
!   1. The approximation is CLOSE to the exact tanh. This test measures the
!      error and prints it. A wrong coefficient shows up here.
!   2. The soft cap is the identity when the cap is off, and it saturates when
!      the cap is small. That is the behavior the attention code relies on.
!
! The gradient test lives with the attention kernels, because it must use the
! same function in the forward and in the backward.
program test_math
  use, intrinsic :: iso_fortran_env, only: real64
  use fortran_kinds_mod, only: wp
  use fortran_math_mod, only: fast_tanh, fast_softcap
  implicit none
  integer :: nfail, i
  real(wp) :: x, cap, y
  real(real64) :: xe, ex, e, worst, worst_at, wd, worst_d

  nfail = 0
  worst = 0.0_real64; worst_at = 0.0_real64
  worst_d = 0.0_real64
  do i = -2000, 2000
    x = real(i, wp)*0.005_wp
    xe = real(x, real64)
    ex = (exp(xe) - exp(-xe))/(exp(xe) + exp(-xe))     ! exact, in real64
    e = abs(real(fast_tanh(x), real64) - ex)
    ! O erro que importa e' o que a atencao ve: o tanh entra no softmax, entao
    ! pesa-lo pela derivada (1 - tanh^2) diz quanto do erro chega ao resultado.
    wd = e*(1.0_real64 - ex*ex)
    if (e > worst) then
      worst = e; worst_at = real(x, real64)
    end if
    if (wd > worst_d) worst_d = wd
  end do
  write (*, '(A,ES12.3,A,F7.3)') '  erro maximo do fast_tanh = ', worst, ' em x = ', worst_at
  write (*, '(A,ES12.3)') '  erro ponderado pela derivada (o que o softmax ve) = ', worst_d
  ! O limiar e' o valor MEDIDO com folga, nao um desejo. O polinomio do fastGPT
  ! erra mais onde o tanh satura, que e' onde a derivada e' pequena.
  call check(worst < 5.0e-3_real64, 'fast_tanh tem erro < 5e-3 (medido: 3.05e-3)')
  call check(worst_d < 5.0e-3_real64, 'o erro que propaga tambem e < 5e-3 (medido: 2.7e-3)')
  call check(abs(fast_tanh(6.0_wp) - 1.0_wp) <= 0.0_wp, 'satura em +1 acima de 5')
  call check(abs(fast_tanh(-6.0_wp) + 1.0_wp) <= 0.0_wp, 'satura em -1 abaixo de -5')
  call check(abs(fast_tanh(0.0_wp)) <= 0.0_wp, 'fast_tanh(0) = 0')
  call check(abs(fast_tanh(1.0_wp) + fast_tanh(-1.0_wp)) <= 1.0e-6_wp, 'e impar')

  ! o cap: desligado = identidade; pequeno = satura
  call check(abs(fast_softcap(3.5_wp, 0.0_wp) - 3.5_wp) <= 0.0_wp, 'cap 0 = identidade')
  y = fast_softcap(100.0_wp, 2.0_wp)
  call check(abs(y - 2.0_wp) < 1.0e-3_wp, 'cap 2 satura em 2 para entrada grande')
  y = fast_softcap(-100.0_wp, 2.0_wp)
  call check(abs(y + 2.0_wp) < 1.0e-3_wp, 'cap 2 satura em -2 para entrada negativa')
  call check(abs(fast_softcap(0.0_wp, 1.0_wp)) <= 0.0_wp, 'cap mantem o zero')

  if (nfail /= 0) error stop 'test_math: FALHOU'
  write (*, '(A)') 'test_math: OK'

contains
  subroutine check(ok, what)
    logical, intent(in) :: ok
    character(*), intent(in) :: what
    if (ok) then
      write (*, '(A,A)') '  ok   ', what
    else
      write (*, '(A,A)') '  FAIL ', what
      nfail = nfail + 1
    end if
  end subroutine check
end program test_math
