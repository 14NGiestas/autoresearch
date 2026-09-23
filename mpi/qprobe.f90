! mpi/qprobe.f90 -- teste unitario LOCAL (sem MPI, ~10 ms) da matematica do
! --sync-fp16: round-trip bf16 e a soma EXATA do ponto fixo de 16 bits.
! Como rodar:  cd mpi && gfortran -O2 -o qprobe qprobe.f90 && ./qprobe
! (vale rodar antes de gastar um job: valida a conta, nao o encanamento do MPI)
program qprobe
  use, intrinsic :: iso_fortran_env, only: int16, int32, real32, real64
  implicit none
  integer, parameter :: wp = real32, N = 100000, NR = 2
  real(wp) :: g(N), g2(N), a(N), back(N), mean_exact(N)
  integer(int16) :: q(N)
  real(real64) :: gmax, step
  real(wp) :: err, rel, abserr
  integer :: i

  ! ---- 1) bf16: round-trip e erro relativo maximo ----
  call random_number(g)
  g = (g - 0.5_wp) * 1e-3_wp
  rel = 0.0_wp
  do i = 1, N
    call bf16_enc1(g(i), q(i))
    call bf16_dec1(q(i), back(i))
    if (abs(g(i)) > 0.0_wp) rel = max(rel, abs(back(i) - g(i))/abs(g(i)))
  end do
  print '(A,ES12.4,A,ES12.4)', "bf16  : erro relativo maximo ", rel, &
      "  (pior caso bf16 = 2^-8 = ", 2.0_wp**(-8), ")"

  ! ---- 2) fix: dois ranks, soma int16 exata, erro da media vs media exata ----
  call random_number(g);  g  = (g - 0.5_wp)*4e-3_wp
  call random_number(g2); g2 = (g2 - 0.5_wp)*4e-3_wp
  gmax = max(maxval(abs(g)), maxval(abs(g2)))
  step = real(gmax/32000.0_real64, wp)
  q = int(nint(g / (step*real(NR, wp))), int16)
  a = real(q, wp)*step           ! "soma" de 1 rank (q do rank 1 entra na mesma conta)
  q = int(nint(g2 / (step*real(NR, wp))), int16)
  a = a + real(q, wp)*step       ! soma dos 2 ranks (int16 exato) -> media
  mean_exact = (g + g2)/real(NR, wp)
  abserr = maxval(abs(a - mean_exact)); err = maxval(abs(a - mean_exact)) / gmax
  print '(A,ES12.4,A,ES12.4,A,ES12.4)', "fix   : erro abs maximo ", abserr, &
      "  relativo ao max|g| ", err, "  (limite teorico nranks*step/2 = ", real(NR,wp)*0.5_wp*step, ")"
  print '(A,I0,A)', "fix   : max|q| = ", maxval(abs(int(q))), &
      "  (tem de caber em int16, i.e. <= 32767)"

contains
  subroutine bf16_enc1(x, qq)
    real(wp), intent(in) :: x
    integer(int16), intent(out) :: qq
    integer(int32) :: i32
    i32 = transfer(x, i32)
    i32 = i32 + 32767_int32 + iand(ishft(i32, -16), 1_int32)
    qq = int(ishft(i32, -16), int16)
  end subroutine
  subroutine bf16_dec1(qq, x)
    integer(int16), intent(in) :: qq
    real(wp), intent(out) :: x
    integer(int32) :: i32
    i32 = ishft(int(qq, int32), 16)
    x = transfer(i32, x)
  end subroutine
end program qprobe
