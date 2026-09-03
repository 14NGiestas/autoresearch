! app/infer.f90 — greedy autoregressive inference demo, pure Fortran.
!
! Tiny random-weight model (deterministic seed): proves the full
! tokenize -> gpt_forward -> sample loop runs end-to-end in Fortran.
! Output is token IDs (no BPE decoder here; that lives in Python).
!
! Build + run:
!   fortran-fpm run infer
!
! Expected: 8 generated IDs printed, all finite, varying (non-degenerate).

program infer
  use iso_c_binding
  use fortran_gpt_mod
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 32, N_HEAD = 4, N_KV = 2, HD = 8
  integer, parameter :: N_LAYER = 2, VV = 256, T0 = 4, N_GEN = 8

  integer :: seed = 1234
  integer :: step, tc, i, j, best
  integer, allocatable :: idx(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), c_q(:), c_k(:), c_v(:), c_pr(:)
  real(sp), allocatable :: c_fc(:), c_pr2(:), lm(:), outp(:)
  real(sp) :: theta, ang, topv

  ! ---- deterministic weights: per-layer stacks (distinct random layers) ---
  allocate(wte(VV*D));          call fillv(wte, VV*D, 0.10_sp)
  allocate(c_q(N_LAYER*N_HEAD*HD*D))
  call fillv(c_q, N_LAYER*N_HEAD*HD*D, 0.05_sp)
  allocate(c_k(N_LAYER*N_KV*HD*D))
  call fillv(c_k, N_LAYER*N_KV*HD*D, 0.05_sp)
  allocate(c_v(N_LAYER*N_KV*HD*D))
  call fillv(c_v, N_LAYER*N_KV*HD*D, 0.05_sp)
  allocate(c_pr(N_LAYER*D*N_HEAD*HD))
  call fillv(c_pr, N_LAYER*D*N_HEAD*HD, 0.05_sp)
  allocate(c_fc(N_LAYER*4*D*D))
  call fillv(c_fc, N_LAYER*4*D*D, 0.05_sp)
  allocate(c_pr2(N_LAYER*D*4*D))
  call fillv(c_pr2, N_LAYER*D*4*D, 0.05_sp)
  allocate(lm(VV*D));           call fillv(lm, VV*D, 0.05_sp)

  ! ---- prompt ------------------------------------------------------------
  allocate(idx(T0))
  idx = [11, 22, 33, 44]
  print '(A,4I5)', "prompt :", idx

  ! ---- autoregressive loop ------------------------------------------------
  do step = 1, N_GEN
    tc = T0 + step - 1

    ! RoPE tables for current length: theta_d = 10000^(-2d/HD)
    if (allocated(cos_b)) deallocate(cos_b, sin_b)
    allocate(cos_b(tc*(HD/2)), sin_b(tc*(HD/2)))
    do i = 1, tc
      do j = 1, HD/2
        theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
        ang = real(i-1, sp) * theta
        cos_b((i-1)*(HD/2)+j) = cos(ang)
        sin_b((i-1)*(HD/2)+j) = sin(ang)
      end do
    end do

    if (allocated(outp)) deallocate(outp)
    allocate(outp(B*tc*VV))

    call gpt_forward(idx, cos_b, sin_b, &
        wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
        outp, B, tc, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)

    ! greedy sample at last position
    best = 1; topv = outp((tc-1)*VV+1)
    do i = 2, VV
      if (outp((tc-1)*VV+i) > topv) then
        topv = outp((tc-1)*VV+i); best = i
      end if
    end do
    if (.not. (topv == topv)) then
      print '(A)', "NaN logit — abort"; call exit(1)
    end if
    ! best is a 1-based position; token IDs are 0-based.
    print '(A,I2,A,I4,A,F9.4)', "step ", step, "  token ", best - 1, "  logit ", topv

    idx = [idx, best - 1]   ! auto-reallocate on assignment (F2003)
  end do

  print '(A)', "generated:"
  print '(8I5)', idx(T0+1:)

contains

  function frand() result(r)
    real(sp) :: r
    seed = mod(seed * 1103515245 + 12345, 2147483647)
    r = real(mod(seed / 65536, 32768), sp) / 32768.0_sp - 0.5_sp
  end function

  subroutine fillv(x, n, s)
    real(sp), intent(out) :: x(*)
    integer, intent(in) :: n
    real(sp), intent(in) :: s
    integer :: k
    do k = 1, n
      x(k) = frand() * s
    end do
  end subroutine

end program infer
