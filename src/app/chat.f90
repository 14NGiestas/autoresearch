! app/chat.f90 — real-weight autoregressive inference, pure Fortran.
!
! Loads depth-12 checkpoint weights exported to .npy by
! scripts/export_weights.py (flat float32, 0-based ids) via
! stdlib_io_npy::load_npy, then greedy-generates.
!
! Usage:
!   fortran-fpm run chat -- --weights DIR --n N --ids 1,2,3
!
! Prints generated token IDs (0-based) space-separated on one line.
! Pair with tiktoken (python) for encode/decode; see scripts/chat_driver.py.
!
! Dims below match checkpoint_depth12_step264_0.3750:
!   D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192

program chat
  use iso_c_binding
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use M_CLI2, only: set_args, sget, iget, igets, specified
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  character(len=512) :: wdir, arg
  integer :: n_gen, n_prompt, step, tc, i, j, best, ios, nargs
  integer :: seed = 777
  integer, allocatable :: idx(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, topv
  character(len=16) :: lstr

  call set_args('--weights WEIGHTS --n 20 --ids 1,2,3', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  chat - ids in/out autoregressive inference (greedy)', &
      'SYNOPSIS', &
      '  chat --weights DIR --n N --ids 1,2,3'], &
      version_text=[character(len=80) :: 'chat 1.0'])
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  idx = igets('ids')
  n_prompt = size(idx)
  if (.not. specified('weights') .or. n_gen < 1 .or. n_prompt < 1) then
    print '(A)', 'require --weights DIR --n N>=1 --ids I,... (--help)'
    call exit(2)
  end if

  ! ---- weights (shared loader module) ------------------------------------
  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)

  ! ---- generate ----------------------------------------------------------
  do step = 1, n_gen
    tc = n_prompt + step - 1
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

    best = 1; topv = outp((tc-1)*VV+1)
    do i = 2, VV
      if (outp((tc-1)*VV+i) > topv) then
        topv = outp((tc-1)*VV+i); best = i
      end if
    end do
    if (.not. (topv == topv)) then
      print '(A)', "NaN logit — abort"; call exit(1)
    end if
    idx = [idx, best - 1]   ! 1-based position -> 0-based token id
  end do

  do i = n_prompt + 1, n_prompt + n_gen
    if (i > n_prompt + 1) write (*, '(A)', advance='no') ' '
    write (*, '(I0)', advance='no') idx(i)
  end do
  print *

end program chat
