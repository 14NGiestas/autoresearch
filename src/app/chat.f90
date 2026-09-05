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
  use M_CLI2, only: set_args, sget, iget, igets, specified, rget
  use sample_mod, only: sample_next
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  character(len=512) :: wdir, arg
  integer :: n_gen, n_prompt, step, tc, i, j, best, ios, nargs
  logical :: dostats = .false.
  integer :: ca, cb, crate
  integer(c_int64_t) :: cms_tot = 0
  integer :: seed = 777
  integer, allocatable :: idx(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, topv, mchk
  real(sp) :: temp = 0.0_sp, topp = 1.0_sp, pres = 0.0_sp, freq = 0.0_sp
  integer :: nblock = 0
  integer(c_int64_t) :: rng = 12345_c_int64_t
  character(len=16) :: lstr

  call set_args('--weights WEIGHTS --n 20 --ids 1,2,3 --stats F' // &
      ' --temp 0.0 --seed 12345 --topp 1.0 --pres 0.0 --freq 0.0' // &
      ' --nblock 0', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  chat - ids in/out autoregressive inference + sampling', &
      'SYNOPSIS', &
      '  chat --weights DIR --n N --ids 1,2,3 [options]', &
      'OPTIONS', &
      '  --stats T     print total ms + tok/s to stderr', &
      '  --temp T --seed S --topp P --pres X --freq X --nblock N'], &
      version_text=[character(len=80) :: 'chat 1.1'])
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  idx = igets('ids')
  temp = rget('temp')
  rng = int(iget('seed'), c_int64_t)
  topp = rget('topp')
  pres = rget('pres')
  freq = rget('freq')
  nblock = iget('nblock')
  n_prompt = size(idx)
  if (.not. specified('weights') .or. n_gen < 1 .or. n_prompt < 1) then
    print '(A)', 'require --weights DIR --n N>=1 --ids I,... (--help)'
    call exit(2)
  end if

  ! ---- weights (shared loader module) ------------------------------------
  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)

  ! ---- generate (full recompute per step; chat_text has the KV path) ---
  dostats = specified('stats')
  if (dostats) call system_clock(count_rate=crate)
  if (dostats) call system_clock(ca)
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

    mchk = maxval(outp((tc-1)*VV+1:tc*VV))
    if (.not. (mchk == mchk)) then
      print '(A)', "NaN logit — abort"; call exit(1)
    end if
    best = sample_next(outp((tc-1)*VV+1:), VV, temp, topp, pres, freq, &
        idx(n_prompt+1:), tc - n_prompt, nblock, rng)
    idx = [idx, best - 1]   ! 1-based position -> 0-based token id
  end do
  if (dostats) then
    call system_clock(cb)
    cms_tot = cb - ca
    write (0, '(A,F10.1,A,F8.2)') "stats: total_ms=", &
        1000.0 * real(cms_tot) / real(crate), " tok_s=", &
        real(n_gen) / max(1.0e-9, real(cms_tot) / real(crate))
  end if

  do i = n_prompt + 1, n_prompt + n_gen
    if (i > n_prompt + 1) write (*, '(A)', advance='no') ' '
    write (*, '(I0)', advance='no') idx(i)
  end do
  print *

end program chat
