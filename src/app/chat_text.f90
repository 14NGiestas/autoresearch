! app/chat_text.f90 — text-in/text-out inference, pure Fortran, zero Python.
!
! Usage:
!   printf 'Alan Turing theorized that' | fortran-fpm run chat_text -- \
!       <tables_dir> <weights_dir> <n_gen> [temp=0] [seed=12345]
!
! Reads prompt bytes from stdin, encodes with the native BPE tokenizer
! (tokenizer_tables_mod/tokenizer_encode_mod, byte-exact vs tiktoken),
! prepends BOS (8188, matching training rows), greedy-generates with the
! depth-12 weights, and writes decoded bytes to stdout.
! Dims: D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192.

program chat_text
  use iso_c_binding
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use sample_mod, only: sample_token, sample_next
  use fortran_kv_mod, only: gpt_step
  use M_CLI2, only: set_args, sget, rget, iget, specified
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, BOS = 8188

  character(len=512) :: tdir, wdir, arg
  character(len=1) :: cb
  integer :: n_gen, u, ios, fsize, i, j, best, tc, step, nprompt
  integer :: ntot, d2, clen
  real(sp), allocatable :: ckv(:), cvv(:), out1(:)
  integer(c_int64_t) :: rng = 12345_c_int64_t
  real(sp) :: temp = 0.0_sp, topp = 1.0_sp, pres = 0.0_sp, freq = 0.0_sp
  integer :: nblock = 0
  integer, allocatable :: pbytes(:), pids(:), idx(:), obytes(:)
  integer :: nbytes
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, mchk
  d2 = HD / 2

  call set_args('--tables TABLES --weights WEIGHTS --n 20 --temp 0.0' // &
      ' --seed 12345 --topp 1.0 --pres 0.0 --freq 0.0 --nblock 0', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  chat_text - pure-Fortran text-in/text-out GPT inference', &
      'SYNOPSIS', &
      '  chat_text --tables DIR --weights DIR --n N [options] < stdin', &
      'OPTIONS', &
      '  --tables DIR  tokenizer tables (ranks.txt, unicode_*.txt)', &
      '  --weights DIR checkpoint .npy set (see export_weights.py)', &
      '  --n N         tokens to generate', &
      '  --temp T      temperature (0 = greedy)', &
      '  --seed S      RNG seed', &
      '  --topp P      nucleus cutoff (1 = off)', &
      '  --pres X      presence penalty', &
      '  --freq X      frequency penalty', &
      '  --nblock N    no-repeat n-gram size (0 = off)'], &
      version_text=[character(len=80) :: 'chat_text 1.0'])
  tdir = trim(sget('tables'))
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  temp = rget('temp')
  rng = int(iget('seed'), c_int64_t)
  topp = rget('topp')
  pres = rget('pres')
  freq = rget('freq')
  nblock = iget('nblock')
  if (.not. specified('tables') .or. .not. specified('weights') &
      .or. n_gen < 1) then
    print '(A)', 'require --tables DIR --weights DIR --n N>=1 (--help for all)'
    call exit(2)
  end if
  call load_tables(trim(tdir))
  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)

  ! prompt bytes from stdin. Pipes have no inquire-able size (gfortran
  ! returns 0), so read byte-by-byte to EOF instead. Buffer holds 1 MiB.
  open (newunit=u, file="/dev/stdin", access="stream", form="unformatted", &
      status="old", action="read", iostat=ios)
  if (ios /= 0) then
    print '(A)', "cannot read stdin"
    call exit(1)
  end if
  allocate(pbytes(1048576))
  fsize = 0
  do
    read (u, iostat=ios) cb
    if (ios /= 0) exit
    fsize = fsize + 1
    if (fsize > size(pbytes)) then
      print '(A)', "prompt exceeds 1 MiB"
      call exit(1)
    end if
    pbytes(fsize) = ichar(cb)
  end do
  close (u)

  call encode(pbytes, fsize, pids)
  nprompt = size(pids) + 1
  allocate(idx(nprompt + n_gen))
  idx(1) = BOS
  idx(2:nprompt) = pids
  deallocate(pbytes, pids)

  ! RoPE tables once for the full (prompt+gen) span; KV cache sized exact.
  ntot = nprompt + n_gen
  allocate(cos_b(ntot*(HD/2)), sin_b(ntot*(HD/2)))
  do i = 1, ntot
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      cos_b((i-1)*(HD/2)+j) = cos(ang)
      sin_b((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do
  allocate(ckv(N_LAYER*ntot*N_KV*HD))
  ckv = 0.0_sp
  allocate(cvv(N_LAYER*ntot*N_KV*HD))
  cvv = 0.0_sp
  clen = 0
  allocate(out1(B*VV))

  ! prefill prompt + generate, one cached step per token. Last forward
  ! needed is t=ntot-1 (its logits predict idx(ntot)); step ntot would
  ! write idx(ntot+1), out of bounds.
  do step = 1, ntot - 1
    tc = step
    call gpt_step(idx(tc:tc), cos_b((tc-1)*d2+1:), sin_b((tc-1)*d2+1:), &
        wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
        ckv, cvv, clen, ntot, out1, &
        B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
    if (step < nprompt) cycle   ! prompt steps only fill the cache
    mchk = maxval(out1)
    if (.not. (mchk == mchk)) then
      print '(A)', "NaN logit — abort"; call exit(1)
    end if
    best = sample_next(out1, VV, temp, topp, pres, freq, &
        idx(nprompt+1:), tc - nprompt, nblock, rng)
    idx(tc+1) = best - 1
  end do

  call decode(idx(nprompt+1:), n_gen, obytes, nbytes)
  open (newunit=u, file="/dev/stdout", access="stream", form="unformatted")
  do i = 1, nbytes
    write (u) char(obytes(i))
  end do
  close (u)

end program chat_text
