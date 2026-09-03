! app/repl.f90 — interactive pure-Fortran chat REPL. Zero Python.
!
! Usage:
!   fortran-fpm run repl -- <tables_dir> <weights_dir> [n_gen=40]
!
! Loads tables + depth-12 weights once, then loops: reads a line of text
! from stdin, encodes (native BPE), prepends BOS, greedy-generates n_gen
! tokens, decodes to stdout. Empty line (or EOF) quits. Prompts go to
! stderr so stdout stays clean for piping.
! Dims: D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192, BOS=8188.

program repl
  use iso_c_binding
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use sample_mod, only: sample_token
  use fortran_kv_mod, only: gpt_step
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, BOS = 8188
  integer, parameter :: MAXT = 2048   ! checkpoint sequence_len; cache cap

  character(len=512) :: tdir, wdir, arg
  character(len=8192) :: linebuf
  integer :: n_gen, u, ios, i, j, best, tc, step, nprompt, nbytes, nlen
  integer :: ntot, d2, clen
  real(sp), allocatable :: ckv(:), cvv(:), out1(:)
  integer(c_int64_t) :: rng = 12345_c_int64_t
  real(sp) :: temp = 0.0_sp
  integer, allocatable :: pbytes(:), pids(:), idx(:), obytes(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, mchk

  n_gen = 40
  if (command_argument_count() < 2) then
    write (0, '(A)') "usage: repl <tables_dir> <weights_dir> [n_gen] [temp] [seed]"
    call exit(2)
  end if
  call get_command_argument(1, tdir)
  call get_command_argument(2, wdir)
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg)
    read (arg, *) n_gen
  end if
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg)
    read (arg, *) temp
  end if
  if (command_argument_count() >= 5) then
    call get_command_argument(5, arg)
    read (arg, *) rng
  end if

  write (0, '(A)') "loading tables + weights (one-time, ~10 s) ..."
  call load_tables(trim(tdir))
  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
  d2 = HD / 2
  write (0, '(A)') "ready. Empty line quits."

  do
    write (0, '(A)', advance='no') ">>> "
    read (*, '(A)', iostat=ios) linebuf
    if (ios /= 0) exit
    nlen = len_trim(linebuf)
    if (nlen == 0) exit
    allocate(pbytes(nlen))
    do i = 1, nlen
      pbytes(i) = ichar(linebuf(i:i))
    end do

    call encode(pbytes, nlen, pids)
    deallocate(pbytes)
    nprompt = size(pids) + 1
    ntot = nprompt + n_gen
    if (ntot > MAXT) then
      write (0, '(A,I0,A)') "prompt+gen ", ntot, " exceeds 2048; shorten input"
      deallocate(pids)
      cycle
    end if
    allocate(idx(ntot))
    idx(1) = BOS
    idx(2:nprompt) = pids
    deallocate(pids)

    if (allocated(cos_b)) deallocate(cos_b, sin_b)
    allocate(cos_b(ntot*(HD/2)), sin_b(ntot*(HD/2)))
    do i = 1, ntot
      do j = 1, HD/2
        theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
        ang = real(i-1, sp) * theta
        cos_b((i-1)*(HD/2)+j) = cos(ang)
        sin_b((i-1)*(HD/2)+j) = sin(ang)
      end do
    end do
    if (allocated(ckv)) deallocate(ckv, cvv, out1)
    allocate(ckv(N_LAYER*MAXT*N_KV*HD))
    ckv = 0.0_sp
    allocate(cvv(N_LAYER*MAXT*N_KV*HD))
    cvv = 0.0_sp
    clen = 0
    allocate(out1(B*VV))

    do step = 1, ntot
      tc = step
      call gpt_step(idx(tc:tc), cos_b((tc-1)*d2+1:), sin_b((tc-1)*d2+1:), &
          wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ckv, cvv, clen, MAXT, out1, &
          B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
      if (step < nprompt) cycle
      mchk = maxval(out1)
      if (.not. (mchk == mchk)) then
        write (0, '(A)') "NaN logit — abort"
        call exit(1)
      end if
      best = sample_token(out1, VV, temp, rng)
      idx(tc+1) = best - 1
    end do
    deallocate(cos_b, sin_b, outp)

    call decode(idx(nprompt+1:), n_gen, obytes, nbytes)
    open (newunit=u, file="/dev/stdout", access="stream", form="unformatted")
    do i = 1, nbytes
      write (u) char(obytes(i))
    end do
    write (u) char(10)
    close (u)
    deallocate(idx, obytes)
  end do
  write (0, '(A)') "bye."

end program repl
