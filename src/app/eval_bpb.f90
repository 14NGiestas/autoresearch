! app/eval_bpb.f90 — bits-per-byte evaluation, pure Fortran forward.
!
! Mirrors prepare.py:evaluate_bpb's forward side: for each packed row of
! T+1 token ids (0-based), runs gpt_forward on ids(1:T) and prints the
! per-position NLLs (nats, space-separated, one line per row). Targets are
! ids(2:T+1). Masking by token byte-length and the nats->bits conversion
! happen in the python driver (scripts/eval_driver.py), which owns the
! tokenizer and the BOS best-fit packing replica.
!
! Usage:
!   fortran-fpm run eval_bpb -- <weights_dir> <rows_file>
!
! Dims match checkpoint_depth12_step264_0.3750 (D=768 NH=6 NKV=6 HD=128
! NL=12 VV=8192, T=2048).

program eval_bpb
  use iso_c_binding
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, TT = 2048

  character(len=512) :: wdir, rowsfile, line
  integer :: ios, unit, tc, i, j, tgt
  integer :: idx(TT)
  integer, allocatable :: full(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, m, s, nll
  character(len=65536) :: buf

  if (command_argument_count() < 2) then
    print '(A)', "usage: eval_bpb <weights_dir> <rows_file>"
    call exit(2)
  end if
  call get_command_argument(1, wdir)
  call get_command_argument(2, rowsfile)

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)

  allocate(full(TT + 1))
  open (newunit=unit, file=trim(rowsfile), status='old', action='read')
  do
    read (unit, '(A)', iostat=ios) buf
    if (ios /= 0) exit
    read (buf, *, iostat=ios) full
    if (ios /= 0) then
      print '(A)', "bad row (need TT+1 ids)"
      call exit(1)
    end if
    idx = full(1:TT)

    ! RoPE tables for TT
    allocate(cos_b(TT*(HD/2)), sin_b(TT*(HD/2)))
    do i = 1, TT
      do j = 1, HD/2
        theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
        ang = real(i-1, sp) * theta
        cos_b((i-1)*(HD/2)+j) = cos(ang)
        sin_b((i-1)*(HD/2)+j) = sin(ang)
      end do
    end do
    allocate(outp(B*TT*VV))

    call gpt_forward(idx, cos_b, sin_b, &
        wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
        outp, B, TT, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)

    ! per-position NLL in nats: logsumexp(logits) - logit[target]
    do tc = 1, TT
      tgt = full(tc + 1) + 1   ! 0-based id -> 1-based position
      m = outp((tc-1)*VV+1)
      do j = 2, VV
        if (outp((tc-1)*VV+j) > m) m = outp((tc-1)*VV+j)
      end do
      s = 0.0_sp
      do j = 1, VV
        s = s + exp(outp((tc-1)*VV+j) - m)
      end do
      nll = (m + log(s)) - outp((tc-1)*VV+tgt)
      if (tc > 1) write (*, '(A)', advance='no') ' '
      write (*, '(ES14.7)', advance='no') nll
    end do
    print *
    deallocate(cos_b, sin_b, outp)
  end do
  close (unit)

end program eval_bpb
