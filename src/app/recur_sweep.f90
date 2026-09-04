! app/recur_sweep.f90 — loop-count sweep for hyp_ba98cd (overthinking curve).
!
! Wires recurrent_forward to REAL depth-12 weights: layer 0 tied for all
! loops + real wte/lm_head. For each packed row and each loop count N,
! prints per-position NLLs (nats). The driver (scripts/sweep_driver.py)
! turns these into bpb(N) — the predicted shape is improve-then-degrade
! (U-curve = depth extrapolation + overthinking); monotonic worsening
! means recurrence can't be bolted on (must be trained in).
!
! Usage:
!   fortran-fpm run recur_sweep -- <weights_dir> <rows_file> <N1> [N2 ...]
! Output per (row, N):
!   ROW <r> LOOPS <n>
!   <nll_1> ... <nll_T>
! Progress on stderr. Dims: depth-12 single layer 0 (D=768 NH=6 NKV=6
! HD=128 VV=8192, T=2048).

program recur_sweep
  use iso_c_binding
  use fortran_recurrent_mod
  use load_weights_mod, only: load1
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: VV = 8192, TT = 2048, MAXN = 8

  character(len=512) :: wdir, rowsfile, arg
  integer :: nargs, nvals, i, j, t, tc, tgt, ios, unit, rownum
  integer :: nlvec(MAXN)
  integer :: idx(TT)
  integer, allocatable :: full(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, m, s, nll
  character(len=65536) :: buf

  nargs = command_argument_count()
  if (nargs < 3) then
    print '(A)', "usage: recur_sweep <weights_dir> <rows_file> <N1> [N2 ...]"
    call exit(2)
  end if
  call get_command_argument(1, wdir)
  call get_command_argument(2, rowsfile)
  nvals = min(nargs - 2, MAXN)
  do i = 1, nvals
    call get_command_argument(2 + i, arg)
    read (arg, *) nlvec(i)
  end do

  ! single tied layer = depth-12 layer 0 + real wte/lm_head
  call load1(trim(wdir) // "/transformer_wte_weight.npy", wte)
  call load1(trim(wdir) // "/lm_head_weight.npy", lm)
  call load1(trim(wdir) // "/transformer_h_0_attn_c_q_weight.npy", c_q)
  call load1(trim(wdir) // "/transformer_h_0_attn_c_k_weight.npy", c_k)
  call load1(trim(wdir) // "/transformer_h_0_attn_c_v_weight.npy", c_v)
  call load1(trim(wdir) // "/transformer_h_0_attn_c_proj_weight.npy", c_pr)
  call load1(trim(wdir) // "/transformer_h_0_mlp_c_fc_weight.npy", c_fc)
  call load1(trim(wdir) // "/transformer_h_0_mlp_c_proj_weight.npy", c_pr2)

  allocate(full(TT + 1))
  open (newunit=unit, file=trim(rowsfile), status='old', action='read')
  rownum = 0
  do
    read (unit, '(A)', iostat=ios) buf
    if (ios /= 0) exit
    read (buf, *, iostat=ios) full
    if (ios /= 0) then
      print '(A)', "bad row (need TT+1 ids)"
      call exit(1)
    end if
    idx = full(1:TT)

    allocate(cos_b(TT*(HD/2)), sin_b(TT*(HD/2)))
    do i = 1, TT
      do j = 1, HD/2
        theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
        ang = real(i-1, sp) * theta
        cos_b((i-1)*(HD/2)+j) = cos(ang)
        sin_b((i-1)*(HD/2)+j) = sin(ang)
      end do
    end do

    do i = 1, nvals
      allocate(outp(B*TT*VV))
      call recurrent_forward(idx, cos_b, sin_b, &
          wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          outp, B, TT, VV, D, N_HEAD, N_KV, HD, nlvec(i), 1.0e-5_sp)
      print '(A,I0,A,I0)', "ROW ", rownum, " LOOPS ", nlvec(i)
      do tc = 1, TT
        tgt = full(tc + 1) + 1
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
      deallocate(outp)
    end do
    write (0, '(A,I0)') "row done: ", rownum
    rownum = rownum + 1
    deallocate(cos_b, sin_b)
  end do
  close (unit)

end program recur_sweep
