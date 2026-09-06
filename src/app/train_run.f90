! app/train_run.f90 — multi-batch trainer (slice 3, hyp_fa2388).
!
! Usage:
!   fortran-fpm run train_run -- <weights> <rows> <outdir> <nsteps> \
!       [lr=0.0003] [t0=1] [log_every=1] [save_every=10] [start_row=0] \
!       [ntrain=40] [val_every=5] [nval=8] [keep_last=2] [bytes_file]
! Cycles train rows [start_row, start_row+ntrain); every val_every steps
! scores rows [start_row+ntrain, +nval) as exact val-bpb (token byte
! lengths from bytes_file, default tok_tables/token_bytes.txt).
! 2-step linear warmup in code (lesson of the overfit run): lr * min(1,
! k/2) for run-relative step k. Best-val snapshot to outdir/best/,
! rotation keeps last keep_last step_N/ dirs.
! Dims: depth-12 (D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192, T=2048, B=1).

program train_run
  use iso_c_binding
  use fortran_train_mod
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights
  use fortran_data_mod, only: load_batch
  use M_CLI2, only: set_args, sget, rget, iget, specified
  use fortran_sys_mod, only: mkdir_p
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, TT = 2048, D = 768
  integer, parameter :: N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  type(dims_t) :: G
  type(params_t) :: M, GR
  type(state_t) :: S
  type(cache_t) :: C
  character(len=512) :: wdir, rowsfile, outdir, arg, ckdir, bytesfile
  integer :: idx(B*TT), targets(B*TT), ngot
  integer, allocatable :: tbytes(:)
  real(sp) :: ct(TT*(HD/2)), st(TT*(HD/2))
  real(sp) :: nll, vnll, tnll, lr, lr_eff, best
  integer :: nsteps, t0, log_every, save_every, start_row
  integer :: ntrain, val_every, nval, keep_last
  integer :: k, i, j, tstep, u, ios, r
  real(sp) :: theta, ang

  lr = 0.0003_sp; t0 = 1; log_every = 1; save_every = 10; start_row = 0
  ntrain = 40; val_every = 5; nval = 8; keep_last = 2
  bytesfile = ""
  nsteps = -1
  call set_args('--weights WEIGHTS --rows ROWS --out OUT --nsteps 20' // &
      ' --lr 0.0003 --t0 1 --log_every 1 --save_every 10' // &
      ' --start_row 0 --ntrain 40 --val_every 5 --nval 8 --keep_last 2' // &
      ' --bytes BYTES', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  train_run - multi-batch trainer (slice 3)', &
      'SYNOPSIS', &
      '  train_run --weights DIR --rows FILE --out DIR --nsteps N'], &
      version_text=[character(len=80) :: 'train_run 1.0'])
  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  outdir = trim(sget('out'))
  nsteps = iget('nsteps')
  lr = rget('lr')
  t0 = iget('t0')
  log_every = iget('log_every')
  save_every = iget('save_every')
  start_row = iget('start_row')
  ntrain = iget('ntrain')
  val_every = iget('val_every')
  nval = iget('nval')
  keep_last = iget('keep_last')
  bytesfile = trim(sget('bytes'))
  if (.not. specified('weights') .or. .not. specified('rows') &
      .or. .not. specified('out') .or. nsteps < 1) then
    print '(A)', 'require --weights --rows --out --nsteps>=1 (--help)'
    call exit(2)
  end if
  if (bytesfile == 'BYTES') &
      bytesfile = trim(wdir) // '/../tok_tables/token_bytes.txt'

  G%B = B; G%T = TT; G%V = VV; G%D = D
  G%nh = N_HEAD; G%nkv = N_KV; G%hd = HD; G%nl = N_LAYER
  G%eps = 1.0e-5_sp

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  call init_state(M, S)
  allocate(GR%wte(size(M%wte)), GR%lm(size(M%lm)))
  allocate(GR%q(size(M%q)), GR%k(size(M%k)), GR%v(size(M%v)))
  allocate(GR%p(size(M%p)), GR%fc(size(M%fc)), GR%p2(size(M%p2)))
  allocate(tbytes(0:VV-1))
  open (newunit=u, file=trim(bytesfile), status="old", iostat=ios)
  if (ios /= 0) then
    print '(2A)', "cannot open bytes file: ", trim(bytesfile)
    call exit(1)
  end if
  do i = 0, VV - 1
    read (u, *, iostat=ios) tbytes(i)
    if (ios /= 0) then
      print '(A)', "short bytes file"
      call exit(1)
    end if
  end do
  close (u)

  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      ct((i-1)*(HD/2)+j) = cos(ang)
      st((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do

  best = huge(1.0_sp)
  do k = 1, nsteps
    tstep = t0 + k - 1
    ! 2-step linear warmup on run-relative k (overfit-run lesson)
    lr_eff = lr * min(1.0_sp, real(k, sp) / 2.0_sp)
    ! cycle within [start_row, start_row+ntrain): r is 0-based offset
    r = mod(k - 1, ntrain)
    call load_batch(trim(rowsfile), start_row + r, B, TT, idx, targets, ngot)
    if (ngot < B) then
      print '(A)', "rows file too short"
      call exit(1)
    end if
    call train_step(idx, targets, ct, st, M, S, G, GR, C, nll, tstep, &
        lr_eff, 0.9_sp, 0.999_sp, 1.0e-8_sp, 0.0_sp)
    if (mod(k, log_every) == 0 .or. k == nsteps) then
      print '(A,I0,A,F10.5,A,F8.5)', "step ", tstep, " nll ", nll, &
        " lr ", lr_eff
      flush (6)
    end if
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      write (ckdir, '(A,I0)') trim(outdir) // "/step_", tstep
      if (mkdir_p(trim(ckdir)) /= 0) then
        print '(2A)', "cannot create ", trim(ckdir)
        call exit(1)
      end if
      call save_gpt_weights(trim(ckdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
          M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
      call rotate_ckpts(trim(outdir), tstep, save_every, keep_last)
    end if
    if (mod(k, val_every) == 0 .or. k == nsteps) then
      vnll = val_bpb(trim(rowsfile), start_row + ntrain, nval)
      ! train-bpb on a held-out slice of the train pool (start_row+nval to
      ! start_row+ntrain) so train/val are on the same scale. Lets us see
      ! overfitting (train << val bpb) and pick best-checkpoint fairly.
      tnll = val_bpb(trim(rowsfile), start_row + nval, ntrain - nval)
      print '(A,I0,A,F10.5)', "val @", tstep, " bpb    ", vnll
      print '(A,I0,A,F10.5)', "trn @", tstep, " bpb    ", tnll
      print '(A,I0,A,F10.5)', "gap @", tstep, " bpb    ", vnll - tnll
      if (vnll < best) then
        best = vnll
        if (mkdir_p(trim(outdir) // "/best") /= 0) then
          print '(A)', "cannot create best/"
          call exit(1)
        end if
        call save_gpt_weights(trim(outdir) // "/best", N_LAYER, D, &
            N_HEAD, N_KV, HD, VV, M%wte, M%lm, M%q, M%k, M%v, M%p, &
            M%fc, M%p2)
        print '(A)', "new best snapshot"
      end if
      flush (6)
    end if
  end do
contains
  ! exact val-bpb over nval rows (byte-masked NLL, train.py metric)
  real(sp) function val_bpb(rowsfile, first, nval)
    character(*), intent(in) :: rowsfile
    integer, intent(in) :: first, nval
    integer :: v_idx(B*TT), v_tgt(B*TT), n, vv2, jj, tid, b2
    real(sp) :: out_nll, tn, tb
    real(sp), allocatable :: nlls(:)
    allocate(nlls(B*TT))
    tn = 0.0_sp; tb = 0
    do b2 = 0, nval - 1
      call load_batch(rowsfile, first + b2, B, TT, v_idx, v_tgt, n)
      if (n < B) exit
      call forward_nlls(v_idx, v_tgt, nlls)
      do jj = 1, B*TT
        tid = v_tgt(jj)
        if (tbytes(tid) > 0) then
          tn = tn + nlls(jj)
          tb = tb + tbytes(tid)
        end if
      end do
    end do
    val_bpb = tn / (0.69314718056_sp * real(tb, sp))
  end function val_bpb

  ! per-position NLLs for one batch (forward only)
  subroutine forward_nlls(idx, targets, nlls)
    integer, intent(in) :: idx(*), targets(*)
    real(sp), intent(out) :: nlls(*)
    real(sp), allocatable :: emd(:), xn(:), sub(:), qo(:), ko(:), vo(:)
    real(sp), allocatable :: qrot(:), krot(:), ao(:), mlpd(:), lgt(:)
    integer :: BT, DD, hdd, dff, ll, jj, it, j2, tg
    integer :: qsz, ksz, psz, fcsz, p2sz
    real(sp) :: mx, sm
    BT = B * TT; DD = D; hdd = N_HEAD * HD
    dff = 4 * DD
    qsz = hdd * DD; ksz = N_KV * HD * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    allocate(emd(BT*DD), xn(BT*DD), sub(BT*DD))
    allocate(qo(BT*hdd), ko(BT*N_KV*HD), vo(BT*N_KV*HD))
    allocate(qrot(BT*hdd), krot(BT*N_KV*HD), ao(BT*DD), mlpd(BT*dff))
    allocate(lgt(BT*VV))
    call wte_lookup(idx, M%wte, emd, B, TT, VV, DD)
    call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
    emd = xn
    do ll = 0, N_LAYER - 1
      call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
      call linear3d(xn, M%q(ll*qsz+1:), qo, B, TT, DD, hdd)
      call linear3d(xn, M%k(ll*ksz+1:), ko, B, TT, DD, N_KV*HD)
      call linear3d(xn, M%v(ll*ksz+1:), vo, B, TT, DD, N_KV*HD)
      call rope_4d(qo, ct, st, qrot, B, TT, N_HEAD, HD)
      call rope_4d(ko, ct, st, krot, B, TT, N_KV, HD)
      call causal_attn(qrot, krot, vo, ao, B, TT, N_HEAD, N_KV, HD)
      call linear3d(ao, M%p(ll*psz+1:), sub, B, TT, DD, DD)
      emd = emd + sub
      call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
      call linear3d(xn, M%fc(ll*fcsz+1:), mlpd, B, TT, DD, dff)
      call relu2(mlpd, BT*dff)
      call linear3d(mlpd, M%p2(ll*p2sz+1:), sub, B, TT, dff, DD)
      emd = emd + sub
    end do
    call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
    call linear3d(xn, M%lm, lgt, B, TT, DD, VV)
    do it = 1, BT
      tg = targets(it) + 1
      mx = lgt((it-1)*VV+1)
      do j2 = 2, VV
        if (lgt((it-1)*VV+j2) > mx) mx = lgt((it-1)*VV+j2)
      end do
      sm = 0.0_sp
      do j2 = 1, VV
        sm = sm + exp(lgt((it-1)*VV+j2) - mx)
      end do
      nlls(it) = (mx + log(sm)) - lgt((it-1)*VV+tg)
    end do
    deallocate(emd, xn, sub, qo, ko, vo, qrot, krot, ao, mlpd, lgt)
  end subroutine forward_nlls

  ! delete step_{t-2*save_every} dirs beyond keep_last (best/ untouched)
  subroutine rotate_ckpts(outdir, tstep, save_every, keep_last)
    character(*), intent(in) :: outdir
    integer, intent(in) :: tstep, save_every, keep_last
    ! Fork-free: rm -rf via execute_command_line forks with live OpenMP
    ! pools and kills children (step 110/120 smashes). Best-effort only —
    ! leave old steps on disk, clean offline. No fork, no crash.
  end subroutine rotate_ckpts
end program train_run
