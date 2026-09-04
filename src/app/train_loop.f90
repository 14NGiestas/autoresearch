! app/train_loop.f90 — multi-step trainer (slice 2): loss logging + ckpts.
!
! Usage:
!   fortran-fpm run train_loop -- <weights> <rows> <outdir> <nsteps> \
!       [lr=0.001] [t0=1] [log_every=1] [save_every=5] [start_row=0]
! Loads weights + fresh AdamW state once, repeats one fixed batch
! (overfit mode; multi-batch cycling is slice 3), logs
! "step N nll" lines to stdout, saves full .npy sets to outdir/step_N/.
! Dims: depth-12 (D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192, T=2048, B=1).

program train_loop
  use iso_c_binding
  use fortran_train_mod
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights
  use fortran_data_mod, only: load_batch
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, TT = 2048, D = 768
  integer, parameter :: N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  type(dims_t) :: G
  type(params_t) :: M, GR
  type(state_t) :: S
  type(cache_t) :: C
  character(len=512) :: wdir, rowsfile, outdir, arg, ckdir
  integer :: idx(B*TT), targets(B*TT), ngot
  real(sp) :: ct(TT*(HD/2)), st(TT*(HD/2))
  real(sp) :: nll
  real(sp) :: lr
  integer :: nsteps, t0, log_every, save_every, start_row
  integer :: k, i, j, tstep
  real(sp) :: theta, ang

  lr = 0.001_sp; t0 = 1; log_every = 1; save_every = 5; start_row = 0
  nsteps = -1
  if (command_argument_count() < 4) then
    print '(A)', "usage: train_loop <weights> <rows> <outdir> <nsteps> " // &
        "[lr] [t0] [log_every] [save_every] [start_row]"
    call exit(2)
  end if
  call get_command_argument(1, wdir)
  call get_command_argument(2, rowsfile)
  call get_command_argument(3, outdir)
  call get_command_argument(4, arg)
  read (arg, *) nsteps
  if (command_argument_count() >= 5) then
    call get_command_argument(5, arg); read (arg, *) lr
  end if
  if (command_argument_count() >= 6) then
    call get_command_argument(6, arg); read (arg, *) t0
  end if
  if (command_argument_count() >= 7) then
    call get_command_argument(7, arg); read (arg, *) log_every
  end if
  if (command_argument_count() >= 8) then
    call get_command_argument(8, arg); read (arg, *) save_every
  end if
  if (command_argument_count() >= 9) then
    call get_command_argument(9, arg); read (arg, *) start_row
  end if

  G%B = B; G%T = TT; G%V = VV; G%D = D
  G%nh = N_HEAD; G%nkv = N_KV; G%hd = HD; G%nl = N_LAYER
  G%eps = 1.0e-5_sp

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  call init_state(M, S)
  allocate(GR%wte(size(M%wte)), GR%lm(size(M%lm)))
  allocate(GR%q(size(M%q)), GR%k(size(M%k)), GR%v(size(M%v)))
  allocate(GR%p(size(M%p)), GR%fc(size(M%fc)), GR%p2(size(M%p2)))

  call load_batch(trim(rowsfile), start_row, B, TT, idx, targets, ngot)
  if (ngot < B) then
    print '(A)', "rows file too short"
    call exit(1)
  end if

  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      ct((i-1)*(HD/2)+j) = cos(ang)
      st((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do

  do k = 1, nsteps
    tstep = t0 + k - 1
    call train_step(idx, targets, ct, st, M, S, G, GR, C, nll, tstep, &
        lr, 0.9_sp, 0.999_sp, 1.0e-8_sp, 0.0_sp)
    if (mod(k, log_every) == 0 .or. k == nsteps) &
      print '(A,I0,A,F10.5)', "step ", tstep, " nll ", nll
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      write (ckdir, '(A,I0)') trim(outdir) // "/step_", tstep
      call execute_command_line("mkdir -p " // trim(ckdir))
      call save_gpt_weights(trim(ckdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
          M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
    end if
  end do
end program train_loop
