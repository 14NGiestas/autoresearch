! app/train_1step.f90 — trainer skeleton, slice 1: single AdamW step.
!
! Usage:
!   fortran-fpm run train_1step -- <weights_dir> <rows_file> <out_dir> \
!       [lr=0.001] [tstep=1]
! Loads depth-12 weights + states (fresh zeros), runs ONE train_step on
! batch row 0 (B=1, T=2048), prints nll before/after, saves updated
! weights to out_dir (resume-compatible .npy set). Dims match the
! depth-12 checkpoint (D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192).

program train_1step
  use iso_c_binding
  use fortran_train_mod
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights
  use fortran_data_mod, only: load_batch
  use M_CLI2, only: set_args, sget, rget, iget, specified
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, TT = 2048, D = 768
  integer, parameter :: N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  type(dims_t) :: G
  type(params_t) :: M, GR
  type(state_t) :: S
  type(cache_t) :: C
  type(temp_t) :: tmp
  character(len=512) :: wdir, rowsfile, outdir, arg
  integer :: idx(B*TT), targets(B*TT), ngot
  real(sp) :: cos_t(TT*(HD/2)), sin_t(TT*(HD/2))
  real(sp) :: nll0, nll1
  real(sp) :: lr
  integer :: tstep, i, j
  real(sp) :: theta, ang

  call set_args('--weights WEIGHTS --rows ROWS --out OUT' // &
      ' --lr 0.001 --tstep 1', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  train_1step - single AdamW step on real weights', &
      'SYNOPSIS', &
      '  train_1step --weights DIR --rows FILE --out DIR [--lr X]'], &
      version_text=[character(len=80) :: 'train_1step 1.0'])
  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  outdir = trim(sget('out'))
  lr = rget('lr')
  tstep = iget('tstep')
  if (.not. specified('weights') .or. .not. specified('rows') &
      .or. .not. specified('out')) then
    print '(A)', 'require --weights --rows --out (--help for all)'
    call exit(2)
  end if

  G%B = B; G%T = TT; G%V = VV; G%D = D
  G%nh = N_HEAD; G%nkv = N_KV; G%hd = HD; G%nl = N_LAYER
  G%eps = 1.0e-5_sp

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  call init_state(M, S)
  call init_temp(G, tmp)
  allocate(GR%wte(size(M%wte)), GR%lm(size(M%lm)))
  allocate(GR%q(size(M%q)), GR%k(size(M%k)), GR%v(size(M%v)))
  allocate(GR%p(size(M%p)), GR%fc(size(M%fc)), GR%p2(size(M%p2)))

  call load_batch(trim(rowsfile), 0, B, TT, idx, targets, ngot)
  if (ngot < B) then
    print '(A)', "rows file too short"
    call exit(1)
  end if

  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      cos_t((i-1)*(HD/2)+j) = cos(ang)
      sin_t((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do

  call forward_save(idx, targets, cos_t, sin_t, M, G, C, tmp, nll0)
  call train_step(idx, targets, cos_t, sin_t, M, S, G, GR, C, tmp, nll1, tstep, &
      lr, 0.9_sp, 0.999_sp, 1.0e-8_sp, 0.0_sp)
  print '(A,F10.5,A,F10.5)', "nll before/after: ", nll0, " ", nll1

  call save_gpt_weights(trim(outdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  print '(A)', "saved to " // trim(outdir)
end program train_1step
