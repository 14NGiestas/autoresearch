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
  use stdlib_io_npy, only: load_npy
  use iso_fortran_env, only: int32
  use M_CLI2, only: set_args, sget, specified
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER, A_VOCAB => VV, A_CTX => TT, &
      write_arch_txt, read_arch_txt, arch_report, require_arch
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: BMAX = 64
  integer, parameter :: D = A_D, N_HEAD = A_HEAD, N_KV = A_KV, HD = A_HD
  integer, parameter :: N_LAYER = A_LAYER, VV = A_VOCAB, TT = A_CTX

  character(len=512) :: wdir, rowsfile
  character(len=32) :: batchstr
  integer :: ios, unit, tc, i, j, r, nb, tgt, rownum, nbatch
  integer :: idx(BMAX*TT)
  integer :: fullb(BMAX, TT + 1)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:), nllbuf(:, :)
  real(sp) :: theta, ang, m, s
  character(len=65536) :: buf
  logical :: attn_blas = .false.
  logical :: attn_qk = .false.
  logical :: attn_stats = .false.
  real(wp), allocatable :: posf(:)
  integer, allocatable :: posc(:)
  integer :: jh, nstats = 0, ncontrib = 0
  integer :: base
  ! npy rows: single (N,TT+1) Fortran-order int32 file, zero text parsing.
  ! (numpy saves logical (N,TT+1) with fortran_order=True; stdlib reads it
  ! straight into Fortran dims (N,TT+1): row i = arr(i,:).)
  ! Detected by extension; .txt path stays until corpora are converted.
  logical :: use_npy = .false., ex
  integer(int32), allocatable :: rows_npy(:, :)
  integer :: nrows_npy = 0, pos_npy = 1, fsize

  call set_args('--weights WEIGHTS --rows ROWS --attn naive --batch 1 --attn-stats F', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  eval_bpb - bits-per-byte evaluation (prints per-position NLLs)', &
      'SYNOPSIS', &
      '  eval_bpb --weights DIR --rows FILE [--attn naive|blas]', &
      '           [--batch N]', &
      'OPTIONS', &
      '  --attn MODE   naive (default, the kernel every recorded bpb used) or', &
      '                blas: attn_sgemm, ~13x faster at T=2048; qkhop: QK-hop (sem V)', &
      '                ~3e-06. Use blas to make a bpb pass affordable; use', &
      '                naive to reproduce an older number exactly.', &
      '  --batch N     rows per forward pass (default 1). N>1 amortizes the', &
      '                40MB weight stream over N rows (GEMV->GEMM); output', &
      '                is identical, row order preserved.'], &
      version_text=[character(len=80) :: 'eval_bpb 1.2'])
  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  attn_blas = trim(sget('attn')) == 'blas'
  attn_stats = specified('attn-stats')
  if (attn_stats) allocate (posf(N_HEAD), source=0.0_wp)
  if (attn_stats) allocate (posc(N_HEAD), source=0)
  attn_qk = trim(sget('attn')) == 'qkhop'
  batchstr = trim(sget('batch'))
  read (batchstr, *, iostat=ios) nbatch
  if (ios /= 0 .or. nbatch < 1 .or. nbatch > BMAX) then
    print '(A)', 'require 1 <= --batch <= 64'
    call exit(2)
  end if
  if (.not. specified('weights') .or. .not. specified('rows')) then
    print '(A)', 'require --weights DIR --rows FILE (--help for all)'
    call exit(2)
  end if

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
  call require_arch(trim(wdir))

  use_npy = len_trim(rowsfile) > 4 .and. &
      rowsfile(len_trim(rowsfile)-3:len_trim(rowsfile)) == '.npy'
  if (use_npy) then
    inquire (file=trim(rowsfile), exist=ex, size=fsize)
    if (.not. ex .or. fsize <= 0) then
      print '(2A)', "rows npy missing or empty: ", trim(rowsfile)
      call exit(1)
    end if
    call load_npy(trim(rowsfile), rows_npy, iostat=ios)
    if (ios /= 0 .or. .not. allocated(rows_npy)) then
      print '(2A)', "rows npy unreadable: ", trim(rowsfile)
      call exit(1)
    end if
    if (size(rows_npy, 2) /= TT + 1) then
      print '(A,2I0)', "rows npy bad width (need TT+1): ", size(rows_npy, 2)
      call exit(1)
    end if
    nrows_npy = size(rows_npy, 1)
  end if

  ! RoPE tables for TT: identical for every row, build once
  allocate(cos_b(TT*(HD/2)), sin_b(TT*(HD/2)))
  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      cos_b((i-1)*(HD/2)+j) = cos(ang)
      sin_b((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do
  allocate(outp(nbatch*TT*VV), nllbuf(nbatch, TT))

  if (.not. use_npy) &
    open (newunit=unit, file=trim(rowsfile), status='old', action='read')
  rownum = 0
  do
    ! fill one batch (last chunk may be partial)
    nb = 0
    do r = 1, nbatch
      if (use_npy) then
        if (pos_npy > nrows_npy) exit
        fullb(r, :) = rows_npy(pos_npy, :)
        pos_npy = pos_npy + 1
      else
        read (unit, '(A)', iostat=ios) buf
        if (ios /= 0) exit
        read (buf, *, iostat=ios) fullb(r, :)
        if (ios /= 0) then
          print '(A)', "bad row (need TT+1 ids)"
          call exit(1)
        end if
      end if
      nb = r
    end do
    if (nb == 0) exit
    do r = 1, nb
      idx((r-1)*TT+1:r*TT) = fullb(r, 1:TT)
    end do

    call gpt_forward(idx(1:nb*TT), cos_b, sin_b, &
        wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
        outp(1:nb*TT*VV), nb, TT, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp, &
        attn_blas=attn_blas, attn_qk=attn_qk, pos_frac=posf, pos_n=posc)
    nstats = nstats + 1
    ncontrib = ncontrib + nb      ! o kernel soma por (batch, cabeca)

    ! per-position NLL in nats: logsumexp(logits) - logit[target].
    ! Rows are independent: parallel over batch, print serially in order.
    !$omp parallel do private(r, tc, tgt, m, s, j, base) schedule(static)
    do r = 1, nb
      base = (r-1)*TT*VV
      do tc = 1, TT
        tgt = fullb(r, tc + 1) + 1   ! 0-based id -> 1-based position
        m = outp(base+(tc-1)*VV+1)
        do j = 2, VV
          if (outp(base+(tc-1)*VV+j) > m) m = outp(base+(tc-1)*VV+j)
        end do
        s = 0.0_sp
        do j = 1, VV
          s = s + exp(outp(base+(tc-1)*VV+j) - m)
        end do
        nllbuf(r, tc) = (m + log(s)) - outp(base+(tc-1)*VV+tgt)
      end do
    end do
    !$omp end parallel do

    do r = 1, nb
      do tc = 1, TT
        if (tc > 1) write (*, '(A)', advance='no') ' '
        write (*, '(ES14.7)', advance='no') nllbuf(r, tc)
      end do
      print *
      ! progress on stderr (unbuffered): the only monitor for hour-long runs
      rownum = rownum + 1
      write (0, '(A,I0)') "row done: ", rownum
    end do
  end do
  if (.not. use_npy) close (unit)

  if (attn_stats) then
    write (*, '(A)') "=== atencao: fracao de scores POSITIVOS por cabeca"
    write (*, '(A)') "  O divisor e' CONTADO (adicoes), nao presumido. O bug antigo: o kernel"
    write (*, '(A)') "  acumulava uma vez por CAMADA, e o print dividia por uma vez por linha."
    write (*, '(A)') "  Eram 12 camadas x 100 linhas = 1200 contra 100, e dai o valor sair 12x"
    write (*, '(A)') "  alto (4,637 = 12 x 0,386, exacto). Uma fracao nao pode passar de 1."
      write (*, '(A,I0,A,I0,A)') "media sobre ", nstats, " chamadas, ", ncontrib, " contribuicoes (batch x cabeca)"
      write (*, '(A)') "cabeca   fracao"
      do jh = 1, size(posf)
        write (*, '(I5,F11.4,A,I0,A,F11.4)') jh, posf(jh)/real(max(1, posc(jh)), wp), &
            '  adicoes=', posc(jh), '  soma=', posf(jh)
      end do
    write (*, '(A,I0,A,I0,A,I0)') "  cabeças com fracao < 1%: ", &
        count(posf/real(max(1, ncontrib), wp) < 0.01_wp), "   < 10%: ", &
        count(posf/real(max(1, ncontrib), wp) < 0.10_wp), "   < 50%: ", &
        count(posf/real(max(1, ncontrib), wp) < 0.50_wp)
  end if
end program eval_bpb
