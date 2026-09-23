! app/spec_bench.f90 — greedy speculative decoding: acceptance + speed pilot.
!
! Draft-and-verify at B=1 on CPU. The drafter is the first DL layers of the
! SAME checkpoint (our weights are stacked per layer, so layers 0..DL-1 are
! the leading slices of c_q/c_k/c_v/... — no second checkpoint, no extra
! load). DL=3 means a draft step costs ~DL/12 of a target step.
!
! Commit rule (temperature 0) is fortran_spec_mod::accept_prefix: the
! committed stream must be EXACTLY the target's greedy stream, which this
! app asserts token-by-token against a plain greedy run on the same state.
!
! Cost model per round (1 pending token + K drafts fed in ONE target pass):
!   1 target pass over K+1 tokens   ~= 1 target step (bandwidth bound)
!   K+1 drafter single-token steps  ~= (K+1)*DL/12 target steps
!   committed tokens                =  n_accept + 1
! so speedup ~= (n_accept+1) / (1 + (K+1)*DL/12), and acceptance is the
! quantity this pilot exists to measure. If acceptance is low, the next
! move is a dedicated small drafter (narrow, trained), not a deeper K.
!
! Usage:
!   spec_bench --weights DIR --rows FILE [--npre 64] [--ngen 32] \
!              [--spec 8] [--draft-layers 3]

program spec_bench
  use iso_c_binding
  use fortran_kv_mod, only: gpt_step, gpt_step_multi
  use load_weights_mod, only: load_gpt_weights
  use fortran_spec_mod, only: accept_prefix, lookup_draft
  use M_CLI2, only: set_args, sget, iget, specified
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192

  character(len=512) :: wdir, rowsfile
  character(len=65536) :: line
  integer :: ids(4096), nids, npre, ngen, kspec, dl, nmax
  integer :: u, ios, i, j, p, tb, srow, d2, clen_t, clen_d, nacc
  integer :: npass, naccept_total, corr, na, gen_done
  integer :: ksum, keff, nmatch
  character(len=16) :: drafter_mode
  logical :: lookup_mode
  integer :: c0, c1, crate
  logical :: verbose = .false.

  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: ck_t(:), cv_t(:), ck_d(:), cv_d(:)
  real(sp), allocatable :: rows(:), chunk_out(:), row1(:)
  integer, allocatable :: seq_plain(:), seq_spec(:), draft(:), targ(:)
  real(sp) :: ms_plain, ms_spec

  call set_args('--weights WEIGHTS --rows ROWS --npre 64 --ngen 32' // &
      ' --spec 8 --draft-layers 3 --chunk 64 --verbose F' // &
      ' --drafter trunc --match 3', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  spec_bench - greedy speculative decode: acceptance + tok/s', &
      'SYNOPSIS', &
      '  spec_bench --weights DIR --rows FILE [options]', &
      'OPTIONS', &
      '  --weights DIR   checkpoint .npy set (target AND drafter prefix)', &
      '  --rows FILE     id rows file; row 1 supplies the prompt ids', &
      '  --npre N        prompt tokens fed as prefill (default 64)', &
      '  --ngen M        tokens to generate (default 32)', &
      '  --spec K        draft tokens per verify pass (default 8)', &
      '  --draft-layers D drafter depth = first D layers (default 3)', &
      '  --verbose T     per-round accept trace (default F)', &
      '  --drafter MODE  trunc = first D layers (default), lookup =', &
      '                  prompt-lookup: copy the continuation of the most', &
      '                  recent past occurrence of the last --match tokens', &
      '  --match M       lookup match length in tokens (default 3)'], &
      version_text=[character(len=80) :: 'spec_bench 1.0'])

  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  npre = iget('npre')
  ngen = iget('ngen')
  kspec = iget('spec')
  dl = iget('draft-layers')
  verbose = .false.
  if (specified('verbose')) verbose = .true.
  drafter_mode = 'trunc'
  if (specified('drafter')) drafter_mode = trim(sget('drafter'))
  lookup_mode = trim(drafter_mode) == 'lookup'
  nmatch = iget('match')
  ksum = 0
  if (dl < 1 .or. dl > N_LAYER) then
    print '(A)', "draft-layers must be in 1..12"
    call exit(1)
  end if
  if (kspec < 1) then
    print '(A)', "spec must be >= 1"
    call exit(1)
  end if

  ! ---- prompt ids (first line of the rows file) -------------------------
  open (newunit=u, file=trim(rowsfile), status='old', action='read', &
      iostat=ios)
  if (ios /= 0) then
    print '(A)', "cannot read rows file"
    call exit(1)
  end if
  read (u, '(A)', iostat=ios) line
  close (u)
  if (ios /= 0) then
    print '(A)', "rows file is empty"
    call exit(1)
  end if
  nids = count_ids(line)
  if (nids < npre + 2) then
    print '(A,I0,A,I0)', "row has ", nids, " ids, need >= ", npre + 2
    call exit(1)
  end if
  read (line, *, iostat=ios) ids(1:nids)
  if (ios /= 0) then
    print '(A)', "cannot parse row as ids"
    call exit(1)
  end if

  nmax = npre + ngen + kspec + 4
  if (nmax > 4096) then
    print '(A)', "npre+ngen+spec too large for the prompt buffer"
    call exit(1)
  end if
  d2 = HD / 2

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_pr2)

  allocate (ck_t(N_LAYER*nmax*N_KV*HD), cv_t(N_LAYER*nmax*N_KV*HD))
  allocate (ck_d(dl*nmax*N_KV*HD), cv_d(dl*nmax*N_KV*HD))
  allocate (cos_b(nmax*d2), sin_b(nmax*d2))
  allocate (rows(nmax*VV), chunk_out(max(64, kspec+2)*VV), row1(VV))
  allocate (seq_plain(ngen), seq_spec(ngen))
  allocate (draft(kspec), targ(kspec+1))
  do i = 1, nmax
    do j = 1, d2
      cos_b((i-1)*d2+j) = cos(real(i-1, sp) * 10000.0_sp**(-2.0_sp*real(j-1, sp)/real(HD, sp)))
      sin_b((i-1)*d2+j) = sin(real(i-1, sp) * 10000.0_sp**(-2.0_sp*real(j-1, sp)/real(HD, sp)))
    end do
  end do

  print '(A,A)', "weights: ", trim(wdir)
  print '(A,I0,A,I0,A,I0,A,I0)', "prompt=", npre, " gen=", ngen, &
      " spec K=", kspec, " drafter layers=", dl
  print '(A,A,A,I0)', "drafter: ", trim(drafter_mode), &
      "  match=", nmatch

  call run_plain(seq_plain, ms_plain)
  call run_spec(seq_spec, ms_spec)

  print '(A,F10.1,A,F10.2)', "plain : ", ms_plain, " ms   ", &
      real(ngen, sp) / max(1.0e-6_sp, ms_plain/1000.0_sp), " tok/s"
  print '(A,F10.1,A,F10.2)', "spec  : ", ms_spec, " ms   ", &
      real(ngen, sp) / max(1.0e-6_sp, ms_spec/1000.0_sp), " tok/s"
  print '(A,F8.2,A)', "speedup: ", ms_plain / max(1.0e-6_sp, ms_spec), "x"
  print '(A,F8.2,A,F8.2,A,F8.2)', "passes=", real(npass, sp), &
      " tokens/pass=", real(ngen, sp)/real(max(1, npass), sp), &
      " mean n_accept=", real(naccept_total, sp)/real(max(1, npass), sp)
  print '(A,F8.2)', "mean acceptance rate (nacc/k_eff) = ", &
      real(naccept_total, sp) / real(max(1, ksum), sp)
  print '(A,F8.2)', "mean drafts proposed per pass = ", &
      real(ksum, sp) / real(max(1, npass), sp)
  if (all(seq_plain(1:ngen) == seq_spec(1:ngen))) then
    print '(A)', "EQUIVALENCE ok: spec stream == plain greedy stream"
  else
    print '(A)', "EQUIVALENCE FAILED: spec stream differs from plain greedy"
    call exit(1)
  end if

contains

  integer function count_ids(s)
    character(*), intent(in) :: s
    integer :: t, nn
    nn = 0
    do t = 1, len_trim(s)
      if (s(t:t) == ' ') then
        if (t > 1 .and. s(t-1:t-1) /= ' ') nn = nn + 1
      end if
    end do
    if (len_trim(s) > 0 .and. s(len_trim(s):len_trim(s)) /= ' ') nn = nn + 1
    count_ids = nn
  end function count_ids

  integer function argmax_i(v, n)
    integer, intent(in) :: n
    real(sp), intent(in) :: v(n)
    integer :: t
    argmax_i = 1
    do t = 2, n
      if (v(t) > v(argmax_i)) argmax_i = t
    end do
  end function argmax_i

  ! prefill both caches with positions 1..npre-1 (chunked): leaves the
  ! invariant "cache valid to p-1, idx(p) is the pending token" at p=npre
  subroutine prefill_both()
    srow = 1
    do while (srow <= npre - 1)
      tb = min(64, npre - srow)
      call gpt_step_multi(ids(srow:srow+tb-1), cos_b((srow-1)*d2+1:), &
          sin_b((srow-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ck_t, cv_t, clen_t, nmax, chunk_out, &
          B, VV, D, N_HEAD, N_KV, HD, N_LAYER, tb, 1.0e-5_sp)
      call gpt_step_multi(ids(srow:srow+tb-1), cos_b((srow-1)*d2+1:), &
          sin_b((srow-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ck_d, cv_d, clen_d, nmax, chunk_out, &
          B, VV, D, N_HEAD, N_KV, HD, dl, tb, 1.0e-5_sp)
      srow = srow + tb
    end do
  end subroutine prefill_both

  ! plain greedy: one cached step per token (the thing we must reproduce)
  subroutine run_plain(seq, ms)
    integer, intent(out) :: seq(ngen)
    real(sp), intent(out) :: ms

    ck_t = 0.0_sp
    cv_t = 0.0_sp
    ck_d = 0.0_sp
    cv_d = 0.0_sp
    clen_t = 0
    clen_d = 0
    call prefill_both()
    call system_clock(c0, crate)
    p = npre
    do i = 1, ngen
      call gpt_step(ids(p:p), cos_b((p-1)*d2+1:), sin_b((p-1)*d2+1:), &
          wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ck_t, cv_t, clen_t, nmax, row1, &
          B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
      ids(p + 1) = argmax_i(row1, VV) - 1
      seq(i) = ids(p + 1)
      p = p + 1
    end do
    call system_clock(c1, crate)
    ms = 1000.0_sp * real(c1 - c0, sp) / real(crate, sp)
  end subroutine run_plain

  ! greedy draft-and-verify. Invariant per round:
  !   p            = number of committed tokens (ids(1:p))
  !   clen_t/clen_d = p-1 (both caches valid up to the pending token - 1)
  !   pending      = ids(p)
  ! One target pass over [pending, d_1..d_K] yields rows r_1..r_{K+1}, the
  ! target distributions for positions p+1..p+K+1, i.e. targ = argmax(r_j).
  ! accept_prefix then yields n_accept and the correction (= targ(n_accept+1),
  ! which is the bonus row when all K drafts are accepted). The caches are
  ! rolled back to p+n_accept (rows 1..p+n_accept stay valid because every
  ! accepted draft is byte-identical to the target's own greedy token), and
  ! the correction becomes the next pending token.
  subroutine run_spec(seq, ms)
    integer, intent(out) :: seq(ngen)
    real(sp), intent(out) :: ms

    ck_t = 0.0_sp
    cv_t = 0.0_sp
    ck_d = 0.0_sp
    cv_d = 0.0_sp
    clen_t = 0
    clen_d = 0
    call prefill_both()
    npass = 0
    naccept_total = 0
    ksum = 0
    p = npre
    gen_done = 0
    call system_clock(c0, crate)
    do while (gen_done < ngen)
      ! ---- draft phase ------------------------------------------------
      if (lookup_mode) then
        ! zero-cost drafter: no model forward, just a context search
        call lookup_draft(ids, p, nmatch, kspec, draft, keff)
      else
        ! feed pending + K-1 drafts, argmax chain -> d_1..d_K
        ! (feed one extra draft so the drafter cache is valid through p+K,
        !  which the all-accepted case needs)
        call gpt_step(ids(p:p), cos_b((p-1)*d2+1:), sin_b((p-1)*d2+1:), &
            wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ck_d, cv_d, clen_d, nmax, row1, &
            B, VV, D, N_HEAD, N_KV, HD, dl, 1.0e-5_sp)
        draft(1) = argmax_i(row1, VV) - 1
        do j = 1, kspec - 1
          call gpt_step(draft(j:j), cos_b((p+j-1)*d2+1:), &
              sin_b((p+j-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
              ck_d, cv_d, clen_d, nmax, row1, &
              B, VV, D, N_HEAD, N_KV, HD, dl, 1.0e-5_sp)
          draft(j + 1) = argmax_i(row1, VV) - 1
        end do
        ! one more drafter step: keeps ck_d/cv_d valid through position p+K
        call gpt_step(draft(kspec:kspec), cos_b((p+kspec-1)*d2+1:), &
            sin_b((p+kspec-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ck_d, cv_d, clen_d, nmax, row1, &
            B, VV, D, N_HEAD, N_KV, HD, dl, 1.0e-5_sp)
        keff = kspec
      end if
      ksum = ksum + keff

      ! ---- verify: ONE target pass over [pending, d_1..d_keff] -----------
      block
        integer :: inp(kspec + 1), px(kspec + 1)
        integer :: jj
        inp(1) = ids(p)
        if (keff > 0) inp(2:keff+1) = draft(1:keff)
        call gpt_step_multi(inp(1:keff+1), cos_b((p-1)*d2+1:), &
            sin_b((p-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ck_t, cv_t, clen_t, nmax, chunk_out, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, keff + 1, 1.0e-5_sp)
        do jj = 1, keff + 1
          px(jj) = argmax_i(chunk_out((jj-1)*VV+1:jj*VV), VV) - 1
        end do
        targ(1:keff+1) = px(1:keff+1)
      end block

      call accept_prefix(draft, targ, keff, na, corr)
      nacc = na
      naccept_total = naccept_total + na
      npass = npass + 1
      if (verbose) print '(A,I6,A,I3,A,I3,A,I3,A,I8)', "  pass ", npass, &
          " n_accept=", nacc, "/", keff, " K=", kspec, " p=", p

      if (nacc > 0) ids(p+1:p+nacc) = draft(1:nacc)
      ids(p + nacc + 1) = corr
      p = p + nacc + 1
      gen_done = p - npre

      ! rollback both caches to "one behind the new pending token"
      clen_t = p - 1
      clen_d = p - 1
    end do
    call system_clock(c1, crate)
    ms = 1000.0_sp * real(c1 - c0, sp) / real(crate, sp)
    do i = 1, ngen
      seq(i) = ids(npre + i)
    end do
  end subroutine run_spec

end program spec_bench
