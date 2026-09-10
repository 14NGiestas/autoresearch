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
  use iso_fortran_env, only: output_unit
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use sample_mod, only: sample_token, sample_next
  use fortran_kv_mod, only: gpt_step, gpt_step_multi
  use fortran_recurrent_mod, only: recurrent_forward
  use M_CLI2, only: set_args, set_mode, sget, rget, iget, specified
  use fortran_chat_mod
  use fortran_spec_mod, only: lookup_draft
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, BOS = 8188

  character(len=512) :: tdir, wdir, arg
  character(len=1) :: cb
  character(len=4096) :: tmpl_raw, sys_raw, stop_raw
  character(len=:), allocatable :: dia
  logical :: diafound
  integer :: n_gen, u, ios, fsize, i, j, best, tc, step, nprompt
  integer :: ntot, d2, clen
  real(sp), allocatable :: ckv(:), cvv(:), out1(:)
  integer(c_int64_t) :: rng = 12345_c_int64_t
  real(sp) :: temp = 0.0_sp, topp = 1.0_sp, pres = 0.0_sp, freq = 0.0_sp, rep = 1.0_sp, plen = 0.0_sp
  integer :: nblock = 0, pwin = 0
  integer :: nloops = 1, pchunk = 64
  integer :: kspec = 0, nmatch = 2, keff, na, corr, jj, pp
  integer :: npass_sp = 0, nacc_sp = 0, spec_probe = 8
  integer :: npass_all = 0, nacc_all = 0, ngen_sp = 0, nplain_left = 0, nhist
  real(sp) :: spec_minacc = 0.5_sp
  logical :: spec_on, mode_spec, round_spec
  integer, allocatable :: draft(:), targ(:)
  real(sp), allocatable :: outspec(:)
  real(sp), allocatable :: outR(:), outc(:)
  integer :: npre, srow, tb
  logical :: dostats = .false.
  logical :: dostream = .true.
  integer :: s0, s1, srate
  integer(c_int64_t) :: cms_pre = 0, cms_dec = 0
  integer, allocatable :: pbytes(:), pids(:), idx(:), obytes(:), sbytes(:)
  integer :: nbytes, snbytes
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, mchk
  d2 = HD / 2

  call set_mode('response_file')
  call set_args('--tables /home/pauli/.cache/autoresearch/tok_tables --weights /tmp/w_long100/best --n 20 --temp 0.0' // &
      ' --seed 12345 --topp 1.0 --pres 0.0 --freq 0.0 --rep 1.0 --pwin 0 --plen 0.0 --nblock 0' // &
      ' --stats F --template TEMPLATE --system SYSTEM --stop STOP --stream T --loops 1 --pchunk 64' // &
      ' --spec 0 --match 2 --spec-probe 8 --spec-min-acc 0.5', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  chat_text - pure-Fortran text-in/text-out GPT inference', &
      'SYNOPSIS', &
      '  chat_text --tables DIR --weights DIR --n N [options] < stdin', &
      '  chat_text @rsp_file (response file, see chat.rsp)', &
      'OPTIONS', &
      '  --tables DIR  tokenizer tables (ranks.txt, unicode_*.txt)', &
      '  --weights DIR checkpoint .npy set (see export_weights.py)', &
      '  --n N         tokens to generate', &
      '  --temp T      temperature (0 = greedy)', &
      '  --seed S      RNG seed', &
      '  --topp P      nucleus cutoff (1 = off)', &
      '  --pres X      presence penalty', &
      '  --freq X      frequency penalty', &
      '  --rep X       CTRL repetition penalty theta (1.0=off, 1.2=on)', &
      '  --pwin N      windowed penalty: only last W tokens (0=full history)', &
      '  --plen X      length penalty (with --pwin)', &
      '  --nblock N    no-repeat n-gram size (0 = off)', &
      '  --loops N     recurrent passes/token on tied layer-0 (1 = plain GPT)', &
      '  --pchunk N    prompt tokens per cached prefill pass (1 = per-token)', &
      '  --spec K      greedy speculative decode with a prompt-lookup drafter', &
      '                (0 = off; requires temp 0 and no penalties)', &
      '  --match M     lookup drafter match length in tokens (default 2)', &
      '  --spec-probe N rounds before judging the drafter (default 8)', &
      '  --spec-min-acc X  fall back to plain decode if mean acceptance', &
      '                after the probe is below X (default 0.5)', &
      '  --template S  chat template with {prompt} (default: chat)', &
      '                use "raw" for no wrapping (completion mode)', &
      '  --system S    system prompt (prepended as ### SYSTEM)', &
      '  --stop S      comma-separated stop sequences', &
      '  --stream T    stream tokens as generated (default: on, use --stream F to disable)', &
      '  --stats T     print prefill/decode ms + tok/s to stderr'], &
      version_text=[character(len=80) :: 'chat_text 1.1'])
  tdir = trim(sget('tables'))
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  nloops = iget('loops')
  pchunk = iget('pchunk')
  kspec = iget('spec')
  nmatch = iget('match')
  spec_probe = iget('spec-probe')
  spec_minacc = rget('spec-min-acc')
  temp = rget('temp')
  rng = int(iget('seed'), c_int64_t)
  topp = rget('topp')
  pres = rget('pres')
  freq = rget('freq')
  rep = rget('rep')
  pwin = iget('pwin')
  plen = rget('plen')
  nblock = iget('nblock')
  spec_on = kspec > 0
  if (spec_on) then
    ! Greedy-only: the verification rows are scored through the SAME
    ! sample_next(master) call as the plain loop (temp 0 -> penalized
    ! argmax), each row with the history the plain path would have had at
    ! that position, so penalties (pres/freq/rep/pwin/plen/nblock) are now
    ! supported and equivalence still holds exactly. Only sampling > 0
    ! (stochastic draft-verify needs rejection sampling) is refused.
    if (temp /= 0.0_sp) then
      print '(A)', "--spec needs greedy decoding: set --temp 0"
      call exit(2)
    end if
    if (nloops /= 1) then
      print '(A)', "--spec has no cache path for --loops > 1"
      call exit(2)
    end if
  end if
  if (n_gen < 1) then
    print '(A)', 'require --n N>=1 (--help for all)'
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

  ! --- templating (customizable, default = chat) ---
  if (specified('template')) then
    tmpl_raw = trim(sget('template'))
    call strip_quotes(tmpl_raw)
  else
    call ckpt_dialect(trim(wdir), dia, diafound)
    if (diafound .and. trim(dia) /= DIALECT_NAME) then
      print '(2A)', "unknown checkpoint dialect (retrain or convert): ", trim(dia)
      call exit(1)
    end if
    tmpl_raw = default_chat_template()
  end if
  if (specified('system')) then
    sys_raw = trim(sget('system'))
    call strip_quotes(sys_raw)
  else
    sys_raw = ""
  end if
  if (trim(tmpl_raw) /= "raw") then
    block
      integer, allocatable :: tmp(:)
      integer :: ntmp
      call apply_template(pbytes, fsize, tmpl_raw, sys_raw, tmp, ntmp)
      deallocate(pbytes)
      call move_alloc(tmp, pbytes)
      fsize = ntmp
    end block
  end if

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
  allocate(outR(B*ntot*VV))

  ! prefill prompt + generate, one cached step per token. Last forward
  ! needed is t=ntot-1 (its logits predict idx(ntot)); step ntot would
  ! write idx(ntot+1), out of bounds.
  dostats = specified('stats')
  dostream = .true.
  if (specified('stream')) dostream = flag_is_true(sget('stream'))
  if (dostats) call system_clock(count_rate=srate)
  cms_pre = 0; cms_dec = 0

  ! ---- prompt prefill, pchunk tokens per pass (--pchunk 1 = per-token) ----
  ! gpt_step_multi is row-for-row identical to the per-token loop
  ! (asserted in test_kv_chunk_equiv), so this changes speed only: T=1
  ! GEMVs become T=pchunk GEMMs. nloops>1 has no cache path; leave it.
  npre = nprompt - 1
  if (dostats) call system_clock(s0)
  if (nloops == 1 .and. npre >= 1 .and. pchunk > 1) then
    allocate(outc(max(1, min(pchunk, npre))*VV))
    srow = 1
    do while (srow <= npre)
      tb = min(pchunk, npre - srow + 1)
      call gpt_step_multi(idx(srow:srow+tb-1), cos_b((srow-1)*d2+1:), &
          sin_b((srow-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ckv, cvv, clen, ntot, outc, &
          B, VV, D, N_HEAD, N_KV, HD, N_LAYER, tb, 1.0e-5_sp)
      srow = srow + tb
    end do
    deallocate(outc)
  else
    do step = 1, npre
      call gpt_step(idx(step:step), cos_b((step-1)*d2+1:), &
          sin_b((step-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
          ckv, cvv, clen, ntot, out1, &
          B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
    end do
  end if
  if (dostats) then
    call system_clock(s1)
    cms_pre = s1 - s0
  end if

  ! ---- decode: positions nprompt..ntot-1 -------------------------------
  ! With --spec the loop alternates between speculative rounds and plain
  ! steps under a sliding-window controller: acceptance is not stationary
  ! (early rounds follow the prompt, later rounds follow the model's own
  ! republishing), so a single startup probe misjudges exactly the case
  ! spec is best at. Every --spec-probe spec rounds the window is scored;
  ! below --spec-min-acc the next window runs plain, then spec is retried.
  ! Both modes keep the same invariant (clen = pp-1, idx(pp) pending), so
  ! switching never changes the emitted stream.
  if (spec_on) then
    allocate (draft(kspec), targ(kspec+1), outspec((kspec+2)*VV))
    pp = nprompt
    mode_spec = .true.
    nplain_left = 0
    do while (pp < ntot)
      if (dostats) call system_clock(s0)
      round_spec = mode_spec
      if (round_spec) then
        call lookup_draft(idx, pp, nmatch, kspec, draft, keff)
        ! Nothing to verify (no context match): a plain step is strictly
        ! cheaper than a verification pass over one token, so skip spec for
        ! this round. This is what removes the downside the controller was
        ! compensating for; the controller is now a second line of defence
        ! for the keff>0-but-low-acceptance regime.
        if (keff == 0) round_spec = .false.
      end if
      if (round_spec) then
        ! ---- one speculative round: propose, verify, commit ----------------
        if (pp + keff > ntot) keff = max(0, ntot - pp)
        block
          integer :: inp(kspec + 1)
          inp(1) = idx(pp)
          if (keff > 0) inp(2:keff+1) = draft(1:keff)
          call gpt_step_multi(inp(1:keff+1), cos_b((pp-1)*d2+1:), &
              sin_b((pp-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
              ckv, cvv, clen, ntot, outspec, &
              B, VV, D, N_HEAD, N_KV, HD, N_LAYER, keff + 1, 1.0e-5_sp)
        end block
        mchk = maxval(outspec(1:(keff+1)*VV))
        if (.not. (mchk == mchk)) then
          print '(A)', "NaN logit — abort"; call exit(1)
        end if
        ! sequential, penalty-aware accept: row jj is scored with exactly the
        ! history the plain loop would have at that position, and accepted
        ! drafts enter idx (and thus the history) as they are accepted. With
        ! penalties off this reduces to accept_prefix's rule.
        na = 0
        do jj = 1, keff + 1
          nhist = pp + jj - 1 - nprompt
          targ(jj) = sample_next(outspec((jj-1)*VV+1:jj*VV), VV, temp, &
              topp, pres, freq, rep, pwin, plen, idx(nprompt+1:), nhist, &
              nblock, rng) - 1
          if (jj <= keff) then
            if (draft(jj) /= targ(jj)) exit
            na = na + 1
            idx(pp + jj) = draft(jj)
          end if
        end do
        corr = targ(na + 1)
        if (pp + na + 1 > ntot) then
          na = ntot - pp - 1
          corr = targ(na + 1)
        end if
        if (na > 0) idx(pp+1:pp+na) = draft(1:na)
        idx(pp + na + 1) = corr
        if (dostream) then
          call decode(idx(pp+1:pp+na+1), na + 1, sbytes, snbytes)
          do i = 1, snbytes
            write (*, '(A)', advance='no') char(sbytes(i))
          end do
          flush(output_unit)
          deallocate(sbytes)
        end if
        pp = pp + na + 1
        clen = pp - 1
        npass_sp = npass_sp + 1
        nacc_sp = nacc_sp + na
        npass_all = npass_all + 1
        nacc_all = nacc_all + na
        ngen_sp = ngen_sp + na + 1
      else
        ! ---- one plain greedy step (controller pause) ---------------------
        call gpt_step(idx(pp:pp), cos_b((pp-1)*d2+1:), sin_b((pp-1)*d2+1:), &
            wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ckv, cvv, clen, ntot, out1, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
        mchk = maxval(out1)
        if (.not. (mchk == mchk)) then
          print '(A)', "NaN logit — abort"; call exit(1)
        end if
        best = sample_next(out1, VV, temp, topp, pres, freq, rep, pwin, plen, &
            idx(nprompt+1:), pp - nprompt, nblock, rng)
        idx(pp + 1) = best - 1
        if (dostream) then
          call decode(idx(pp+1:pp+1), 1, sbytes, snbytes)
          do i = 1, snbytes
            write (*, '(A)', advance='no') char(sbytes(i))
          end do
          flush(output_unit)
          deallocate(sbytes)
        end if
        pp = pp + 1
        clen = pp - 1
      end if
      if (dostats) then
        call system_clock(s1)
        cms_dec = cms_dec + (s1 - s0)
      end if
      ! sliding-window controller
      if (mode_spec) then
        if (npass_sp >= spec_probe) then
          if (real(nacc_sp, sp) < spec_minacc*real(npass_sp, sp)) then
            write (0, '(A,F5.2,A,F5.2,A)') "spec: window mean_accept=", &
                real(nacc_sp, sp)/real(npass_sp, sp), " < ", spec_minacc, &
                " — pausing spec for one probe window"
            mode_spec = .false.
            nplain_left = spec_probe
          end if
          npass_sp = 0
          nacc_sp = 0
        end if
      else
        nplain_left = nplain_left - 1
        if (nplain_left <= 0) then
          mode_spec = .true.
          npass_sp = 0
          nacc_sp = 0
        end if
      end if
    end do
    deallocate (draft, targ, outspec)
  else
    do step = nprompt, ntot - 1
      tc = step
      if (dostats) call system_clock(s0)
      if (nloops > 1) then
        call recurrent_forward(idx(1:tc), cos_b, sin_b, &
            wte, c_q(1:N_HEAD*HD*D), c_k(1:N_KV*HD*D), c_v(1:N_KV*HD*D), &
            c_pr(1:D*N_HEAD*HD), c_fc(1:4*D*D), c_pr2(1:D*4*D), lm, &
            outR, B, tc, VV, D, N_HEAD, N_KV, HD, nloops, 1.0e-5_sp)
        out1 = outR((tc-1)*VV+1:tc*VV)
      else
        call gpt_step(idx(tc:tc), cos_b((tc-1)*d2+1:), sin_b((tc-1)*d2+1:), &
            wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ckv, cvv, clen, ntot, out1, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
      end if
      if (dostats) then
        call system_clock(s1)
        cms_dec = cms_dec + (s1 - s0)
      end if
      mchk = maxval(out1)
      if (.not. (mchk == mchk)) then
        print '(A)', "NaN logit — abort"; call exit(1)
      end if
      best = sample_next(out1, VV, temp, topp, pres, freq, rep, pwin, plen, &
          idx(nprompt+1:), tc - nprompt, nblock, rng)
      idx(tc+1) = best - 1
      if (dostream) then
        call decode(idx(tc+1:tc+1), 1, sbytes, snbytes)
        do i = 1, snbytes
          write (*, '(A)', advance='no') char(sbytes(i))
        end do
        flush(output_unit)
        deallocate(sbytes)
      end if
    end do
  end if

  if (dostats) write (0, '(A,F10.1,A,F10.1,A,F8.2)') "stats: prefill_ms=", &
      1000.0 * real(cms_pre) / real(srate), " decode_ms=", &
      1000.0 * real(cms_dec) / real(srate), " tok_s=", &
      real(n_gen) / max(1.0e-9, real(cms_pre + cms_dec) / real(srate))
  if (dostats .and. spec_on) write (0, '(A,I0,A,F6.2,A,F6.2,A,I0)') &
      "spec: passes=", npass_all, " tokens/pass=", &
      real(ngen_sp, sp) / real(max(1, npass_all), sp), " mean_accept=", &
      real(nacc_all, sp) / real(max(1, npass_all), sp), " spec_tokens=", &
      ngen_sp
  if (dostream) then
    write (*, '(A)') ""
  else
    call decode(idx(nprompt+1:), n_gen, obytes, nbytes)
    ! stop-sequence truncation (only for non-streaming; streaming already flushed)
    if (specified('stop')) then
      stop_raw = trim(sget('stop'))
    else
      stop_raw = "### USER,### SYSTEM"
    end if
    if (len_trim(stop_raw) > 0) call truncate_at_stop(obytes, nbytes, stop_raw)
    open (newunit=u, file="/dev/stdout", access="stream", form="unformatted")
    do i = 1, nbytes
      write (u) char(obytes(i))
    end do
    close (u)
  end if

end program chat_text
