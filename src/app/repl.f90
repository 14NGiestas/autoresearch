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
  use iso_fortran_env, only: output_unit
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use sample_mod, only: sample_token, sample_next
  use fortran_kv_mod, only: gpt_step, gpt_step_multi
  use M_CLI2, only: set_args, set_mode, sget, rget, iget, specified
  use fortran_chat_mod
  use fortran_spec_mod, only: lookup_draft
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, BOS = 8188
  integer, parameter :: MAXT = 2048   ! checkpoint sequence_len; cache cap

  character(len=512) :: tdir, wdir, arg
  character(len=8192) :: linebuf
  character(len=4096) :: tmpl_raw, sys_raw, stop_raw
  character(len=:), allocatable :: dia
  logical :: diafound
  integer :: n_gen, u, ios, i, j, best, tc, step, nprompt, nbytes, nlen
  integer :: ntot, d2, clen
  logical :: dostats = .false.
  logical :: dostream = .false.
  integer :: ca, cb, crate
  integer(c_int64_t) :: cms_pre = 0, cms_dec = 0
  real(sp), allocatable :: ckv(:), cvv(:), out1(:)
  integer(c_int64_t) :: rng = 12345_c_int64_t
  real(sp) :: temp = 0.0_sp, topp = 1.0_sp, pres = 0.0_sp, freq = 0.0_sp, rep = 1.0_sp, plen = 0.0_sp
  integer :: nblock = 0, pwin = 0
  integer, allocatable :: pbytes(:), pids(:), idx(:), obytes(:), sbytes(:)
  integer :: snbytes
  integer :: pchunk = 64, npre, srow, tb
  integer :: kspec = 0, nmatch = 2, keff, na, corr, jj, pp
  integer :: npass_sp = 0, nacc_sp = 0, spec_probe = 8
  integer :: npass_all = 0, nacc_all = 0, ngen_sp = 0, nplain_left = 0, nhist
  real(sp) :: spec_minacc = 0.5_sp
  logical :: spec_on, mode_spec, round_spec
  integer, allocatable :: draft(:), targ(:)
  real(sp), allocatable :: outspec(:)
  real(sp), allocatable :: outc(:)
  real(sp), allocatable :: cos_b(:), sin_b(:)
  real(sp), allocatable :: wte(:), lm(:)
  real(sp), allocatable :: c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_pr2(:)
  real(sp), allocatable :: outp(:)
  real(sp) :: theta, ang, mchk

  n_gen = 40
  call set_mode('response_file')
  call set_args('--tables /home/pauli/.cache/autoresearch/tok_tables --weights /tmp/w_long100/best --n 40 --temp 0.0' // &
      ' --seed 12345 --topp 1.0 --pres 0.0 --freq 0.0 --rep 1.0 --pwin 0 --plen 0.0 --nblock 0' // &
      ' --stats F --template TEMPLATE --system SYSTEM --stop STOP --stream F --pchunk 64' // &
      ' --spec 0 --match 2 --spec-probe 8 --spec-min-acc 0.5', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  repl - interactive pure-Fortran chat REPL', &
      'SYNOPSIS', &
      '  repl --tables DIR --weights DIR [options]', &
      '  repl @rsp_file (response file, see chat.rsp)', &
      'OPTIONS', &
      '  --tables DIR  tokenizer tables', &
      '  --weights DIR checkpoint .npy set', &
      '  --n N         tokens per turn (empty line quits)', &
      '  --temp T --seed S --topp P --pres X --freq X --rep X --pwin W --plen X --nblock N', &
      '                sampling controls (see chat_text --help)', &
      '  --template S  chat template with {prompt} (default: chat)', &
      '  --system S    system prompt (prepended as ### SYSTEM)', &
      '  --stop S      comma-separated stop sequences', &
      '  --stream T    stream tokens as generated', &
      '  --pchunk N    prompt tokens per cached prefill pass (1 = per-token)', &
      '  --spec K      greedy speculative decode with a prompt-lookup drafter', &
      '                (0 = off; needs temp 0 and no penalties)', &
      '  --match M     lookup drafter match length in tokens (default 2)', &
      '  --spec-probe N --spec-min-acc X  adaptive guard: score each window of', &
      '                N spec rounds, pause spec for one window if mean', &
      '                acceptance < X (default 8 / 0.5; X=0 = never pause)', &
      '  --stats T     print per-turn prefill/decode ms + tok/s'], &
      version_text=[character(len=80) :: 'repl 1.1'])
  tdir = trim(sget('tables'))
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  temp = rget('temp')
  rng = int(iget('seed'), c_int64_t)
  topp = rget('topp')
  pres = rget('pres')
  freq = rget('freq')
  rep = rget('rep')
  pwin = iget('pwin')
  plen = rget('plen')
  nblock = iget('nblock')
  kspec = iget('spec')
  nmatch = iget('match')
  spec_probe = iget('spec-probe')
  spec_minacc = rget('spec-min-acc')
  spec_on = kspec > 0
  if (spec_on) then
    ! The verification rows go through the same sample_next call as the
    ! plain loop (temp 0 -> penalized argmax) with the history the plain
    ! path would have at that position, so penalties are supported and
    ! equivalence holds exactly. Stochastic sampling is not.
    if (temp /= 0.0_sp) then
      write (0, '(A)') '--spec needs greedy decoding: set --temp 0'
      call exit(2)
    end if
  end if
  ! tables/weights have defaults (see set_args) so bare 'fpm run repl' works;
  ! response file @chat can still override them.
  if (n_gen < 1) then
    write (0, '(A)') 'require --n N>=1 (--help for all)'
    call exit(2)
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
    ! templating (default = chat)
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
        call apply_template(pbytes, nlen, tmpl_raw, sys_raw, tmp, ntmp)
        deallocate(pbytes)
        call move_alloc(tmp, pbytes)
        nlen = ntmp
      end block
    end if

    call encode_bytes(pbytes, nlen, pids)
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

    ! ---- prompt prefill, pchunk tokens per pass (--pchunk 1 = per-token) ----
    ! gpt_step_multi is row-for-row identical to the per-token loop
    ! (asserted in test_kv_chunk_equiv): speed only.
    npre = nprompt - 1
    dostats = specified('stats')
    dostream = .true.
    if (specified('stream')) dostream = flag_is_true(sget('stream'))
    if (specified('pchunk')) pchunk = iget('pchunk')
    if (dostats) call system_clock(count_rate=crate)
    cms_pre = 0; cms_dec = 0
    if (dostats) call system_clock(ca)
    if (npre >= 1 .and. pchunk > 1) then
      allocate(outc(max(1, min(pchunk, npre))*VV))
      srow = 1
      do while (srow <= npre)
        tb = min(pchunk, npre - srow + 1)
        call gpt_step_multi(idx(srow:srow+tb-1), cos_b((srow-1)*d2+1:), &
            sin_b((srow-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ckv, cvv, clen, MAXT, outc, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, tb, 1.0e-5_sp)
        srow = srow + tb
      end do
      deallocate(outc)
    else
      do step = 1, npre
        call gpt_step(idx(step:step), cos_b((step-1)*d2+1:), &
            sin_b((step-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ckv, cvv, clen, MAXT, out1, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
      end do
    end if
    if (dostats) then
      call system_clock(cb)
      cms_pre = cb - ca
    end if

    ! ---- decode: positions nprompt..ntot-1 ---------------------------------
    ! --spec: speculative rounds interleaved with plain steps. A round with
    ! no lookup proposal (keff=0) skips the verification pass entirely — a
    ! plain step is strictly cheaper than a pass over one token, which is
    ! what keeps spec from losing on novel text. A sliding-window controller
    ! (--spec-probe / --spec-min-acc) pauses spec for one window when the
    ! observed acceptance is too low, then retries: acceptance is NOT
    ! stationary, it grows as the model starts repeating itself. Both modes
    ! share the invariant clen = pp-1, so switching never changes output.
    if (spec_on) then
      allocate (draft(kspec), targ(kspec+1), outspec((kspec+2)*VV))
      pp = nprompt
      mode_spec = .true.
      nplain_left = 0
      do while (pp < ntot)
        if (dostats) call system_clock(ca)
        round_spec = mode_spec
        if (round_spec) then
          call lookup_draft(idx, pp, nmatch, kspec, draft, keff)
          if (keff == 0) round_spec = .false.
        end if
        if (round_spec) then
          if (pp + keff > ntot) keff = max(0, ntot - pp)
          block
            integer :: inp(kspec + 1)
            inp(1) = idx(pp)
            if (keff > 0) inp(2:keff+1) = draft(1:keff)
            call gpt_step_multi(inp(1:keff+1), cos_b((pp-1)*d2+1:), &
                sin_b((pp-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
                ckv, cvv, clen, MAXT, outspec, &
                B, VV, D, N_HEAD, N_KV, HD, N_LAYER, keff + 1, 1.0e-5_sp)
          end block
          mchk = maxval(outspec(1:(keff+1)*VV))
          if (.not. (mchk == mchk)) then
            write (0, '(A)') "NaN logit — abort"
            call exit(1)
          end if
          ! penalty-aware sequential accept (same rule as the plain loop)
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
            call decode_bytes(idx(pp+1:pp+na+1), na + 1, sbytes, snbytes)
            do j = 1, snbytes
              write (*, '(A)', advance='no') char(sbytes(j))
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
          call gpt_step(idx(pp:pp), cos_b((pp-1)*d2+1:), &
              sin_b((pp-1)*d2+1:), wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
              ckv, cvv, clen, MAXT, out1, &
              B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
          mchk = maxval(out1)
          if (.not. (mchk == mchk)) then
            write (0, '(A)') "NaN logit — abort"
            call exit(1)
          end if
          best = sample_next(out1, VV, temp, topp, pres, freq, rep, pwin, plen, &
              idx(nprompt+1:), pp - nprompt, nblock, rng)
          idx(pp + 1) = best - 1
          if (dostream) then
            call decode_bytes(idx(pp+1:pp+1), 1, sbytes, snbytes)
            do j = 1, snbytes
              write (*, '(A)', advance='no') char(sbytes(j))
            end do
            flush(output_unit)
            deallocate(sbytes)
          end if
          pp = pp + 1
          clen = pp - 1
        end if
        if (dostats) then
          call system_clock(cb)
          cms_dec = cms_dec + (cb - ca)
        end if
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
        if (dostats) call system_clock(ca)
        call gpt_step(idx(tc:tc), cos_b((tc-1)*d2+1:), sin_b((tc-1)*d2+1:), &
            wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
            ckv, cvv, clen, MAXT, out1, &
            B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
        if (dostats) then
          call system_clock(cb)
          cms_dec = cms_dec + (cb - ca)
        end if
        mchk = maxval(out1)
        if (.not. (mchk == mchk)) then
          write (0, '(A)') "NaN logit — abort"
          call exit(1)
        end if
        best = sample_next(out1, VV, temp, topp, pres, freq, rep, pwin, plen, &
            idx(nprompt+1:), tc - nprompt, nblock, rng)
        idx(tc+1) = best - 1
        if (dostream) then
          call decode_bytes(idx(tc+1:tc+1), 1, sbytes, snbytes)
          do j = 1, snbytes
            write (*, '(A)', advance='no') char(sbytes(j))
          end do
          flush(output_unit)
          deallocate(sbytes)
        end if
      end do
    end if
    deallocate(cos_b, sin_b)
    ! ckv/cvv/out1 are deallocated together at top of next turn
    ! (line 119: if (allocated(ckv)) deallocate(ckv, cvv, out1))

    if (dostats) write (0, '(A,F10.1,A,F10.1,A,F8.2)') &
        "stats: prefill_ms=", &
        1000.0 * real(cms_pre) / real(crate), " decode_ms=", &
        1000.0 * real(cms_dec) / real(crate), " tok_s=", &
        real(n_gen) / max(1.0e-9, real(cms_pre + cms_dec) / real(crate))
    if (dostream) then
      write (*, '(A)') ""
      deallocate(idx)
    else
      call decode_bytes(idx(nprompt+1:), n_gen, obytes, nbytes)
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
      write (u) char(10)
      close (u)
      deallocate(idx, obytes)
    end if
  end do
  write (0, '(A)') "bye."

end program repl
