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
  use fortran_gpt_mod
  use load_weights_mod, only: load_gpt_weights
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use sample_mod, only: sample_token, sample_next
  use fortran_kv_mod, only: gpt_step
  use M_CLI2, only: set_args, set_mode, sget, rget, iget, specified
  implicit none

  integer, parameter :: sp = c_float
  integer, parameter :: B = 1, D = 768, N_HEAD = 6, N_KV = 6, HD = 128
  integer, parameter :: N_LAYER = 12, VV = 8192, BOS = 8188

  character(len=512) :: tdir, wdir, arg
  character(len=1) :: cb
  character(len=4096) :: tmpl_raw, sys_raw, stop_raw
  integer :: n_gen, u, ios, fsize, i, j, best, tc, step, nprompt
  integer :: ntot, d2, clen
  real(sp), allocatable :: ckv(:), cvv(:), out1(:)
  integer(c_int64_t) :: rng = 12345_c_int64_t
  real(sp) :: temp = 0.0_sp, topp = 1.0_sp, pres = 0.0_sp, freq = 0.0_sp
  integer :: nblock = 0
  logical :: dostats = .false.
  logical :: dostream = .false.
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
      ' --seed 12345 --topp 1.0 --pres 0.0 --freq 0.0 --nblock 0' // &
      ' --stats F --template TEMPLATE --system SYSTEM --stop STOP --stream F', &
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
      '  --nblock N    no-repeat n-gram size (0 = off)', &
      '  --template S  chat template with {prompt} (default: chat)', &
      '                use "raw" for no wrapping (completion mode)', &
      '  --system S    system prompt (prepended as ### SYSTEM)', &
      '  --stop S      comma-separated stop sequences', &
      '  --stream T    stream tokens as generated (flush each)', &
      '  --stats T     print prefill/decode ms + tok/s to stderr'], &
      version_text=[character(len=80) :: 'chat_text 1.1'])
  tdir = trim(sget('tables'))
  wdir = trim(sget('weights'))
  n_gen = iget('n')
  temp = rget('temp')
  rng = int(iget('seed'), c_int64_t)
  topp = rget('topp')
  pres = rget('pres')
  freq = rget('freq')
  nblock = iget('nblock')
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
    tmpl_raw = "### USER" // char(10) // "{prompt}" // char(10) // char(10) // "### ASSISTANT" // char(10)
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

  ! prefill prompt + generate, one cached step per token. Last forward
  ! needed is t=ntot-1 (its logits predict idx(ntot)); step ntot would
  ! write idx(ntot+1), out of bounds.
  dostats = specified('stats')
  dostream = specified('stream')
  if (dostats) call system_clock(count_rate=srate)
  if (dostream) open (newunit=u, file="/dev/stdout", access="stream", form="unformatted")
  cms_pre = 0; cms_dec = 0
  do step = 1, ntot - 1
    tc = step
    if (dostats) call system_clock(s0)
    call gpt_step(idx(tc:tc), cos_b((tc-1)*d2+1:), sin_b((tc-1)*d2+1:), &
        wte, c_q, c_k, c_v, c_pr, c_fc, c_pr2, lm, &
        ckv, cvv, clen, ntot, out1, &
        B, VV, D, N_HEAD, N_KV, HD, N_LAYER, 1.0e-5_sp)
    if (dostats) then
      call system_clock(s1)
      if (step < nprompt) then
        cms_pre = cms_pre + (s1 - s0)
      else
        cms_dec = cms_dec + (s1 - s0)
      end if
    end if
    if (step < nprompt) cycle   ! prompt steps only fill the cache
    mchk = maxval(out1)
    if (.not. (mchk == mchk)) then
      print '(A)', "NaN logit — abort"; call exit(1)
    end if
    best = sample_next(out1, VV, temp, topp, pres, freq, &
        idx(nprompt+1:), tc - nprompt, nblock, rng)
    idx(tc+1) = best - 1
    if (dostream) then
      call decode(idx(tc+1:tc+1), 1, sbytes, snbytes)
      do i = 1, snbytes
        write (u) char(sbytes(i))
      end do
      flush(u)
      deallocate(sbytes)
    end if
  end do

  if (dostats) write (0, '(A,F10.1,A,F10.1,A,F8.2)') "stats: prefill_ms=", &
      1000.0 * real(cms_pre) / real(srate), " decode_ms=", &
      1000.0 * real(cms_dec) / real(srate), " tok_s=", &
      real(n_gen) / max(1.0e-9, real(cms_pre + cms_dec) / real(srate))
  if (dostream) then
    close (u)
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

contains

  subroutine apply_template(inp, n_in, tmpl, sys, out, n_out)
    integer, intent(in) :: inp(*), n_in
    character(*), intent(in) :: tmpl, sys
    integer, allocatable, intent(out) :: out(:)
    integer, intent(out) :: n_out
    character(len=:), allocatable :: pre, post, sys_part, t2
    integer :: at, k, n1, n2, ns, ii
    allocate(character(len=len_trim(tmpl)) :: t2)
    t2 = trim(tmpl)
    call unescape_nl(t2)
    if (len_trim(sys) > 0) then
      sys_part = "### SYSTEM" // char(10) // trim(sys) // char(10) // char(10)
    else
      sys_part = ""
    end if
    at = index(t2, "{prompt}")
    if (at > 0) then
      pre = sys_part // t2(1:at-1)
      post = t2(at+8:)
    else
      pre = sys_part // t2
      post = ""
    end if
    n1 = len(pre); n2 = len(post); ns = len_trim(sys_part)
    ! count without trimming inner spaces: use len of pre/post as built
    n_out = len(pre) + n_in + len(post)
    allocate(out(n_out))
    k = 1
    do ii = 1, len(pre)
      out(k) = ichar(pre(ii:ii)); k = k + 1
    end do
    do ii = 1, n_in
      out(k) = inp(ii); k = k + 1
    end do
    do ii = 1, len(post)
      out(k) = ichar(post(ii:ii)); k = k + 1
    end do
  end subroutine apply_template

  subroutine unescape_nl(s)
    character(len=:), allocatable, intent(inout) :: s
    character(len=:), allocatable :: r
    integer :: i, j
    allocate(character(len=len(s)) :: r)
    j = 0
    i = 1
    do while (i <= len(s))
      if (s(i:i) == "\\" .and. i < len(s) .and. s(i+1:i+1) == "n") then
        j = j + 1; r(j:j) = char(10); i = i + 2
      else if (s(i:i) == "\\" .and. i < len(s) .and. s(i+1:i+1) == "t") then
        j = j + 1; r(j:j) = char(9); i = i + 2
      else
        j = j + 1; r(j:j) = s(i:i); i = i + 1
      end if
    end do
    s = r(1:j)
  end subroutine unescape_nl

  subroutine strip_quotes(s)
    character(len=*), intent(inout) :: s
    integer :: n
    n = len_trim(s)
    if (n >= 2) then
      if ((s(1:1) == '"' .and. s(n:n) == '"') .or. (s(1:1) == "'" .and. s(n:n) == "'")) then
        s = s(2:n-1)
      end if
    end if
  end subroutine strip_quotes

  subroutine truncate_at_stop(bytes, n, stops)
    integer, intent(inout) :: n
    integer, intent(in) :: bytes(*)
    character(*), intent(in) :: stops
    character(len=:), allocatable :: txt, stop1
    integer :: p, q, cut, best_cut
    allocate(character(len=n) :: txt)
    do p = 1, n; txt(p:p) = char(bytes(p)); end do
    best_cut = 0
    p = 1
    do
      q = index(stops(p:), ",")
      if (q == 0) then
        stop1 = trim(adjustl(stops(p:)))
        if (len_trim(stop1) > 0) then
          cut = index(txt, trim(stop1))
          if (cut > 0 .and. (best_cut == 0 .or. cut < best_cut)) best_cut = cut
        end if
        exit
      else
        stop1 = trim(adjustl(stops(p:p+q-2)))
        if (len_trim(stop1) > 0) then
          cut = index(txt, trim(stop1))
          if (cut > 0 .and. (best_cut == 0 .or. cut < best_cut)) best_cut = cut
        end if
        p = p + q
      end if
    end do
    if (best_cut > 0) n = best_cut - 1
  end subroutine truncate_at_stop

end program chat_text
