! lib/tokenizer_encode.f90 — pre-tokenizer scanner + BPE loop + decode.
!
! Faithful port of the GPT-4-style split pattern used at training
! (prepare.py SPLIT_PATTERN):
!   '(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}|
!     ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+
! Branches tried in order, first match wins at each position (regex
! leftmost semantics); possessive quantifiers never give characters back.
! BPE core is tiktoken's algorithm: repeatedly merge the adjacent pair
! whose concatenation has the lowest vocab rank.

module tokenizer_encode_mod
  use tokenizer_tables_mod
  implicit none
contains

  ! Split bytes(1:n) into pieces; returns starts/lens (byte units) + count.
  subroutine pretokenize(bytes, n, pstart, plen, npieces)
    integer, intent(in) :: bytes(*), n
    integer, intent(out) :: pstart(*), plen(*), npieces
    integer :: pos, epos
    npieces = 0
    pos = 1
    do while (pos <= n)
      call next_piece(bytes, n, pos, epos)
      npieces = npieces + 1
      pstart(npieces) = pos
      plen(npieces) = epos - pos
      pos = epos
    end do
  end subroutine pretokenize

  subroutine next_piece(bytes, n, pos, epos)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: epos
    integer :: cp, nb, cur
    call codepoint_at(bytes, n, pos, cp, nb)
    ! (a) contraction: ' + s/d/m/t/ll/ve/re (ASCII case-insensitive)
    if (bytes(pos) == 39) then
      if (match_contraction(bytes, n, pos, epos)) return
    end if
    ! (b) [^\\r\\n\\p{L}\\p{N}]?+ \\p{L}+
    if (match_letters(bytes, n, pos, epos)) return
    ! (c) \\p{N}{1,2}
    if (cat_of(cp) == CAT_NUMBER) then
      epos = pos + nb
      if (epos <= n) then
        call codepoint_at(bytes, n, epos, cp, nb)
        if (cat_of(cp) == CAT_NUMBER) epos = epos + nb
      end if
      return
    end if
    ! (d) ' '? punct+ \\r\\n*
    if (match_punct(bytes, n, pos, epos)) return
    ! (e) \\s* [\\r\\n]
    if (match_nl(bytes, n, pos, epos)) return
    ! (f) \\s+(?!\\S): maximal whitespace run; the negative lookahead
    ! forces backtracking to run-minus-last-codepoint unless the run
    ! reaches end of input (lookahead then holds trivially). A 1-char run
    ! followed by non-space fails (nothing left to match with).
    cur = pos
    do while (cur <= n)
      call codepoint_at(bytes, n, cur, cp, nb)
      if (.not. is_space(cp)) exit
      cur = cur + nb
    end do
    if (cur > pos) then
      if (cur > n) then
        epos = n + 1   ! run reaches end: lookahead holds
        return
      end if
      ! step back one codepoint from cur (first non-space)
      epos = cur - 1
      do while (epos > pos .and. iand(bytes(epos), 192) == 128)
        epos = epos - 1
      end do
      if (epos > pos) return   ! run-minus-one matched
      ! else single-char run + non-space ahead: fail -> (g)
    end if
    ! (g) maximal whitespace run (pos itself is space here)
    epos = pos
    do while (epos <= n)
      call codepoint_at(bytes, n, epos, cp, nb)
      if (.not. is_space(cp)) exit
      epos = epos + nb
    end do
  end subroutine next_piece

  logical function match_contraction(bytes, n, pos, epos)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: epos
    integer :: c1, c2
    match_contraction = .false.
    if (pos + 1 > n) return
    c1 = fold(bytes(pos+1))
    if (c1 == 115 .or. c1 == 100 .or. c1 == 109 .or. c1 == 116) then
      epos = pos + 2   ! 's 'd 'm 't
      match_contraction = .true.
      return
    end if
    if (pos + 2 > n) return
    c2 = fold(bytes(pos+2))
    if ((c1 == 108 .and. c2 == 108) .or. &   ! 'll
        (c1 == 118 .and. c2 == 101) .or. &   ! 've
        (c1 == 114 .and. c2 == 101)) then    ! 're
      epos = pos + 3
      match_contraction = .true.
    end if
  end function match_contraction

  integer function fold(b)
    integer, intent(in) :: b
    if (b >= 65 .and. b <= 90) then
      fold = b + 32
    else
      fold = b
    end if
  end function fold

  logical function match_letters(bytes, n, pos, epos)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: epos
    integer :: cp, nb, cur
    match_letters = .false.
    call codepoint_at(bytes, n, pos, cp, nb)
    cur = pos
    ! Possessive optional single prefix: one non-\\r\\n non-letter
    ! non-number, locked in (no retry without it). Adjudicated against
    ! PCRE2/Viktor oracles: multi-space+letter splits come from branch (f)
    ! below and BPE cascade, NOT from a longer prefix.
    if (cp /= 13 .and. cp /= 10 .and. cat_of(cp) == CAT_OTHER) then
      cur = cur + nb
      if (cur > n) return   ! consumed prefix, no room for L+ -> fail
      call codepoint_at(bytes, n, cur, cp, nb)
    end if
    if (cat_of(cp) /= CAT_LETTER) return
    do while (cur <= n)
      call codepoint_at(bytes, n, cur, cp, nb)
      if (cat_of(cp) /= CAT_LETTER) exit
      cur = cur + nb
    end do
    epos = cur
    match_letters = .true.
  end function match_letters

  logical function match_punct(bytes, n, pos, epos)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: epos
    integer :: cp, nb, cur, npunct
    match_punct = .false.
    cur = pos
    if (bytes(cur) == 32) cur = cur + 1   ! optional single space
    npunct = 0
    do while (cur <= n)
      call codepoint_at(bytes, n, cur, cp, nb)
      if (is_space(cp) .or. cat_of(cp) /= CAT_OTHER) exit
      cur = cur + nb
      npunct = npunct + 1
    end do
    if (npunct < 1) return
    do while (cur <= n)
      if (bytes(cur) /= 13 .and. bytes(cur) /= 10) exit
      cur = cur + 1
    end do
    epos = cur
    match_punct = .true.
  end function match_punct

  logical function match_nl(bytes, n, pos, epos)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: epos
    integer :: cp, nb, cur, lastnl
    match_nl = .false.
    cur = pos
    lastnl = -1
    do while (cur <= n)
      call codepoint_at(bytes, n, cur, cp, nb)
      if (.not. is_space(cp)) exit
      if (bytes(cur) == 13 .or. bytes(cur) == 10) lastnl = cur
      cur = cur + nb
    end do
    if (lastnl < 0) return
    epos = lastnl + 1
    match_nl = .true.
  end function match_nl

  ! BPE merge loop over piece bytes(s:s+plen-1); appends ranks to out().
  subroutine bpe_piece(bytes, s, plen, out, nout)
    integer, intent(in) :: bytes(*), s, plen
    integer, intent(inout) :: out(*)
    integer, intent(inout) :: nout
    integer :: starts(1024), lens(1024), nparts, i, r, best, best_rank
    integer :: key(256), klen, k
    if (plen > 1024) then
      print '(A)', "bpe_piece: piece too long"
      call exit(1)
    end if
    nparts = plen
    do i = 1, plen
      starts(i) = s + i - 1
      lens(i) = 1
    end do
    do while (nparts > 1)
      best = -1; best_rank = TOK_N + 1
      do i = 1, nparts - 1
        klen = lens(i) + lens(i+1)
        do k = 1, lens(i)
          key(k) = bytes(starts(i)+k-1)
        end do
        do k = 1, lens(i+1)
          key(lens(i)+k) = bytes(starts(i+1)+k-1)
        end do
        r = rank_of(key, klen)
        if (r >= 0 .and. r < best_rank) then
          best_rank = r; best = i
        end if
      end do
      if (best < 0) exit
      lens(best) = lens(best) + lens(best+1)
      do i = best + 1, nparts - 1
        starts(i) = starts(i+1)
        lens(i) = lens(i+1)
      end do
      nparts = nparts - 1
    end do
    do i = 1, nparts
      if (lens(i) == 1) then
        nout = nout + 1
        out(nout) = bytes(starts(i))   ! single bytes rank as themselves
      else
        klen = lens(i)
        do k = 1, klen
          key(k) = bytes(starts(i)+k-1)
        end do
        r = rank_of(key, klen)
        if (r < 0) then
          print '(A)', "bpe_piece: part missing from vocab"
          call exit(1)
        end if
        nout = nout + 1
        out(nout) = r
      end if
    end do
  end subroutine bpe_piece

  ! Full encode: bytes -> 0-based ids (exact-size allocatable).
  subroutine encode(bytes, n, ids)
    integer, intent(in) :: bytes(*), n
    integer, allocatable, intent(out) :: ids(:)
    integer, allocatable :: pstart(:), plen(:), tmp(:)
    integer :: npieces, i, nout
    allocate(pstart(n), plen(n), tmp(n))
    call pretokenize(bytes, n, pstart, plen, npieces)
    nout = 0
    do i = 1, npieces
      call bpe_piece(bytes, pstart(i), plen(i), tmp, nout)
    end do
    allocate(ids(nout))
    ids = tmp(1:nout)
  end subroutine encode

  ! Decode: 0-based ids -> bytes (specials emitted as literal names).
  subroutine decode(ids, nids, bytes, nbytes)
    integer, intent(in) :: ids(*), nids
    integer, allocatable, intent(out) :: bytes(:)
    integer, intent(out) :: nbytes
    integer :: i, tl, k, pos
    character(len=14) :: sp
    nbytes = 0
    do i = 1, nids
      if (ids(i) < TOK_N) then
        nbytes = nbytes + (tok_off(ids(i)+1) - tok_off(ids(i)))
      else
        write (sp, '(A,I0)') '<?>', ids(i)
        nbytes = nbytes + len_trim(sp)
      end if
    end do
    allocate(bytes(nbytes))
    pos = 1
    do i = 1, nids
      if (ids(i) < TOK_N) then
        tl = tok_off(ids(i)+1) - tok_off(ids(i))
        do k = 1, tl
          bytes(pos) = tok_data(tok_off(ids(i))+k)
          pos = pos + 1
        end do
      else
        write (sp, '(A,I0)') '<?>', ids(i)
        do k = 1, len_trim(sp)
          bytes(pos) = ichar(sp(k:k))
          pos = pos + 1
        end do
      end if
    end do
  end subroutine decode

end module tokenizer_encode_mod
