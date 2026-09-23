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
    integer, intent(in) :: bytes(:), n
    integer, intent(out) :: pstart(:), plen(:), npieces
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
    integer, intent(in) :: bytes(:), n, pos
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
    integer, intent(in) :: bytes(:), n, pos
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
    integer, intent(in) :: bytes(:), n, pos
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
    integer, intent(in) :: bytes(:), n, pos
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
    integer, intent(in) :: bytes(:), n, pos
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
    integer, intent(in) :: bytes(:), s, plen
    integer, intent(inout) :: out(:)
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
    integer, intent(in) :: bytes(:), n
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
    integer, intent(in) :: ids(:), nids
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

  ! ------------------------------------------------------------------------
  ! Corpus space = the mapping the TRAINING CORPORA actually use, taken from
  ! the rows themselves. It is NOT the byte rule I first assumed, and NOT what
  ! scripts/tokenize_corpus.py says either -- three different rules existed:
  !
  !   cp < 128          -> id = cp            (ASCII byte)
  !   128 <= cp < 256   -> id = 256 + cp      ('a'-tilde -> 483, NOT 227)
  !   cp >= 256         -> id = 256 + b for each UTF-8 byte of the char
  !
  ! Evidence, from 200 sampled rows of /tmp/prose: ZERO ids in 128..255, and
  ! 2271 occurrences of 483 (= 256+227, 'a'-tilde -- the commonest accented
  ! letter of Portuguese), plus 481/487/489/499/500 for acute-a, c-cedilla,
  ! acute-e, acute-o, circumflex-o. Decoding with this rule gives readable text
  ! ("em suas maos, ... De avos a netos"); UTF-8 accumulation gives mojibake.
  ! The script's `ord(ch) < 256` rule would have emitted 227, so it never ran
  ! on these corpora -- and the rows are the truth: they are what the model saw.
  !
  ! The consequence for inference is the third bug of this family: taking the
  ! terminal's UTF-8 bytes raw (as we did) turns 'a'-tilde (C3 A3) into ids
  ! 451/419 instead of 483, and emitting raw latin-1 bytes on the way out made
  ! accents unreadable on a UTF-8 terminal. Both directions are fixed here.
  !
  ! The BPE routines above remain the legacy tokenizer: using them at inference
  ! was the original bug (prompt in as 6 BPE ids instead of 13 bytes).
  integer function byte_to_id(b)
    integer, intent(in) :: b
    if (b < 128) then
      byte_to_id = b
    else
      byte_to_id = 256 + b
    end if
  end function byte_to_id

  ! UTF-8 bytes of a codepoint, for turning ids back into terminal-ready text.
  subroutine utf8_of_cp(cp, buf, nb)
    integer, intent(in) :: cp
    integer, intent(out) :: buf(4), nb
    if (cp < 128) then
      nb = 1; buf(1) = cp
    else if (cp < 2048) then
      nb = 2; buf(1) = 192 + cp / 64; buf(2) = 128 + mod(cp, 64)
    else if (cp < 65536) then
      nb = 3; buf(1) = 224 + cp / 4096
      buf(2) = 128 + mod(cp / 64, 64); buf(3) = 128 + mod(cp, 64)
    else
      nb = 4; buf(1) = 240 + cp / 262144
      buf(2) = 128 + mod(cp / 4096, 64); buf(3) = 128 + mod(cp / 64, 64)
      buf(4) = 128 + mod(cp, 64)
    end if
  end subroutine utf8_of_cp

  ! -1 = not a text id (BOS 8188 and anything undefined are not text)
  integer function id_to_byte(i)
    integer, intent(in) :: i
    if (i >= 0 .and. i < 128) then
      id_to_byte = i
    else if (i >= 256 .and. i < 512) then
      id_to_byte = i - 256
    else
      id_to_byte = -1
    end if
  end function id_to_byte

  ! Text (UTF-8 bytes) -> corpus ids. One id per ASCII byte, but for non-ASCII
  ! the CODEPOINT decides: below 256 it becomes 256+cp (one id), at or above it
  ! it becomes one id per UTF-8 byte (256+b). That asymmetry is the corpus's,
  ! not ours -- it is what the books produced when they were tokenized.
  subroutine encode_bytes(bytes, n, ids)
    integer, intent(in) :: bytes(:), n
    integer, allocatable, intent(out) :: ids(:)
    integer, allocatable :: tmp(:)
    integer :: pos, cp, nb, k, m
    allocate(tmp(4 * n + 8))
    m = 0
    pos = 1
    do while (pos <= n)
      call codepoint_at(bytes, n, pos, cp, nb)
      if (cp < 128) then
        m = m + 1; tmp(m) = cp
      else if (cp < 256) then
        m = m + 1; tmp(m) = 256 + cp
      else
        do k = 1, nb
          m = m + 1; tmp(m) = 256 + bytes(pos + k - 1)
        end do
      end if
      pos = pos + nb
    end do
    allocate(ids(m))
    if (m > 0) ids = tmp(:m)
  end subroutine encode_bytes

  ! Corpus ids -> UTF-8 bytes for display. Drops non-text ids (no more "<?>N").
  ! An id in 256..511 is a codepoint (id-256) by the rule above; but when a run
  ! of such ids forms a VALID UTF-8 sequence whose codepoint is >= 256, that is
  ! what it must have been (a character like an em dash, whose UTF-8 bytes were
  ! stored as 256+b) and we reassemble it -- otherwise "--" style punctuation
  ! would print as mojibake. Accented letters never trigger this: a lone
  ! 256..511 id decodes to a codepoint < 256, which is the common case.
  subroutine decode_bytes(ids, nids, bytes, nbytes)
    integer, intent(in) :: ids(:), nids
    integer, allocatable, intent(out) :: bytes(:)
    integer, intent(out) :: nbytes
    integer :: i, b, cp, k, need, j, ok, buf(4), nb
    integer :: seq(4)
    allocate(bytes(4 * nids + 8))
    nbytes = 0
    i = 1
    do while (i <= nids)
      b = id_to_byte(ids(i))
      if (b < 0) then
        i = i + 1
        cycle
      end if
      if (ids(i) < 128) then
        nbytes = nbytes + 1; bytes(nbytes) = b
        i = i + 1
        cycle
      end if
      ! try to reassemble a multi-byte character from following 256..511 ids
      if (b >= 194 .and. b <= 244) then
        if (b < 224) then; need = 1
        else if (b < 240) then; need = 2
        else; need = 3
        end if
        ok = 0
        if (i + need <= nids) then
          seq(1) = b
          ok = 1
          do j = 1, need
            if (ids(i + j) < 256 .or. ids(i + j) >= 512) then
              ok = 0
              exit
            end if
            seq(j + 1) = ids(i + j) - 256
            if (seq(j + 1) < 128 .or. seq(j + 1) > 191) then
              ok = 0
              exit
            end if
          end do
        end if
        if (ok == 1) then
          if (need == 1) then
            cp = mod(seq(1), 32) * 64 + mod(seq(2), 64)
          else if (need == 2) then
            cp = mod(seq(1), 16) * 4096 + mod(seq(2), 64) * 64 + mod(seq(3), 64)
          else
            cp = mod(seq(1), 8) * 262144 + mod(seq(2), 64) * 4096 + &
                mod(seq(3), 64) * 64 + mod(seq(4), 64)
          end if
          if (cp >= 256) then
            call utf8_of_cp(cp, buf, nb)
            do k = 1, nb
              nbytes = nbytes + 1; bytes(nbytes) = buf(k)
            end do
            i = i + need + 1
            cycle
          end if
        end if
      end if
      ! plain single codepoint 256..511 -> UTF-8 (so accents reach the terminal)
      call utf8_of_cp(b, buf, nb)
      do k = 1, nb
        nbytes = nbytes + 1; bytes(nbytes) = buf(k)
      end do
      i = i + 1
    end do
    bytes = bytes(:nbytes)
  end subroutine decode_bytes

end module tokenizer_encode_mod
