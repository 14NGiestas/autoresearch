! lib/tokenizer_tables.f90 — data tables for the pure-Fortran BPE tokenizer.
!
! One-time exports (scripts/export_tokenizer.py, runtime needs no Python):
!   ranks.txt      8188 lines: <rank> <nbytes> <b0> ..
!   unicode_L.txt  <n> then <lo> <hi> codepoint ranges (\\p{L})
!   unicode_N.txt  <n> then <lo> <hi> codepoint ranges (\\p{N})
! \\s (Unicode White_Space, 25 codepoints) is hardcoded below.
!
! Provides: load_tables, codepoint_at (UTF-8), cat_of/is_space, rank_of
! (binary search over byte-sorted vocab), token bytes for decode.

module tokenizer_tables_mod
  implicit none

  integer, parameter :: TOK_N = 8188
  integer, parameter :: CAT_OTHER = 0, CAT_LETTER = 1, CAT_NUMBER = 2

  ! vocab store: token id -> bytes (0..255 per element)
  integer, allocatable :: tok_off(:)   ! tok_off(id) 0-based start, size TOK_N+1
  integer, allocatable :: tok_data(:)  ! concatenated bytes
  integer, allocatable :: tok_order(:) ! ids sorted by bytes (for bsearch)

  ! unicode ranges (sorted, binary-searched)
  integer, allocatable :: L_lo(:), L_hi(:), N_lo(:), N_hi(:)

  ! \\s = Unicode White_Space (fixed list, regex crate semantics)
  integer, parameter :: WS(25) = [9, 10, 11, 12, 13, 32, 133, 160, 5760, &
      8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, &
      8232, 8233, 8239, 8287, 12288]

contains

  subroutine load_tables(dir)
    character(*), intent(in) :: dir
    integer :: u, i, r, nb, b, ios, total
    character(len=512) :: line
    integer, allocatable :: lens(:)

    allocate(lens(0:TOK_N-1))
    open (newunit=u, file=trim(dir) // "/ranks.txt", status="old")
    read (u, *) total
    if (total /= TOK_N) then
      print '(A,I0)', "ranks.txt count mismatch: ", total
      call exit(1)
    end if
    ! first pass: lengths (lines have variable field counts)
    do i = 1, TOK_N
      read (u, '(A)') line
      read (line, *) r, nb
      lens(r) = nb
    end do
    close (u)
    allocate(tok_off(0:TOK_N))
    tok_off(0) = 0
    do i = 0, TOK_N - 1
      tok_off(i+1) = tok_off(i) + lens(i)
    end do
    allocate(tok_data(tok_off(TOK_N)))
    open (newunit=u, file=trim(dir) // "/ranks.txt", status="old")
    read (u, *)
    do i = 1, TOK_N
      read (u, '(A)') line
      read (line, *) r, nb, tok_data(tok_off(r)+1:tok_off(r)+nb)
    end do
    close (u)
    deallocate(lens)

    ! byte-sorted permutation (insertion sort is O(n^2); n=8188 one-time ok?
    ! no — use merge sort via recursive subroutine below)
    allocate(tok_order(0:TOK_N-1))
    do i = 0, TOK_N - 1
      tok_order(i) = i
    end do
    call mergesort_order(tok_order, 0, TOK_N - 1)

    call load_ranges(trim(dir) // "/unicode_L.txt", L_lo, L_hi)
    call load_ranges(trim(dir) // "/unicode_N.txt", N_lo, N_hi)
  end subroutine load_tables

  subroutine load_ranges(path, lo, hi)
    character(*), intent(in) :: path
    integer, allocatable, intent(out) :: lo(:), hi(:)
    integer :: u, n, i
    open (newunit=u, file=path, status="old")
    read (u, *) n
    allocate(lo(n), hi(n))
    do i = 1, n
      read (u, *) lo(i), hi(i)
    end do
    close (u)
  end subroutine load_ranges

  ! Sort tok_order by token bytes. a is 0-based absolute throughout:
  ! recursive calls pass the FULL array (never sections), so indices stay
  ! absolute. (Passing sections with relative bounds silently mis-sorts.)
  recursive subroutine mergesort_order(a, lo, hi)
    integer, intent(inout) :: a(0:)
    integer, intent(in) :: lo, hi
    integer :: mid, i, j, k
    integer, allocatable :: tmp(:)
    if (hi - lo < 1) return
    mid = (lo + hi) / 2
    call mergesort_order(a, lo, mid)
    call mergesort_order(a, mid+1, hi)
    allocate(tmp(0:hi-lo))
    i = lo; j = mid + 1; k = 0
    do while (i <= mid .and. j <= hi)
      if (tok_cmp(a(i), a(j)) <= 0) then
        tmp(k) = a(i); i = i + 1
      else
        tmp(k) = a(j); j = j + 1
      end if
      k = k + 1
    end do
    do while (i <= mid)
      tmp(k) = a(i); i = i + 1; k = k + 1
    end do
    do while (j <= hi)
      tmp(k) = a(j); j = j + 1; k = k + 1
    end do
    a(lo:hi) = tmp(0:hi-lo)
  end subroutine mergesort_order

  ! compare token bytes: -1/0/+1
  integer function tok_cmp(id1, id2)
    integer, intent(in) :: id1, id2
    integer :: l1, l2, k, n
    l1 = tok_off(id1+1) - tok_off(id1)
    l2 = tok_off(id2+1) - tok_off(id2)
    n = min(l1, l2)
    do k = 0, n - 1
      if (tok_data(tok_off(id1)+1+k) /= tok_data(tok_off(id2)+1+k)) then
        if (tok_data(tok_off(id1)+1+k) < tok_data(tok_off(id2)+1+k)) then
          tok_cmp = -1
        else
          tok_cmp = 1
        end if
        return
      end if
    end do
    if (l1 == l2) then
      tok_cmp = 0
    else if (l1 < l2) then
      tok_cmp = -1
    else
      tok_cmp = 1
    end if
  end function tok_cmp

  ! rank of byte string key(1:klen), or -1 if absent (binary search)
  integer function rank_of(key, klen)
    integer, intent(in) :: key(*), klen
    integer :: lo, hi, mid, c
    lo = 0; hi = TOK_N - 1
    rank_of = -1
    do while (lo <= hi)
      mid = (lo + hi) / 2
      c = key_cmp(key, klen, tok_order(mid))
      if (c == 0) then
        rank_of = tok_order(mid)
        return
      else if (c < 0) then
        hi = mid - 1
      else
        lo = mid + 1
      end if
    end do
  end function rank_of

  integer function key_cmp(key, klen, id)
    integer, intent(in) :: key(*), klen, id
    integer :: tl, k, n
    tl = tok_off(id+1) - tok_off(id)
    n = min(klen, tl)
    do k = 1, n
      if (key(k) /= tok_data(tok_off(id)+k)) then
        if (key(k) < tok_data(tok_off(id)+k)) then
          key_cmp = -1
        else
          key_cmp = 1
        end if
        return
      end if
    end do
    if (klen == tl) then
      key_cmp = 0
    else if (klen < tl) then
      key_cmp = -1
    else
      key_cmp = 1
    end if
  end function key_cmp

  ! decode one UTF-8 codepoint at bytes(pos..), 1-based. Invalid -> (b,1).
  subroutine codepoint_at(bytes, n, pos, cp, nb)
    integer, intent(in) :: bytes(*), n, pos
    integer, intent(out) :: cp, nb
    integer :: b0, b1, b2, b3
    b0 = bytes(pos)
    if (b0 < 128) then
      cp = b0; nb = 1
    else if (b0 >= 194 .and. b0 <= 223 .and. pos + 1 <= n) then
      b1 = bytes(pos+1)
      if (iand(b1, 192) /= 128) then
        cp = b0; nb = 1
      else
        cp = ior(ishft(iand(b0, 31), 6), iand(b1, 63)); nb = 2
      end if
    else if (b0 >= 224 .and. b0 <= 239 .and. pos + 2 <= n) then
      b1 = bytes(pos+1); b2 = bytes(pos+2)
      if (iand(b1, 192) /= 128 .or. iand(b2, 192) /= 128) then
        cp = b0; nb = 1
      else
        cp = ior(ior(ishft(iand(b0, 15), 12), ishft(iand(b1, 63), 6)), &
            iand(b2, 63)); nb = 3
      end if
    else if (b0 >= 240 .and. b0 <= 244 .and. pos + 3 <= n) then
      b1 = bytes(pos+1); b2 = bytes(pos+2); b3 = bytes(pos+3)
      if (iand(b1, 192) /= 128 .or. iand(b2, 192) /= 128 .or. &
          iand(b3, 192) /= 128) then
        cp = b0; nb = 1
      else
        cp = ior(ior(ior(ishft(iand(b0, 7), 18), ishft(iand(b1, 63), 12)), &
            ishft(iand(b2, 63), 6)), iand(b3, 63)); nb = 4
      end if
    else
      cp = b0; nb = 1
    end if
  end subroutine codepoint_at

  logical function in_ranges(cp, lo, hi)
    integer, intent(in) :: cp, lo(:), hi(:)
    integer :: a, b, m
    a = 1; b = size(lo)
    in_ranges = .false.
    do while (a <= b)
      m = (a + b) / 2
      if (cp < lo(m)) then
        b = m - 1
      else if (cp > hi(m)) then
        a = m + 1
      else
        in_ranges = .true.
        return
      end if
    end do
  end function in_ranges

  integer function cat_of(cp)
    integer, intent(in) :: cp
    if (in_ranges(cp, L_lo, L_hi)) then
      cat_of = CAT_LETTER
    else if (in_ranges(cp, N_lo, N_hi)) then
      cat_of = CAT_NUMBER
    else
      cat_of = CAT_OTHER
    end if
  end function cat_of

  logical function is_space(cp)
    integer, intent(in) :: cp
    integer :: i
    is_space = .false.
    do i = 1, size(WS)
      if (WS(i) == cp) then
        is_space = .true.
        return
      else if (WS(i) > cp) then
        return
      end if
    end do
  end function is_space

end module tokenizer_tables_mod
