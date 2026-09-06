! lib/fortran_chat.f90 — shared chat templating + stop truncation.
! Used by chat, chat_text, repl (was duplicated in 3 apps).

module fortran_chat_mod
  implicit none
contains

  subroutine apply_template(inp, n_in, tmpl, sys, out, n_out)
    integer, intent(in) :: inp(*), n_in
    character(*), intent(in) :: tmpl, sys
    integer, allocatable, intent(out) :: out(:)
    integer, intent(out) :: n_out
    character(len=:), allocatable :: pre, post, sys_part, t2
    integer :: at, k, ii
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
    j = 0; i = 1
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
    best_cut = 0; p = 1
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

end module fortran_chat_mod
