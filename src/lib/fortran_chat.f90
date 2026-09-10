! lib/fortran_chat.f90 — shared chat templating + stop truncation.
! Used by chat, chat_text, repl (was duplicated in 3 apps).

module fortran_chat_mod
  implicit none
  ! Canonical chat dialect (autoresearch-v1): single source of truth for
  ! role delimiters. Every checkpoint dir carries template.txt with these
  ! headers (write_template_txt, called on save); inference refuses
  ! unknown dialects loudly instead of misrendering (ckpt_dialect).
  character(len=*), parameter :: DIALECT_NAME = "autoresearch-v1"
  character(len=*), parameter :: HDR_SYSTEM = "### SYSTEM"
  character(len=*), parameter :: HDR_USER = "### USER"
  character(len=*), parameter :: HDR_ASSISTANT = "### ASSISTANT"
  character(len=*), parameter :: HDR_TOOL = "### TOOL"
contains

  subroutine apply_template(inp, n_in, tmpl, sys, out, n_out)
    integer, intent(in) :: inp(:), n_in
    character(*), intent(in) :: tmpl, sys
    integer, allocatable, intent(out) :: out(:)
    integer, intent(out) :: n_out
    character(len=:), allocatable :: pre, post, sys_part, t2
    integer :: at, k, ii
    allocate(character(len=len_trim(tmpl)) :: t2)
    t2 = trim(tmpl)
    call unescape_nl(t2)
    if (len_trim(sys) > 0) then
      sys_part = HDR_SYSTEM // char(10) // trim(sys) // char(10) // char(10)
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
    integer, intent(in) :: bytes(:)
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

  ! Default single-turn template, built from the canonical headers so the
  ! literal exists exactly once (was duplicated in chat_text/repl).
  function default_chat_template() result(t)
    character(len=:), allocatable :: t
    t = HDR_USER // char(10) // "{prompt}" // char(10) // char(10) // &
        HDR_ASSISTANT // char(10)
  end function default_chat_template

  ! Multi-turn renderer: roles(i) in {'system','user','assistant','tool'}.
  ! gen_prompt appends the assistant opener (serving equivalent of HF
  ! add_generation_prompt). Unknown roles fail LOUDLY (provenance rule:
  ! never silently misrender a speaker).
  subroutine apply_messages(roles, texts, nmsg, sys, gen_prompt, out, n_out)
    character(len=*), intent(in) :: roles(:), texts(:)
    integer, intent(in) :: nmsg
    character(*), intent(in) :: sys
    logical, intent(in) :: gen_prompt
    character(len=:), allocatable, intent(out) :: out
    integer, intent(out) :: n_out
    character(len=:), allocatable :: buf, hdr
    integer :: i
    buf = ""
    if (len_trim(sys) > 0) &
      buf = HDR_SYSTEM // char(10) // trim(sys) // char(10) // char(10)
    do i = 1, nmsg
      select case (trim(roles(i)))
      case ("system"); hdr = HDR_SYSTEM
      case ("user"); hdr = HDR_USER
      case ("assistant"); hdr = HDR_ASSISTANT
      case ("tool"); hdr = HDR_TOOL
      case default
        print '(2A)', "apply_messages: unknown role: ", trim(roles(i))
        call exit(1)
      end select
      buf = buf // hdr // char(10) // trim(texts(i)) // char(10) // char(10)
    end do
    if (gen_prompt) buf = buf // HDR_ASSISTANT // char(10)
    n_out = len(buf)
    out = buf
  end subroutine apply_messages

  ! Version the dialect WITH the weights (our minimal model card):
  ! template.txt lives in every checkpoint dir, written at save time.
  subroutine write_template_txt(wdir)
    character(*), intent(in) :: wdir
    integer :: u, ios
    open (newunit=u, file=trim(wdir) // "/template.txt", status="replace", &
        action="write", iostat=ios)
    if (ios /= 0) then
      print '(2A)', "cannot write template.txt in ", trim(wdir)
      call exit(1)
    end if
    write (u, '(A)', iostat=ios) "dialect=" // DIALECT_NAME
    write (u, '(A)', iostat=ios) "system=" // HDR_SYSTEM
    write (u, '(A)', iostat=ios) "user=" // HDR_USER
    write (u, '(A)', iostat=ios) "assistant=" // HDR_ASSISTANT
    write (u, '(A)', iostat=ios) "tool=" // HDR_TOOL
    close (u, iostat=ios)
    if (ios /= 0) then
      print '(2A)', "template.txt write/close failed in ", trim(wdir)
      call exit(1)
    end if
  end subroutine write_template_txt

  ! Read back a checkpoint's dialect. found=.false. for checkpoints that
  ! predate template.txt (defaults apply). Callers must refuse loudly on
  ! unknown dialect names.
  subroutine ckpt_dialect(wdir, dialect, found)
    character(*), intent(in) :: wdir
    character(len=:), allocatable, intent(out) :: dialect
    logical, intent(out) :: found
    integer :: u, ios
    character(len=256) :: line
    found = .false.
    dialect = ""
    open (newunit=u, file=trim(wdir) // "/template.txt", status="old", &
        action="read", iostat=ios)
    if (ios /= 0) return
    do
      read (u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (line(1:8) == "dialect=") then
        dialect = trim(line(9:))
        found = .true.
      end if
    end do
    close (u)
  end subroutine ckpt_dialect

  ! Tri-state CLI flags done right: --flag absent => default; present =>
  ! parse the value (T/t/1 = true, anything else = false). Presence alone
  ! must NOT enable the flag — that bug made '--stream F' turn streaming ON.
  logical function flag_is_true(s)
    character(*), intent(in) :: s
    character(len=len(s)) :: t
    t = adjustl(s)
    flag_is_true = len_trim(t) > 0 .and. index("Tt1Yy", t(1:1)) > 0
  end function flag_is_true

end module fortran_chat_mod
