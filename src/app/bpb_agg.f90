! app/bpb_agg.f90 — aggregate eval_bpb output into held-out bits-per-byte.
!
! Fortran twin of scripts/bpb_from_eval.py (which stays as the reference).
! Same inputs, same weighting, same output line. The CLI is positional on
! purpose (get_command_argument, not M_CLI2 flags) so any existing invocation
! works by swapping the binary name:
!   bpb_agg EVAL_OUT ROWS_FILE TOKEN_BYTES [MAX_ROWS]
!
! Semantics (== Python): bpb = sum(nll over positions with tbytes>0) /
!   sum(tbytes) / ln2. Row contract: T+1 ids per row, inputs ids[0:T],
!   targets ids[1:T+1]; T is read from the rows file, never hardcoded.
!   token_bytes.txt line i = byte length of id i.
!
! Like the Python, eval lines that aren't exactly T parseable reals (e.g. the
! "row done: N" progress markers, which eval_bpb prints to STDERR) are filtered
! out BEFORE pairing, so a merged stdout+stderr capture gives the same answer
! as separated streams. Out-of-range token ids abort loudly instead of risking
! silent out-of-bounds reads.
!
! Accumulation in real64 (model math stays wp/real32 per fortran_kinds_mod;
! this is I/O aggregation, where the extra precision is free and exact).
program bpb_agg
  use iso_fortran_env, only: real64, int64, error_unit
  implicit none
  character(len=4096) :: eval_path, rows_path, tbytes_path, arg
  character(len=:), allocatable :: line
  integer, allocatable :: ids(:)
  real(real64), allocatable :: vals(:)
  integer, allocatable :: tbytes(:)
  integer :: nargs, ios, u_e, u_r, u_t, n, i, r, T, used
  integer(int64) :: tot_bytes, max_rows
  integer :: tid, b
  real(real64) :: tot_nll, v, bpb, ln2v

  ln2v = log(2.0_real64)
  nargs = command_argument_count()
  if (nargs >= 1) then
    call get_command_argument(1, arg)
    if (trim(arg) == '--help' .or. trim(arg) == '-h') then
      print '(A)', 'usage: bpb_agg EVAL_OUT ROWS_FILE TOKEN_BYTES [MAX_ROWS]'
      stop 0
    end if
  end if
  if (nargs < 3) then
    print '(A)', 'usage: bpb_agg EVAL_OUT ROWS_FILE TOKEN_BYTES [MAX_ROWS]'
    stop 2
  end if
  call get_command_argument(1, eval_path)
  call get_command_argument(2, rows_path)
  call get_command_argument(3, tbytes_path)
  max_rows = 1000000000_int64
  if (nargs >= 4) then
    call get_command_argument(4, arg)
    read (arg, *, iostat=ios) max_rows
    if (ios /= 0 .or. max_rows < 0) then
      write (error_unit, '(A)') 'bad MAX_ROWS'
      stop 2
    end if
  end if

  ! token byte-lengths (whole file; vocab-sized, tiny)
  allocate(tbytes(1048576))
  n = 0
  open (newunit=u_t, file=trim(tbytes_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write (error_unit, '(2A)') 'cannot open ', trim(tbytes_path)
    stop 2
  end if
  do
    if (n >= size(tbytes)) then
      write (error_unit, '(A)') 'token_bytes too large (>1M entries)'
      stop 2
    end if
    read (u_t, *, iostat=ios) b
    if (ios /= 0) exit
    n = n + 1
    tbytes(n) = b
  end do
  close (u_t)
  if (n == 0) then
    write (error_unit, '(A)') 'empty token_bytes'
    stop 2
  end if

  allocate(character(len=2000000) :: line)
  open (newunit=u_r, file=trim(rows_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write (error_unit, '(2A)') 'cannot open ', trim(rows_path)
    stop 2
  end if
  open (newunit=u_e, file=trim(eval_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write (error_unit, '(2A)') 'cannot open ', trim(eval_path)
    stop 2
  end if

  tot_nll = 0.0_real64
  tot_bytes = 0_int64
  used = 0
  T = -1
  do
    if (int(used, int64) >= max_rows) exit
    ! next non-blank row line; T from the first one
    do
      read (u_r, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (count_tokens(line) > 0) exit
    end do
    if (ios /= 0) exit
    if (T < 0) then
      T = count_tokens(line) - 1
      if (T < 1) then
        write (error_unit, '(A)') 'first row has <2 ids'
        stop 2
      end if
      allocate(ids(T + 1))
      allocate(vals(T))
    end if
    if (count_tokens(line) /= T + 1) then
      write (error_unit, '(A,I0,A,I0)') 'row with wrong field count, skipping'
      cycle
    end if
    read (line, *, iostat=ios) ids
    if (ios /= 0) then
      write (error_unit, '(A)') 'row ids do not parse, skipping'
      cycle
    end if
    ! next eval line that parses as exactly T reals; anything else
    ! (progress markers, blank lines) is skipped WITHOUT consuming the row
    do
      read (u_e, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (count_tokens(line) /= T) cycle
      read (line, *, iostat=ios) vals
      if (ios /= 0) cycle
      exit
    end do
    if (ios /= 0) exit
    do i = 1, T
      tid = ids(i + 1)
      if (tid < 0 .or. tid >= n) then
        write (error_unit, '(A,I0)') 'token id out of token_bytes range: ', tid
        stop 1
      end if
      b = tbytes(tid + 1)
      if (b > 0) then
        v = vals(i)
        tot_nll = tot_nll + v
        tot_bytes = tot_bytes + int(b, int64)
      end if
    end do
    used = used + 1
  end do
  close (u_r)
  close (u_e)
  if (tot_bytes == 0) then
    write (error_unit, '(A)') 'no counted bytes'
    stop 1
  end if
  bpb = tot_nll / real(tot_bytes, real64) / ln2v
  write (*, '(A,I0,A,I0,A,F0.2,A,F0.5)') 'rows=', used, ' bytes=', tot_bytes, &
      ' nll_sum=', tot_nll, ' bpb=', bpb

contains

  integer function count_tokens(s)
    character(len=*), intent(in) :: s
    integer :: k
    logical :: in_tok
    count_tokens = 0
    in_tok = .false.
    do k = 1, len(s)
      if (s(k:k) == ' ' .or. s(k:k) == char(9)) then
        in_tok = .false.
      else if (.not. in_tok) then
        in_tok = .true.
        count_tokens = count_tokens + 1
      end if
    end do
  end function count_tokens

end program bpb_agg
