! lib/fortran_data.f90 — pre-tokenized id-stream loading for training.
!
! Rows files: text, one row per line, T+1 space-separated 0-based ids
! (input = first T, targets = last T). Produced by scripts/eval_driver.py
! packing or any equivalent pre-tokenizer. No parquet in Fortran yet
! (see scripts/ for the torch-free Python packing side).

module fortran_data_mod
  implicit none
  integer, parameter :: ROW_BUF = 65536
contains

  ! number of non-empty lines in path (0 if missing/unreadable -> -1)
  integer function count_rows(path)
    character(*), intent(in) :: path
    integer :: u, ios
    character(len=ROW_BUF) :: line
    count_rows = 0
    open (newunit=u, file=trim(path), status='old', action='read', &
        iostat=ios)
    if (ios /= 0) then
      count_rows = -1
      return
    end if
    do
      read (u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) > 0) count_rows = count_rows + 1
    end do
    close (u)
  end function count_rows

  ! Load B consecutive rows starting at start_row (0-based) into
  ! idx(B,T), targets(B,T). Skips blank lines. stops short at EOF
  ! (returns actual n got). Each row must hold at least T+1 ids.
  subroutine load_batch(path, start_row, B, T, idx, targets, ngot)
    character(*), intent(in) :: path
    integer, intent(in) :: start_row, B, T
    integer, intent(out) :: idx(B*T), targets(B*T)
    integer, intent(out) :: ngot
    integer :: u, ios, r, k
    integer, allocatable :: full(:)
    character(len=ROW_BUF) :: line
    allocate(full(T + 1))
    ngot = 0
    open (newunit=u, file=trim(path), status='old', action='read', &
        iostat=ios)
    if (ios /= 0) return
    r = 0
    do
      read (u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle
      if (r < start_row) then
        r = r + 1
        cycle
      end if
      if (ngot >= B) exit
      read (line, *, iostat=ios) full
      if (ios /= 0) then
        print '(A,I0)', "short row at ", r
        call exit(1)
      end if
      do k = 1, T
        idx(ngot*T+k) = full(k)
        targets(ngot*T+k) = full(k+1)
      end do
      ngot = ngot + 1
      r = r + 1
    end do
    close (u)
  end subroutine load_batch

end module fortran_data_mod
