! lib/fortran_data.f90 — pre-tokenized id-stream loading for training.
!
! Rows files: text, one row per line, T+1 space-separated 0-based ids
! (input = first T, targets = last T). Produced by scripts/eval_driver.py
! packing or any equivalent pre-tokenizer. No parquet in Fortran yet
! (see scripts/ for the torch-free Python packing side).
!
! ...or a single npy: int32 (N,W) Fortran-order file (see
! scripts/rows_to_npy.py), detected by extension. The whole corpus loads
! ONCE into a module cache; every batch is then a memcpy. This kills both
! the per-step text re-parse AND the O(N^2) re-scan from the top
! (load_batch used to reopen + skip start_row lines every call).

module fortran_data_mod
  use iso_fortran_env, only: int32
  use stdlib_io_npy, only: load_npy
  implicit none
  integer, parameter :: ROW_BUF = 65536
  ! npy corpus cache: whole (N,W) file, keyed by path.
  character(len=512), save :: cache_path = ''
  integer(int32), allocatable, save :: cache_rows(:, :)
contains

  logical function is_npy(path)
    character(*), intent(in) :: path
    integer :: L
    L = len_trim(path)
    is_npy = L > 4 .and. path(L-3:L) == '.npy'
  end function is_npy

  ! Load path into cache unless already there. Fails loud, never partial.
  subroutine ensure_cache(path)
    character(*), intent(in) :: path
    integer :: ios, fsize
    logical :: ex
    if (allocated(cache_rows) .and. cache_path == path) return
    if (allocated(cache_rows)) deallocate (cache_rows)
    inquire (file=trim(path), exist=ex, size=fsize)
    if (.not. ex .or. fsize <= 0) then
      print '(2A)', "rows npy missing or empty: ", trim(path)
      call exit(1)
    end if
    call load_npy(trim(path), cache_rows, iostat=ios)
    if (ios /= 0 .or. .not. allocated(cache_rows)) then
      print '(2A)', "rows npy unreadable: ", trim(path)
      call exit(1)
    end if
    cache_path = path
  end subroutine ensure_cache

  ! number of non-empty lines in path (0 if missing/unreadable -> -1)
  integer function count_rows(path)
    character(*), intent(in) :: path
    integer :: u, ios
    character(len=ROW_BUF) :: line
    if (is_npy(path)) then
      call ensure_cache(path)
      count_rows = size(cache_rows, 1)
      return
    end if
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
    integer, intent(out) :: idx(:), targets(:)
    integer, intent(out) :: ngot
    integer :: u, ios, r, k
    integer :: full(T + 1)
    character(len=ROW_BUF) :: line
    if (is_npy(path)) then
      call ensure_cache(path)
      if (size(cache_rows, 2) < T + 1) then
        print '(A,2I0)', "rows npy too narrow (need T+1): ", size(cache_rows, 2)
        call exit(1)
      end if
      ngot = 0
      do k = 1, B
        r = start_row + k   ! 0-based start -> 1-based row
        if (r > size(cache_rows, 1)) exit
        idx((k-1)*T+1:k*T) = cache_rows(r, 1:T)
        targets((k-1)*T+1:k*T) = cache_rows(r, 2:T+1)
        ngot = k
      end do
      return
    end if
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
