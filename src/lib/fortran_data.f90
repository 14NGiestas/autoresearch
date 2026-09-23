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

  ! Cabecalho do .npy: ordem de memoria e shape. Existe porque um pool gravado em
  ! ordem C (numpy default) vira uma matriz TRANSPOSTA na cache: o loader lê
  ! (1025, 11797) em vez de (11797, 1025) e as rodadas morrem mais tarde com
  ! "rows file too short" num passo que parece aleatorio (foi o que aconteceu em
  ! 2026-09-19: rows_perm*.npy gravados com np.save sem ordem F derrubaram 6 arms
  ! do job 137). A convencao do lab e' fortran_order=True.
  subroutine npy_header(path, fortran_order, nrows, ncols, ok)
    character(*), intent(in) :: path
    logical, intent(out) :: fortran_order, ok
    integer, intent(out) :: nrows, ncols
    character(len=4096) :: raw
    integer :: u, ios, p1, p2, p3
    fortran_order = .false.; nrows = -1; ncols = -1; ok = .false.
    open (newunit=u, file=trim(path), status='old', access='stream', &
        form='unformatted', action='read', iostat=ios)
    if (ios /= 0) return
    read (u, iostat=ios) raw
    close (u)
    if (ios /= 0) return
    p1 = index(raw, 'fortran_order')
    if (p1 <= 0) return
    p2 = index(raw(p1:), ':')
    if (p2 <= 0) return
    fortran_order = index(raw(p1 + p2:p1 + p2 + 8), 'True') > 0
    p3 = index(raw, 'shape')
    if (p3 <= 0) return
    read (raw(p3:), *, iostat=ios) nrows, ncols
    ! 'shape' e' "(a, b)": a leitura acima pega os dois inteiros da tupla
    ok = ios == 0 .and. nrows > 0 .and. ncols > 0
  end subroutine npy_header

  ! Load path into cache unless already there. Fails loud, never partial.
  subroutine ensure_cache(path)
    character(*), intent(in) :: path
    integer :: ios, fsize, h_rows, h_cols
    logical :: ex, f_ord, h_ok
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
    ! Formato: ordem C vira matriz transposta silenciosamente (ver npy_header).
    call npy_header(trim(path), f_ord, h_rows, h_cols, h_ok)
    if (h_ok) then
      if (.not. f_ord) then
        print '(2A)', "rows npy nao esta em ordem Fortran (np.save default): ", trim(path)
        print '(A)', "  grave o pool com np.asfortranarray(...) -- ver scripts/rows_to_npy.py"
        call exit(1)
      end if
      if (size(cache_rows, 1) /= h_rows .or. size(cache_rows, 2) /= h_cols) then
        print '(A,4I0)', "rows npy: shape do header != shape lida: ", &
            h_rows, h_cols, size(cache_rows, 1), size(cache_rows, 2)
        call exit(1)
      end if
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
