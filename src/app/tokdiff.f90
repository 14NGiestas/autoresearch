! app/tokdiff.f90 — differential test for the pure-Fortran tokenizer.
!
! Usage: fortran-fpm run tokdiff -- <tables_dir> <list_file>
! list_file: one doc path per line. Prints one line per doc:
! space-separated 0-based ids from the Fortran encoder.
! The python driver (scripts/tokdiff_driver.py) diffs against tiktoken.

program tokdiff
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  implicit none

  character(len=512) :: tdir, listfile, path
  integer :: u, lu, ios, fsize, i
  character(len=:), allocatable :: raw
  integer, allocatable :: bytes(:), ids(:)

  if (command_argument_count() < 2) then
    print '(A)', "usage: tokdiff <tables_dir> <list_file>"
    call exit(2)
  end if
  call get_command_argument(1, tdir)
  call get_command_argument(2, listfile)
  call load_tables(trim(tdir))

  open (newunit=lu, file=trim(listfile), status="old", action="read")
  do
    read (lu, '(A)', iostat=ios) path
    if (ios /= 0) exit
    open (newunit=u, file=trim(path), access="stream", form="unformatted", &
        status="old", action="read", iostat=ios)
    if (ios /= 0) then
      print '(2A)', "cannot open ", trim(path)
      call exit(1)
    end if
    inquire (unit=u, size=fsize)
    allocate (character(len=fsize) :: raw)
    if (fsize > 0) read (u) raw
    close (u)
    allocate(bytes(fsize))
    do i = 1, fsize
      bytes(i) = ichar(raw(i:i))
    end do
    deallocate(raw)
    if (fsize == 0) then
      print *
    else
      call encode(bytes, fsize, ids)
      do i = 1, size(ids)
        if (i > 1) write (*, '(A)', advance='no') ' '
        write (*, '(I0)', advance='no') ids(i)
      end do
      print *
      deallocate(ids)
    end if
    deallocate(bytes)
  end do
  close (lu)

end program tokdiff
