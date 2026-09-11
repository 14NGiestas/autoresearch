! app/tokdiff.f90 — differential test for the pure-Fortran tokenizer.
!
! Usage: fortran-fpm run tokdiff -- <tables_dir> <list_file>
! list_file: one doc path per line. Prints one line per doc:
! space-separated 0-based ids from the Fortran encoder.
! The python driver (scripts/tokdiff_driver.py) diffs against tiktoken.

program tokdiff
  use tokenizer_tables_mod
  use tokenizer_encode_mod
  use M_CLI2, only: set_args, sget, specified
  implicit none

  character(len=512) :: tdir, listfile, path, rowsfile, space
  integer :: u, lu, ios, fsize, i, nrt, ntok, j
  character(len=:), allocatable :: raw
  character(len=100000) :: lin
  integer :: nlin
  integer, allocatable :: bytes(:), ids(:), back(:)
  integer, allocatable :: toks(:)
  integer :: nb

  call set_args('--tables TABLES --list LIST --space SPACE --rows ROWS --roundtrip N', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  tokdiff - differential test for the Fortran tokenizer', &
      'SYNOPSIS', &
      '  tokdiff --tables DIR --list FILE (one doc path per line)'], &
      version_text=[character(len=80) :: 'tokdiff 1.0'])
  tdir = trim(sget('tables'))
  listfile = trim(sget('list'))
  rowsfile = trim(sget('rows'))
  space = trim(sget('space'))
  if (len_trim(space) == 0) space = 'bpe'
  nrt = 0
  if (specified('roundtrip')) nrt = int(read_num(sget('roundtrip')))
  if (.not. specified('tables') .or. &
      (.not. specified('list') .and. .not. specified('rows'))) then
    print '(A)', 'require --tables DIR and --list FILE (or --rows FILE --roundtrip N)'
    call exit(2)
  end if
  call load_tables(trim(tdir))


  ! ---- byte/corpus space: round-trip over a rows file -------------------
  ! This is the parity test the model's OWN space needs: take real corpus id
  ! rows, decode them through decode_bytes, encode the result back through
  ! encode_bytes, and print the ids. Any mismatch means inference speaks a
  ! language the model never learned (the BPE-space bug).
  if (len_trim(rowsfile) > 0) then
    open (newunit=lu, file=trim(rowsfile), status="old", action="read")
    do j = 1, nrt
      lin = ''
      read (lu, '(A)', iostat=ios) lin
      if (ios /= 0) exit
      ! parse the ids
      nlin = len_trim(lin)
      allocate(toks(nlin + 2))
      ntok = 0
      i = 1
      do while (i <= nlin)
        do while (i <= nlin .and. lin(i:i) == ' ')
          i = i + 1
        end do
        if (i > nlin) exit
        nb = 0
        do while (i <= nlin .and. lin(i:i) /= ' ')
          nb = nb * 10 + (iachar(lin(i:i)) - 48)
          i = i + 1
        end do
        ntok = ntok + 1
        toks(ntok) = nb
      end do
      if (ntok == 0) then
        print *
        cycle
      end if
      ! toks(1) is BOS: drop it (it is not text) and keep the rest
      call decode_bytes(toks(2:ntok), ntok - 1, back, nb)
      call encode_bytes(back, nb, ids)
      do i = 1, size(ids)
        if (i > 1) write (*, '(A)', advance='no') ' '
        write (*, '(I0)', advance='no') ids(i)
      end do
      print *
      deallocate(toks, back, ids)
    end do
    close (lu)
    call exit(0)
  end if

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
      if (trim(space) == 'byte') then
        call encode_bytes(bytes, fsize, ids)
      else
        call encode(bytes, fsize, ids)
      end if
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

contains

  integer function read_num(str)
    character(len=*), intent(in) :: str
    integer :: k, c
    read_num = 0
    do k = 1, len_trim(str)
      c = iachar(str(k:k))
      if (c < 48 .or. c > 57) exit
      read_num = read_num * 10 + (c - 48)
    end do
  end function read_num

end program tokdiff
