! app/texture_scan.f90 — report the texture of a text file.
!
! Usage: texture_scan FILE [--json T] [--key texture]
!   --json T  Print one JSON line. This line goes into the card and into the
!             __metadata__ of a checkpoint.
!
! The app gives the same value as the checkpoint annotation will give.
program texture_scan
  use, intrinsic :: iso_fortran_env, only: real64
  use fortran_texture_mod, only: texture_t, texture_panel, texture_json
  use M_CLI2, only: set_args, sget, lget
  implicit none
  character(len=1024) :: path, key
  type(texture_t) :: t
  integer :: u, ios, n
  character(len=:), allocatable :: text
  logical :: as_json

  call set_args('--json F --key texture', help_text=[character(len=80) :: &
      'NAME', '  texture_scan - the texture of one file, in pure Fortran', &
      '', 'SYNOPSIS', '  texture_scan FILE [--json T] [--key texture]', &
      '', 'DESCRIPTION', &
      '  The app measures the three steps: 1. byte floor, 2. usable', &
      '  generator, 3. reasoner. It measures bytes. One character is', &
      '  one byte, and iachar gives the byte value.'])
  if (command_argument_count() < 1) then
    print '(A)', 'uso: texture_scan ARQ [--json T] [--key texture]'
    call exit(2)
  end if
  call get_command_argument(1, path)
  as_json = lget('json'); key = sget('key')

  open (newunit=u, file=trim(path), access='stream', form='unformatted', &
      status='old', action='read', iostat=ios)
  if (ios /= 0) then
    print '(2A)', 'texture_scan: nao abriu ', trim(path)
    call exit(1)
  end if
  inquire (u, size=n)
  allocate (character(len=n) :: text)
  if (n > 0) read (u) text
  close (u)

  call texture_panel(text, t)
  if (as_json) then
    print '(A)', texture_json(t, trim(key))
  else
    print '(A,I0,A,A)', '=== TEXTURE (', t%n, ' bytes) -> ', trim(t%estagio)
    print '(A,F7.4)', '  alpha              ', t%alpha
    print '(A,F7.4)', '  byte_alto          ', t%byte_alto
    print '(A,I0)', '  words              ', t%palavras
    print '(A,F7.4)', '  plausible_word     ', t%palavra_plausivel
    print '(A,F7.4,F7.4,F7.4)', '  distinct 1/2/3     ', t%distinct1, t%distinct2, t%distinct3
    print '(A,I0,F7.4)', '  long_loop/rep      ', t%maior_laco, t%rep_frac
    print '(A)', '  The controls of this repository: real prose gives'
    print '(A)', '  palavra_plausivel near 0.95 and byte_alto near 0.005.'
    print '(A)', '  A trained 3M model gives 0.20 and 0.146. That is step 1.'
  end if
end program texture_scan
