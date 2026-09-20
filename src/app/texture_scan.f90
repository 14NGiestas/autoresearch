! app/texture_scan.f90 — caracteriza um texto ARQUIVO com o painel em Fortran puro.
!
! Uso: texture_scan ARQ [--json] [--key texture]
!   --json   imprime a linha JSON (a mesma que vai para o card/__metadata__)
!
! Existe para: (a) medir texto de qualquer origem sem depender de Python, e
! (b) dar o mesmo numero que a caracterizacao do checkpoint vai gravar.
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
      'NAME', '  texture_scan - textura de um arquivo (bytes), em Fortran puro', &
      '', 'SYNOPSIS', '  texture_scan ARQ [--json T] [--key texture]', &
      '', 'DESCRIPTION', &
      '  Mede a ESCADA de geracao: 1) chao de bytes, 2) gerador usavel,', &
      '  3) raciocinador. Tudo sobre BYTES (caractere = byte via iachar).'])
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
    print '(A,I0,A,A)', '=== TEXTURA (', t%n, ' bytes) -> ', trim(t%estagio)
    print '(A,F7.4)', '  alpha              ', t%alpha
    print '(A,F7.4)', '  byte_alto          ', t%byte_alto
    print '(A,I0)', '  palavras           ', t%palavras
    print '(A,F7.4)', '  palavra_plausivel  ', t%palavra_plausivel
    print '(A,F7.4,F7.4,F7.4)', '  distinct 1/2/3     ', t%distinct1, t%distinct2, t%distinct3
    print '(A,I0,F7.4)', '  maior_laco/rep     ', t%maior_laco, t%rep_frac
    print '(A)', '  controles deste repo: prosa real palavra_plausivel ~0.95 / byte_alto ~0.005;'
    print '(A)', '  modelo 3M treinado ~0.20 / 0.146 (degrau 1).'
  end if
end program texture_scan
