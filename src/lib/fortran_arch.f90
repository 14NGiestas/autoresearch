! lib/fortran_arch.f90 — A ARQUITETURA em UM lugar só (fonte de verdade única).
!
! Antes: os mesmos números (N_LAYER = 12, VV = 8192, N_HEAD = 6, HD = 128)
! duplicados em 10 arquivos, e um par binário/checkpoint incompatível produzia
! LIXO SILENCIOSO -- foi o que aconteceu quando um binário de outra pasta de build
! devolveu zero linhas sem erro. Duas coisas diferentes estavam erradas:
!
!   1. duplicação   -> resolvido aqui: define-se uma vez só;
!   2. expectativa nunca conferida -> resolvido por check_shape, que compara o que
!      o arquivo .npy diz com o que a arquitetura espera e FALHA ALTO.
!
! `parameter` (e não variável) porque: os tamanhos entram em limites de array (o
! resto do código não precisa de allocatable) e o compilador otimiza melhor. A
! segunda falha acima NÃO dependia de runtime -- dependia de conferir.
!
! Para treinar OUTRO tamanho: scripts/set_arch.sh <d> <heads> <kv> <layers> <vocab>
! <ctx> -- valida a combinação, reescreve os números AQUI e rebuilda. Sem
! pré-processador e sem duplicação: a mudança aparece no git diff.
module fortran_arch_mod
  use fortran_kinds_mod, only: wp
  implicit none

  integer, parameter :: D_MODEL = 768
  integer, parameter :: N_HEAD = 6
  integer, parameter :: N_KV = 6
  integer, parameter :: N_LAYER = 12
  integer, parameter :: VV = 8192
  integer, parameter :: TT = 2048
  integer, parameter :: BOS = 8188
  integer, parameter :: HD = D_MODEL / N_HEAD   ! derivado, nunca digitado
contains

  ! Transforma lixo silencioso em erro alto: `want` é o que a arquitetura deste
  ! binário espera, `got` o que o arquivo de pesos tem.
  subroutine check_shape(name, want, got, ok)
    character(len=*), intent(in) :: name
    integer, intent(in) :: want, got
    logical, intent(out) :: ok
    ok = (want == got)
    if (.not. ok) then
      write (*, '(A)') 'FATAL: shape mismatch em '//trim(name)
      write (*, '(A,I0,A,I0)') '  arquitetura espera ', want, ' | arquivo tem ', got
      write (*, '(A)') '  este binário foi compilado para outro tamanho de modelo;'
      write (*, '(A)') '  use scripts/set_arch.sh (ou aponte para o checkpoint certo)'
    end if
  end subroutine check_shape

  ! A arquitetura efetiva vai para o log: nenhum checkpoint fica órfão de config.
  subroutine arch_report(unit)
    integer, intent(in) :: unit
    write (unit, '(A,I0,A,I0,A,I0,A,I0)') 'arch: d=', D_MODEL, ' heads=', N_HEAD, &
        ' kv_heads=', N_KV, ' layers=', N_LAYER
    write (unit, '(A,I0,A,I0,A,I0)') 'arch: vocab=', VV, ' ctx=', TT, ' bos=', BOS
  end subroutine arch_report

  ! ------------------------------------------------------------------------
  ! Arquitetura viaja COM o checkpoint (mesma convenção do template.txt que já
  ! escrevemos). Formato `chave = valor`: Fortran lê com leitura list-directed,
  ! Python lê em três linhas, e o git diff mostra a mudança. Namelist seria
  ! nativo mas só o Fortran lê, e nosso pipeline é Python.
  subroutine write_arch_txt(dir)
    character(len=*), intent(in) :: dir
    integer :: u, ios
    open (newunit=u, file=trim(dir)//'/arch.txt', status='replace', &
        action='write', iostat=ios)
    if (ios /= 0) return
    write (u, '(A,I0)') 'd_model = ', D_MODEL
    write (u, '(A,I0)') 'n_head = ', N_HEAD
    write (u, '(A,I0)') 'n_kv = ', N_KV
    write (u, '(A,I0)') 'n_layer = ', N_LAYER
    write (u, '(A,I0)') 'vocab = ', VV
    write (u, '(A,I0)') 'ctx = ', TT
    write (u, '(A,I0)') 'bos = ', BOS
    write (u, '(A,I0)') 'head_dim = ', HD
    write (u, '(A)') '# escrito no save; lido e validado no load (check_shape)'
    close (u)
  end subroutine write_arch_txt

  ! Lê arch.txt, se existir, e confere CADA campo contra a arquitetura deste
  ! binário. Ausente = ok (checkpoints antigos não têm). Divergente = falha alta,
  ! dizendo qual campo e as duas formas -- em vez do lixo silencioso de antes.
  subroutine read_arch_txt(dir, ok)
    character(len=*), intent(in) :: dir
    logical, intent(out) :: ok
    character(len=128) :: line
    character(len=64) :: key
    integer :: u, ios, eq, val, want
    logical :: cok
    ok = .true.
    open (newunit=u, file=trim(dir)//'/arch.txt', status='old', action='read', &
        iostat=ios)
    if (ios /= 0) return
    do
      read (u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      eq = index(line, '=')
      if (eq == 0 .or. line(1:1) == '#') cycle
      key = trim(adjustl(line(:eq - 1)))
      read (line(eq + 1:), *, iostat=ios) val
      if (ios /= 0) cycle
      want = -1
      select case (key)
      case ('d_model'); want = D_MODEL
      case ('n_head'); want = N_HEAD
      case ('n_kv'); want = N_KV
      case ('n_layer'); want = N_LAYER
      case ('vocab'); want = VV
      case ('ctx'); want = TT
      case ('bos'); want = BOS
      case ('head_dim'); want = HD
      end select
      if (want < 0) cycle          ! campo desconhecido: ignora (retrocompatível)
      call check_shape('arch.txt '//trim(key), want, val, cok)
      if (.not. cok) then
        write (*, '(A)') '  o checkpoint em '//trim(dir)//' foi treinado com outra arquitetura'
        ok = .false.
      end if
    end do
    close (u)
  end subroutine read_arch_txt

end module fortran_arch_mod

