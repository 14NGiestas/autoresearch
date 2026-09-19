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
  use, intrinsic :: iso_fortran_env, only: int64
  use fortran_kinds_mod, only: wp
  implicit none

  integer, parameter :: D_MODEL = 216
  integer, parameter :: N_HEAD = 6
  integer, parameter :: N_KV = 2
  integer, parameter :: N_LAYER = 12
  integer, parameter :: VV = 8192
  integer, parameter :: TT = 1024
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

  ! Reusável: todo app que carrega pesos chama ISTO, em vez de repetir a checagem.
  ! Um checkpoint de outra arquitetura tem de parar a execução, não gerar lixo.
  ! ---------------------------------------------------------------------
  ! IDENTIDADE da arquitetura.
  !
  ! A verdade sobre a arch mora em TRES lugares (os `parameter` deste modulo, o
  ! __metadata__ do safetensors e o sidecar arch.txt) e o vinculo
  ! binario<->checkpoint era um NOME DE PASTA (arch_d96_h6_kv2_l12_v8192_c1024,
  ! schema repetido em fedavg_rounds.py, compose_grid.py e set_arch.sh). Foi
  ! assim que a arvore ficou com a fonte em d216 e os experimentos em d96.
  !
  ! Aqui a arch ganha UMA identidade derivada dos proprios parametros:
  !   arch_canonical()  string canonica (ordem fixa, sem espaco) -- vai no
  !                     metadata como arch.canonical;
  !   arch_id()         16 digitos hex, para selecao e checagem.
  !
  ! O id NAO e' criptografico: e' rotacao+xor sobre os bytes da canonica,
  ! escolhido porque (a) nao depende de overflow de inteiro assinado (que a
  ! norma nao define e o compilador pode explorar) e (b) tem implementacao
  ! identica de 5 linhas em Fortran e em Python, o que permite TESTE de
  ! concordancia entre as linguagens (bin/arch_check.sh). Ele so' precisa
  ! garantir que configs diferentes nao colidam -- e isso e' testado.
  ! ---------------------------------------------------------------------
  ! Versao do schema da identidade (vai no metadata como arch.schema). Mudar o
  ! CONJUNTO de campos exige subir isto -- e' o que diz ao leitor que o formato
  ! mudou, em vez de deixar campo faltando passar batido.
  function arch_schema() result(s)
    character(len=:), allocatable :: s
    s = 'fortran_gpt/arch/1'
  end function arch_schema

  function arch_canonical() result(s)
    character(len=:), allocatable :: s
    s = arch_canonical_of(D_MODEL, N_HEAD, N_KV, HD, N_LAYER, VV, TT, BOS)
  end function arch_canonical

  ! Mesma string canonica para numeros EXPLICITOS: e' o que permite comparar
  ! Fortran e Python (e checar um checkpoint) sem depender de qual arch este
  ! binario foi compilado.
  function arch_canonical_of(d, nh, nkv, hd_, nl, vv_, ctx, bos_) result(s)
    integer, intent(in) :: d, nh, nkv, hd_, nl, vv_, ctx, bos_
    character(len=:), allocatable :: s
    character(len=256) :: buf
    write (buf, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A)') &
        '{"schema":"fortran_gpt/arch/1","d_model":', d, &
        ',"n_head":', nh, ',"n_kv":', nkv, ',"head_dim":', hd_, &
        ',"n_layer":', nl, ',"vocab":', vv_, ',"ctx":', ctx, &
        ',"bos":', bos_, '}'
    s = trim(buf)
  end function arch_canonical_of

  function arch_id() result(h)
    character(len=16) :: h
    h = arch_id_of(arch_canonical())
  end function arch_id

  function arch_id_of(s) result(h)
    character(*), intent(in) :: s
    character(len=16) :: h
    integer(int64) :: acc, m17, v1
    integer :: i, b
    acc = int(z'0123456789abcdef', int64)      ! semente fixa
    m17 = int(z'1ffff', int64)                 ! 17 bits (os que saem na rotacao)
    do i = 1, len(s)
      b = iachar(s(i:i))
      acc = ieor(acc, int(b, int64))
      ! rotacao a esquerda de 17 bits (so' shift e or: sem overflow)
      acc = ior(shiftl(acc, 17), iand(shiftr(acc, 47), m17))
      ! mistura
      acc = ieor(acc, shiftr(acc, 29))
    end do
    do i = 0, 15
      v1 = iand(shiftr(acc, int(4*(15 - i), int64)), 15_int64)
      h(i + 1:i + 1) = '0123456789abcdef'(int(v1) + 1:int(v1) + 1)
    end do
  end function arch_id_of

  subroutine require_arch(dir)
    character(len=*), intent(in) :: dir
    logical :: ok
    call read_arch_txt(dir, ok)
    if (.not. ok) then
      write (*, '(A)') 'abortando: o checkpoint não bate com a arquitetura deste binário'
      call exit(1)
    end if
  end subroutine require_arch

end module fortran_arch_mod

