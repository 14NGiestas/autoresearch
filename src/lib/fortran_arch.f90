! lib/fortran_arch.f90 — A ARQUITETURA em um lugar só, agora em RUNTIME.
!
! Antes: os mesmos números (N_LAYER = 12, VV = 8192, N_HEAD = 6, HD = 128)
! duplicados em 10 arquivos. Mudar o tamanho do modelo exigia editar todos, e um
! par binário/checkpoint incompatível produzia LIXO SILENCIOSO -- foi o que
! aconteceu quando um binário de outra pasta de build devolveu zero linhas sem
! erro no tokdiff. Agora as dimensões são VARIÁVEIS, com os valores históricos
! como default, e a validação é explícita:
!
!   - set_arch() checa consistência (d divisível por heads, 0 < nkv <= nhead,
!     dimensões positivas) e devolve ok=.false. em vez de seguir com lixo;
!   - check_shape() compara o que o arquivo .npy diz com o que a arquitetura
!     espera e falha ALTO, imprimindo as duas formas;
!   - arch_report() imprime a arquitetura efetiva, para todo log de treino/eval
!     carregar a procedência (um checkpoint nunca mais fica órfão de config).
!
! Um mecanismo só: runtime. Nada de defines de compilação fazendo o mesmo
! trabalho -- duas fontes de verdade para a mesma coisa é a família de bugs desta
! sessão (BPE vs bytes; binário vs checkpoint). Se algum dia os laços elementwise
! pedirem constantes, elas saem GERADAS daqui, não hand-maintained.
module fortran_arch_mod
  use fortran_kinds_mod, only: wp
  implicit none

  ! defaults = valores históricos (todo bpb já medido continua válido)
  integer :: D_MODEL = 768
  integer :: N_HEAD = 6
  integer :: N_KV = 6
  integer :: N_LAYER = 12
  integer :: VV = 8192
  integer :: TT = 2048
  integer :: BOS = 8188
  integer :: HD = 768 / 6
contains

  ! Seta a arquitetura em runtime. Argumentos <= 0 mantêm o default.
  ! ok = .false. se a combinação for inconsistente.
  ! Nomes dos argumentos com prefixo set_ de propósito: Fortran é insensível a
  ! maiúsculas, então um argumento `bos` COLIDE com a variável do módulo `BOS`
  ! (e `n_head` colidiria com `N_HEAD`). O prefixo elimina a classe inteira.
  subroutine set_arch(set_d, set_h, set_kv, set_l, set_v, set_c, set_b, ok)
    integer, intent(in) :: set_d, set_h, set_kv, set_l, set_v, set_c, set_b
    logical, intent(out) :: ok
    ok = .true.
    if (set_d > 0) D_MODEL = set_d
    if (set_h > 0) N_HEAD = set_h
    if (set_kv > 0) N_KV = set_kv
    if (set_l > 0) N_LAYER = set_l
    if (set_v > 0) VV = set_v
    if (set_c > 0) TT = set_c
    if (set_b > 0) BOS = set_b
    if (D_MODEL <= 0 .or. N_HEAD <= 0 .or. N_LAYER <= 0 .or. VV <= 0 .or. TT <= 0) then
      ok = .false.
      return
    end if
    if (mod(D_MODEL, N_HEAD) /= 0) then
      ok = .false.
      return
    end if
    if (N_KV < 1 .or. N_KV > N_HEAD) then
      ok = .false.
      return
    end if
    if (BOS >= VV) then
      ok = .false.
      return
    end if
    HD = D_MODEL / N_HEAD
  end subroutine set_arch

  ! Por que existe: um checkpoint de 97,5M carregado num binário que espera d=96
  ! não deve produzir texto estranho -- deve parar com as duas formas no texto.
  ! `want` é o que a arquitetura espera, `got` o que o arquivo tem.
  subroutine check_shape(name, want, got, ok)
    character(len=*), intent(in) :: name
    integer, intent(in) :: want, got
    logical, intent(out) :: ok
    ok = (want == got)
    if (.not. ok) then
      write (*, '(A)') 'FATAL: shape mismatch em '//trim(name)
      write (*, '(A,I0,A,I0)') '  arquitetura espera ', want, ' | arquivo tem ', got
      write (*, '(A)') '  (dims de runtime: passe --d/--layers/--vocab ou aponte para o checkpoint certo)'
    end if
  end subroutine check_shape

  subroutine arch_report(unit)
    integer, intent(in) :: unit
    write (unit, '(A,I0,A,I0,A,I0,A,I0)') 'arch: d=', D_MODEL, ' heads=', N_HEAD, &
        ' kv_heads=', N_KV, ' layers=', N_LAYER
    write (unit, '(A,I0,A,I0,A,I0)') 'arch: vocab=', VV, ' ctx=', TT, ' bos=', BOS
  end subroutine arch_report

end module fortran_arch_mod
