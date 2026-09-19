! app/arch_id.f90 — a IDENTIDADE da arquitetura: string canonica + id.
!
!   arch_id                 identidade da arch COMPILADA neste binario
!   arch_id --from DIR      identidade da arch declarada por aquele checkpoint
!
! Os dois usos existem de proposito:
!   1. humano/depuracao: "para que arch este binario foi compilado?" (antes isso
!      era respondido por NOME de pasta de build -- ver src/lib/fortran_arch.f90);
!   2. teste de concordancia Fortran<->Python: bin/arch_check.sh compara a saida
!      de `arch_id --from DIR` com `scripts/arch.py DIR` para a MESMA arch. Se as
!      duas implementacoes discordarem, a identidade quebra em silencio -- e isso
!      e' bug, nao estilo.
program arch_id_app
  use fortran_arch_mod, only: arch_canonical, arch_id, arch_canonical_of, arch_id_of
  use load_weights_mod, only: read_arch_any
  use M_CLI2, only: set_args, sget
  implicit none
  character(len=256) :: from, src
  integer :: d, nh, nkv, hd, nl, vv, ctx, bos
  character(len=:), allocatable :: can

  call set_args('--from none', help_text=[character(len=80) :: &
      'NAME', '  arch_id - identity of an architecture (canonical string + id)', &
      '', 'SYNOPSIS', '  arch_id [--from CHECKPOINT_DIR]', &
      '', 'DESCRIPTION', &
      '  Sem --from: a arch COMPILADA neste binario.', &
      '  Com --from: a arch declarada por aquele checkpoint (metadata primeiro,', &
      '  arch.txt como fallback -- mesma leitura do resto do codigo).'])
  from = sget('from')
  if (trim(from) /= 'none' .and. len_trim(from) > 0) then
    call read_arch_any(trim(from), d, nh, nkv, hd, nl, vv, ctx, bos, src)
    can = arch_canonical_of(d, nh, nkv, hd, nl, vv, ctx, bos)
  else
    can = arch_canonical()
  end if
  print '(A,A)', 'canonical ', can
  print '(A,A)', 'id ', arch_id_of(can)
end program arch_id_app
