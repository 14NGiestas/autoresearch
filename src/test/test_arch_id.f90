! test_arch_id.f90 — a identidade da arquitetura tem de ser ESTAVEL e DISCRIMINANTE.
!
! Nao testa um valor dourado (que mudaria a cada set_arch.sh): testa as
! PROPRIEDADES que fazem a identidade servir para alguma coisa.
!
!   1. determinismo: mesma entrada -> mesmo id, sempre;
!   2. sensibilidade: mudar QUALQUER um dos campos muda o id (um campo que nao
!      entra na canônica e' um campo que nao existe para a identidade);
!   3. nao-colisao entre as configuracoes conhecidas deste lab;
!   4. forma da canônica: uma linha, sem espaco, com o schema;
!   5. a canônica da arch compilada bate com os parametros compilados.
program test_arch_id
  use fortran_arch_mod, only: arch_canonical_of, arch_id_of, arch_canonical, arch_id, &
      arch_schema, D_MODEL, N_HEAD, N_KV, HD, N_LAYER, VV, TT, BOS
  implicit none
  integer :: nfail, i, k
  character(len=:), allocatable :: c1, c2, base
  character(len=16) :: id1, id2
  ! campos: d, nh, nkv, hd, nl, vv, ctx, bos
  integer, parameter :: base_a(8) = [96, 6, 2, 16, 12, 8192, 1024, 8188]
  integer, parameter :: known(8, 3) = reshape([ &
      96, 6, 2, 16, 12, 8192, 1024, 8188, &
      216, 6, 2, 36, 12, 8192, 1024, 8188, &
      96, 6, 2, 16, 6, 8192, 1024, 8188], [8, 3])
  integer :: v(8)

  nfail = 0
  base = arch_canonical_of(base_a(1), base_a(2), base_a(3), base_a(4), &
      base_a(5), base_a(6), base_a(7), base_a(8))

  ! 1. determinismo
  do i = 1, 10
    call check(arch_id_of(base) == arch_id_of(base), 'determinismo (mesma entrada)')
  end do

  ! 2. sensibilidade campo a campo
  do k = 1, 8
    v = base_a
    v(k) = v(k) + 1
    c2 = arch_canonical_of(v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8))
    call check(arch_id_of(c2) /= arch_id_of(base), 'campo muda -> id muda')
  end do

  ! 3. nao-colisao entre configs conhecidas
  do i = 1, 3
    do k = i + 1, 3
      c1 = arch_canonical_of(known(1, i), known(2, i), known(3, i), known(4, i), &
          known(5, i), known(6, i), known(7, i), known(8, i))
      c2 = arch_canonical_of(known(1, k), known(2, k), known(3, k), known(4, k), &
          known(5, k), known(6, k), known(7, k), known(8, k))
      call check(arch_id_of(c1) /= arch_id_of(c2), 'configs conhecidas nao colidem')
    end do
  end do

  ! 4. forma da canônica
  call check(index(base, ' ') == 0, 'canônica sem espaco')
  call check(index(base, '{"schema":"'//arch_schema()//'"') == 1, 'schema primeiro campo')
  call check(len(arch_id_of(base)) == 16, 'id tem 16 digitos')
  c1 = ''
  id2 = arch_id_of(base)
  do i = 1, 16
    if (index('0123456789abcdef', id2(i:i)) == 0) c1 = c1//'x'
  end do
  call check(len_trim(c1) == 0, 'id e hexadecimal')

  ! 5. a canônica compilada reflete os parametros compilados
  c1 = arch_canonical()
  id1 = arch_id()
  id2 = arch_id_of(arch_canonical_of(D_MODEL, N_HEAD, N_KV, HD, N_LAYER, VV, TT, BOS))
  call check(id1 == id2, 'arch_id() == arch_id_of(canonica compilada)')
  call check(index(c1, '"d_model":'//trim(i2s(D_MODEL))) > 0, 'canônica compilada tem d_model')

  write (*, '(A,A)') 'arch compilada: ', c1
  write (*, '(A,A)') 'id: ', id1
  if (nfail /= 0) error stop 'test_arch_id: FALHOU'
  write (*, '(A)') 'test_arch_id: OK'

contains

  subroutine check(ok, what)
    logical, intent(in) :: ok
    character(*), intent(in) :: what
    if (ok) then
      write (*, '(A,A)') '  ok   ', what
    else
      write (*, '(A,A)') '  FAIL ', what
      nfail = nfail + 1
    end if
  end subroutine check

  function i2s(n) result(s)
    integer, intent(in) :: n
    character(len=16) :: s
    write (s, '(I0)') n
  end function i2s

end program test_arch_id
