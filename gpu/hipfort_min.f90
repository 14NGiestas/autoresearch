! Teste minimo: o lado do HOST em Fortran puro, via hipfort.
! Prova que o Fortran fala com a GPU sem C++ nenhum do nosso lado.
program hipfort_min
  use iso_c_binding
  use hipfort
  implicit none
  integer :: ierr, n
  real(c_float), allocatable, target :: h(:)
  type(c_ptr) :: d
  real(c_float) :: soma
  n = 1024*768
  allocate(h(n)); h = 1.5_c_float
  ierr = hipMalloc(d, int(n, c_size_t)*4_c_size_t)
  if (ierr /= 0) then; print '(A,I0)', 'hipMalloc falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(d, c_loc(h), int(n, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  if (ierr /= 0) then; print '(A,I0)', 'H2D falhou: ', ierr; stop 2; end if
  h = 0.0_c_float
  ierr = hipMemcpy(c_loc(h), d, int(n, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  if (ierr /= 0) then; print '(A,I0)', 'D2H falhou: ', ierr; stop 3; end if
  soma = sum(h)/real(n, c_float)
  print '(A,F7.3,A,I0)', 'media depois da ida e volta: ', soma, '   (esperado 1.500)   n=', n
  print '(A)', 'OK: o Fortran alocou, copiou e leu memoria de device, sem C++ nosso.'
  ierr = hipFree(d)
end program

! COMO COMPILAR, medido (nao suposto):
!   nix develop .#rocm --command bash -c '
!     H=$(dirname $(dirname $(command -v hipfc)))
!     gfortran -O2 -I$H/include/hipfort/amdgcn -I$H/include gpu/hipfort_min.f90 \
!       -o /tmp/hipfort_min -L$H/lib -lhipfort-amdgcn -lrocblas -lamdhip64 && /tmp/hipfort_min'
!
! Tres coisas que isto resolve, e que eu tinha afirmado mal:
!   1. o hipfort EXISTE no nixpkgs, como rocmPackages.hipfort (7.2.3, a mesma
!      versao do clr e do rocblas). Eu procurei nixpkgs#hipfort e conclui de mais.
!   2. nao e' preciso amdflang: as interfaces f2003 sao iso_c_binding puro, e o
!      gfortran 15.3 do shell chega. O hipfc nem e' usado.
!   3. logo o lado do HOST pode ser Fortran, e o C++ fica so' nos kernels, que
!      e' o desenho do proprio hipfort e nao uma escolha nossa.
