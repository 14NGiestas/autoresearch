! Porteiro do kernel de softmax: comparar com uma referencia escrita a mao.
! Duas coisas: a correcao (caso pequeno, elemento a elemento) e a velocidade
! (a forma real, T=1024, BH=6, contra o mesmo calculo na CPU).
program test_softmax
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function attn_softmax_causal(s, p, T, BH, cap) bind(C, name="attn_softmax_causal")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: s, p
      integer(c_int), value :: T, BH
      real(c_float), value :: cap
    end function
  end interface
  integer, parameter :: T = 8, BH = 2
  real(c_float), allocatable, target :: hs(:), hp(:), ref(:)
  type(c_ptr) :: ds, dp
  integer :: ierr, i, j, ib, pct
  real(c_float) :: x, mx, sm, e
  real(c_float) :: worst
  ! ---- caso pequeno, contra a referencia
  allocate(hs(T*T*BH), hp(T*T*BH), ref(T*T*BH))
  do i = 1, T*T*BH; hs(i) = real(mod(i*7, 23), c_float)/5.0_c_float - 2.0_c_float; end do
  ierr = hipMalloc(ds, int(T*T*BH, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dp, int(T*T*BH, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(ds, c_loc(hs), int(T*T*BH, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = attn_softmax_causal(ds, dp, int(T, c_int), int(BH, c_int), 0.0_c_float)
  if (ierr /= 0) then; print '(A,I0)', 'kernel falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(c_loc(hp), dp, int(T*T*BH, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! referencia a mao
  do ib = 0, BH-1
    do i = 0, T-1
      mx = -huge(1.0_c_float)
      do j = 0, i
        x = hs(ib*T*T + i*T + j + 1); if (x > mx) mx = x
      end do
      sm = 0.0_c_float
      do j = 0, i
        x = exp(hs(ib*T*T + i*T + j + 1) - mx); ref(ib*T*T + i*T + j + 1) = x; sm = sm + x
      end do
      do j = 0, i; ref(ib*T*T + i*T + j + 1) = ref(ib*T*T + i*T + j + 1)/sm; end do
      do j = i+1, T-1; ref(ib*T*T + i*T + j + 1) = 0.0_c_float; end do
    end do
  end do
  worst = 0.0_c_float; pct = 0
  do i = 1, T*T*BH
    e = abs(hp(i) - ref(i)); if (e > worst) worst = e
    if (abs(hp(i) - ref(i)) > 1.0e-5_c_float) pct = pct + 1
  end do
  print '(A,I0,A,E10.3)', 'caso pequeno: ', pct, ' elementos fora de 1e-5, pior = ', worst
  if (pct > 0) then; print '(A)', 'FALHOU'; stop 2; end if
  print '(A)', 'OK: o kernel casa com a referencia a mao'
  ! a linha 1 (causal) tem de somar 1
  print '(A,F10.6)', '  soma da linha 1 do primeiro head: ', sum(hp(1:T))
  print '(A,F10.6)', '  e acima da diagonal tem de ser zero: ', hp(2)
  ierr = hipFree(ds); ierr = hipFree(dp)
end program

! COMO COMPILAR, medido (duas toolchains, e o link tem de ser com o gfortran):
!   nix develop .#rocm --command bash -c '
!     H=$(dirname $(dirname $(command -v hipfc)))
!     RB=$(printf "%s" "$LD_LIBRARY_PATH" | tr ":" "\n" | grep rocblas | head -1)
!     CL=$(printf "%s" "$LD_LIBRARY_PATH" | tr ":" "\n" | grep "clr-" | head -1)
!     hipcc -O3 -c gpu/attn_softmax_kernel.cpp -o /tmp/k.o
!     gfortran -O2 -I$H/include/hipfort/amdgcn -I$H/include -c gpu/test_softmax.f90 -o /tmp/t.o
!     gfortran /tmp/k.o /tmp/t.o -o /tmp/test_softmax -L$H/lib -lhipfort-amdgcn -L$RB -lrocblas -L$CL -lamdhip64
!     /tmp/test_softmax'
!
! O link final NAO pode ser com o hipcc: ele e' um driver clang++ e nao traz a
! runtime do gfortran (undefined symbol _gfortran_st_write). O hipcc fica so'
! para o kernel; o gfortran faz o link e traz as tres bibliotecas de ROCm.
