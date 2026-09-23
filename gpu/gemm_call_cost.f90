! gemm_call_cost.f90 -- quanto custa UMA chamada ao rocBLAS, no host?
!
! O usuario observou o que a medida escondia: a GPU da' um spike curto e o que
! demora e' o host. Se for isso, o custo por chamada domina e nao o compute, e a
! cura nao e' um kernel melhor: e' MENOS chamadas, ou chamadas em lote.
!
! O teste separa as duas coisas: aquece a forma, mede 1 chamada, e depois mede
! 100 chamadas. Se 100 custarem ~100x uma, o custo e' por chamada. Se custarem
! quase o mesmo, o custo e' de inicializacao.
program gemm_call_cost
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  integer, parameter :: T = 1024, D = 768, FF = 3072
  integer :: ierr, k, t0, t1, rate
  real(c_double) :: ms1, ms100
  real(c_float), allocatable, target :: hx(:), hw(:)
  type(c_ptr) :: dx, dy, dw, h
  allocate(hx(T*D), hw(FF*D))
  hx = 0.01_c_float; hw = 0.01_c_float
  ierr = hipMalloc(dx, int(T*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy, int(T*FF, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dw, int(FF*D, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx, c_loc(hx), int(T*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dw, c_loc(hw), int(FF*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = rocblas_create_handle(h)
  ! AQUECIMENTO: a primeira chamada de cada forma paga a Tensile.
  do k = 1, 3
    ierr = sgemm(h, dw, dx, dy, FF, D, T)
  end do
  ierr = hipDeviceSynchronize()
  call system_clock(t0, rate)
  ierr = sgemm(h, dw, dx, dy, FF, D, T)
  ierr = hipDeviceSynchronize()
  call system_clock(t1, rate)
  ms1 = real(t1-t0, c_double)/real(rate, c_double)*1000.0_c_double
  call system_clock(t0, rate)
  do k = 1, 100
    ierr = sgemm(h, dw, dx, dy, FF, D, T)
  end do
  ierr = hipDeviceSynchronize()
  call system_clock(t1, rate)
  ms100 = real(t1-t0, c_double)/real(rate, c_double)*1000.0_c_double
  print '(A,F10.4,A)', '1 chamada, ja aquecida:      ', ms1, ' ms'
  print '(A,F10.4,A)', '100 chamadas + 1 sync:       ', ms100, ' ms'
  print '(A,F10.4,A)', '  por chamada:               ', ms100/100.0_c_double, ' ms'
  print '(A,F10.1,A)', '  e o trabalho de uma:       ', &
      2.0_c_double*T*FF*D/(ms100/100.0_c_double/1000.0_c_double)/1e9, ' GFLOP/s'
  print '(A,F10.2,A)', '  razao 100x1:               ', ms100/max(ms1,1.0e-9_c_double), ' x'
  ierr = hipFree(dx); ierr = hipFree(dy); ierr = hipFree(dw)
contains
  integer function sgemm(hh, w, x, y, OF, IF, TT) result(r)
    type(c_ptr), intent(in) :: hh, w, x, y
    integer, intent(in) :: OF, IF, TT
    r = rocblas_sgemm(hh, rocblas_operation_none, rocblas_operation_none, &
        int(OF, c_int), int(TT, c_int), int(IF, c_int), 1.0_c_float, w, int(IF, c_int), &
        x, int(IF, c_int), 0.0_c_float, y, int(OF, c_int))
  end function
end program
