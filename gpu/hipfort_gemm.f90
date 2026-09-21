! A/B: rocBLAS via hipfort (Fortran) contra o meu shim (C++), mesmas formas.
program hipfort_gemm
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  integer :: BT, IF, OF, ierr, k, reps
  real(c_float), allocatable, target :: x(:), w(:), y(:)
  type(c_ptr) :: dx, dw, dy, h
  integer(c_int64_t) :: t0, t1, rate
  real(c_double) :: ms, fl
  BT = 1024; IF = 768; OF = 3072; reps = 20
  allocate(x(BT*IF), w(OF*IF), y(BT*OF))
  x = 0.01_c_float; w = 0.02_c_float
  ierr = hipMalloc(dx, int(BT*IF, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dw, int(OF*IF, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy, int(BT*OF, c_size_t)*4_c_size_t)
  ierr = rocblas_create_handle(h)
  if (ierr /= 0) then; print '(A,I0)', 'rocblas_create_handle falhou: ', ierr; stop 1; end if
  ! os pesos ficam RESIDENTES, como no shim
  ierr = hipMemcpy(dw, c_loc(w), int(OF*IF, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ! aquecimento
  ierr = hipMemcpy(dx, c_loc(x), int(BT*IF, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, &
      int(OF, c_int), int(BT, c_int), int(IF, c_int), 1.0_c_float, dw, int(IF, c_int), &
      dx, int(IF, c_int), 0.0_c_float, dy, int(OF, c_int))
  if (ierr /= 0) then; print '(A,I0)', 'sgemm falhou: ', ierr; stop 2; end if
  call system_clock(t0, rate)
  do k = 1, reps
    ierr = hipMemcpy(dx, c_loc(x), int(BT*IF, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
    ierr = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, &
        int(OF, c_int), int(BT, c_int), int(IF, c_int), 1.0_c_float, dw, int(IF, c_int), &
        dx, int(IF, c_int), 0.0_c_float, dy, int(OF, c_int))
    ierr = hipMemcpy(c_loc(y), dy, int(BT*OF, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  end do
  call system_clock(t1, rate)
  ms = real(t1-t0, c_double)/real(rate, c_double)*1000.0_c_double/real(reps, c_double)
  fl = 2.0_c_double*BT*IF*OF
  print '(A,F8.3,A,F9.1,A)', 'hipfort rocBLAS (Fortran): ', ms, ' ms -> ', fl/(ms/1000.0_c_double)/1e9, ' GFLOP/s (com transferencias)'
  print '(A,F7.3)', '  e o primeiro elemento de y: ', y(1)
  ierr = rocblas_destroy_handle(h)
  ierr = hipFree(dx); ierr = hipFree(dw); ierr = hipFree(dy)
end program
