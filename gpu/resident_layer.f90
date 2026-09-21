! O teste que decide o port: os pesos e as activacoes RESIDENTES.
!
! Se a parede era a transferencia, entao uma camada com tudo residente deve
! aproximar-se do teto da GPU, e nao dos 50 s por passo do caminho por chamada.
!
! Uma camada do d768: Q, K, V, O e o MLP (up e down), 6 GEMMs. Doze camadas
! vezes 6 mais a cabeca dao as ~80 chamadas que o passo tem.
!
! Tudo alocado UMA vez no device. A unica transferencia e' a entrada e a saida.
program resident_layer
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  integer, parameter :: T = 1024, D = 768, HDD = 768, KV = 256, FF = 3072, NHEAD = 6
  integer :: ierr, k, reps
  real(c_float), allocatable, target :: hx(:)
  type(c_ptr) :: dx, dq, dk, dv, dao, dsub, dup
  type(c_ptr) :: wq, wk, wv, wo, wup, wdn
  type(c_ptr) :: h
  integer :: t0, t1, rate   ! o COUNT do system_clock e' integer DEFAULT: com c_int64_t o gfortran nao escreve
  real(c_double) :: ms, fl
  reps = 10
  allocate(hx(T*D)); hx = 0.01_c_float
  ! activacoes: residentes
  ierr = hipMalloc(dx, int(T*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dq, int(T*HDD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dk, int(T*KV, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dv, int(T*KV, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dao, int(T*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dsub, int(T*FF, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dup, int(T*D, c_size_t)*4_c_size_t)
  ! pesos: residentes, uma copia so'
  ierr = hipMalloc(wq, int(HDD*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(wk, int(KV*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(wv, int(KV*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(wo, int(D*HDD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(wup, int(FF*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(wdn, int(D*FF, c_size_t)*4_c_size_t)
  if (any([ierr] /= 0)) stop 1
  ierr = rocblas_create_handle(h)
  if (ierr /= 0) then; print '(A,I0)', 'create_handle falhou: ', ierr; stop 2; end if
  ! a unica transferencia por passo: a entrada
  ierr = hipMemcpy(dx, c_loc(hx), int(T*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ! uma camada, com tudo residente
  call system_clock(t0, rate)
  do k = 1, reps
    ierr = rocblas_sgemm_handle(1.0_c_float, wq, dx, 0.0_c_float, dq, HDD, D, T)   ! Q
    ierr = rocblas_sgemm_handle(1.0_c_float, wk, dx, 0.0_c_float, dk, KV,  D, T)   ! K
    ierr = rocblas_sgemm_handle(1.0_c_float, wv, dx, 0.0_c_float, dv, KV,  D, T)   ! V
    ierr = rocblas_sgemm_handle(1.0_c_float, wo, dq, 0.0_c_float, dao, D,  HDD, T) ! O
    ierr = rocblas_sgemm_handle(1.0_c_float, wup, dx, 0.0_c_float, dsub, FF, D, T) ! MLP up
    ierr = rocblas_sgemm_handle(1.0_c_float, wdn, dsub, 0.0_c_float, dup, D, FF, T)! MLP down
  end do
  call system_clock(t1, rate)
  ms = real(t1-t0, c_double)/real(rate, c_double)*1000.0_c_double/real(reps, c_double)
  fl = 2.0_c_double*T*(HDD*D + 2*KV*D + D*HDD + FF*D + D*FF)
  print '(A,F9.3,A)', 'uma camada, tudo residente: ', ms, ' ms'
  print '(A,F9.1,A)', '  -> ', fl/(ms/1000.0_c_double)/1e9, ' GFLOP/s  (sem transferencias no meio)'
  print '(A,F9.2,A)', '  -> o passo (12 camadas + cabeca): ', ms*12.0_c_double, ' ms'
  ierr = hipFree(dx); ierr = hipFree(dq); ierr = hipFree(dk); ierr = hipFree(dv)
  ierr = hipFree(dao); ierr = hipFree(dsub); ierr = hipFree(dup)
  ierr = hipFree(wq); ierr = hipFree(wk); ierr = hipFree(wv)
  ierr = hipFree(wo); ierr = hipFree(wup); ierr = hipFree(wdn)
contains
  ! wrapper simples: y(OF,T) = W(OF,IF) . x(IF,T), tudo residente
  integer function rocblas_sgemm_handle(alpha, w, x, beta, y, OF, IF, TT) result(r)
    real(c_float), intent(in) :: alpha, beta
    type(c_ptr), intent(in) :: w, x, y
    integer, intent(in) :: OF, IF, TT
    r = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
        int(OF, c_int), int(TT, c_int), int(IF, c_int), alpha, w, int(IF, c_int), &
        x, int(IF, c_int), beta, y, int(OF, c_int))
  end function
end program
