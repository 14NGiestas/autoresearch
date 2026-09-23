! layer_wall2 -- a camada d768 medida a SERIO.
!
! Os tres defeitos das tentativas anteriores, corrigidos:
!  1. O lda era escrito a mao e valia K. Aqui sai das formas, pelo modulo, logo
!     lda>=M e' estrutural. O MLP-up tem M=3072 e K=768, que era o caso que
!     falhava, e agora e' correcto por construcao.
!  2. Nao aquecia, logo media a carga da Tensile. Aqui aquece 3 voltas das seis
!     formas antes de medir.
!  3. Usava system_clock, que aqui da' tempo de CPU e le' zero com o host
!     bloqueado a' espera da placa. Aqui usa date_and_time, que da' a hora de
!     parede.
program layer_wall2
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  use hip_gemm_mod, only: gemm_f32, mfi_handles_init, mfi_handles_done
  implicit none
  integer, parameter :: T = 1024, D = 768, HDD = 768, KV = 256, FF = 3072, REPS = 20
  integer :: ierr, it, ok
  real(c_float), allocatable, target :: hbuf(:)
  real(c_float), pointer :: xn(:,:), qo(:,:), ko(:,:), vo(:,:), ao(:,:), sub(:,:)
  real(c_float), pointer :: wq(:,:), wk(:,:), wv(:,:), wo(:,:), wup(:,:), wdn(:,:)
  type(c_ptr) :: dxn, dqo, dko, dvo, dao, dsub
  type(c_ptr) :: dwq, dwk, dwv, dwo, dwup, dwdn
  real(c_double) :: ms
  real(c_double) :: fl, gflops
  integer :: d0(8), d1(8), k
  allocate(hbuf(max(FF*D, T*D)))
  hbuf = 0.01_c_float
  ierr = hipMalloc(dxn,  int(T*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dqo,  int(T*HDD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dko,  int(T*KV, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dvo,  int(T*KV, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dao,  int(T*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dsub, int(T*FF, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwq,  int(HDD*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwk,  int(KV*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwv,  int(KV*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwo,  int(D*HDD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwup, int(FF*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dwdn, int(D*FF, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dxn, c_loc(hbuf), int(T*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwq, c_loc(hbuf), int(HDD*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwk, c_loc(hbuf), int(KV*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwv, c_loc(hbuf), int(KV*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwo, c_loc(hbuf), int(D*HDD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwup, c_loc(hbuf), int(FF*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwdn, c_loc(hbuf), int(D*FF, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ! As VISTAS TIPADAS: e' delas que o modulo tira o lda e o m.
  call c_f_pointer(dxn,  xn,  [D, T])
  call c_f_pointer(dqo,  qo,  [HDD, T])
  call c_f_pointer(dko,  ko,  [KV, T])
  call c_f_pointer(dvo,  vo,  [KV, T])
  call c_f_pointer(dao,  ao,  [D, T])
  call c_f_pointer(dsub, sub, [FF, T])
  call c_f_pointer(dwq,  wq,  [HDD, D])
  call c_f_pointer(dwk,  wk,  [KV, D])
  call c_f_pointer(dwv,  wv,  [KV, D])
  call c_f_pointer(dwo,  wo,  [D, HDD])
  call c_f_pointer(dwup, wup, [FF, D])
  call c_f_pointer(dwdn, wdn, [D, FF])
  call mfi_handles_init(ok=ok)
  print '(A,I0,A,I0)', 'handles criados, ok=', ok, '  e o da thread 0 e o indice ', 1
  ! AQUECIMENTO: as seis formas, para pagar a Tensile antes de medir.
  do k = 1, 3
    call camada()
  end do
  ierr = hipDeviceSynchronize()
  ! E agora, com o relogio de PAREDE.
  call date_and_time(values=d0)
  do it = 1, REPS
    call camada()
  end do
  ierr = hipDeviceSynchronize()
  call date_and_time(values=d1)
  ms = (real(d1(5)-d0(5), c_double)*3600.0_c_double + &
        real(d1(6)-d0(6), c_double)*60.0_c_double + &
        real(d1(7)-d0(7), c_double) + &
        real(d1(8)-d0(8), c_double)/1000.0_c_double)*1000.0_c_double
  fl = 2.0_c_double*T*(HDD*D + 2*KV*D + D*HDD + FF*D + D*FF)
  gflops = fl*REPS/(ms/1000.0_c_double)/1e9_c_double
  print '(A,F10.3,A)', 'uma camada d768, relogio de PAREDE: ', ms/real(REPS, c_double), ' ms'
  print '(A,F10.1,A)', '  -> ', gflops, ' GFLOP/s'
  print '(A,F10.2,A)', '  -> o passo (12 camadas):         ', ms/real(REPS, c_double)*12.0_c_double, ' ms'
  call mfi_handles_done()
contains
  subroutine camada()
    call gemm_f32(wq, xn, qo, ok=ok)
    call gemm_f32(wk, xn, ko, ok=ok)
    call gemm_f32(wv, xn, vo, ok=ok)
    call gemm_f32(wo, qo, ao, ok=ok)
    call gemm_f32(wup, xn, sub, ok=ok)      ! M=3072, K=768: o caso que falhava
    call gemm_f32(wdn, sub, ao, ok=ok)
  end subroutine camada
end program layer_wall2
