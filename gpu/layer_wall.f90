! layer_wall.f90 -- a camada residente, medida pelo relogio da PLACA.
!
! Por que este ficheiro existe: o resident_layer usava system_clock, e aqui esse
! relogio devolve tempo de CPU. Com o host bloqueado a' espera da placa o tempo
! de CPU e' zero, e por isso ele imprimia 1,400 ms para uma camada que leva
! minutos. Tambem nao aquecia: a primeira chamada de cada forma paga a carga dos
! code objects da Tensile, que sao ~170 s por processo.
!
! As duas correcoes: aquecer TODAS as formas antes de medir, e medir com
! hipEvent, que e' o relogio da placa e nao o do processo.
program layer_wall
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  integer, parameter :: T = 1024, D = 768, HDD = 768, KV = 256, FF = 3072
  integer, parameter :: REPS = 20
  integer :: ierr, k, ncall
  real(c_float), allocatable, target :: hx(:), hw(:)
  type(c_ptr) :: dx, dq, dk, dv, dao, dsub, dup
  type(c_ptr) :: wq, wk, wv, wo, wup, wdn
  type(c_ptr) :: h
  real(c_double) :: ms
  real(c_double) :: fl, gflops
  integer :: t0, t1, rate
  integer :: d0(8), d1(8)
  ncall = 0
  allocate(hx(T*D), hw(FF*D))
  hx = 0.01_c_float; hw = 0.01_c_float
  ierr = hipMalloc(dx, int(T*D, c_size_t)*4_c_size_t); call chk('dx', ierr)
  ierr = hipMalloc(dq, int(T*HDD, c_size_t)*4_c_size_t); call chk('dq', ierr)
  ierr = hipMalloc(dk, int(T*KV, c_size_t)*4_c_size_t); call chk('dk', ierr)
  ierr = hipMalloc(dv, int(T*KV, c_size_t)*4_c_size_t); call chk('dv', ierr)
  ierr = hipMalloc(dao, int(T*D, c_size_t)*4_c_size_t); call chk('dao', ierr)
  ierr = hipMalloc(dsub, int(T*FF, c_size_t)*4_c_size_t); call chk('dsub', ierr)
  ierr = hipMalloc(dup, int(T*D, c_size_t)*4_c_size_t); call chk('dup', ierr)
  ierr = hipMalloc(wq, int(HDD*D, c_size_t)*4_c_size_t); call chk('wq', ierr)
  ierr = hipMalloc(wk, int(KV*D, c_size_t)*4_c_size_t); call chk('wk', ierr)
  ierr = hipMalloc(wv, int(KV*D, c_size_t)*4_c_size_t); call chk('wv', ierr)
  ierr = hipMalloc(wo, int(D*HDD, c_size_t)*4_c_size_t); call chk('wo', ierr)
  ierr = hipMalloc(wup, int(FF*D, c_size_t)*4_c_size_t); call chk('wup', ierr)
  ierr = hipMalloc(wdn, int(D*FF, c_size_t)*4_c_size_t); call chk('wdn', ierr)
  ierr = hipMemcpy(dx, c_loc(hx), int(T*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-dx', ierr)
  ierr = hipMemcpy(wq, c_loc(hw), int(HDD*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wq', ierr)
  ierr = hipMemcpy(wk, c_loc(hw), int(KV*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wk', ierr)
  ierr = hipMemcpy(wv, c_loc(hw), int(KV*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wv', ierr)
  ierr = hipMemcpy(wo, c_loc(hw), int(D*HDD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wo', ierr)
  ierr = hipMemcpy(wup, c_loc(hw), int(FF*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wup', ierr)
  ierr = hipMemcpy(wdn, c_loc(hw), int(D*FF, c_size_t)*4_c_size_t, hipMemcpyHostToDevice); call chk('memcpy-wdn', ierr)
  ierr = rocblas_create_handle(h); call chk('create_handle', ierr)
  ! O relogio de parede da shell, para cruzar com o da placa.
  call system_clock(t0, rate)
  ! AQUECIMENTO: as 6 formas, para pagar a Tensile antes de medir.
  do k = 1, 3
    ierr = sgemm(wq, dx, dq, HDD, D, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec Q  k=', k, ' ierr=', ierr, ' MNT=', HDD, D, T
    ierr = sgemm(wk, dx, dk, KV, D, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec K  k=', k, ' ierr=', ierr, ' MNT=', KV, D, T
    ierr = sgemm(wv, dx, dv, KV, D, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec V  k=', k, ' ierr=', ierr, ' MNT=', KV, D, T
    ierr = sgemm(wo, dq, dao, D, HDD, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec O  k=', k, ' ierr=', ierr, ' MNT=', D, HDD, T
    ierr = sgemm(wup, dx, dsub, FF, D, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec up k=', k, ' ierr=', ierr, ' MNT=', FF, D, T
    ierr = sgemm(wdn, dsub, dup, D, FF, T)
    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'aquec dn k=', k, ' ierr=', ierr, ' MNT=', D, FF, T
    flush(6)
  end do
  print '(A,I0)', 'fim do aquecimento, ultimo ierr=', ierr
  flush(6)
  ierr = hipDeviceSynchronize(); call chk('sync-do-aquecimento', ierr)
  print '(A,I0)', '  ierr logo apos o sync do aquecimento = ', ierr
  flush(6)
  call system_clock(t1, rate)
  print '(A,F10.3,A)', 'aquecimento (3 voltas, 18 GEMMs): ', &
      real(t1-t0, c_double)/real(rate, c_double), ' s de CPU'
  print '(A,I0)', '  ierr logo apos o aquecimento = ', ierr
  flush(6)
  ! Agora sim. O relogio e' o date_and_time, que da' a hora de PAREDE: o
  ! system_clock aqui devolve tempo de CPU e le' zero enquanto o host espera.
  call date_and_time(values=d0)
  print '(A,I0)', '  ierr antes do ciclo medido = ', ierr
  flush(6)
  do k = 1, REPS
    print '(A,I0)', '--- volta ', k
    flush(6)
    ncall = ncall + 1; print '(A,I0,A,I0)', '  chamada ', ncall, ' Q  ierr_anterior=', ierr; flush(6)
    ierr = sgemm(wq, dx, dq, HDD, D, T)
    ierr = sgemm(wk, dx, dk, KV, D, T);    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'FALHOU K   k=', k, ' ierr=', ierr, ' MNT=', KV, D, T
    ierr = sgemm(wv, dx, dv, KV, D, T);    if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'FALHOU V   k=', k, ' ierr=', ierr, ' MNT=', KV, D, T
    ierr = sgemm(wo, dq, dao, D, HDD, T);  if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'FALHOU O   k=', k, ' ierr=', ierr, ' MNT=', D, HDD, T
    ierr = sgemm(wup, dx, dsub, FF, D, T); if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'FALHOU up  k=', k, ' ierr=', ierr, ' MNT=', FF, D, T
    ierr = sgemm(wdn, dsub, dup, D, FF, T); if (ierr /= 0) print '(A,I0,A,I0,A,3I6)', 'FALHOU dn  k=', k, ' ierr=', ierr, ' MNT=', D, FF, T
    flush(6)
  end do
  ierr = hipDeviceSynchronize(); call chk('sync-final', ierr)
  call date_and_time(values=d1)
  ms = (real(d1(5)-d0(5), c_double)*3600.0_c_double + &
        real(d1(6)-d0(6), c_double)*60.0_c_double + &
        real(d1(7)-d0(7), c_double) + &
        real(d1(8)-d0(8), c_double)/1000.0_c_double) * 1000.0_c_double
  fl = 2.0_c_double*T*(HDD*D + 2*KV*D + D*HDD + FF*D + D*FF)
  gflops = fl*REPS/(ms/1000.0_c_double)/1e9_c_double
  print '(A,F10.3,A)', 'uma camada, relogio de PAREDE:    ', ms/real(REPS, c_double), ' ms'
  print '(A,F10.1,A)', '  -> ', gflops, ' GFLOP/s  (sem transferencias no meio)'
  print '(A,F10.2,A)', '  -> o passo (12 camadas): ', ms/real(REPS, c_double)*12.0_c_double, ' ms'
  print '(A,F10.3,A)', '  e as REPS inteiras:             ', ms, ' ms'
  ierr = hipFree(dx); ierr = hipFree(dq); ierr = hipFree(dk); ierr = hipFree(dv)
  ierr = hipFree(dao); ierr = hipFree(dsub); ierr = hipFree(dup)
  ierr = hipFree(wq); ierr = hipFree(wk); ierr = hipFree(wv)
  ierr = hipFree(wo); ierr = hipFree(wup); ierr = hipFree(wdn)
contains
  subroutine chk(nome, r)
    character(*), intent(in) :: nome
    integer, intent(in) :: r
    if (r /= 0) then
      print '(A,A,A,I0)', 'FALHA em ', nome, ' ierr=', r
      flush(6)
    end if
  end subroutine chk
  integer function sgemm(w, x, y, OF, IF, TT) result(r)
    type(c_ptr), intent(in) :: w, x, y
    integer, intent(in) :: OF, IF, TT
    r = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
        int(OF, c_int), int(TT, c_int), int(IF, c_int), 1.0_c_float, w, int(IF, c_int), &
        x, int(IF, c_int), 0.0_c_float, y, int(OF, c_int))
  end function
end program
