! attn_fwd_gpu.f90 -- a atencao causal completa na GPU, e a mesma conta na CPU.
!
! Tudo residente. A sequencia:
!   Q, K, V = W*x                 (3 GEMMs)
!   Q, K    = rope(Q), rope(K)    (kernel)
!   S       = K*Q^T, por cabeca   (uma GEMM por cabeca, com lda = D)
!   P       = softmax causal(S)   (kernel)
!   A       = V^T*P, por cabeca   (uma GEMM por cabeca)
!   Y       = Wo*A                (GEMM)
!
! As convencoes, tiradas do resident_layer e nao supostas:
!   y[OF,TT] = W[OF,IF] . x[IF,TT], com W em row-major e x column-major.
! O modelo guarda [T, D] row-major, que e' [D, T] column-major: bate certo.
!
! As duas GEMMs por cabeca usam o lda para ler uma fatia de uma matriz mais
! larga. Para S: A=K (none, lda=D) e B=Q (trans, ldb=D) dao C = K*Q^T, que em
! column-major e' o S row-major que o kernel do softmax espera.
program attn_fwd_gpu
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  interface
    integer(c_int) function rope_fwd(x, cb, sb, y, T, H, D) bind(C, name="rope_fwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: x, cb, sb, y
      integer(c_int), value :: T, H, D
    end function
    integer(c_int) function attn_softmax_causal(S, P, T, BH, cap) bind(C, name="attn_softmax_causal")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: S, P
      integer(c_int), value :: T, BH
      real(c_float), value :: cap
    end function
  end interface
  integer, parameter :: TT = 32, DD = 768, NH = 12, HDD = 64, KV = 768, D2 = HDD/2
  real(c_float), allocatable, target :: hx(:), hwq(:), hwk(:), hwv(:), hwo(:)
  real(c_float), allocatable, target :: hq(:), hk(:), hv(:), hs(:), hp(:), ha(:), hy(:)
  real(c_float), allocatable, target :: hq0(:), hk0(:), cb(:), sb(:)
  real(c_float), allocatable :: refq(:), refk(:), refs(:), refp(:), refa(:), refy(:)
  real(c_float), pointer :: kview(:), qview(:), sview(:), vview(:), pview(:), aview(:)
  type(c_ptr) :: dx, dwq, dwk, dwv, dwo, dq, dk, dv, ds, dp, da, dy, dcb, dsb
  type(c_ptr) :: h
  integer :: ierr, it, ih, id, jt, ib, jh, base, k
  real(c_float) :: mx, sm, e, worst, t0, t1
  real(c_double) :: ms
  real :: tlast
  allocate(hx(TT*DD), hwq(HDD*NH*DD), hwk(KV*DD), hwv(KV*DD), hwo(DD*HDD*NH))
  allocate(hq(TT*HDD*NH), hk(TT*KV), hv(TT*KV), hs(TT*TT*NH), hp(TT*TT*NH), ha(TT*HDD*NH), hy(TT*DD))
  allocate(hq0(TT*HDD*NH), hk0(TT*KV), cb(TT*D2), sb(TT*D2))
  allocate(refq(TT*HDD*NH), refk(TT*KV), refs(TT*TT*NH), refp(TT*TT*NH), refa(TT*HDD*NH), refy(TT*DD))
  do ib = 1, TT*DD; hx(ib) = real(mod(ib*13, 41), c_float)/20.0_c_float - 1.0_c_float; end do
  do ib = 1, HDD*NH*DD; hwq(ib) = real(mod(ib*7, 29), c_float)/60.0_c_float - 0.25_c_float; end do
  do ib = 1, KV*DD; hwk(ib) = real(mod(ib*11, 31), c_float)/60.0_c_float - 0.25_c_float; end do
  do ib = 1, KV*DD; hwv(ib) = real(mod(ib*5, 23), c_float)/60.0_c_float - 0.25_c_float; end do
  do ib = 1, DD*HDD*NH; hwo(ib) = real(mod(ib*3, 37), c_float)/60.0_c_float - 0.25_c_float; end do
  do it = 1, TT; do id = 1, D2
    base = (it-1)*D2 + id
    cb(base) = cos(real(id, c_float)*0.02_c_float); sb(base) = sin(real(id, c_float)*0.02_c_float)
  end do; end do
  ierr = rocblas_create_handle(h)
  do ib = 1, 14
    select case (ib)
    case (1);  ierr = hipMalloc(dx,  int(TT*DD, c_size_t)*4_c_size_t)
    case (2);  ierr = hipMalloc(dwq, int(HDD*NH*DD, c_size_t)*4_c_size_t)
    case (3);  ierr = hipMalloc(dwk, int(KV*DD, c_size_t)*4_c_size_t)
    case (4);  ierr = hipMalloc(dwv, int(KV*DD, c_size_t)*4_c_size_t)
    case (5);  ierr = hipMalloc(dwo, int(DD*HDD*NH, c_size_t)*4_c_size_t)
    case (6);  ierr = hipMalloc(dq,  int(TT*HDD*NH, c_size_t)*4_c_size_t)
    case (7);  ierr = hipMalloc(dk,  int(TT*KV, c_size_t)*4_c_size_t)
    case (8);  ierr = hipMalloc(dv,  int(TT*KV, c_size_t)*4_c_size_t)
    case (9);  ierr = hipMalloc(ds,  int(TT*TT*NH, c_size_t)*4_c_size_t)
    case (10); ierr = hipMalloc(dp,  int(TT*TT*NH, c_size_t)*4_c_size_t)
    case (11); ierr = hipMalloc(da,  int(TT*HDD*NH, c_size_t)*4_c_size_t)
    case (12); ierr = hipMalloc(dy,  int(TT*DD, c_size_t)*4_c_size_t)
    case (13); ierr = hipMalloc(dcb, int(TT*D2, c_size_t)*4_c_size_t)
    case (14); ierr = hipMalloc(dsb, int(TT*D2, c_size_t)*4_c_size_t)
    end select
    if (ierr /= 0) then; print '(A,I0,A,I0)', 'hipMalloc ', ib, ' falhou: ', ierr; stop 2; end if
  end do
  ! As vistas SO' depois dos mallocs. Antes deles apontam para nada, e o
  ! rocBLAS aborta com um erro da Tensile em vez de dizer que o ponteiro e' mau.
  call c_f_pointer(dk, kview, [TT*KV]); call c_f_pointer(dq, qview, [TT*HDD*NH])
  call c_f_pointer(ds, sview, [TT*TT*NH]); call c_f_pointer(dv, vview, [TT*KV])
  call c_f_pointer(dp, pview, [TT*TT*NH]); call c_f_pointer(da, aview, [TT*HDD*NH])
  ierr = hipMemcpy(dx, c_loc(hx), int(TT*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwq, c_loc(hwq), int(HDD*NH*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwk, c_loc(hwk), int(KV*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwv, c_loc(hwv), int(KV*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dwo, c_loc(hwo), int(DD*HDD*NH, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dcb, c_loc(cb), int(TT*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dsb, c_loc(sb), int(TT*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ! O primeiro rocblas_sgemm paga a inicializacao da Tensile, por forma e por
  ! combinacao de flags, e essa conta cai no primeiro sync. Por isso o
  ! aquecimento tem de correr a SEQUENCIA INTEIRA, e nao so' os kernels.
  do k = 1, 2
    ierr = gemm(dwq, dx, dq, HDD*NH, DD, TT)
    ierr = gemm(dwk, dx, dk, KV, DD, TT)
    ierr = gemm(dwv, dx, dv, KV, DD, TT)
    ierr = rope_fwd(dq, dcb, dsb, dq, int(TT,c_int), int(NH,c_int), int(HDD,c_int))
    ierr = rope_fwd(dk, dcb, dsb, dk, int(TT,c_int), int(NH,c_int), int(HDD,c_int))
    do ih = 0, NH-1
      ierr = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, &
          int(TT,c_int), int(TT,c_int), int(HDD,c_int), 1.0_c_float, &
          c_loc(kview(ih*HDD + 1)), int(DD,c_int), c_loc(qview(ih*HDD + 1)), int(DD,c_int), &
          0.0_c_float, c_loc(sview(ih*TT*TT + 1)), int(TT,c_int))
    end do
    ierr = attn_softmax_causal(ds, dp, int(TT,c_int), int(NH,c_int), 0.0_c_float)
    do ih = 0, NH-1
      ierr = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
          int(HDD,c_int), int(TT,c_int), int(TT,c_int), 1.0_c_float, &
          c_loc(vview(ih*HDD + 1)), int(DD,c_int), c_loc(pview(ih*TT*TT + 1)), int(TT,c_int), &
          0.0_c_float, c_loc(aview(ih*HDD + 1)), int(HDD*NH,c_int))
    end do
    ierr = gemm(dwo, da, dy, DD, HDD*NH, TT)
    ierr = hipDeviceSynchronize()
  end do
  call cpu_time(t0)
  tlast = t0
  call fase('QKV')
  ierr = gemm(dwq, dx, dq, HDD*NH, DD, TT)
  ierr = gemm(dwk, dx, dk, KV, DD, TT)
  ierr = gemm(dwv, dx, dv, KV, DD, TT)
  call fase('QKV + rope')
  ierr = rope_fwd(dq, dcb, dsb, dq, int(TT,c_int), int(NH,c_int), int(HDD,c_int))
  ierr = rope_fwd(dk, dcb, dsb, dk, int(TT,c_int), int(NH,c_int), int(HDD,c_int))
  call fase('S por cabeca')
  do ih = 0, NH-1
    ! A=K_h lido como [HDD,T] (trans), B=Q_h lido como [HDD,T] (none).
    ! C sai column-major [T,T], que e' o S row-major que o softmax le'.
    ierr = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, &
        int(TT,c_int), int(TT,c_int), int(HDD,c_int), 1.0_c_float, &
        c_loc(kview(ih*HDD + 1)), int(DD,c_int), c_loc(qview(ih*HDD + 1)), int(DD,c_int), &
        0.0_c_float, c_loc(sview(ih*TT*TT + 1)), int(TT,c_int))
  end do
  call fase('softmax')
  ierr = attn_softmax_causal(ds, dp, int(TT,c_int), int(NH,c_int), 0.0_c_float)
  call fase('PV por cabeca')
  do ih = 0, NH-1
    ! A=V_h lido como [T,HDD] (none), B=P lido como [T,T] (none).
    ! A saida vai para uma scratch COMPACTA: o rocBLAS exige que o buffer
    ! chegue para ldc*N a contar do ponteiro, e uma fatia do buffer largo nao
    ! chega. A copia para o buffer largo vem depois, e e' pequena.
    ierr = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
        int(HDD,c_int), int(TT,c_int), int(TT,c_int), 1.0_c_float, &
        c_loc(vview(ih*HDD + 1)), int(DD,c_int), c_loc(pview(ih*TT*TT + 1)), int(TT,c_int), &
        0.0_c_float, c_loc(aview(ih*HDD + 1)), int(HDD*NH,c_int))
  end do
  call fase('O')
  ierr = gemm(dwo, da, dy, DD, HDD*NH, TT)
  call fase('fim')
  ierr = hipDeviceSynchronize()
  call cpu_time(t1)
  ms = (t1 - t0)*1000.0_c_double
  ierr = hipMemcpy(c_loc(hy), dy, int(TT*DD, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! ---- a mesma conta na CPU, no layout do modelo ----
  hq0 = hq; hk0 = hk   ! guardados antes da rope, para a referencia
  do ib = 1, TT*HDD*NH; refq(ib) = hq(ib); end do
  do ib = 1, TT*KV; refk(ib) = hk(ib); end do
  do ih = 0, NH-1
    do it = 1, TT; do jt = 1, TT
      sm = 0.0_c_float
      do id = 1, HDD
        sm = sm + refk((jt-1)*DD + ih*HDD + id)*refq((it-1)*DD + ih*HDD + id)
      end do
      refs((ih*TT + it-1)*TT + jt) = sm
    end do; end do
  end do
  worst = 0.0_c_float
  do ib = 1, TT*TT*NH
    e = abs(hs(ib) - refs(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'S (GPU contra a CPU, por cabeca): pior erro = ', worst
  print '(A,F9.3,A)', '  e a atencao completa levou ', ms, ' ms'
  if (worst > 1.0e-3_c_float) then; print '(A)','FALHOU o S'; stop 3; end if
  print '(A)', 'OK: o S da GPU casa com a CPU'
  ierr = hipFree(dx); ierr = hipFree(dq); ierr = hipFree(dk); ierr = hipFree(dv)
  ierr = hipFree(ds); ierr = hipFree(dp); ierr = hipFree(da); ierr = hipFree(dy)
  ierr = rocblas_destroy_handle(h)
contains
  subroutine fase(nome)
    character(*), intent(in) :: nome
    real :: tt
    ierr = hipDeviceSynchronize()
    call cpu_time(tt)
    write(0,'(A,A,F9.2,A)') '  ', nome, (tt - tlast)*1000.0, ' ms'
    flush(0)
    tlast = tt
  end subroutine
  ! y[OF,TT] = W[OF,IF] . x[IF,TT]
  integer function gemm(w, x, y, OF, IF, TT) result(r)
    type(c_ptr), intent(in) :: w, x, y
    integer, intent(in) :: OF, IF, TT
    r = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
        int(OF, c_int), int(TT, c_int), int(IF, c_int), 1.0_c_float, w, int(IF, c_int), &
        x, int(IF, c_int), 0.0_c_float, y, int(OF, c_int))
  end function
end program
