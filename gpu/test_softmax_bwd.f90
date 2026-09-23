! Porteiro do backward do softmax: a formula escrita a mao, na CPU.
program test_softmax_bwd
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function attn_softmax_bwd(P, dP, dS, T, BH) bind(C, name="attn_softmax_bwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: P, dP, dS
      integer(c_int), value :: T, BH
    end function
  end interface
  integer, parameter :: TT = 8, BH = 2
  real(c_float), allocatable, target :: hp(:), hdp(:), hds(:), ref(:)
  type(c_ptr) :: dp_, ddp_, dds_
  integer :: ierr, ib, i, j, bhx
  real(c_float) :: rowsum, e, worst
  integer :: base
  allocate(hp(TT*TT*BH), hdp(TT*TT*BH), hds(TT*TT*BH), ref(TT*TT*BH))
  do ib = 1, TT*TT*BH
    hp(ib)  = real(mod(ib*3, 17), c_float)/17.0_c_float
    hdp(ib) = real(mod(ib*7, 13), c_float)/13.0_c_float - 0.5_c_float
  end do
  ierr = hipMalloc(dp_,  int(TT*TT*BH, c_size_t)*4_c_size_t)
  ierr = hipMalloc(ddp_, int(TT*TT*BH, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dds_, int(TT*TT*BH, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dp_,  c_loc(hp),  int(TT*TT*BH, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(ddp_, c_loc(hdp), int(TT*TT*BH, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = attn_softmax_bwd(dp_, ddp_, dds_, int(TT, c_int), int(BH, c_int))
  if (ierr /= 0) then; print '(A,I0)', 'kernel falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(c_loc(hds), dds_, int(TT*TT*BH, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! referencia a mao, a mesma formula
  do bhx = 0, BH-1
    do i = 0, TT-1
      base = bhx*TT*TT + i*TT
      rowsum = 0.0_c_float
      do j = 0, i; rowsum = rowsum + hp(base + j + 1)*hdp(base + j + 1); end do
      do j = 0, i
        ref(base + j + 1) = hp(base + j + 1)*(hdp(base + j + 1) - rowsum)
      end do
      do j = i+1, TT-1; ref(base + j + 1) = 0.0_c_float; end do
    end do
  end do
  worst = 0.0_c_float
  do ib = 1, TT*TT*BH
    e = abs(hds(ib) - ref(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'backward do softmax contra a formula a mao, pior erro = ', worst
  if (worst > 1.0e-5_c_float) then; print '(A)','FALHOU'; stop 2; end if
  print '(A)', 'OK: casa com a formula'
  ierr = hipFree(dp_); ierr = hipFree(ddp_); ierr = hipFree(dds_)
end program
