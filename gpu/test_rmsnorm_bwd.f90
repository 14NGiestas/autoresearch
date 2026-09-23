program test_rmsnorm_bwd
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function rmsnorm_bwd(dy, x, dx, rows, D, eps) bind(C, name="rmsnorm_bwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: dy, x, dx
      integer(c_int), value :: rows, D
      real(c_float), value :: eps
    end function
  end interface
  integer, parameter :: ROWS = 5, DD = 768
  real(c_float), allocatable, target :: hx(:), hdy(:), hdx(:), ref(:)
  type(c_ptr) :: dx_, ddy_, ddx_
  integer :: ierr, ir, ic, base
  real(c_float) :: ss, inv, dot, coef, e, worst, eps
  eps = 1.0e-5_c_float
  allocate(hx(ROWS*DD), hdy(ROWS*DD), hdx(ROWS*DD), ref(ROWS*DD))
  do ir = 1, ROWS*DD
    hx(ir)  = real(mod(ir*13, 37), c_float)/9.0_c_float - 2.0_c_float
    hdy(ir) = real(mod(ir*7, 23), c_float)/11.0_c_float - 1.0_c_float
  end do
  ierr = hipMalloc(dx_,  int(ROWS*DD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(ddy_, int(ROWS*DD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(ddx_, int(ROWS*DD, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx_,  c_loc(hx),  int(ROWS*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(ddy_, c_loc(hdy), int(ROWS*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = rmsnorm_bwd(ddy_, dx_, ddx_, int(ROWS, c_int), int(DD, c_int), eps)
  if (ierr /= 0) then; print '(A,I0)', 'kernel falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(c_loc(hdx), ddx_, int(ROWS*DD, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! referencia a mao, a formula do modelo
  do ir = 0, ROWS-1
    base = ir*DD
    ss = 0.0_c_float; dot = 0.0_c_float
    do ic = 1, DD
      ss = ss + hx(base + ic)**2
      dot = dot + hdy(base + ic)*hx(base + ic)
    end do
    ss = ss/real(DD, c_float) + eps
    inv = 1.0_c_float/sqrt(ss)
    coef = dot/(real(DD, c_float)*ss*sqrt(ss))
    do ic = 1, DD
      ref(base + ic) = hdy(base + ic)*inv - hx(base + ic)*coef
    end do
  end do
  worst = 0.0_c_float
  do ir = 1, ROWS*DD
    e = abs(hdx(ir) - ref(ir)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'backward da RMSNorm contra a formula do modelo, pior erro = ', worst
  if (worst > 1.0e-5_c_float) then; print '(A)','FALHOU'; stop 2; end if
  print '(A)', 'OK: casa com o rmsnorm0_bwd'
  ierr = hipFree(dx_); ierr = hipFree(ddy_); ierr = hipFree(ddx_)
end program
