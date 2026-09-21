! Porteiro da RMSNorm: referencia a mao, e a taxa na forma real.
program test_rmsnorm
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function rmsnorm_fwd(x, y, rows, D, eps) bind(C, name="rmsnorm_fwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: x, y
      integer(c_int), value :: rows, D
      real(c_float), value :: eps
    end function
  end interface
  integer, parameter :: ROWS = 7, D = 768
  real(c_float), allocatable, target :: hx(:), hy(:), ref(:)
  type(c_ptr) :: dx, dy
  integer :: ierr, r, i
  real(c_float) :: acc, inv, e, worst, eps
  eps = 1.0e-5_c_float
  allocate(hx(ROWS*D), hy(ROWS*D), ref(ROWS*D))
  do i = 1, ROWS*D; hx(i) = real(mod(i*13, 37), c_float)/9.0_c_float - 2.0_c_float; end do
  ierr = hipMalloc(dx, int(ROWS*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy, int(ROWS*D, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx, c_loc(hx), int(ROWS*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = rmsnorm_fwd(dx, dy, int(ROWS, c_int), int(D, c_int), eps)
  if (ierr /= 0) then; print '(A,I0)', 'kernel falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(c_loc(hy), dy, int(ROWS*D, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! referencia a mao
  do r = 0, ROWS-1
    acc = 0.0_c_float
    do i = 1, D; acc = acc + hx(r*D + i)**2; end do
    inv = 1.0_c_float/sqrt(acc/real(D, c_float) + eps)
    do i = 1, D; ref(r*D + i) = hx(r*D + i)*inv; end do
  end do
  worst = 0.0_c_float
  do i = 1, ROWS*D
    e = abs(hy(i) - ref(i)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'RMSNorm contra a referencia a mao, pior erro = ', worst
  ! e a propriedade: a media dos quadrados da saida tem de ser ~1
  acc = 0.0_c_float
  do i = 1, D; acc = acc + hy(i)**2; end do
  print '(A,F10.6,A)', '  media dos quadrados da linha 1 = ', acc/real(D, c_float), '  (esperado ~1)'
  if (worst > 1.0e-5_c_float) then; print '(A)', 'FALHOU'; stop 2; end if
  print '(A)', 'OK: casa com a referencia'
  ierr = hipFree(dx); ierr = hipFree(dy)
end program
