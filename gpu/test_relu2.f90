program test_relu2
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function relu2_fwd(x, y, n) bind(C, name="relu2_fwd")
      import :: c_int, c_float, c_ptr, c_int64_t
      type(c_ptr), value :: x, y
      integer(c_int64_t), value :: n
    end function
  end interface
  integer, parameter :: N = 1000
  real(c_float), allocatable, target :: hx(:), hy(:), ref(:)
  type(c_ptr) :: dx, dy
  integer :: ierr, ib
  real(c_float) :: e, worst
  allocate(hx(N), hy(N), ref(N))
  do ib = 1, N; hx(ib) = real(mod(ib*17, 41), c_float)/5.0_c_float - 4.0_c_float; end do
  ierr = hipMalloc(dx, int(N, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy, int(N, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx, c_loc(hx), int(N, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = relu2_fwd(dx, dy, int(N, c_int64_t))
  ierr = hipMemcpy(c_loc(hy), dy, int(N, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  do ib = 1, N
    ref(ib) = merge(hx(ib)*hx(ib), 0.0_c_float, hx(ib) > 0.0_c_float)
  end do
  worst = 0.0_c_float
  do ib = 1, N
    e = abs(hy(ib) - ref(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'relu2 contra a referencia a mao, pior erro = ', worst
  print "(A,2F10.3)", "  um negativo tem de dar zero: x(1), y(1) = ", hx(1), hy(1)
  if (worst > 1.0e-6_c_float) then; print '(A)','FALHOU'; stop 2; end if
  print '(A)', 'OK: relu2 casa'
  ierr = hipFree(dx); ierr = hipFree(dy)
end program
