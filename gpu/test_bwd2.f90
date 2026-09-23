! Porteiros dos dois ultimos kernels: o round-trip da RoPE, e a formula do relu2.
program test_bwd2
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function rope_fwd(x, cb, sb, y, T, H, D) bind(C, name="rope_fwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: x, cb, sb, y
      integer(c_int), value :: T, H, D
    end function
    integer(c_int) function rope_bwd(dy, cb, sb, dx, T, H, D) bind(C, name="rope_bwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: dy, cb, sb, dx
      integer(c_int), value :: T, H, D
    end function
    integer(c_int) function relu2_bwd(dy, x, dx, n) bind(C, name="relu2_bwd")
      import :: c_int, c_float, c_ptr, c_int64_t
      type(c_ptr), value :: dy, x, dx
      integer(c_int64_t), value :: n
    end function
  end interface
  integer, parameter :: TT = 8, HH = 3, DD = 16, D2 = DD/2
  integer, parameter :: NN = 1000
  real(c_float), allocatable, target :: hx(:), hy(:), hz(:), cb(:), sb(:)
  real(c_float), allocatable, target :: hdy(:), hdx(:), href(:)
  type(c_ptr) :: dx_, dy_, dz_, dcb, dsb, ddy_, ddx_, dout_
  integer :: ierr, ib, it, ih, id2, base
  real(c_float) :: e, worst, x1, x2
  allocate(hx(TT*HH*DD), hy(TT*HH*DD), hz(TT*HH*DD), cb(TT*D2), sb(TT*D2))
  do ib = 1, TT*HH*DD; hx(ib) = real(mod(ib*11, 29), c_float)/7.0_c_float - 2.0_c_float; end do
  do ib = 1, TT*D2
    cb(ib) = cos(real(ib, c_float)*0.01_c_float); sb(ib) = sin(real(ib, c_float)*0.01_c_float)
  end do
  ierr = hipMalloc(dx_, int(TT*HH*DD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy_, int(TT*HH*DD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dz_, int(TT*HH*DD, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dcb, int(TT*D2, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dsb, int(TT*D2, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx_, c_loc(hx), int(TT*HH*DD, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dcb, c_loc(cb), int(TT*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dsb, c_loc(sb), int(TT*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ! round-trip: forward e depois inversa tem de dar a identidade
  ierr = rope_fwd(dx_, dcb, dsb, dy_, int(TT,c_int), int(HH,c_int), int(DD,c_int))
  ierr = rope_bwd(dy_, dcb, dsb, dz_, int(TT,c_int), int(HH,c_int), int(DD,c_int))
  ierr = hipMemcpy(c_loc(hz), dz_, int(TT*HH*DD, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  worst = 0.0_c_float
  do ib = 1, TT*HH*DD
    e = abs(hz(ib) - hx(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'RoPE round-trip (forward depois inversa) contra a entrada, pior erro = ', worst
  if (worst > 1.0e-5_c_float) then; print '(A)','FALHOU o round-trip'; stop 2; end if
  print '(A)', 'OK: a inversa desfaz a rotacao'
  ! e o backward do relu2: dx = 2*max(0,x)*dy
  allocate(hdy(NN), hdx(NN), href(NN))
  do ib = 1, NN
    hdy(ib) = real(mod(ib*5, 19), c_float)/19.0_c_float - 0.5_c_float
    href(ib) = real(mod(ib*3, 11), c_float)/11.0_c_float - 0.5_c_float
  end do
  ierr = hipMalloc(ddy_, int(NN, c_size_t)*4_c_size_t)
  ierr = hipMalloc(ddx_, int(NN, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dout_, int(NN, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(ddy_, c_loc(hdy), int(NN, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(ddx_, c_loc(href), int(NN, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = relu2_bwd(ddy_, ddx_, dout_, int(NN, c_int64_t))
  ierr = hipMemcpy(c_loc(hdx), dout_, int(NN, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  do ib = 1, NN
    href(ib) = merge(2.0_c_float*href(ib)*hdy(ib), 0.0_c_float, href(ib) > 0.0_c_float)
  end do
  worst = 0.0_c_float
  do ib = 1, NN
    e = abs(hdx(ib) - href(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'relu2 backward contra a formula 2*max(0,x)*dy, pior erro = ', worst
  if (worst > 1.0e-6_c_float) then; print '(A)','FALHOU o relu2'; stop 3; end if
  print '(A)', 'OK: relu2 backward casa'
  ierr = hipFree(ddy_); ierr = hipFree(ddx_); ierr = hipFree(dout_)
  ierr = hipFree(dx_); ierr = hipFree(dy_); ierr = hipFree(dz_); ierr = hipFree(dcb); ierr = hipFree(dsb)
end program
