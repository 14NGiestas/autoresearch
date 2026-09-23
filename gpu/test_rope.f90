! Porteiro da RoPE: referencia a mao, e a propriedade da rotacao (norma preservada).
program test_rope
  use iso_c_binding
  use hipfort
  implicit none
  interface
    integer(c_int) function rope_fwd(x, cb, sb, y, T, H, D) bind(C, name="rope_fwd")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: x, cb, sb, y
      integer(c_int), value :: T, H, D
    end function
  end interface
  integer, parameter :: T = 8, H = 3, D = 16
  integer, parameter :: D2 = D/2
  real(c_float), allocatable, target :: hx(:), hy(:), ref(:), cb(:), sb(:)
  type(c_ptr) :: dx, dy, dcb, dsb
  integer :: ierr, it, ih, id2
  real(c_float) :: x1, x2, c, s, e, worst
  real(c_float) :: n_in, n_out, n_worst
  integer :: n, base, ib
  allocate(hx(T*H*D), hy(T*H*D), ref(T*H*D), cb(T*D2), sb(T*D2))
  do ib = 1, T*H*D; hx(ib) = real(mod(ib*11, 29), c_float)/7.0_c_float - 2.0_c_float; end do
  do ib = 1, T*D2
    cb(ib) = cos(real(ib, c_float)*0.01_c_float)
    sb(ib) = sin(real(ib, c_float)*0.01_c_float)
  end do
  ierr = hipMalloc(dx, int(T*H*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dy, int(T*H*D, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dcb, int(T*D2, c_size_t)*4_c_size_t)
  ierr = hipMalloc(dsb, int(T*D2, c_size_t)*4_c_size_t)
  ierr = hipMemcpy(dx, c_loc(hx), int(T*H*D, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dcb, c_loc(cb), int(T*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = hipMemcpy(dsb, c_loc(sb), int(T*D2, c_size_t)*4_c_size_t, hipMemcpyHostToDevice)
  ierr = rope_fwd(dx, dcb, dsb, dy, int(T, c_int), int(H, c_int), int(D, c_int))
  if (ierr /= 0) then; print '(A,I0)', 'rope falhou: ', ierr; stop 1; end if
  ierr = hipMemcpy(c_loc(hy), dy, int(T*H*D, c_size_t)*4_c_size_t, hipMemcpyDeviceToHost)
  ! referencia a mao, a mesma convencao
  n_worst = 0.0_c_float
  do it = 0, T-1
    do ih = 0, H-1
      base = (it*H + ih)*D
      do id2 = 0, D2-1
        x1 = hx(base + id2 + 1); x2 = hx(base + id2 + D2 + 1)
        c = cb(it*D2 + id2 + 1); s = sb(it*D2 + id2 + 1)
        ref(base + id2 + 1) = x1*c + x2*s
        ref(base + id2 + D2 + 1) = -x1*s + x2*c
      end do
      n_in = 0.0_c_float; n_out = 0.0_c_float
      do id2 = 0, D-1
        n_in = n_in + hx(base + id2 + 1)**2
        n_out = n_out + hy(base + id2 + 1)**2
      end do
      e = abs(n_in - n_out); if (e > n_worst) n_worst = e
    end do
  end do
  worst = 0.0_c_float
  do ib = 1, T*H*D
    e = abs(hy(ib) - ref(ib)); if (e > worst) worst = e
  end do
  print '(A,E10.3)', 'RoPE contra a referencia a mao, pior erro = ', worst
  print '(A,E10.3)', '  e a norma preservada, pior diferenca = ', n_worst
  if (worst > 1.0e-5_c_float .or. n_worst > 1.0e-4_c_float) then; print '(A)','FALHOU'; stop 2; end if
  print '(A)', 'OK: a rotacao casa e preserva a norma'
  ierr = hipFree(dx); ierr = hipFree(dy); ierr = hipFree(dcb); ierr = hipFree(dsb)
end program
