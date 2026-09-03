! Fortran transformer-math kernels — called from Python via ctypes.
! Parallelised with OpenMP for multi-threaded CPU performance.
!
! Mirrors the numeric core of train.py's GPT forward (the "math part").
! All arrays are passed as 1-D, row-major (C order) flat buffers.
module fortran_math_mod
  use iso_c_binding
  implicit none
contains

  subroutine rmsnorm_f(x, w, y, n, c, eps) bind(c, name="rmsnorm_f")
    integer(c_int), value :: n, c
    real(c_float), value :: eps
    real(c_float), intent(in)  :: x(n * c)
    real(c_float), intent(in)  :: w(c)
    real(c_float), intent(out) :: y(n * c)
    integer :: i, j
    real(c_float) :: ss, inv
    !$omp parallel do private(ss, inv, j)
    do i = 1, n
      ss = 0.0_c_float
      do j = 1, c
        ss = ss + x((i - 1) * c + j) * x((i - 1) * c + j)
      end do
      inv = 1.0_c_float / sqrt(ss / real(c, kind=c_float) + eps)
      do j = 1, c
        y((i - 1) * c + j) = x((i - 1) * c + j) * inv * w(j)
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm_f

  subroutine rmsnorm0_f(x, y, n, c, eps) bind(c, name="rmsnorm0_f")
    integer(c_int), value :: n, c
    real(c_float), value :: eps
    real(c_float), intent(in)  :: x(n * c)
    real(c_float), intent(out) :: y(n * c)
    integer :: i, j
    real(c_float) :: ss, inv
    !$omp parallel do private(ss, inv, j)
    do i = 1, n
      ss = 0.0_c_float
      do j = 1, c
        ss = ss + x((i - 1) * c + j) * x((i - 1) * c + j)
      end do
      inv = 1.0_c_float / sqrt(ss / real(c, kind=c_float) + eps)
      do j = 1, c
        y((i - 1) * c + j) = x((i - 1) * c + j) * inv
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm0_f

  ! y(i,o) = sum_j x(i,j) * w(o,j)   ; x:(n,in)  w:(out,in)  y:(n,out)
  subroutine linear_f(x, w, y, n, inn, out) bind(c, name="linear_f")
    integer(c_int), value :: n, inn, out
    real(c_float), intent(in)  :: x(n * inn)
    real(c_float), intent(in)  :: w(out * inn)
    real(c_float), intent(out) :: y(n * out)
    integer :: i, o, j
    real(c_float) :: acc
    !$omp parallel do collapse(2) private(acc, j)
    do i = 1, n
      do o = 1, out
        acc = 0.0_c_float
        do j = 1, inn
          acc = acc + x((i - 1) * inn + j) * w((o - 1) * inn + j)
        end do
        y((i - 1) * out + o) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear_f

  ! Alias for the BLAS path (now using OpenMP loops for reliability)
  subroutine linear_blas_f(x, Wt, y, n, inn, out) bind(c, name="linear_blas_f")
    integer(c_int), value :: n, inn, out
    real(c_float), intent(in)  :: x(n * inn)
    real(c_float), intent(in)  :: Wt(inn * out)
    real(c_float), intent(out) :: y(n * out)
    integer :: i, o, j
    real(c_float) :: acc
    !$omp parallel do collapse(2) private(acc, j)
    do i = 1, n
      do o = 1, out
        acc = 0.0_c_float
        do j = 1, inn
          acc = acc + x((i - 1) * inn + j) * Wt((j - 1) * out + o)
        end do
        y((i - 1) * out + o) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear_blas_f

  subroutine rope_f(x, cos, sin, y, B, T, H, D) bind(c, name="rope_f")
    integer(c_int), value :: B, T, H, D
    real(c_float), intent(in)  :: x(B * T * H * D)
    real(c_float), intent(in)  :: cos(T * (D / 2))
    real(c_float), intent(in)  :: sin(T * (D / 2))
    real(c_float), intent(out) :: y(B * T * H * D)
    integer :: d2, bb, tt, hh, dd
    real(c_float) :: x1, x2, c_, s_
    d2 = D / 2
    !$omp parallel do collapse(3) private(dd, x1, x2, c_, s_)
    do bb = 1, B
      do tt = 1, T
        do hh = 1, H
          do dd = 1, d2
            x1 = x((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd)
            x2 = x((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd + d2)
            c_ = cos((tt-1)*d2 + dd)
            s_ = sin((tt-1)*d2 + dd)
            y((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd)      = x1 * c_ + x2 * s_
            y((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd + d2) = x1 * (-s_) + x2 * c_
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine rope_f

  subroutine causal_attn_f(q, k, v, y, B, T, H, D) bind(c, name="causal_attn_f")
    integer(c_int), value :: B, T, H, D
    real(c_float), intent(in)  :: q(B * T * H * D)
    real(c_float), intent(in)  :: k(B * T * H * D)
    real(c_float), intent(in)  :: v(B * T * H * D)
    real(c_float), intent(out) :: y(B * T * H * D)
    integer :: bb, hh, tt, ss, dd
    real(c_float) :: m, sm, inv, acc, scale, val
    real(c_float) :: sc(T)
    scale = 1.0_c_float / sqrt(real(D, kind=c_float))
    !$omp parallel do collapse(2) private(tt, ss, dd, sc, m, sm, inv, acc, val)
    do bb = 1, B
      do hh = 1, H
        do tt = 1, T
          m = -huge(1.0_c_float)
          do ss = 1, tt
            val = 0.0_c_float
            do dd = 1, D
              val = val + q((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd) * &
                          k((((bb-1)*T + (ss-1))*H + (hh-1))*D + dd)
            end do
            sc(ss) = val * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_c_float
          do ss = 1, tt
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          inv = 1.0_c_float / sm
          do dd = 1, D
            acc = 0.0_c_float
            do ss = 1, tt
              acc = acc + sc(ss) * inv * v((((bb-1)*T + (ss-1))*H + (hh-1))*D + dd)
            end do
            y((((bb-1)*T + (tt-1))*H + (hh-1))*D + dd) = acc
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine causal_attn_f

  subroutine mlp_act_f(x, y, n) bind(c, name="mlp_act_f")
    integer(c_int), value :: n
    real(c_float), intent(in)  :: x(n)
    real(c_float), intent(out) :: y(n)
    integer :: i
    real(c_float) :: a
    !$omp parallel do private(a)
    do i = 1, n
      a = x(i)
      if (a < 0.0_c_float) a = 0.0_c_float
      y(i) = a * a
    end do
    !$omp end parallel do
  end subroutine mlp_act_f

end module fortran_math_mod
