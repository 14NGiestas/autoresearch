! RoPE (Rotary Positional Encoding) — pure Fortran.
!
! Implements:
!   rope_4d: applies RoPE to a 4D tensor
!     x: (B, T, H, D)   cos/sin: (T, D/2)   y: (B, T, H, D)
!
!   For each (b, t, h, d) with d in [0, D/2):
!     y_d      = x_d      * cos(t, d) + x_{d+D/2} * sin(t, d)
!     y_{d+D/2} = x_{d+D/2} * cos(t, d) - x_d      * sin(t, d)
!
!   This exactly matches the PyTorch apply_rotary_emb in train.py:
!     y1 = x1 * cos + x2 * sin
!     y2 = x1 * (-sin) + x2 * cos
!   where  x1 = x[..., :d], x2 = x[..., d:]
!
! All arrays are row-major (C order) flat float32 buffers.
! Parallelized with OpenMP.

module fortran_rope_mod
  use iso_c_binding
  implicit none
contains

  subroutine rope_4d(x, cos_buf, sin_buf, y, B, T, H, D) bind(c, name='rope_4d')
    integer(c_int), intent(in) :: B, T, H, D
    real(c_float), intent(in)  :: x(B*T*H*D)
    real(c_float), intent(in)  :: cos_buf(T*(D/2))
    real(c_float), intent(in)  :: sin_buf(T*(D/2))
    real(c_float), intent(out) :: y(B*T*H*D)
    integer :: aa, bb, cc, dd, d2
    real(c_float) :: x1, x2, c_, s_

    d2 = D / 2

    !$omp parallel do collapse(3) private(aa, bb, cc, dd, x1, x2, c_, s_)
    do aa = 1, B
      do bb = 1, T
        do cc = 1, H
          do dd = 1, d2
            x1 = x(((aa-1)*T + (bb-1))*H*D + (cc-1)*D + dd)
            x2 = x(((aa-1)*T + (bb-1))*H*D + (cc-1)*D + dd + d2)
            c_ = cos_buf((bb-1)*d2 + dd)
            s_ = sin_buf((bb-1)*d2 + dd)
            y(((aa-1)*T + (bb-1))*H*D + (cc-1)*D + dd)      = x1 * c_  + x2 * s_
            y(((aa-1)*T + (bb-1))*H*D + (cc-1)*D + dd + d2) = x1 * (-s_) + x2 * c_
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine rope_4d

end module fortran_rope_mod