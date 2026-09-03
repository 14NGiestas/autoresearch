! Attention and activation kernels — pure Fortran.
!
! Implements:
!   causal_attn:  scaled dot-product causal attention
!   relu2:        ReLU squared activation  (y = max(0, x)^2)
!
! All arrays are row-major (C order) flat float32 buffers.
! Parallelized with OpenMP.
!
! Causal attention mirrors train.py's IS_ROCM branch:
!   q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
!   y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
!   y = y.transpose(1,2).contiguous().view(B, T, -1)
!
! With causal mask (only attend to positions <= t) and scale = 1/sqrt(D).

module fortran_attn_mod
  use iso_c_binding
  implicit none
contains

  ! Causal scaled dot-product attention
  ! q, k, v: (B, T, H, D)  out: (B, T, H, D)
  subroutine causal_attn(q, k, v, y, B, T, H, K_H, D) bind(c, name='causal_attn')
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(c_float), intent(in)  :: q(B*T*H*D), k(B*T*K_H*D), v(B*T*K_H*D)
    real(c_float), intent(out) :: y(B*T*H*D)
    integer :: aa, bb, cc, ss, dd, kb, rep
    real(c_float) :: scale, sm, inv, acc
    real(c_float) :: sc(T), m, val

    scale = 1.0_c_float / sqrt(real(D))
    rep = H / K_H   ! GQA group size: q head bb attends kv head (bb-1)/rep + 1

    !$omp parallel do collapse(2) private(aa, bb, cc, ss, dd, kb, sc, m, sm, inv, acc, val)
    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          m = -huge(1.0_c_float)
          do ss = 1, cc
            acc = 0.0_c_float
            do dd = 1, D
              acc = acc + q(((aa-1)*T + (cc-1))*H*D + (bb-1)*D + dd) &
                         * k(((aa-1)*T + (ss-1))*K_H*D + (kb-1)*D + dd)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_c_float
          do ss = 1, cc
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          inv = 1.0_c_float / sm
          do dd = 1, D
            acc = 0.0_c_float
            do ss = 1, cc
              acc = acc + sc(ss) * inv * v(((aa-1)*T + (ss-1))*K_H*D + (kb-1)*D + dd)
            end do
            y(((aa-1)*T + (cc-1))*H*D + (bb-1)*D + dd) = acc
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine causal_attn

  ! ReLU^2 activation:  y = max(0, x)^2
  subroutine relu2(x, N) bind(c, name='relu2')
    integer(c_int), intent(in) :: N
    real(c_float), intent(inout) :: x(N)
    integer :: ii
    real(c_float) :: aa

    !$omp parallel do private(aa)
    do ii = 1, N
      aa = x(ii)
      if (aa < 0.0_c_float) aa = 0.0_c_float
      x(ii) = aa * aa
    end do
    !$omp end parallel do
  end subroutine relu2

end module fortran_attn_mod