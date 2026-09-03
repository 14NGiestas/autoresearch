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

  ! Single-query attention over a KV cache (decoding step).
  ! q: (B, H, D) current query (already RoPE'd)  K, V: (B, Tc, K_H, D)
  ! y: (B, H, D). No causal mask: the cache holds only past positions.
  ! Must match causal_attn's last row bit-exactly (same op order).
  subroutine attn_step(q, K, V, y, BB, HH, K_HH, DD, TC) bind(c, name='attn_step')
    integer(c_int), intent(in) :: BB, HH, K_HH, DD, TC
    real(c_float), intent(in)  :: q(BB*HH*DD)
    real(c_float), intent(in)  :: K(BB*TC*K_HH*DD), V(BB*TC*K_HH*DD)
    real(c_float), intent(out) :: y(BB*HH*DD)
    integer :: ia, ib, ss, id, kb, rep
    real(c_float) :: scale, sm, inv, acc
    real(c_float) :: sc(TC), m

    scale = 1.0_c_float / sqrt(real(DD))
    rep = HH / K_HH

    !$omp parallel do collapse(2) private(ia, ib, ss, id, kb, sc, m, sm, inv, acc)
    do ia = 1, BB
      do ib = 1, HH
        kb = (ib - 1) / rep + 1
        m = -huge(1.0_c_float)
        do ss = 1, TC
          acc = 0.0_c_float
          do id = 1, DD
            acc = acc + q(((ia-1)*HH + (ib-1))*DD + id) &
                       * K(((ia-1)*TC + (ss-1))*K_HH*DD + (kb-1)*DD + id)
          end do
          sc(ss) = acc * scale
          if (sc(ss) > m) m = sc(ss)
        end do
        sm = 0.0_c_float
        do ss = 1, TC
          sc(ss) = exp(sc(ss) - m)
          sm = sm + sc(ss)
        end do
        inv = 1.0_c_float / sm
        do id = 1, DD
          acc = 0.0_c_float
          do ss = 1, TC
            acc = acc + sc(ss) * inv * V(((ia-1)*TC + (ss-1))*K_HH*DD + (kb-1)*DD + id)
          end do
          y(((ia-1)*HH + (ib-1))*DD + id) = acc
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine attn_step

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