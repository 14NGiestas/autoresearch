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

  ! SDPA backward with GQA (recomputes scores/softmax: checkpoint style).
  ! Forward per (b,h,t): s_i = (q_t.k_i)/sqrt(D), i<=t; p = softmax(s);
  !   y_d = sum_i p_i * v_{i,d}.
  !   dv_{i,d} += p_i * dy_d
  !   ds_i = p_i * (dp_i - sum_j dp_j*p_j),  dp_i = sum_d dy_d*v_{i,d}
  !   dq_d = sum_i ds_i * k_{i,d} / sqrt(D)
  !   dk_{i,d} += ds_i * q_d / sqrt(D)
  ! dq positions are unique per (b,h,t) (plain writes); kv heads are
  ! shared across each GQA group, so dk/dv use atomics.
  subroutine attn_bwd(dy, q, k, v, dq, dk, dv, BB, TT, HH, K_HH, DD) &
      bind(c, name='attn_bwd')
    integer(c_int), intent(in) :: BB, TT, HH, K_HH, DD
    real(c_float), intent(in)  :: dy(BB*TT*HH*DD)
    real(c_float), intent(in)  :: q(BB*TT*HH*DD)
    real(c_float), intent(in)  :: k(BB*TT*K_HH*DD), v(BB*TT*K_HH*DD)
    real(c_float), intent(out) :: dq(BB*TT*HH*DD)
    real(c_float), intent(inout) :: dk(BB*TT*K_HH*DD), dv(BB*TT*K_HH*DD)
    integer :: ia, ib, ic, ss, id, kb, rep
    real(c_float) :: scale, sm, ssum, acc, ds
    real(c_float) :: sc(TT), dpv(TT), m

    scale = 1.0_c_float / sqrt(real(DD))
    rep = HH / K_HH

    !$omp parallel do collapse(2) private(ia, ib, ic, ss, id, kb, sc, dpv, &
    !$omp& m, sm, ssum, acc, ds)
    do ia = 1, BB
      do ib = 1, HH
        kb = (ib - 1) / rep + 1
        do ic = 1, TT
          ! forward replay: scores + softmax for query row ic
          m = -huge(1.0_c_float)
          do ss = 1, ic
            acc = 0.0_c_float
            do id = 1, DD
              acc = acc + q(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) &
                         * k(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_c_float
          do ss = 1, ic
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          do ss = 1, ic
            sc(ss) = sc(ss) / sm
          end do
          ! dp_i = dy . v_i  (reused as dpv), S = sum dp*p
          ssum = 0.0_c_float
          do ss = 1, ic
            acc = 0.0_c_float
            do id = 1, DD
              acc = acc + dy(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) &
                         * v(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id)
            end do
            dpv(ss) = acc
            ssum = ssum + acc * sc(ss)
          end do
          ! dq (unique: plain write) + dk/dv (shared: atomics)
          do id = 1, DD
            acc = 0.0_c_float
            do ss = 1, ic
              ds = sc(ss) * (dpv(ss) - ssum)
              acc = acc + ds * k(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id)
              !$omp atomic
              dk(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id) = &
                  dk(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id) + &
                  ds * q(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) * scale
              !$omp end atomic
              !$omp atomic
              dv(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id) = &
                  dv(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id) + &
                  sc(ss) * dy(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id)
              !$omp end atomic
            end do
            dq(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) = acc * scale
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine attn_bwd

  ! ReLU^2 backward: y = max(0,x)^2  ->  dx = 2*max(0,x) * dy.
  subroutine relu2_bwd(dy, x, dx, N) bind(c, name='relu2_bwd')
    integer(c_int), intent(in) :: N
    real(c_float), intent(in)  :: dy(N), x(N)
    real(c_float), intent(out) :: dx(N)
    integer :: ii
    real(c_float) :: aa

    !$omp parallel do private(aa)
    do ii = 1, N
      aa = x(ii)
      if (aa < 0.0_c_float) aa = 0.0_c_float
      dx(ii) = 2.0_c_float * aa * dy(ii)
    end do
    !$omp end parallel do
  end subroutine relu2_bwd

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