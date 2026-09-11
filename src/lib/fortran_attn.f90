! Attention and activation kernels — pure Fortran.
!
! Implements:
!   causal_attn:  scaled dot-product causal attention
!   relu2:        ReLU squared activation  (y = max(0, x)^2)
!
! All arrays are row-major flat real(wp) buffers.
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
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: sgemm
  implicit none
contains

  ! Causal scaled dot-product attention
  ! q, k, v: (B, T, H, D)  out: (B, T, H, D)
  subroutine causal_attn(q, k, v, y, B, T, H, K_H, D)
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in)  :: q(:), k(:), v(:)
    real(wp), intent(out) :: y(:)
    integer :: aa, bb, cc, ss, dd, kb, rep
    real(wp) :: scale, sm, inv, acc
    real(wp) :: sc(T), m, val

    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H   ! GQA group size: q head bb attends kv head (bb-1)/rep + 1

    !$omp parallel do collapse(2) private(aa, bb, cc, ss, dd, kb, sc, m, sm, inv, acc, val)
    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          m = -huge(1.0_wp)
          do ss = 1, cc
            acc = 0.0_wp
            do dd = 1, D
              acc = acc + q(((aa-1)*T + (cc-1))*H*D + (bb-1)*D + dd) &
                         * k(((aa-1)*T + (ss-1))*K_H*D + (kb-1)*D + dd)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_wp
          do ss = 1, cc
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          inv = 1.0_wp / sm
          do dd = 1, D
            acc = 0.0_wp
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

  ! Document-masked causal attention. Identical to causal_attn except that a
  ! query only attends to positions >= docstart(query): our rows are packed
  ! token streams where BOS separates documents, and without this mask the
  ! model spends most of its attention budget on pairs that never co-occur at
  ! inference (measured on math_reasoning.txt: ~12.5 docs/row, only 8.1% of
  ! causal pairs are within-document). docstart is (B,T) flat, 1-based.
  ! With docstart == 1 everywhere this must equal causal_attn bit-exactly
  ! (same accumulation order), which is what test_causal_attn_doc asserts.
  subroutine causal_attn_doc(q, k, v, y, B, T, H, K_H, D, docstart)
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in)  :: q(:), k(:), v(:)
    integer(c_int), intent(in) :: docstart(:)
    real(wp), intent(out) :: y(:)
    integer :: aa, bb, cc, ss, dd, kb, rep, s0
    real(wp) :: scale, sm, inv, acc
    real(wp) :: sc(T), m

    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H

    !$omp parallel do collapse(2) private(bb, cc, ss, dd, kb, s0, sc, m, sm, inv, acc)
    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          s0 = docstart((aa-1)*T + cc)
          m = -huge(1.0_wp)
          do ss = s0, cc
            acc = 0.0_wp
            do dd = 1, D
              acc = acc + q(((aa-1)*T + (cc-1))*H*D + (bb-1)*D + dd) &
                         * k(((aa-1)*T + (ss-1))*K_H*D + (kb-1)*D + dd)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_wp
          do ss = s0, cc
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          inv = 1.0_wp / sm
          do dd = 1, D
            acc = 0.0_wp
            do ss = s0, cc
              acc = acc + sc(ss) * inv &
                  * v(((aa-1)*T + (ss-1))*K_H*D + (kb-1)*D + dd)
            end do
            y(((aa-1)*T + (cc-1))*H*D + (bb-1)*D + dd) = acc
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine causal_attn_doc

  ! BLAS-backed causal attention (forward). Same math as causal_attn, but both
  ! matmuls go through sgemm instead of hand-written loops. Motivation,
  ! measured on this codebase: the naive kernels run at ~6 GFLOP/s where sgemm
  ! reaches ~50+, and attention is ~30% of a training step's flops (155 GFLOP
  ! forward + 309 backward at B=1,T=2048,L=12) while being the dominant cost of
  ! eval_bpb (a 60-row bpb pass takes ~20 min per checkpoint). The linears were
  ! already BLAS; this is the same fix the chunked prefill applied (11.8x).
  !
  ! Layout note (same trick as linear3d_sgemm): our buffers are row-major, so a
  ! row-major (T,D) buffer IS the column-major matrix (D,T). Therefore
  !   S = Q K^T  is computed as  S^T = K Q^T  ->  sgemm('T','N', T,T,D, K, Q)
  ! and the resulting row-major S(i,j) is exactly score(query i, key j).
  !   Y = P V    is computed as  Y^T = V^T P^T  ->  sgemm('N','N', D,T,T, V, P)
  ! Per (batch, head): S is a (T,T) scratch the caller owns and we reuse.
  ! Summation order differs from causal_attn, so expect ~1e-6 drift, not bit
  ! equality (asserted in test_attn_sgemm).
  subroutine attn_sgemm(q, k, v, y, B, T, H, K_H, D, S)
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in)  :: q(:), k(:), v(:)
    real(wp), intent(out) :: y(:)
    real(wp), intent(inout) :: S(:)          ! (T,T) scratch
    integer :: aa, bb, kb, rep, ii, jj
    integer(c_int64_t) :: m, n, kk, lda, ldb, ldc
    real(wp) :: scale, mx, sm, inv

    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H

    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        ! ---- S = Q K^T (scaled), one sgemm per (batch, head) ----
        m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
        lda = int(K_H*D, c_int64_t); ldb = int(H*D, c_int64_t)
        ldc = int(T, c_int64_t)
        call sgemm('T', 'N', m, n, kk, scale, &
             k((aa-1)*T*K_H*D + (kb-1)*D + 1:), lda, &
             q((aa-1)*T*H*D + (bb-1)*D + 1:), ldb, &
             0.0_wp, S, ldc)
        ! ---- causal mask + softmax, row by row ----
        do ii = 1, T
          mx = -huge(1.0_wp)
          do jj = 1, ii
            if (S((ii-1)*T + jj) > mx) mx = S((ii-1)*T + jj)
          end do
          sm = 0.0_wp
          do jj = 1, ii
            S((ii-1)*T + jj) = exp(S((ii-1)*T + jj) - mx)
            sm = sm + S((ii-1)*T + jj)
          end do
          inv = 1.0_wp / sm
          do jj = 1, ii
            S((ii-1)*T + jj) = S((ii-1)*T + jj) * inv
          end do
          do jj = ii + 1, T
            S((ii-1)*T + jj) = 0.0_wp
          end do
        end do
        ! ---- Y = P V ----
        m = int(D, c_int64_t); n = int(T, c_int64_t); kk = int(T, c_int64_t)
        lda = int(K_H*D, c_int64_t); ldb = int(T, c_int64_t)
        ldc = int(H*D, c_int64_t)
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             v((aa-1)*T*K_H*D + (kb-1)*D + 1:), lda, S, ldb, &
             0.0_wp, y((aa-1)*T*H*D + (bb-1)*D + 1:), ldc)
      end do
    end do
  end subroutine attn_sgemm

  ! Single-query attention over a KV cache (decoding step).
  ! q: (B, H, D) current query (already RoPE'd)  K, V: (B, Tc, K_H, D)
  ! y: (B, H, D). No causal mask: the cache holds only past positions.
  ! Must match causal_attn's last row bit-exactly (same op order).
  subroutine attn_step(q, K, V, y, BB, HH, K_HH, DD, TC)
    integer(c_int), intent(in) :: BB, HH, K_HH, DD, TC
    real(wp), intent(in)  :: q(:)
    real(wp), intent(in)  :: K(:), V(:)
    real(wp), intent(out) :: y(:)
    integer :: ia, ib, ss, id, kb, rep
    real(wp) :: scale, sm, inv, acc
    real(wp) :: sc(TC), m

    scale = 1.0_wp / sqrt(real(DD, wp))
    rep = HH / K_HH

    !$omp parallel do collapse(2) private(ia, ib, ss, id, kb, sc, m, sm, inv, acc)
    do ia = 1, BB
      do ib = 1, HH
        kb = (ib - 1) / rep + 1
        m = -huge(1.0_wp)
        do ss = 1, TC
          acc = 0.0_wp
          do id = 1, DD
            acc = acc + q(((ia-1)*HH + (ib-1))*DD + id) &
                       * K(((ia-1)*TC + (ss-1))*K_HH*DD + (kb-1)*DD + id)
          end do
          sc(ss) = acc * scale
          if (sc(ss) > m) m = sc(ss)
        end do
        sm = 0.0_wp
        do ss = 1, TC
          sc(ss) = exp(sc(ss) - m)
          sm = sm + sc(ss)
        end do
        inv = 1.0_wp / sm
        do id = 1, DD
          acc = 0.0_wp
          do ss = 1, TC
            acc = acc + sc(ss) * inv * V(((ia-1)*TC + (ss-1))*K_HH*DD + (kb-1)*DD + id)
          end do
          y(((ia-1)*HH + (ib-1))*DD + id) = acc
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine attn_step

  ! Chunked cached attention: TB queries against a KV cache that already
  ! contains the chunk's own K/V entries (caller appends them first).
  !   q, y: (B, TB, H, D)
  !   K, V: cache slice from the layer base, contiguous positions 1..TCPREV+TB
  !         (B=1 layout, as in attn_step — the batch stride is not carried)
  ! Query at chunk position iq (1-based) attends to cache positions
  ! 1..TCPREV+iq: full past, causal inside the chunk.
  ! Equivalences (both asserted in src/test/test_kernels.f90):
  !   TB=1          -> attn_step(..., TC=TCPREV+1)
  !   TCPREV=0      -> causal_attn on the same rows (same op order)
  ! This is the prefill/spec-verify kernel: chunked passes replace one call
  ! per token, turning T=1 GEMVs into T=TB GEMMs at identical semantics.
  subroutine attn_chunk(q, K, V, y, BB, HH, K_HH, DD, TC_PREV, TB)
    integer(c_int), intent(in) :: BB, HH, K_HH, DD, TC_PREV, TB
    real(wp), intent(in)  :: q(:)
    real(wp), intent(in)  :: K(:), V(:)
    real(wp), intent(out) :: y(:)
    integer :: ia, iq, ib, nvalid, ss, id, kb, rep
    real(wp) :: scale, sm, inv, acc, m
    real(wp) :: sc(TC_PREV + TB)

    scale = 1.0_wp / sqrt(real(DD, wp))
    rep = HH / K_HH

    !$omp parallel do collapse(2) private(iq, ib, nvalid, ss, id, kb, sc, m, sm, inv, acc)
    do ia = 1, BB
      do iq = 1, TB
        nvalid = TC_PREV + iq
        do ib = 1, HH
          kb = (ib - 1) / rep + 1
          m = -huge(1.0_wp)
          do ss = 1, nvalid
            acc = 0.0_wp
            do id = 1, DD
              acc = acc + q(((ia-1)*TB + (iq-1))*HH*DD + (ib-1)*DD + id) &
                  * K(((ss-1)*K_HH + (kb-1))*DD + id)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_wp
          do ss = 1, nvalid
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          inv = 1.0_wp / sm
          do id = 1, DD
            acc = 0.0_wp
            do ss = 1, nvalid
              acc = acc + sc(ss) * inv &
                  * V(((ss-1)*K_HH + (kb-1))*DD + id)
            end do
            y(((ia-1)*TB + (iq-1))*HH*DD + (ib-1)*DD + id) = acc
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine attn_chunk

  ! ---------------------------------------------------------------------------
  ! TODO(next session): attn_bwd_sgemm -- the missing half of the attention work.
  !
  ! Why it is the big prize: attn_bwd is a hand-written O(T^2) loop with
  ! !$omp atomic on dk/dv, and it is ~309 GFLOP of a training step's ~1.5 TFLOP
  ! (B=1,T=2048,L=12) at ~8 GFLOP/s, where the forward's attn_sgemm runs at
  ! ~103 GFLOP/s. Forward alone: ~20 s/step -> ~1.5 s. Both: step ~46 s ->
  ! ~10-15 s, i.e. every future phase gets ~3-4x cheaper.
  !
  ! Formulation, with S = Q K^T (unscaled), P = softmax(scale*S + causal mask),
  ! Y = P V, and per (batch, kv-head):
  !   dV   = P^T dY                 sgemm('T','N', D,T,T, P, dY)
  !   dP   = dY V^T                 sgemm('N','T', T,T,D, dY, V)
  !   dS   = P * (dP - rowsum(P*dP))      elementwise, O(T^2) memory-bound
  !   dQ   = scale * dS K           sgemm('N','N', T,D,T, dS, K)
  !   dK   = scale * dS^T Q         sgemm('T','N', T,D,T, dS, Q)
  ! remembering the codebase's row-major-as-column-major trick: a row-major
  ! (T,D) buffer IS the column-major matrix (D,T), so every operand above needs
  ! its transposed view spelled out the way attn_sgemm does it.
  !
  ! GQA: with H > K_H several query heads share a kv head. The naive kernel
  ! uses atomics; do NOT copy that -- accumulate the rep heads' dK/dV into a
  ! scratch per kv head and add once, which is both faster and deterministic.
  !
  ! Verification is already scaffolded: test_attn_bwd does finite differences on
  ! the naive kernel. Mirror it for the sgemm twin: same tolerances, plus the
  ! T=1 and no-GQA cases, and only then wire it behind an explicit switch
  ! (never swap attention kernels under a live phase -- the last-bit drift would
  ! silently invalidate the val curve and the bpb self-check).
  !
  ! SDPA backward with GQA (recomputes scores/softmax: checkpoint style).
  ! Forward per (b,h,t): s_i = (q_t.k_i)/sqrt(D), i<=t; p = softmax(s);
  !   y_d = sum_i p_i * v_{i,d}.
  !   dv_{i,d} += p_i * dy_d
  !   ds_i = p_i * (dp_i - sum_j dp_j*p_j),  dp_i = sum_d dy_d*v_{i,d}
  !   dq_d = sum_i ds_i * k_{i,d} / sqrt(D)
  !   dk_{i,d} += ds_i * q_d / sqrt(D)
  ! dq positions are unique per (b,h,t) (plain writes); kv heads are
  ! shared across each GQA group, so dk/dv use atomics.
  subroutine attn_bwd(dy, q, k, v, dq, dk, dv, BB, TT, HH, K_HH, DD)
    integer(c_int), intent(in) :: BB, TT, HH, K_HH, DD
    real(wp), intent(in)  :: dy(:)
    real(wp), intent(in)  :: q(:)
    real(wp), intent(in)  :: k(:), v(:)
    real(wp), intent(out) :: dq(:)
    real(wp), intent(inout) :: dk(:), dv(:)
    integer :: ia, ib, ic, ss, id, kb, rep
    real(wp) :: scale, sm, ssum, acc, ds
    real(wp) :: sc(TT), dpv(TT), m

    scale = 1.0_wp / sqrt(real(DD, wp))
    rep = HH / K_HH

    !$omp parallel do collapse(2) private(ia, ib, ic, ss, id, kb, sc, dpv, &
    !$omp& m, sm, ssum, acc, ds)
    do ia = 1, BB
      do ib = 1, HH
        kb = (ib - 1) / rep + 1
        do ic = 1, TT
          ! forward replay: scores + softmax for query row ic
          m = -huge(1.0_wp)
          do ss = 1, ic
            acc = 0.0_wp
            do id = 1, DD
              acc = acc + q(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) &
                         * k(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id)
            end do
            sc(ss) = acc * scale
            if (sc(ss) > m) m = sc(ss)
          end do
          sm = 0.0_wp
          do ss = 1, ic
            sc(ss) = exp(sc(ss) - m)
            sm = sm + sc(ss)
          end do
          do ss = 1, ic
            sc(ss) = sc(ss) / sm
          end do
          ! dp_i = dy . v_i  (reused as dpv), S = sum dp*p
          ssum = 0.0_wp
          do ss = 1, ic
            acc = 0.0_wp
            do id = 1, DD
              acc = acc + dy(((ia-1)*TT + (ic-1))*HH*DD + (ib-1)*DD + id) &
                         * v(((ia-1)*TT + (ss-1))*K_HH*DD + (kb-1)*DD + id)
            end do
            dpv(ss) = acc
            ssum = ssum + acc * sc(ss)
          end do
          ! dq (unique: plain write) + dk/dv (shared: atomics)
          do id = 1, DD
            acc = 0.0_wp
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
  subroutine relu2_bwd(dy, x, dx, N)
    integer(c_int), intent(in) :: N
    real(wp), intent(in)  :: dy(:), x(:)
    real(wp), intent(out) :: dx(:)
    integer :: ii
    real(wp) :: aa

    !$omp parallel do private(aa)
    do ii = 1, N
      aa = x(ii)
      if (aa < 0.0_wp) aa = 0.0_wp
      dx(ii) = 2.0_wp * aa * dy(ii)
    end do
    !$omp end parallel do
  end subroutine relu2_bwd

  ! ReLU^2 activation:  y = max(0, x)^2
  subroutine relu2(x, N)
    integer(c_int), intent(in) :: N
    real(wp), intent(inout) :: x(:)
    integer :: ii
    real(wp) :: aa

    !$omp parallel do private(aa)
    do ii = 1, N
      aa = x(ii)
      if (aa < 0.0_wp) aa = 0.0_wp
      x(ii) = aa * aa
    end do
    !$omp end parallel do
  end subroutine relu2

end module fortran_attn_mod