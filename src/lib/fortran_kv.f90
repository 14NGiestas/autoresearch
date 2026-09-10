! lib/fortran_kv.f90 — single-token decoding step with KV cache.
!
! gpt_step processes ONE new token per call: linears/norms/RoPE run on the
! single position (TT=1 through the existing generic kernels), attention
! reads past K/V from the cache via attn_step, and the new k/v are
! appended. This turns generation from O(T^2) linears (full-prefix
! recompute every step, as chat does today) into O(T).
!
! Cache layout (flat, caller-owned, zeroed by caller before step 1):
!   cache_k/v: n_layer * maxT * (n_kv_head*head_dim) reals.
!   layer l base = l*maxT*Dkh (l 0-based); position t (1-based) at
!   base+(t-1)*Dkh. cache_len counts filled positions (in/out).
! Per-layer weights stacked exactly as in gpt_forward.

module fortran_kv_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod, only: attn_step, attn_chunk, relu2
  implicit none
contains

  subroutine gpt_step(idx1, cos1, sin1, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       cache_k, cache_v, cache_len, maxT, &
       out1, &
       BB, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_layer, eps)

    integer(c_int), intent(in) :: BB, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_layer
    integer(c_int), intent(in) :: maxT
    integer(c_int), intent(inout) :: cache_len
    real(wp), value :: eps

    integer(c_int), intent(in) :: idx1(:)
    real(wp), intent(in) :: cos1(:), sin1(:)

    real(wp), intent(in) :: wte(:)
    real(wp), intent(in) :: c_q(:)
    real(wp), intent(in) :: c_k(:)
    real(wp), intent(in) :: c_v(:)
    real(wp), intent(in) :: c_proj(:)
    real(wp), intent(in) :: c_fc(:)
    real(wp), intent(in) :: c_proj2(:)
    real(wp), intent(in) :: lm_head(:)

    real(wp), intent(inout) :: cache_k(:)
    real(wp), intent(inout) :: cache_v(:)
    real(wp), intent(out) :: out1(:)

    real(wp), allocatable :: emd(:), xn(:), sub_out(:)
    real(wp), allocatable :: q(:), k1(:), v1(:)
    real(wp), allocatable :: qrot(:), krot(:)
    real(wp), allocatable :: attn_out(:), mlpd(:)

    integer :: d_ff, d2, ll, jj, tc, lk
    integer :: qsz, ksz, psz, fcsz, p2sz, dkh

    d_ff = 4 * d_model
    d2   = head_dim / 2
    dkh  = n_kv_head*head_dim
    qsz  = n_head*head_dim*d_model
    ksz  = n_kv_head*head_dim*d_model
    psz  = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model

    if (cache_len >= maxT) then
      print '(A)', "gpt_step: cache full (raise maxT)"
      call exit(1)
    end if
    tc = cache_len + 1   ! 1-based write position; cache holds 1..tc after

    allocate(emd(BB*d_model))
    allocate(xn(BB*d_model))
    allocate(sub_out(BB*d_model))

    allocate(q(BB*n_head*head_dim))
    allocate(k1(BB*n_kv_head*head_dim))
    allocate(v1(BB*n_kv_head*head_dim))
    allocate(qrot(BB*n_head*head_dim))
    allocate(krot(BB*n_kv_head*head_dim))
    allocate(attn_out(BB*d_model))
    allocate(mlpd(BB*d_ff))

    ! ---- 1. token embedding (single position) --------------
    call wte_lookup(idx1, wte, emd, BB, 1, vocab_size, d_model)

    ! ---- 2. initial RMSNorm (train.py norms embeddings before blocks)
    call rmsnorm0(emd, xn, BB, d_model, eps)
    do jj = 1, BB*d_model
      emd(jj) = xn(jj)
    end do

    ! ---- 2. transformer blocks -------------------------------
    do ll = 0, n_layer - 1

      call rmsnorm0(emd, xn, BB, d_model, eps)
      call linear3d_sgemm(xn, c_q(ll*qsz+1:), q, BB, 1, d_model, n_head*head_dim)
      call linear3d_sgemm(xn, c_k(ll*ksz+1:), k1, BB, 1, d_model, n_kv_head*head_dim)
      call linear3d_sgemm(xn, c_v(ll*ksz+1:), v1, BB, 1, d_model, n_kv_head*head_dim)

      call rope_4d(q, cos1, sin1, qrot, BB, 1, n_head, head_dim)
      call rope_4d(k1, cos1, sin1, krot, BB, 1, n_kv_head, head_dim)

      ! append rotated k + v to this layer's cache (flat, batch-major,
      ! same layout attn_step reads back)
      lk = ll*maxT*dkh + (tc-1)*dkh
      call cache_copy(krot, cache_k(lk+1:), BB*dkh)
      call cache_copy(v1, cache_v(lk+1:), BB*dkh)

      call attn_step(qrot, cache_k(ll*maxT*dkh+1:), &
          cache_v(ll*maxT*dkh+1:), attn_out, BB, n_head, n_kv_head, &
          head_dim, tc)

      call linear3d_sgemm(attn_out, c_proj(ll*psz+1:), sub_out, BB, 1, &
          d_model, d_model)

      do jj = 1, BB*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      call rmsnorm0(emd, xn, BB, d_model, eps)
      call linear3d_sgemm(xn, c_fc(ll*fcsz+1:), mlpd, BB, 1, d_model, d_ff)
      call relu2(mlpd, BB*d_ff)
      call linear3d_sgemm(mlpd, c_proj2(ll*p2sz+1:), sub_out, BB, 1, d_ff, &
          d_model)

      do jj = 1, BB*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 3. final RMSNorm + LM head --------------------------
    call rmsnorm0(emd, xn, BB, d_model, eps)
    call linear3d_sgemm(xn, lm_head, out1, BB, 1, d_model, vocab_size)

    cache_len = tc

    deallocate(emd, xn, sub_out, q, k1, v1, qrot, krot, attn_out, mlpd)

  end subroutine gpt_step

  ! Multi-token cached step: TB new tokens at positions cache_len+1 ..
  ! cache_len+TB processed in ONE pass (chunked prefill / speculative
  ! verification). out holds TB logit rows: row t is the distribution for
  ! position cache_len+t+1, so out(TB*V+1:) is the next-token row after the
  ! chunk — identical to the last gpt_step of a per-token loop.
  ! Semantics are the same as calling gpt_step TB times (asserted in
  ! src/test/test_kernels.f90, test_kv_chunk_equiv); only the GEMM shapes
  ! differ, so expect ~1e-7 float drift, not bit equality, from BLAS.
  ! B=1 only (cache append and attn_chunk indexing follow attn_step).
  subroutine gpt_step_multi(idx, cos_buf, sin_buf, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       cache_k, cache_v, cache_len, maxT, &
       out, &
       BB, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_layer, TB, eps)

    integer(c_int), intent(in) :: BB, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_layer
    integer(c_int), intent(in) :: maxT, TB
    integer(c_int), intent(inout) :: cache_len
    real(wp), value :: eps

    integer(c_int), intent(in) :: idx(:)
    real(wp), intent(in) :: cos_buf(:), sin_buf(:)

    real(wp), intent(in) :: wte(:)
    real(wp), intent(in) :: c_q(:)
    real(wp), intent(in) :: c_k(:)
    real(wp), intent(in) :: c_v(:)
    real(wp), intent(in) :: c_proj(:)
    real(wp), intent(in) :: c_fc(:)
    real(wp), intent(in) :: c_proj2(:)
    real(wp), intent(in) :: lm_head(:)

    real(wp), intent(inout) :: cache_k(:)
    real(wp), intent(inout) :: cache_v(:)
    real(wp), intent(out) :: out(:)

    real(wp), allocatable :: emd(:), xn(:), sub_out(:)
    real(wp), allocatable :: q(:), k1(:), v1(:)
    real(wp), allocatable :: qrot(:), krot(:)
    real(wp), allocatable :: attn_out(:), mlpd(:)

    integer :: d_ff, d2, ll, jj, tc0, lk, bt
    integer :: qsz, ksz, psz, fcsz, p2sz, dkh

    d_ff = 4 * d_model
    d2   = head_dim / 2
    dkh  = n_kv_head*head_dim
    qsz  = n_head*head_dim*d_model
    ksz  = n_kv_head*head_dim*d_model
    psz  = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model
    bt   = BB * TB

    if (TB < 1) then
      print '(A)', "gpt_step_multi: TB must be >= 1"
      call exit(1)
    end if
    if (cache_len + TB > maxT) then
      print '(A)', "gpt_step_multi: cache full (raise maxT)"
      call exit(1)
    end if
    tc0 = cache_len   ! positions tc0+1 .. tc0+TB are written by this call

    allocate(emd(bt*d_model))
    allocate(xn(bt*d_model))
    allocate(sub_out(bt*d_model))
    allocate(q(bt*n_head*head_dim))
    allocate(k1(bt*n_kv_head*head_dim))
    allocate(v1(bt*n_kv_head*head_dim))
    allocate(qrot(bt*n_head*head_dim))
    allocate(krot(bt*n_kv_head*head_dim))
    allocate(attn_out(bt*d_model))
    allocate(mlpd(bt*d_ff))

    ! ---- 1. token embeddings (TB positions) ------------------
    call wte_lookup(idx, wte, emd, BB, TB, vocab_size, d_model)

    ! ---- 2. initial RMSNorm ----------------------------------
    call rmsnorm0(emd, xn, bt, d_model, eps)
    do jj = 1, bt*d_model
      emd(jj) = xn(jj)
    end do

    ! ---- 3. transformer blocks -------------------------------
    do ll = 0, n_layer - 1

      call rmsnorm0(emd, xn, bt, d_model, eps)
      call linear3d_sgemm(xn, c_q(ll*qsz+1:), q, BB, TB, d_model, &
          n_head*head_dim)
      call linear3d_sgemm(xn, c_k(ll*ksz+1:), k1, BB, TB, d_model, &
          n_kv_head*head_dim)
      call linear3d_sgemm(xn, c_v(ll*ksz+1:), v1, BB, TB, d_model, &
          n_kv_head*head_dim)

      call rope_4d(q, cos_buf, sin_buf, qrot, BB, TB, n_head, head_dim)
      call rope_4d(k1, cos_buf, sin_buf, krot, BB, TB, n_kv_head, head_dim)

      ! append the chunk's rotated k + v (B=1: contiguous)
      lk = ll*maxT*dkh + tc0*dkh
      call cache_copy(krot, cache_k(lk+1:), TB*dkh)
      call cache_copy(v1, cache_v(lk+1:), TB*dkh)

      call attn_chunk(qrot, cache_k(ll*maxT*dkh+1:), &
          cache_v(ll*maxT*dkh+1:), attn_out, BB, n_head, n_kv_head, &
          head_dim, tc0, TB)

      call linear3d_sgemm(attn_out, c_proj(ll*psz+1:), sub_out, BB, TB, &
          d_model, d_model)

      do jj = 1, bt*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      call rmsnorm0(emd, xn, bt, d_model, eps)
      call linear3d_sgemm(xn, c_fc(ll*fcsz+1:), mlpd, BB, TB, d_model, d_ff)
      call relu2(mlpd, bt*d_ff)
      call linear3d_sgemm(mlpd, c_proj2(ll*p2sz+1:), sub_out, BB, TB, d_ff, &
          d_model)

      do jj = 1, bt*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 4. final RMSNorm + LM head (TB rows) ----------------
    call rmsnorm0(emd, xn, bt, d_model, eps)
    call linear3d_sgemm(xn, lm_head, out, BB, TB, d_model, vocab_size)

    cache_len = tc0 + TB

    deallocate(emd, xn, sub_out, q, k1, v1, qrot, krot, attn_out, mlpd)

  end subroutine gpt_step_multi

  ! flat copy used for cache appends (keeps the call sites readable)
  subroutine cache_copy(src, dst, n)
    real(wp), intent(in) :: src(:)
    real(wp), intent(out) :: dst(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      dst(i) = src(i)
    end do
  end subroutine cache_copy

end module fortran_kv_mod
