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
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none
contains

  subroutine gpt_step(idx1, cos1, sin1, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       cache_k, cache_v, cache_len, maxT, &
       out1, &
       BB, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_layer, eps) bind(c, name='gpt_step')

    integer(c_int), intent(in) :: BB, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_layer
    integer(c_int), intent(in) :: maxT
    integer(c_int), intent(inout) :: cache_len
    real(c_float), value :: eps

    integer(c_int), intent(in) :: idx1(BB)
    real(c_float), intent(in) :: cos1(head_dim/2), sin1(head_dim/2)

    real(c_float), intent(in) :: wte(vocab_size*d_model)
    real(c_float), intent(in) :: c_q(n_layer*n_head*head_dim*d_model)
    real(c_float), intent(in) :: c_k(n_layer*n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_v(n_layer*n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_proj(n_layer*d_model*n_head*head_dim)
    real(c_float), intent(in) :: c_fc(n_layer*4*d_model*d_model)
    real(c_float), intent(in) :: c_proj2(n_layer*d_model*4*d_model)
    real(c_float), intent(in) :: lm_head(vocab_size*d_model)

    real(c_float), intent(inout) :: cache_k(n_layer*maxT*n_kv_head*head_dim)
    real(c_float), intent(inout) :: cache_v(n_layer*maxT*n_kv_head*head_dim)
    real(c_float), intent(out) :: out1(BB*vocab_size)

    real(c_float), allocatable :: emd(:), xn(:), sub_out(:)
    real(c_float), allocatable :: q(:), k1(:), v1(:)
    real(c_float), allocatable :: qrot(:), krot(:)
    real(c_float), allocatable :: attn_out(:), mlpd(:)

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
      call linear3d(xn, c_q(ll*qsz+1:), q, BB, 1, d_model, n_head*head_dim)
      call linear3d(xn, c_k(ll*ksz+1:), k1, BB, 1, d_model, n_kv_head*head_dim)
      call linear3d(xn, c_v(ll*ksz+1:), v1, BB, 1, d_model, n_kv_head*head_dim)

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

      call linear3d(attn_out, c_proj(ll*psz+1:), sub_out, BB, 1, &
          d_model, d_model)

      do jj = 1, BB*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      call rmsnorm0(emd, xn, BB, d_model, eps)
      call linear3d(xn, c_fc(ll*fcsz+1:), mlpd, BB, 1, d_model, d_ff)
      call relu2(mlpd, BB*d_ff)
      call linear3d(mlpd, c_proj2(ll*p2sz+1:), sub_out, BB, 1, d_ff, &
          d_model)

      do jj = 1, BB*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 3. final RMSNorm + LM head --------------------------
    call rmsnorm0(emd, xn, BB, d_model, eps)
    call linear3d(xn, lm_head, out1, BB, 1, d_model, vocab_size)

    cache_len = tc

    deallocate(emd, xn, sub_out, q, k1, v1, qrot, krot, attn_out, mlpd)

  end subroutine gpt_step

  ! flat copy used for cache appends (keeps the call sites readable)
  subroutine cache_copy(src, dst, n)
    real(c_float), intent(in) :: src(n)
    real(c_float), intent(out) :: dst(n)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      dst(i) = src(i)
    end do
  end subroutine cache_copy

end module fortran_kv_mod
