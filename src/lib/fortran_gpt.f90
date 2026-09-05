! Main GPT forward pass — wires all kernels together.
!
! Mirrors train.py's GPT.forward() exactly:
!
!   x = wte(idx)                                    ! token embedding
!   x = norm(x)                                     ! RMSNorm pre-norm
!   for block in h:
!     x = x + attn(norm(x))                        ! attention + residual
!     x = x + mlp(norm(x))                         ! MLP + residual
!   x = norm(x)
!   logits = lm_head(x)                             ! (B, T, vocab_size)
!
! Parameters (all flat row-major C-order float32 buffers):
!
!   idx:        (B*T)           token IDs
!   cos_buf:    (T * head_dim/2)   rotary cos table
!   sin_buf:    (T * head_dim/2)   rotary sin table
!   wte:        (vocab_size * d_model)  token embeddings
!   c_q:        (n_head * head_dim * d_model)  Q projection
!   c_k:        (n_kv_head * head_dim * d_model)  K projection
!   c_v:        (n_kv_head * head_dim * d_model)  V projection
!   c_proj:     (d_model * n_head * head_dim)  attention output projection
!   c_fc:       (4*d_model * d_model)  FC1 (SwiGLU gate)
!   c_proj2:    (d_model * 4*d_model)  FC2 (SwiGLU up)
!   lm_head:    (vocab_size * d_model)  unembedding
!
! Output:  outp:  (B * T * vocab_size)  logits

module fortran_gpt_mod
  use iso_c_binding
  implicit none

  ! C-mangled interfaces to subroutines in other modules
  interface
    subroutine wte_lookup(idx, wte, out, B, T, V, D) bind(c, name='wte_lookup')
      integer, intent(in) :: idx(*), B, T, V, D
      real, intent(in) :: wte(*)
      real, intent(out) :: out(*)
    end subroutine
    subroutine rmsnorm0(x, y, N, C, eps) bind(c, name='rmsnorm0')
      integer, intent(in) :: N, C
      real, intent(in) :: x(*)
      real, intent(out) :: y(*)
      real, value :: eps
    end subroutine
    subroutine linear3d(x, w, y, B, T, IF, OF) bind(c, name='linear3d')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine linear3d_sgemm(x, w, y, B, T, IF, OF) \
        bind(c, name='linear3d_sgemm')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine linear3dT(x, w, y, B, T, IF, OF) bind(c, name='linear3dT')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine rope_4d(x, cos, sin, y, B, T, H, D) bind(c, name='rope_4d')
      integer, intent(in) :: B, T, H, D
      real, intent(in) :: x(*), cos(*), sin(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine causal_attn(q, k, v, y, B, T, H, K_H, D) bind(c, name='causal_attn')
      integer, intent(in) :: B, T, H, K_H, D
      real, intent(in) :: q(*), k(*), v(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine relu2(x, N) bind(c, name='relu2')
      integer, intent(in) :: N
      real, intent(inout) :: x(*)
    end subroutine
  end interface
contains

  subroutine gpt_forward(idx, cos_buf, sin_buf, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       outp, &
       BB, TT, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_layer, eps) bind(c, name='gpt_forward')

    integer(c_int), intent(in) :: BB, TT, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_layer
    real(c_float), value :: eps

    integer(c_int), intent(in) :: idx(BB*TT)
    real(c_float), intent(in) :: cos_buf(TT*(head_dim/2))
    real(c_float), intent(in) :: sin_buf(TT*(head_dim/2))

    real(c_float), intent(in) :: wte(vocab_size*d_model)
    ! Per-layer weights, layer ll occupying [(ll-1)*per+1 : ll*per], row-major.
    ! (Real checkpoints have distinct weights per layer; tied weights = pass
    !  the same buffer n_layer times, but the signature models the general case.)
    real(c_float), intent(in) :: c_q(n_layer*n_head*head_dim*d_model)
    real(c_float), intent(in) :: c_k(n_layer*n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_v(n_layer*n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_proj(n_layer*d_model*n_head*head_dim)
    real(c_float), intent(in) :: c_fc(n_layer*4*d_model*d_model)
    real(c_float), intent(in) :: c_proj2(n_layer*d_model*4*d_model)
    real(c_float), intent(in) :: lm_head(vocab_size*d_model)

    real(c_float), intent(out) :: outp(BB*TT*vocab_size)

    ! working buffers: all flat row-major (B,T,...) to match PyTorch layout.
    ! (Declared 1-D on purpose: Fortran N-D arrays are column-major and would
    !  silently transpose the buffers the kernels read as row-major.)
    real(c_float), allocatable :: emd(:), xn(:), sub_out(:)
    real(c_float), allocatable :: q(:), k(:), v(:)
    real(c_float), allocatable :: qrot(:), krot(:)
    real(c_float), allocatable :: attn_out(:), mlpd(:)

    integer :: d_ff, d2, ll, jj
    integer :: qsz, ksz, psz, fcsz, p2sz

    d_ff = 4 * d_model
    d2   = head_dim / 2
    qsz  = n_head*head_dim*d_model
    ksz  = n_kv_head*head_dim*d_model
    psz  = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model

    allocate(emd(BB*TT*d_model))
    allocate(xn(BB*TT*d_model))
    allocate(sub_out(BB*TT*d_model))

    allocate(q(BB*TT*n_head*head_dim))
    allocate(k(BB*TT*n_kv_head*head_dim))
    allocate(v(BB*TT*n_kv_head*head_dim))
    allocate(qrot(BB*TT*n_head*head_dim))
    allocate(krot(BB*TT*n_kv_head*head_dim))
    allocate(attn_out(BB*TT*d_model))
    allocate(mlpd(BB*TT*d_ff))

    ! ---- 1. token embedding --------------------------------
    call wte_lookup(idx, wte, emd, BB, TT, vocab_size, d_model)

    ! ---- 2. initial RMSNorm (train.py norms embeddings before blocks)
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)
    do jj = 1, BB*TT*d_model
      emd(jj) = xn(jj)
    end do

    ! ---- 3. transformer blocks: x = x + attn(norm(x)); x = x + mlp(norm(x)) ---
    do ll = 1, n_layer

      ! attention sub-layer (all nn.Linear -> linear3d, i.e. y = x @ W^T)
      ! Layer ll slice: contiguous section -> sequence association, no copy.
      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d_sgemm(xn, c_q((ll-1)*qsz+1:), q, BB, TT, d_model, n_head*head_dim)
      call linear3d_sgemm(xn, c_k((ll-1)*ksz+1:), k, BB, TT, d_model, n_kv_head*head_dim)
      call linear3d_sgemm(xn, c_v((ll-1)*ksz+1:), v, BB, TT, d_model, n_kv_head*head_dim)

      call rope_4d(q, cos_buf, sin_buf, qrot, BB, TT, n_head, head_dim)
      call rope_4d(k, cos_buf, sin_buf, krot, BB, TT, n_kv_head, head_dim)

      call causal_attn(qrot, krot, v, attn_out, BB, TT, n_head, n_kv_head, head_dim)

      call linear3d_sgemm(attn_out, c_proj((ll-1)*psz+1:), sub_out, BB, TT, d_model, d_model)

      ! residual: emd = emd + attn_out
      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      ! MLP sub-layer: fc -> relu^2 -> proj
      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d_sgemm(xn, c_fc((ll-1)*fcsz+1:), mlpd, BB, TT, d_model, d_ff)
      call relu2(mlpd, BB*TT*d_ff)
      call linear3d_sgemm(mlpd, c_proj2((ll-1)*p2sz+1:), sub_out, BB, TT, d_ff, d_model)

      ! residual: emd = emd + mlp_out
      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 4. final RMSNorm ----------------------------------
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)

    ! ---- 5. LM head (nn.Linear n_embd -> vocab) ------------
    call linear3d_sgemm(xn, lm_head, outp, BB, TT, d_model, vocab_size)

    deallocate(emd, xn, sub_out, q, k, v, qrot, krot, attn_out, mlpd)

  end subroutine gpt_forward

end module fortran_gpt_mod
