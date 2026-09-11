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
! Parameters (all flat row-major real(wp) buffers):
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
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: linear3d_sgemm
  use fortran_linear_mod, only: wte_lookup
  use fortran_rmsnorm_mod, only: rmsnorm0
  use fortran_rope_mod, only: rope_4d
  use fortran_attn_mod, only: causal_attn, relu2, attn_sgemm
  implicit none

  ! Inference workspace: allocated once, reused forever (no per-call
  ! malloc). Serial callers only; dims-checked, reallocated on change.
  type :: gpt_ws_t
    real(wp), pointer :: emd(:) => null(), xn(:) => null()
    real(wp), pointer :: sub_out(:) => null()
    real(wp), pointer :: q(:) => null(), k(:) => null(), v(:) => null()
    real(wp), pointer :: qrot(:) => null(), krot(:) => null()
    real(wp), pointer :: ao(:) => null(), mlpd(:) => null()
  end type gpt_ws_t
  type(gpt_ws_t), save :: WS
  integer, save :: WS_BT = -1, WS_DD = -1, WS_hdd = -1
  integer, save :: WS_kvd = -1, WS_dff = -1
contains

  subroutine gpt_forward(idx, cos_buf, sin_buf, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       outp, &
       BB, TT, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_layer, eps, attn_blas)

    integer(c_int), intent(in) :: BB, TT, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_layer
    real(wp), value :: eps
    ! Optional: route the attention through sgemm (attn_sgemm) instead of the
    ! hand-written causal_attn. ~13x faster at T=2048 (7.9 -> 102.8 GFLOP/s)
    ! and equal to ~2.6e-06, but a different summation order -- so it is opt-in
    ! and stays off by default, which is what keeps previously recorded bpb
    ! numbers comparable (see eval_bpb --attn).
    logical, intent(in), optional :: attn_blas
    logical :: useblas
    real(wp), allocatable :: Satt(:)

    integer(c_int), intent(in) :: idx(:)
    real(wp), intent(in) :: cos_buf(:)
    real(wp), intent(in) :: sin_buf(:)

    real(wp), intent(in) :: wte(:)
    ! Per-layer weights, layer ll occupying [(ll-1)*per+1 : ll*per], row-major.
    ! (Real checkpoints have distinct weights per layer; tied weights = pass
    !  the same buffer n_layer times, but the signature models the general case.)
    real(wp), intent(in) :: c_q(:)
    real(wp), intent(in) :: c_k(:)
    real(wp), intent(in) :: c_v(:)
    real(wp), intent(in) :: c_proj(:)
    real(wp), intent(in) :: c_fc(:)
    real(wp), intent(in) :: c_proj2(:)
    real(wp), intent(in) :: lm_head(:)

    real(wp), intent(out) :: outp(:)

    ! working buffers: all flat row-major (B,T,...) to match PyTorch layout.
    ! (Declared 1-D on purpose: Fortran N-D arrays are column-major and would
    !  silently transpose the buffers the kernels read as row-major.)
    real(wp), pointer :: emd(:), xn(:), sub_out(:)
    real(wp), pointer :: q(:), k(:), v(:)
    real(wp), pointer :: qrot(:), krot(:)
    real(wp), pointer :: attn_out(:), mlpd(:)

    integer :: d_ff, d2, ll, jj
    integer :: qsz, ksz, psz, fcsz, p2sz

    d_ff = 4 * d_model
    useblas = .false.
    if (present(attn_blas)) useblas = attn_blas
    if (useblas) allocate (Satt(TT*TT))
    d2   = head_dim / 2
    qsz  = n_head*head_dim*d_model
    ksz  = n_kv_head*head_dim*d_model
    psz  = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model

    if (WS_BT /= BB*TT .or. WS_DD /= d_model .or. WS_hdd /= n_head*head_dim &
        .or. WS_kvd /= n_kv_head*head_dim .or. WS_dff /= 4*d_model) then
      WS_BT = BB*TT; WS_DD = d_model; WS_hdd = n_head*head_dim
      WS_kvd = n_kv_head*head_dim; WS_dff = 4*d_model
      if (associated(WS%emd)) deallocate(WS%emd, WS%xn, WS%sub_out, &
          WS%q, WS%k, WS%v, WS%qrot, WS%krot, WS%ao, WS%mlpd)
      allocate(WS%emd(WS_BT*WS_DD), WS%xn(WS_BT*WS_DD), &
          WS%sub_out(WS_BT*WS_DD))
      allocate(WS%q(WS_BT*WS_hdd), WS%k(WS_BT*WS_kvd), WS%v(WS_BT*WS_kvd))
      allocate(WS%qrot(WS_BT*WS_hdd), WS%krot(WS_BT*WS_kvd))
      allocate(WS%ao(WS_BT*WS_DD), WS%mlpd(WS_BT*WS_dff))
    end if
    emd => WS%emd; xn => WS%xn; sub_out => WS%sub_out
    q => WS%q; k => WS%k; v => WS%v
    qrot => WS%qrot; krot => WS%krot; attn_out => WS%ao; mlpd => WS%mlpd

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

      if (useblas) then
        call attn_sgemm(qrot, krot, v, attn_out, BB, TT, n_head, n_kv_head, &
            head_dim, Satt)
      else
        call causal_attn(qrot, krot, v, attn_out, BB, TT, n_head, n_kv_head, &
            head_dim)
      end if

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



    if (allocated(Satt)) deallocate (Satt)
  end subroutine gpt_forward

end module fortran_gpt_mod
