! lib/fortran_recurrent.f90 — weight-tied recurrent forward (hyp_ba98cd).
!
! The same transformer block as gpt_forward, but ONE layer's weights
! applied n_loops times (cf. Kohli et al. 2604.07822, Chen 2603.21676:
! recurrent-depth transformers, depth extrapolation, overthinking).
!
! With n_loops=1 the output must equal gpt_forward(n_layer=1) bit-exactly
! (same kernels, same order) — that equivalence is the regression test.
! Overthinking studies call with n_loops=1..K and compare drift.
!
! All buffers flat row-major (B,T,...), PyTorch layout, 0-based ids.

module fortran_recurrent_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none

  ! Inference workspace: allocated once, reused forever (no per-call
  ! malloc). Serial callers only; dims-checked, reallocated on change.
  type :: rec_ws_t
    real(wp), pointer :: emd(:) => null(), xn(:) => null()
    real(wp), pointer :: sub_out(:) => null()
    real(wp), pointer :: q(:) => null(), k(:) => null(), v(:) => null()
    real(wp), pointer :: qrot(:) => null(), krot(:) => null()
    real(wp), pointer :: ao(:) => null(), mlpd(:) => null()
  end type rec_ws_t
  type(rec_ws_t), save :: RWS
  integer, save :: RWS_BT = -1, RWS_DD = -1, RWS_hdd = -1
  integer, save :: RWS_kvd = -1, RWS_dff = -1
contains

  subroutine recurrent_forward(idx, cos_buf, sin_buf, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       outp, &
       BB, TT, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_loops, eps)

    integer(c_int), intent(in) :: BB, TT, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_loops
    real(wp), value :: eps

    integer(c_int), intent(in) :: idx(:)
    real(wp), intent(in) :: cos_buf(:)
    real(wp), intent(in) :: sin_buf(:)

    real(wp), intent(in) :: wte(:)
    real(wp), intent(in) :: c_q(:)
    real(wp), intent(in) :: c_k(:)
    real(wp), intent(in) :: c_v(:)
    real(wp), intent(in) :: c_proj(:)
    real(wp), intent(in) :: c_fc(:)
    real(wp), intent(in) :: c_proj2(:)
    real(wp), intent(in) :: lm_head(:)

    real(wp), intent(out) :: outp(:)

    real(wp), pointer :: emd(:), xn(:), sub_out(:)
    real(wp), pointer :: q(:), k(:), v(:)
    real(wp), pointer :: qrot(:), krot(:)
    real(wp), pointer :: attn_out(:), mlpd(:)

    integer :: d_ff, lr, jj

    d_ff = 4 * d_model

    if (RWS_BT /= BB*TT .or. RWS_DD /= d_model .or. RWS_hdd /= n_head*head_dim &
        .or. RWS_kvd /= n_kv_head*head_dim .or. RWS_dff /= 4*d_model) then
      RWS_BT = BB*TT; RWS_DD = d_model; RWS_hdd = n_head*head_dim
      RWS_kvd = n_kv_head*head_dim; RWS_dff = 4*d_model
      if (associated(RWS%emd)) deallocate(RWS%emd, RWS%xn, RWS%sub_out, &
          RWS%q, RWS%k, RWS%v, RWS%qrot, RWS%krot, RWS%ao, RWS%mlpd)
      allocate(RWS%emd(RWS_BT*RWS_DD), RWS%xn(RWS_BT*RWS_DD), &
          RWS%sub_out(RWS_BT*RWS_DD))
      allocate(RWS%q(RWS_BT*RWS_hdd), RWS%k(RWS_BT*RWS_kvd), &
          RWS%v(RWS_BT*RWS_kvd))
      allocate(RWS%qrot(RWS_BT*RWS_hdd), RWS%krot(RWS_BT*RWS_kvd))
      allocate(RWS%ao(RWS_BT*RWS_DD), RWS%mlpd(RWS_BT*RWS_dff))
    end if
    emd => RWS%emd; xn => RWS%xn; sub_out => RWS%sub_out
    q => RWS%q; k => RWS%k; v => RWS%v
    qrot => RWS%qrot; krot => RWS%krot; attn_out => RWS%ao; mlpd => RWS%mlpd

    ! ---- 1. token embedding --------------------------------
    call wte_lookup(idx, wte, emd, BB, TT, vocab_size, d_model)

    ! ---- 2. initial RMSNorm (matches gpt_forward / train.py) ---
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)
    do jj = 1, BB*TT*d_model
      emd(jj) = xn(jj)
    end do

    ! ---- 2. recurrent block: same weights, n_loops passes --
    do lr = 1, n_loops

      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d_sgemm(xn, c_q, q, BB, TT, d_model, n_head*head_dim)
      call linear3d_sgemm(xn, c_k, k, BB, TT, d_model, n_kv_head*head_dim)
      call linear3d_sgemm(xn, c_v, v, BB, TT, d_model, n_kv_head*head_dim)

      call rope_4d(q, cos_buf, sin_buf, qrot, BB, TT, n_head, head_dim)
      call rope_4d(k, cos_buf, sin_buf, krot, BB, TT, n_kv_head, head_dim)

      call causal_attn(qrot, krot, v, attn_out, BB, TT, n_head, &
          n_kv_head, head_dim)

      call linear3d_sgemm(attn_out, c_proj, sub_out, BB, TT, d_model, d_model)

      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d_sgemm(xn, c_fc, mlpd, BB, TT, d_model, d_ff)
      call relu2(mlpd, BB*TT*d_ff)
      call linear3d_sgemm(mlpd, c_proj2, sub_out, BB, TT, d_ff, d_model)

      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 3. final RMSNorm + LM head -------------------------
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)
    call linear3d_sgemm(xn, lm_head, outp, BB, TT, d_model, vocab_size)



  end subroutine recurrent_forward

end module fortran_recurrent_mod
