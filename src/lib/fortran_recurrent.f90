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
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none
contains

  subroutine recurrent_forward(idx, cos_buf, sin_buf, &
       wte, c_q, c_k, c_v, c_proj, &
       c_fc, c_proj2, lm_head, &
       outp, &
       BB, TT, vocab_size, d_model, &
       n_head, n_kv_head, head_dim, &
       n_loops, eps) bind(c, name='recurrent_forward')

    integer(c_int), intent(in) :: BB, TT, vocab_size, d_model
    integer(c_int), intent(in) :: n_head, n_kv_head, head_dim, n_loops
    real(c_float), value :: eps

    integer(c_int), intent(in) :: idx(BB*TT)
    real(c_float), intent(in) :: cos_buf(TT*(head_dim/2))
    real(c_float), intent(in) :: sin_buf(TT*(head_dim/2))

    real(c_float), intent(in) :: wte(vocab_size*d_model)
    real(c_float), intent(in) :: c_q(n_head*head_dim*d_model)
    real(c_float), intent(in) :: c_k(n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_v(n_kv_head*head_dim*d_model)
    real(c_float), intent(in) :: c_proj(d_model*n_head*head_dim)
    real(c_float), intent(in) :: c_fc(4*d_model*d_model)
    real(c_float), intent(in) :: c_proj2(d_model*4*d_model)
    real(c_float), intent(in) :: lm_head(vocab_size*d_model)

    real(c_float), intent(out) :: outp(BB*TT*vocab_size)

    real(c_float), allocatable :: emd(:), xn(:), sub_out(:)
    real(c_float), allocatable :: q(:), k(:), v(:)
    real(c_float), allocatable :: qrot(:), krot(:)
    real(c_float), allocatable :: attn_out(:), mlpd(:)

    integer :: d_ff, lr, jj

    d_ff = 4 * d_model

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

    ! ---- 2. initial RMSNorm (matches gpt_forward / train.py) ---
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)
    do jj = 1, BB*TT*d_model
      emd(jj) = xn(jj)
    end do

    ! ---- 2. recurrent block: same weights, n_loops passes --
    do lr = 1, n_loops

      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d(xn, c_q, q, BB, TT, d_model, n_head*head_dim)
      call linear3d(xn, c_k, k, BB, TT, d_model, n_kv_head*head_dim)
      call linear3d(xn, c_v, v, BB, TT, d_model, n_kv_head*head_dim)

      call rope_4d(q, cos_buf, sin_buf, qrot, BB, TT, n_head, head_dim)
      call rope_4d(k, cos_buf, sin_buf, krot, BB, TT, n_kv_head, head_dim)

      call causal_attn(qrot, krot, v, attn_out, BB, TT, n_head, &
          n_kv_head, head_dim)

      call linear3d(attn_out, c_proj, sub_out, BB, TT, d_model, d_model)

      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

      call rmsnorm0(emd, xn, BB*TT, d_model, eps)
      call linear3d(xn, c_fc, mlpd, BB, TT, d_model, d_ff)
      call relu2(mlpd, BB*TT*d_ff)
      call linear3d(mlpd, c_proj2, sub_out, BB, TT, d_ff, d_model)

      do jj = 1, BB*TT*d_model
        emd(jj) = emd(jj) + sub_out(jj)
      end do

    end do

    ! ---- 3. final RMSNorm + LM head -------------------------
    call rmsnorm0(emd, xn, BB*TT, d_model, eps)
    call linear3d(xn, lm_head, outp, BB, TT, d_model, vocab_size)

    deallocate(emd, xn, sub_out, q, k, v, qrot, krot, attn_out, mlpd)

  end subroutine recurrent_forward

end module fortran_recurrent_mod
