! lib/fortran_backward.f90 — reverse-mode kernels (hyp_34ea7c training loop).
!
! Mirrors the forward kernels in row-major flat layout. Tested by central
! finite differences (no torch needed), not by transcription.
!
! linear3d forward:  y(bt,o) = sum_i x(bt,i) * w(o,i)
!   dx(bt,i) = sum_o dy(bt,o) * w(o,i)
!   dw(o,i)  = sum_bt dy(bt,o) * x(bt,i)
! rmsnorm0 forward:  y = x * inv,  inv = 1/sqrt(mean(x^2) + eps)
!   dx_i = dy_i*inv - x_i * dot(dy,x) / (C * s^1.5),  s = mean(x^2)+eps
! wte_lookup backward: scatter-add dout rows into dwte at idx positions
!   (critical section: colliding token ids must accumulate, not race).

module fortran_backward_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  implicit none
contains

  subroutine linear3d_bwd(dy, x, w, dx, dw, BB, TT, IF, OF)
    integer(c_int), intent(in) :: BB, TT, IF, OF
    real(wp), intent(in)  :: dy(:), x(:), w(:)
    real(wp), intent(out) :: dx(:), dw(:)
    integer :: bt, oo, ii
    real(wp) :: acc

    !$omp parallel do collapse(2) private(acc, ii)
    do bt = 1, BB*TT
      do ii = 1, IF
        acc = 0.0_wp
        do oo = 1, OF
          acc = acc + dy((bt-1)*OF + oo) * w((oo-1)*IF + ii)
        end do
        dx((bt-1)*IF + ii) = acc
      end do
    end do
    !$omp end parallel do

    !$omp parallel do collapse(2) private(acc, bt)
    do oo = 1, OF
      do ii = 1, IF
        acc = 0.0_wp
        do bt = 1, BB*TT
          acc = acc + dy((bt-1)*OF + oo) * x((bt-1)*IF + ii)
        end do
        dw((oo-1)*IF + ii) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear3d_bwd

  subroutine rmsnorm0_bwd(dy, x, dx, NN, CC, eps_val)
    integer(c_int), intent(in) :: NN, CC
    real(wp), intent(in)  :: dy(:), x(:)
    real(wp), intent(out) :: dx(:)
    real(wp), value :: eps_val
    integer :: ii, jj
    real(wp) :: ss, inv, dot, coef

    !$omp parallel do private(ss, inv, dot, coef)
    do ii = 1, NN
      ss = 0.0_wp
      dot = 0.0_wp
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
        dot = dot + dy((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      ss = ss / real(CC, wp) + eps_val
      inv = 1.0_wp / sqrt(ss)
      coef = dot / (real(CC, wp) * ss * sqrt(ss))
      do jj = 1, CC
        dx((ii-1)*CC + jj) = dy((ii-1)*CC + jj) * inv &
            - x((ii-1)*CC + jj) * coef
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm0_bwd

  ! RoPE backward (transpose of the forward rotation):
  !   dx1 = dy1*cos + dy2*(-sin),  dx2 = dy1*sin + dy2*cos
  subroutine rope_4d_bwd(dy, cos_buf, sin_buf, dx, BB, TT, HH, DD)
    integer(c_int), intent(in) :: BB, TT, HH, DD
    real(wp), intent(in)  :: dy(:)
    real(wp), intent(in)  :: cos_buf(:)
    real(wp), intent(in)  :: sin_buf(:)
    real(wp), intent(out) :: dx(:)
    integer :: ia, ib, ic, id, d2, i1, i2, cb
    real(wp) :: e1, e2, c_, s_

    d2 = DD / 2

    !$omp parallel do collapse(3) private(ia, ib, ic, id, i1, i2, cb, e1, e2, c_, s_)
    do ia = 1, BB
      do ib = 1, TT
        do ic = 1, HH
          do id = 1, d2
            i1 = ((ia-1)*TT + (ib-1))*HH*DD + (ic-1)*DD + id
            i2 = i1 + d2
            cb = (ib-1)*d2 + id
            e1 = dy(i1); e2 = dy(i2)
            c_ = cos_buf(cb); s_ = sin_buf(cb)
            dx(i1) = e1 * c_ + e2 * (-s_)
            dx(i2) = e1 * s_ + e2 * c_
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine rope_4d_bwd

  ! Per-position NLL (nats): nll(bt) = logsumexp(logits) - logit[target].
  ! Targets are 0-based ids; no ignore-index (callers mask outside).
  subroutine xent_fwd(logits, targets, nll, BB, TT, VV)
    integer(c_int), intent(in) :: BB, TT, VV
    real(wp), intent(in)  :: logits(:)
    integer(c_int), intent(in) :: targets(:)
    real(wp), intent(out) :: nll(:)
    integer :: bt, jj, tgt
    real(wp) :: m, s

    !$omp parallel do private(bt, jj, tgt, m, s)
    do bt = 1, BB*TT
      tgt = targets(bt) + 1
      m = logits((bt-1)*VV+1)
      do jj = 2, VV
        if (logits((bt-1)*VV+jj) > m) m = logits((bt-1)*VV+jj)
      end do
      s = 0.0_wp
      do jj = 1, VV
        s = s + exp(logits((bt-1)*VV+jj) - m)
      end do
      nll(bt) = (m + log(s)) - logits((bt-1)*VV+tgt)
    end do
    !$omp end parallel do
  end subroutine xent_fwd

  ! Softmax-CE backward (mean reduction):
  !   dlogits = (softmax(logits) - onehot(target)) * scale, scale = 1/N.
  subroutine xent_bwd(logits, targets, dlogits, BB, TT, VV, scale_val)
    integer(c_int), intent(in) :: BB, TT, VV
    real(wp), intent(in)  :: logits(:)
    integer(c_int), intent(in) :: targets(:)
    real(wp), intent(out) :: dlogits(:)
    real(wp), value :: scale_val
    integer :: bt, jj, tgt
    real(wp) :: m, s, p

    !$omp parallel do private(bt, jj, tgt, m, s, p)
    do bt = 1, BB*TT
      tgt = targets(bt) + 1
      m = logits((bt-1)*VV+1)
      do jj = 2, VV
        if (logits((bt-1)*VV+jj) > m) m = logits((bt-1)*VV+jj)
      end do
      s = 0.0_wp
      do jj = 1, VV
        s = s + exp(logits((bt-1)*VV+jj) - m)
      end do
      do jj = 1, VV
        p = exp(logits((bt-1)*VV+jj) - m) / s
        if (jj == tgt) p = p - 1.0_wp
        dlogits((bt-1)*VV+jj) = p * scale_val
      end do
    end do
    !$omp end parallel do
  end subroutine xent_bwd

  ! dwte(v,:) += sum over (b,t) with idx=v of dout(b,t,:). 0-based ids.
  subroutine wte_bwd(idx, dout, dwte, BR, TC, VMAX, DMX)
    integer(c_int), intent(in) :: idx(:), BR, TC, VMAX, DMX
    real(wp), intent(in)  :: dout(:)
    real(wp), intent(inout) :: dwte(:)
    integer :: ia, ib, ic, id

    !$omp parallel do collapse(2) private(ic, id)
    do ia = 1, BR
      do ib = 1, TC
        id = idx((ia-1)*TC + ib)
        do ic = 1, DMX
          !$omp critical
          dwte(id*DMX + ic) = dwte(id*DMX + ic) + &
              dout(((ia-1)*TC + (ib-1))*DMX + ic)
          !$omp end critical
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine wte_bwd

end module fortran_backward_mod
