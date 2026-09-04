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
  implicit none
contains

  subroutine linear3d_bwd(dy, x, w, dx, dw, BB, TT, IF, OF) &
      bind(c, name='linear3d_bwd')
    integer(c_int), intent(in) :: BB, TT, IF, OF
    real(c_float), intent(in)  :: dy(BB*TT*OF), x(BB*TT*IF), w(OF*IF)
    real(c_float), intent(out) :: dx(BB*TT*IF), dw(OF*IF)
    integer :: bt, oo, ii
    real(c_float) :: acc

    !$omp parallel do collapse(2) private(acc, ii)
    do bt = 1, BB*TT
      do ii = 1, IF
        acc = 0.0_c_float
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
        acc = 0.0_c_float
        do bt = 1, BB*TT
          acc = acc + dy((bt-1)*OF + oo) * x((bt-1)*IF + ii)
        end do
        dw((oo-1)*IF + ii) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear3d_bwd

  subroutine rmsnorm0_bwd(dy, x, dx, NN, CC, eps_val) &
      bind(c, name='rmsnorm0_bwd')
    integer(c_int), intent(in) :: NN, CC
    real(c_float), intent(in)  :: dy(NN*CC), x(NN*CC)
    real(c_float), intent(out) :: dx(NN*CC)
    real(c_float), value :: eps_val
    integer :: ii, jj
    real(c_float) :: ss, inv, dot, coef

    !$omp parallel do private(ss, inv, dot, coef)
    do ii = 1, NN
      ss = 0.0_c_float
      dot = 0.0_c_float
      do jj = 1, CC
        ss = ss + x((ii-1)*CC + jj) * x((ii-1)*CC + jj)
        dot = dot + dy((ii-1)*CC + jj) * x((ii-1)*CC + jj)
      end do
      ss = ss / real(CC) + eps_val
      inv = 1.0_c_float / sqrt(ss)
      coef = dot / (real(CC) * ss * sqrt(ss))
      do jj = 1, CC
        dx((ii-1)*CC + jj) = dy((ii-1)*CC + jj) * inv &
            - x((ii-1)*CC + jj) * coef
      end do
    end do
    !$omp end parallel do
  end subroutine rmsnorm0_bwd

  ! dwte(v,:) += sum over (b,t) with idx=v of dout(b,t,:). 0-based ids.
  subroutine wte_bwd(idx, dout, dwte, BR, TC, VMAX, DMX) &
      bind(c, name='wte_bwd')
    integer(c_int), intent(in) :: idx(BR*TC), BR, TC, VMAX, DMX
    real(c_float), intent(in)  :: dout(BR*TC*DMX)
    real(c_float), intent(inout) :: dwte(VMAX*DMX)
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
