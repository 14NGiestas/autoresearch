! Linear (matrix multiply) kernels — pure Fortran.
!
! Implements:
!   linear3d:  y(bt, o) = sum_i x(bt, i) * w(o, i)  for x:(B,T,IF), w:(OF,IF), y:(B,T,OF)
!   linear3dT: y(bt, o) = sum_i x(bt, i) * w(i, o)  for x:(B,T,IF), w:(IF,OF), y:(B,T,OF)
!   wte_lookup: emb(b,t,j) = wte(idx(b,t), j)  for idx:(B,T), wte:(V,D), emb:(B,T,D)
!
! All arrays are row-major flat real(wp) buffers.
! Parallelized with OpenMP.

module fortran_linear_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  implicit none
contains

  ! y = x @ w^T   where  x:(B,T,IF)  w:(OF,IF)  y:(B,T,OF)
  subroutine linear3d(x, w, y, BB, TT, IF, OF)
    integer(c_int), intent(in) :: BB, TT, IF, OF
    real(wp), intent(in)  :: x(:), w(:)
    real(wp), intent(out) :: y(:)
    integer :: bt, oo, ii
    real(wp) :: acc

    !$omp parallel do collapse(2) private(acc, ii)
    do bt = 1, BB*TT
      do oo = 1, OF
        acc = 0.0_wp
        do ii = 1, IF
          acc = acc + x((bt-1)*IF + ii) * w((oo-1)*IF + ii)
        end do
        y((bt-1)*OF + oo) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear3d

  ! y = x @ w   where  x:(B,T,IF)  w:(IF,OF)  y:(B,T,OF)
  subroutine linear3dT(x, w, y, BB, TT, IF, OF)
    integer(c_int), intent(in) :: BB, TT, IF, OF
    real(wp), intent(in)  :: x(:), w(:)
    real(wp), intent(out) :: y(:)
    integer :: bt, oo, ii
    real(wp) :: acc

    !$omp parallel do collapse(2) private(acc, ii)
    do bt = 1, BB*TT
      do oo = 1, OF
        acc = 0.0_wp
        do ii = 1, IF
          acc = acc + x((bt-1)*IF + ii) * w((ii-1)*OF + oo)
        end do
        y((bt-1)*OF + oo) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine linear3dT

  ! Token embedding lookup: out(b,t,:) = wte(idx(b,t), :)
  ! idx is 0-based (PyTorch convention): row id = idx, NOT idx-1.
  ! Flat 1-D indexing preserves row-major layout; sizes as scalar args.
  subroutine wte_lookup(idx, wte, out, BR, TC, VMAX, DMX)
    integer(c_int), intent(in) :: BR, TC, VMAX, DMX
    integer(c_int), intent(in) :: idx(:)
    real(wp), intent(in) :: wte(:)
    real(wp), intent(out) :: out(:)
    integer :: ia, ib, ic

    do ia = 1, BR
      do ib = 1, TC
        do ic = 1, DMX
          out(((ia-1)*TC + (ib-1))*DMX + ic) = &
            wte(idx((ia-1)*TC + ib)*DMX + ic)
        end do
      end do
    end do
  end subroutine wte_lookup

end module fortran_linear_mod