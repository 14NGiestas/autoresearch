! lib/fortran_muon.f90 — Muon optimizer math (Keller Jordan, 2024).
!
! Faithful Fortran port of the canonical muon.py:
!   zeropower_via_newtonschulz5 (a,b,c = 3.4445,-4.7750,2.0315) +
!   muon_update (momentum 0.95, Nesterov-via-lerp, spectral rescale).
! Verified against numpy goldens in test (test_muon_ns): bit-faithful to
! ~1e-5 in fp32. Recipe scope (unchanged): Muon for hidden 2D matrices
! ONLY; embeddings / lm_head / vectors stay on Adam(W).
!
! Phase 1: pure math + test. Wiring into train_run (state mgmt mirroring
! fortran_adam_state) comes after the LR micro-sweep decides adoption.
!
! Layout: native 2D column-major working arrays. Integration maps the
! row-major flat checkpoint buffers at the boundary (transpose-if-tall
! rule keys on (rows,cols) = (fanout,fanin) as in the source).

module fortran_muon_mod
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: sgemm
  implicit none
  private
  public :: ns_orthogonalize, muon_update_mat

  real(wp), parameter :: NS_A = 3.4445_wp
  real(wp), parameter :: NS_B = -4.7750_wp
  real(wp), parameter :: NS_C = 2.0315_wp

contains

  ! In-place Newton-Schulz (quintic) orthogonalization of X.
  ! Tall input (rows>cols) is worked transposed, exactly like the source
  ! (X = X.mT ... X = X.mT bracketing the iteration).
  subroutine ns_orthogonalize(X, steps)
    real(wp), contiguous, intent(inout) :: X(:, :)
    integer, intent(in), optional :: steps
    integer :: nst, r, c, k, kw, it
    real(wp) :: nrm
    real(wp), allocatable :: W(:, :), A(:, :), T2(:, :), B(:, :), W2(:, :)
    logical :: was_tall

    nst = 5
    if (present(steps)) nst = steps
    r = size(X, 1)
    c = size(X, 2)
    was_tall = r > c
    if (was_tall) then
      W = transpose(X)   ! wide (c,r)
    else
      W = X
    end if
    k = size(W, 1)
    kw = size(W, 2)
    nrm = sqrt(sum(W*W))
    W = W / (nrm + 1.0e-7_wp)
    allocate (A(k, k), T2(k, k), B(k, k), W2(k, kw))
    do it = 1, nst
      ! A = W @ W^T ; T2 = A @ A ; B = b*A + c*T2 ; W = a*W + B @ W
      call sgemm('N', 'T', int(k, 8), int(k, 8), int(kw, 8), 1.0_wp, &
          W, int(k, 8), W, int(k, 8), 0.0_wp, A, int(k, 8))
      call sgemm('N', 'N', int(k, 8), int(k, 8), int(k, 8), 1.0_wp, &
          A, int(k, 8), A, int(k, 8), 0.0_wp, T2, int(k, 8))
      B = NS_B*A + NS_C*T2
      call sgemm('N', 'N', int(k, 8), int(kw, 8), int(k, 8), 1.0_wp, &
          B, int(k, 8), W, int(k, 8), 0.0_wp, W2, int(k, 8))
      W = NS_A*W + W2
    end do
    deallocate (A, T2, B, W2)
    if (was_tall) then
      X = transpose(W)
    else
      X = W
    end if
  end subroutine ns_orthogonalize

  ! One Muon update for a single 2D matrix (functional; state lives outside):
  !   mbuf = beta*mbuf + (1-beta)*g            (momentum, in-place)
  !   upd  = (1-beta)*g + beta*mbuf            (Nesterov-via-lerp, exact source order)
  !   upd  = zeropower(upd)                    (NS orthogonalize)
  !   upd *= sqrt(max(1, srows/scols))         (spectral rescale)
  ! srows/scols are the LOGICAL (fanout,fanin) dims: callers passing a
  ! transposed native view (our flat buffers are row-major) give them
  ! explicitly, so the scale keys on the model geometry, not the view.
  ! Caller does p = p*(1-lr*wd) - lr*upd (decoupled decay, AdamW-style).
  subroutine muon_update_mat(g, mbuf, beta, upd, srows, scols)
    real(wp), contiguous, intent(in) :: g(:, :)
    real(wp), contiguous, intent(inout) :: mbuf(:, :)
    real(wp), intent(in) :: beta
    real(wp), contiguous, intent(out) :: upd(:, :)
    integer, intent(in) :: srows, scols
    real(wp) :: scale

    mbuf = beta*mbuf + (1.0_wp - beta)*g
    upd = (1.0_wp - beta)*g + beta*mbuf
    call ns_orthogonalize(upd)
    scale = sqrt(max(1.0_wp, real(srows, wp) / real(scols, wp)))
    upd = upd*scale
  end subroutine muon_update_mat

end module fortran_muon_mod
