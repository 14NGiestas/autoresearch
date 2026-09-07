! lib/fortran_blas.f90 — BLAS-backed matmul (plain hand interface).
!
! Same math as linear3d (y = x @ W^T, row-major flats) via a single
! sgemm call. Layout trick: row-major Y(BT,OF) IS column-major Yf(OF,BT),
! so with Xf(IF,BT) = X^T and Wf(IF,OF) = W^T in the same memory:
!   Yf = Wf^T . Xf  ->  sgemm('T','N', OF,BT,IF, 1, W,IF, X,IF, 0, Y,OF)
! Call OUTSIDE OpenMP regions (OpenBLAS threads internally; nesting
! oversubscribes). Needs -lopenblas (flake) + [build] link (fpm.toml).
!
! ABI: nixpkgs OpenBLAS is ILP64 (openblas_get_config reports
! USE64BITINT), so all integer args are 64-bit — int32 silently reads
! stack garbage (SIGFPE in gemm_driver). No bind(C) anywhere: plain
! interface body `sgemm` mangles to OpenBLAS's `sgemm_` symbol, plain
! characters ('T'/'N'). NOTE: the mfi fpm package was tried here and
! REVERTED: its interfaces are LP64 (default integer) and segfault/FPE
! against this ILP64 OpenBLAS. Do not re-add without an ILP64 BLAS.

module fortran_blas_mod
  use iso_c_binding, only: c_int64_t
  use fortran_kinds_mod, only: wp
  implicit none

  interface
    subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, &
        beta, c, ldc)
      import :: c_int64_t, wp
      character(len=1), intent(in) :: transa, transb
      integer(c_int64_t), intent(in) :: m, n, k, lda, ldb, ldc
      real(wp), intent(in) :: alpha
      real(wp), intent(in) :: a(*), b(*)
      real(wp), intent(in) :: beta
      real(wp), intent(inout) :: c(*)
    end subroutine sgemm
  end interface
contains

  ! y(bt,o) = sum_i x(bt,i) * w(o,i); x:(BT,IF) w:(OF,IF) y:(BT,OF).
  subroutine linear3d_sgemm(x, w, y, BB, TT, IF, OF)
    integer, intent(in) :: BB, TT, IF, OF
    real(wp), intent(in)  :: x(:), w(:)
    real(wp), intent(out) :: y(:)
    integer(c_int64_t) :: m, n, k, lda, ldb, ldc
    m = int(OF, c_int64_t)
    n = int(BB, c_int64_t) * int(TT, c_int64_t)
    k = int(IF, c_int64_t)
    lda = int(IF, c_int64_t)
    ldb = int(IF, c_int64_t)
    ldc = int(OF, c_int64_t)
    call sgemm('T', 'N', m, n, k, &
        1.0_wp, w, lda, x, ldb, 0.0_wp, y, ldc)
  end subroutine linear3d_sgemm

  ! Reverse-mode twin (same row-major-as-col-major trick):
  !   dx(bt,i) = sum_o dy(bt,o) * w(o,i) -> DXf = Wf . DYf
  !     sgemm('N','N', IF,BT,OF, 1, W,IF, dy,OF, 0, dx,IF)
  !   dw(o,i)  = sum_bt dy(bt,o) * x(bt,i) -> DWf = Xf . DYf^T
  !     sgemm('N','T', IF,OF,BT, 1, x,IF, dy,OF, 0, dw,IF)
  ! Call OUTSIDE OpenMP regions (compute_grads is serial). FP32-only
  ! (sgemm_); a wp->real64 flip needs a dgemm_ twin.
  subroutine linear3d_bwd_sgemm(dy, x, w, dx, dw, BB, TT, IF, OF)
    integer, intent(in) :: BB, TT, IF, OF
    real(wp), intent(in)  :: dy(:), x(:), w(:)
    real(wp), intent(out) :: dx(:), dw(:)
    integer(c_int64_t) :: m, n, k, lda, ldb, ldc, bt64
    bt64 = int(BB, c_int64_t) * int(TT, c_int64_t)
    ! dx = dy . W  (plain, NOT transposed)
    m = int(IF, c_int64_t); n = bt64; k = int(OF, c_int64_t)
    lda = int(IF, c_int64_t); ldb = int(OF, c_int64_t); ldc = int(IF, c_int64_t)
    call sgemm('N', 'N', m, n, k, &
        1.0_wp, w, lda, dy, ldb, 0.0_wp, dx, ldc)
    ! dw = dy^T . x
    m = int(IF, c_int64_t); n = int(OF, c_int64_t); k = bt64
    lda = int(IF, c_int64_t); ldb = int(OF, c_int64_t); ldc = int(IF, c_int64_t)
    call sgemm('N', 'T', m, n, k, &
        1.0_wp, x, lda, dy, ldb, 0.0_wp, dw, ldc)
  end subroutine linear3d_bwd_sgemm

end module fortran_blas_mod
