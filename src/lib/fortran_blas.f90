! lib/fortran_blas.f90 — BLAS-backed matmul (perf slice, hyp_34ea7c).
!
! Same math as linear3d (y = x @ W^T, row-major flats) via a single
! sgemm call. Layout trick: row-major Y(BT,OF) IS column-major Yf(OF,BT),
! so with Xf(IF,BT) = X^T and Wf(IF,OF) = W^T in the same memory:
!   Yf = Wf^T . Xf  ->  sgemm('T','N', OF,BT,IF, 1, W,IF, X,IF, 0, Y,OF)
! Call OUTSIDE OpenMP regions (OpenBLAS threads internally; nesting
! oversubscribes). Needs -lopenblas (flake) + [build] link (fpm.toml).
! If OpenBLAS is absent at link time, delete this file and keep linear3d.

module fortran_blas_mod
  use iso_c_binding
  implicit none

  ! NOTE: nixpkgs OpenBLAS builds ILP64 (USE64BITINT): all Fortran
  ! integer args are 64-bit. Declaring them c_int (32-bit) makes sgemm
  ! read adjacent stack garbage (param-13/LDC error). int64 here.
  interface
    subroutine sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, &
        beta, c, ldc) bind(c, name='sgemm_')
      import :: c_char, c_float, c_int64_t
      character(kind=c_char), intent(in) :: transa, transb
      integer(c_int64_t), intent(in) :: m, n, k, lda, ldb, ldc
      real(c_float), intent(in) :: alpha
      real(c_float), intent(in) :: a(*), b(*)
      real(c_float), intent(in) :: beta
      real(c_float), intent(inout) :: c(*)
    end subroutine sgemm_
  end interface
contains

  ! y(bt,o) = sum_i x(bt,i) * w(o,i); x:(BT,IF) w:(OF,IF) y:(BT,OF).
  subroutine linear3d_sgemm(x, w, y, BB, TT, IF, OF) &
      bind(c, name='linear3d_sgemm')
    integer(c_int), intent(in) :: BB, TT, IF, OF
    real(c_float), intent(in)  :: x(BB*TT*IF), w(OF*IF)
    real(c_float), intent(out) :: y(BB*TT*OF)
    integer(c_int64_t) :: m, n, k, lda, ldb, ldc
    m = int(OF, c_int64_t)
    n = int(BB, c_int64_t) * int(TT, c_int64_t)
    k = int(IF, c_int64_t)
    lda = int(IF, c_int64_t)
    ldb = int(IF, c_int64_t)
    ldc = int(OF, c_int64_t)
    call sgemm_(char(84, c_char), char(78, c_char), m, n, k, &
        1.0_c_float, w, lda, x, ldb, 0.0_c_float, y, ldc)
  end subroutine linear3d_sgemm

end module fortran_blas_mod
