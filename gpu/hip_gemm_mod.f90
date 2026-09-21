! hip_gemm_mod -- a interface MODERNA para o sgemm do rocBLAS.
!
! Por que existe: o wrapper antigo recebia type(c_ptr) crus e o lda era um numero
! escrito a mao. Com lda=K o rocBLAS recusa quando M>K, e a noite de hoje foi
! gasta a culpar a Tensile, uma flag e o relogio, quando o defeito era esse
! numero. Aqui o lda e' DERIVADO da forma do array: nao ha' como errar.
!
! Com arrays tipados e de forma assumida, o compilador confere as dimensoes, o
! tipo e a intencao. O que era um erro de runtime passa a erro de compilacao.
module hip_gemm_mod
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  private
  public :: gemm_f32

contains
  ! c(m,n) = a(m,k) . b(k,n), tudo column-major como o Fortran guarda.
  ! O lda, o ldb e o ldc saem das formas: lda = size(a,1), e assim por diante.
  subroutine gemm_f32(a, b, c, h, ok)
    real(c_float), intent(in), target, contiguous :: a(:, :), b(:, :)
    real(c_float), intent(inout), target, contiguous :: c(:, :)
    type(c_ptr), intent(in) :: h
    integer, intent(out) :: ok
    integer :: m, n, k
    m = int(size(a, 1), c_int)
    k = int(size(a, 2), c_int)
    n = int(size(b, 2), c_int)
    if (size(b, 1) /= k) then
      ok = -1000        ! as formas nao encaixam: erro que o chamador ve'
      return
    end if
    if (size(c, 1) /= m .or. size(c, 2) /= n) then
      ok = -1001
      return
    end if
    ok = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, &
        m, n, k, 1.0_c_float, &
        c_loc(a), int(size(a, 1), c_int), &
        c_loc(b), int(size(b, 1), c_int), &
        0.0_c_float, &
        c_loc(c), int(size(c, 1), c_int))
  end subroutine gemm_f32
end module hip_gemm_mod
