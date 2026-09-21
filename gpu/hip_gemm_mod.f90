! hip_gemm_mod -- GEMMs tipados sobre o rocBLAS NATIVO, no padrao do MFI.
!
! O que se aprendeu com o MFI (github.com/14NGiestas/mfi), que ainda nao fala HIP
! direto e por isso nao se pode usar aqui:
!
!  1. TRES argumentos. O lda sai de size(a,1) e o m de size(c,1), logo a relacao
!     lda >= M e' estrutural e nao um numero escrito a mao. A noite de 21/set
!     gastou-se inteira nesse numero.
!  2. Arrays tipados e de forma assumida: o compilador confere dimensoes, tipo e
!     intencao. O que era erro de runtime passa a erro de compilacao.
!  3. Opcionais com defaults: transa, transb, alpha, beta.
!  4. UM HANDLE POR THREAD, criado a' primeira utilizacao. E' a licao que faltava:
!     o host e' multi-threaded e um handle partilhado serializa as chamadas.
!  5. Init preguicoso: nao ha' codigo de setup, e ha' um force_cpu/force_gpu.
!
! Aqui nao ha' camada de traducao: e' rocBLAS nativo.
module hip_gemm_mod
  use iso_c_binding
  use hipfort
  use hipfort_rocblas
  implicit none
  private
  public :: gemm_f32, mfi_handles_init, mfi_handles_done, mfi_gpu, mfi_cpu, mfi_use_gpu

  ! O estado do modulo, no espirito do cublas_wrap.c do MFI.
  type(c_ptr), allocatable, save :: g_handles(:)
  integer, save :: g_ready = 0
  integer, save :: g_active = 1        ! 1 = usa a placa, 0 = nao ha' placa
  integer, save :: g_calls = 0

contains
  ! Cria um handle por thread, uma vez. Idempotente.
  subroutine mfi_handles_init(nthreads, ok)
    integer, intent(in), optional :: nthreads
    integer, intent(out), optional :: ok
    integer :: i, nt, ierr
    if (g_ready == 1) then
      if (present(ok)) ok = 0
      return
    end if
    nt = 1
    if (present(nthreads)) nt = max(1, nthreads)
    allocate (g_handles(nt))
    do i = 1, nt
      ierr = rocblas_create_handle(g_handles(i))
      if (ierr /= 0) then
        if (present(ok)) ok = ierr
        return
      end if
      ierr = rocblas_set_pointer_mode(g_handles(i), rocblas_pointer_mode_host)
    end do
    g_ready = 1
    if (present(ok)) ok = 0
  end subroutine mfi_handles_init

  subroutine mfi_handles_done()
    integer :: i, ierr
    if (g_ready /= 1) return
    do i = 1, size(g_handles)
      ierr = rocblas_destroy_handle(g_handles(i))
    end do
    deallocate (g_handles)
    g_ready = 0
  end subroutine mfi_handles_done

  subroutine mfi_gpu()
    g_active = 1
  end subroutine mfi_gpu

  subroutine mfi_cpu()
    g_active = 0
  end subroutine mfi_cpu

  logical function mfi_use_gpu() result(r)
    r = (g_active == 1)
  end function mfi_use_gpu

  ! c(m,n) = a . b, com o lda, o ldb e o ldc tirados das formas.
  ! transa/transb aceitam 'N' e 'T', como no MFI.
  subroutine gemm_f32(a, b, c, transa, transb, alpha, beta, ok)
    real(c_float), intent(in), target, contiguous :: a(:, :), b(:, :)
    real(c_float), intent(inout), target, contiguous :: c(:, :)
    character, intent(in), optional :: transa, transb
    real(c_float), intent(in), optional :: alpha, beta
    integer, intent(out), optional :: ok
    character :: ta, tb
    real(c_float) :: al, be
    integer :: m, n, k, ierr
    integer(c_int) :: opa, opb
    ta = 'N'; if (present(transa)) ta = transa
    tb = 'N'; if (present(transb)) tb = transb
    al = 1.0_c_float; if (present(alpha)) al = alpha
    be = 0.0_c_float; if (present(beta)) be = beta
    m = int(size(c, 1), c_int)
    n = int(size(c, 2), c_int)
    if (ta == 'N' .or. ta == 'n') then
      k = int(size(a, 2), c_int)
      if (size(a, 1) /= m) then
        if (present(ok)) ok = -1000
        return
      end if
      opa = rocblas_operation_none
    else
      k = int(size(a, 1), c_int)
      if (size(a, 2) /= m) then
        if (present(ok)) ok = -1000
        return
      end if
      opa = rocblas_operation_transpose
    end if
    if (tb == 'N' .or. tb == 'n') then
      if (size(b, 1) /= k .or. size(b, 2) /= n) then
        if (present(ok)) ok = -1001
        return
      end if
      opb = rocblas_operation_none
    else
      if (size(b, 1) /= n .or. size(b, 2) /= k) then
        if (present(ok)) ok = -1001
        return
      end if
      opb = rocblas_operation_transpose
    end if
    if (g_ready /= 1) call mfi_handles_init(1)
    g_calls = g_calls + 1
    ierr = rocblas_sgemm(g_handles(1), opa, opb, m, n, k, al, &
        c_loc(a), int(size(a, 1), c_int), &
        c_loc(b), int(size(b, 1), c_int), &
        be, c_loc(c), int(size(c, 1), c_int))
    if (present(ok)) ok = ierr
  end subroutine gemm_f32
end module hip_gemm_mod
