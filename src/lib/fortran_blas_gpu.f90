! lib/fortran_blas_gpu.f90 -- o caminho GPU das duas rotinas de BLAS.
!
! Por que existe separado: o shim (gpu/rocblas_shim.c) compila com hipcc, que e'
! um compilador DIFERENTE. O fpm tem o hook para isso (FPM_CXX=hipcc), mas um
! build comum nao deve precisar de ROCm. Entao este modulo tem duas metades:
!
!   - a interface bind(C) com o shim, sob a macro ARCH_GPU. Sem a macro, nada
!     aqui chama a GPU, e o pacote builda em qualquer maquina. Isto e' o que
!     mantem o fpm test verde.
!   - o despacho: com a GPU ligada em tempo de execucao (gpu_on), linear3d vai
!     pelo shim; desligada, vai pelo sgemm do OpenBLAS. Os DOIS caminhos ficam
!     vivos, porque o CPU e' a referencia de correcao e a comparacao e' a
!     medicao.
!
! Os tipos sao explicitos (int64_t, real(4)), e nao ha' interface implicita. A
! armadilha registrada em fortran_blas.f90 (OpenBLAS ILP64 le' lixo com interface
! LP64) nao se aplica aqui, porque bind(C) declara tudo.
module fortran_blas_gpu_mod
  use iso_c_binding, only: c_int32_t, c_int64_t, c_float, c_ptr, c_f_pointer
  use fortran_kinds_mod, only: wp
  implicit none
  private

  logical :: gpu_on = .false.
  public :: gpu_available, gpu_enable, gpu_disable, gpu_is_on
  public :: gpu_count

#ifdef ARCH_GPU
  interface
    function gpublas_sgemm_fwd(x, w, bt, y, if_, of_) bind(C, name="gpublas_sgemm_fwd") result(rc)
      import :: c_int32_t, c_int64_t, c_float
      real(c_float), intent(in)  :: x(*), w(*)
      integer(c_int64_t), value  :: bt, if_, of_
      real(c_float), intent(out) :: y(*)
      integer(c_int32_t) :: rc
    end function gpublas_sgemm_fwd

    function gpublas_sgemm_bwd_dx(dy, w, bt, dx, if_, of_) bind(C, name="gpublas_sgemm_bwd_dx") result(rc)
      import :: c_int32_t, c_int64_t, c_float
      real(c_float), intent(in)  :: dy(*), w(*)
      integer(c_int64_t), value  :: bt, if_, of_
      real(c_float), intent(out) :: dx(*)
      integer(c_int32_t) :: rc
    end function gpublas_sgemm_bwd_dx

    function gpublas_sgemm_bwd_dw(dy, x, bt, dw, if_, of_) bind(C, name="gpublas_sgemm_bwd_dw") result(rc)
      import :: c_int32_t, c_int64_t, c_float
      real(c_float), intent(in)  :: dy(*), x(*)
      integer(c_int64_t), value  :: bt, if_, of_
      real(c_float), intent(out) :: dw(*)
      integer(c_int32_t) :: rc
    end function gpublas_sgemm_bwd_dw

    function gpublas_count() bind(C, name="gpublas_count") result(n)
      import :: c_int32_t
      integer(c_int32_t) :: n
    end function gpublas_count
  end interface
#endif

contains

  ! A GPU existe neste binario? Sem a macro, nao.
  logical function gpu_available()
#ifdef ARCH_GPU
    gpu_available = .true.
#else
    gpu_available = .false.
#endif
  end function gpu_available

  integer function gpu_count()
#ifdef ARCH_GPU
    gpu_count = int(gpublas_count())
#else
    gpu_count = 0
#endif
  end function gpu_count

  ! Liga e desliga em tempo de execucao. Se o binario nao tem GPU, pedir ligar
  ! e' erro do chamador, e nao falha silenciosa.
  subroutine gpu_enable(ok)
    logical, intent(out) :: ok
    if (.not. gpu_available()) then
      ok = .false.; gpu_on = .false.; return
    end if
    gpu_on = .true.; ok = .true.
  end subroutine gpu_enable

  subroutine gpu_disable()
    gpu_on = .false.
  end subroutine gpu_disable

  logical function gpu_is_on()
    gpu_is_on = gpu_on
  end function gpu_is_on

  ! Despacho do forward. Devolve .true. se a GPU tratou a chamada.
  logical function gpu_fwd(x, w, y, BT, IF, OF)
    real(wp), intent(in)  :: x(*), w(*)
    real(wp), intent(out) :: y(*)
    integer, intent(in)   :: BT, IF, OF
    gpu_fwd = .false.
#ifdef ARCH_GPU
    if (.not. gpu_on) return
    gpu_fwd = gpublas_sgemm_fwd(x, w, int(BT, c_int64_t), y, &
                                int(IF, c_int64_t), int(OF, c_int64_t)) == 0
#endif
  end function gpu_fwd

  ! Despacho do dx do backward.
  logical function gpu_bwd_dx(dy, w, dx, BT, IF, OF)
    real(wp), intent(in)  :: dy(*), w(*)
    real(wp), intent(out) :: dx(*)
    integer, intent(in)   :: BT, IF, OF
    gpu_bwd_dx = .false.
#ifdef ARCH_GPU
    if (.not. gpu_on) return
    gpu_bwd_dx = gpublas_sgemm_bwd_dx(dy, w, int(BT, c_int64_t), dx, &
                                      int(IF, c_int64_t), int(OF, c_int64_t)) == 0
#endif
  end function gpu_bwd_dx

  ! Despacho do dw do backward.
  logical function gpu_bwd_dw(dy, x, dw, BT, IF, OF)
    real(wp), intent(in)  :: dy(*), x(*)
    real(wp), intent(out) :: dw(*)
    integer, intent(in)   :: BT, IF, OF
    gpu_bwd_dw = .false.
#ifdef ARCH_GPU
    if (.not. gpu_on) return
    gpu_bwd_dw = gpublas_sgemm_bwd_dw(dy, x, int(BT, c_int64_t), dw, &
                                      int(IF, c_int64_t), int(OF, c_int64_t)) == 0
#endif
  end function gpu_bwd_dw

end module fortran_blas_gpu_mod
