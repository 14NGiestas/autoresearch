! lib/fortran_kinds.f90 — single source of truth for kind parameters.
!
! Every module in this library gets its working precision from here:
!   use fortran_kinds_mod, only: wp
! To retune precision (e.g. fp32 -> fp64), change ONE line below.
! NOTE: the sgemm_ interface in fortran_blas (OpenBLAS here is FP32)
! and any other C-facing code keep explicit c_float/c_int kinds from
! iso_c_binding regardless of wp. Index integers stay c_int.

module fortran_kinds_mod
  use iso_fortran_env, only: real32, real64
  implicit none
  private
  public :: wp
  integer, parameter :: wp = real32   ! <-- flip to real64 for fp64
end module fortran_kinds_mod
