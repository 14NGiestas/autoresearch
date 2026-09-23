! lib/fortran_sys.f90 — OS primitives with zero bind(C), zero iso_c_binding.
!
! mkdir_p delegates to stdlib's make_directory_all (direct libc mkdir,
! no fork). Fork-based creation (execute_command_line) is banned here:
! forking with live OpenMP/OpenBLAS thread pools produced crashed
! children. Drivers additionally PRE-CREATE checkpoint dirs at startup
! (pools cold), so mid-run mkdir_p only ever hits the dir_exists guard.
!   mkdir_p    - mkdir -p equivalent; 0 on ok-or-exists, else 1
!   dir_exists - inquire-based check

module fortran_sys_mod
  use stdlib_error, only: state_type
  use stdlib_system, only: make_directory_all
  implicit none
contains

  ! mkdir -p path; 0 on ok-or-exists, else 1.
  integer function mkdir_p(path)
    character(*), intent(in) :: path
    type(state_type) :: err
    if (dir_exists(path)) then
      mkdir_p = 0
      return
    end if
    call make_directory_all(path, err)
    if (err%error()) then
      mkdir_p = 1
    else
      mkdir_p = 0
    end if
  end function mkdir_p

  logical function dir_exists(path)
    character(*), intent(in) :: path
    logical :: ex
    inquire (file=trim(path) // "/.", exist=ex)
    dir_exists = ex
  end function dir_exists

end module fortran_sys_mod
