! lib/fortran_sys.f90 — fork-free OS primitives for threaded programs.
!
! execute_command_line() forks. Forking a process with live OpenMP /
! OpenBLAS thread pools produces crashed children (observed: stack
! smashing in every mkdir child while the parent continued normally).
! These wrappers call libc directly (no fork, no threads involved):
!   mkdir_p  - mkdir -p equivalent (creates parents, EEXIST ok)
!   dir_exists - inquire-based check
! Rotation deletes (rm -rf) stay fork-based but are best-effort only;
! a dead rm child is harmless (dirs pile up, nothing corrupts).

module fortran_sys_mod
  use iso_c_binding
  implicit none

  interface
    function c_mkdir(path, mode) bind(c, name='mkdir')
      import :: c_char, c_int, c_int32_t
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int32_t), value :: mode
      integer(c_int) :: c_mkdir
    end function c_mkdir
    function c_errno_ptr() bind(c, name='__errno_location')
      import :: c_ptr
      type(c_ptr) :: c_errno_ptr
    end function c_errno_ptr
  end interface
contains

  integer function errno_val()
    use iso_c_binding
    integer(c_int), pointer :: p
    call c_f_pointer(c_errno_ptr(), p)
    errno_val = p
  end function errno_val

  ! mkdir -p path (mode 0755); returns 0 on ok-or-exists, else errno.
  integer function mkdir_p(path)
    character(*), intent(in) :: path
    integer :: i, rc
    character(len=len_trim(path)+1, kind=c_char) :: cp
    mkdir_p = 0
    do i = 1, len_trim(path)
      if (path(i:i) == '/' .and. i > 1) then
        cp = trim(path(1:i-1)) // c_null_char
        rc = c_mkdir(cp, int(o'755', c_int32_t))
        if (rc /= 0 .and. errno_val() /= 17) then  ! 17 = EEXIST
          mkdir_p = errno_val()
          return
        end if
      end if
    end do
    cp = trim(path) // c_null_char
    rc = c_mkdir(cp, int(o'755', c_int32_t))
    if (rc /= 0 .and. errno_val() /= 17) mkdir_p = errno_val()
  end function mkdir_p

  logical function dir_exists(path)
    character(*), intent(in) :: path
    logical :: ex
    inquire (file=trim(path) // "/.", exist=ex)
    dir_exists = ex
  end function dir_exists

end module fortran_sys_mod
