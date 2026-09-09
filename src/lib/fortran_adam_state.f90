! lib/fortran_adam_state.f90 — persist AdamW moments across phases.
!
! Cognivolve (arXiv:2505.11643): optimizer resets kill curriculum gains.
! Checkpoints hold weights only, so every phase so far re-init moments
! (init_state zeros). This module saves/loads the 16 state_t arrays as
! adam_m_<name>.npy / adam_v_<name>.npy next to the weights. Old
! checkpoints without them load as zeros (backward compatible).
! stdlib save/load_npy only — no C bindings, no forking.

module fortran_adam_state_mod
  use fortran_kinds_mod, only: wp
  use fortran_train_mod, only: state_t
  use stdlib_io_npy, only: load_npy, save_npy
  implicit none
contains

  subroutine mast_save1(path, a)
    character(*), intent(in) :: path
    real(wp), intent(in) :: a(:)
    integer :: ios
    character(len=:), allocatable :: msg
    call save_npy(path, a, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', "adam save failed: ", trim(path), " " // trim(msg)
      call exit(1)
    end if
  end subroutine mast_save1

  subroutine mast_load1(path, a, ok)
    character(*), intent(in) :: path
    real(wp), intent(inout) :: a(:)
    logical, intent(inout) :: ok
    real(wp), allocatable :: tmp(:)
    integer :: ios
    character(len=:), allocatable :: msg
    call load_npy(path, tmp, iostat=ios, iomsg=msg)
    if (ios /= 0 .or. size(tmp) /= size(a)) then
      ok = .false.
      return
    end if
    a = tmp
    deallocate(tmp)
  end subroutine mast_load1

  ! Save all 16 moment arrays into wdir (must exist).
  subroutine save_adam_state(wdir, S)
    character(*), intent(in) :: wdir
    type(state_t), intent(in) :: S
    call mast_save1(trim(wdir) // "/adam_m_wte.npy", S%wte)
    call mast_save1(trim(wdir) // "/adam_m_lm.npy", S%lm)
    call mast_save1(trim(wdir) // "/adam_m_q.npy", S%q)
    call mast_save1(trim(wdir) // "/adam_m_k.npy", S%k)
    call mast_save1(trim(wdir) // "/adam_m_v.npy", S%v)
    call mast_save1(trim(wdir) // "/adam_m_p.npy", S%p)
    call mast_save1(trim(wdir) // "/adam_m_fc.npy", S%fc)
    call mast_save1(trim(wdir) // "/adam_m_p2.npy", S%p2)
    call mast_save1(trim(wdir) // "/adam_v_wte.npy", S%vwte)
    call mast_save1(trim(wdir) // "/adam_v_lm.npy", S%vlm)
    call mast_save1(trim(wdir) // "/adam_v_q.npy", S%vq)
    call mast_save1(trim(wdir) // "/adam_v_k.npy", S%vk)
    call mast_save1(trim(wdir) // "/adam_v_v.npy", S%vv)
    call mast_save1(trim(wdir) // "/adam_v_p.npy", S%vp)
    call mast_save1(trim(wdir) // "/adam_v_fc.npy", S%vfc)
    call mast_save1(trim(wdir) // "/adam_v_p2.npy", S%vp2)
  end subroutine save_adam_state

  ! Load moments over preallocated (zeroed) S; found=.false. if any file
  ! is missing or mis-sized (caller keeps zeros = old behavior).
  subroutine load_adam_state(wdir, S, found)
    character(*), intent(in) :: wdir
    type(state_t), intent(inout) :: S
    logical, intent(out) :: found
    logical :: ok
    ok = .true.
    call mast_load1(trim(wdir) // "/adam_m_wte.npy", S%wte, ok)
    call mast_load1(trim(wdir) // "/adam_m_lm.npy", S%lm, ok)
    call mast_load1(trim(wdir) // "/adam_m_q.npy", S%q, ok)
    call mast_load1(trim(wdir) // "/adam_m_k.npy", S%k, ok)
    call mast_load1(trim(wdir) // "/adam_m_v.npy", S%v, ok)
    call mast_load1(trim(wdir) // "/adam_m_p.npy", S%p, ok)
    call mast_load1(trim(wdir) // "/adam_m_fc.npy", S%fc, ok)
    call mast_load1(trim(wdir) // "/adam_m_p2.npy", S%p2, ok)
    call mast_load1(trim(wdir) // "/adam_v_wte.npy", S%vwte, ok)
    call mast_load1(trim(wdir) // "/adam_v_lm.npy", S%vlm, ok)
    call mast_load1(trim(wdir) // "/adam_v_q.npy", S%vq, ok)
    call mast_load1(trim(wdir) // "/adam_v_k.npy", S%vk, ok)
    call mast_load1(trim(wdir) // "/adam_v_v.npy", S%vv, ok)
    call mast_load1(trim(wdir) // "/adam_v_p.npy", S%vp, ok)
    call mast_load1(trim(wdir) // "/adam_v_fc.npy", S%vfc, ok)
    call mast_load1(trim(wdir) // "/adam_v_p2.npy", S%vp2, ok)
    found = ok
  end subroutine load_adam_state

end module fortran_adam_state_mod
