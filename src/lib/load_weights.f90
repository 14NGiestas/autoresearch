! lib/load_weights.f90 — shared checkpoint weight loader.
!
! Loads .npy flats exported by scripts/export_weights.py via
! stdlib_io_npy::load_npy. Per-layer weights are stacked: layer ll occupies
! [(ll-1)*per+1 : ll*per], matching fortran_gpt_mod's expectation.
!
! Filenames: transformer_wte_weight.npy, lm_head_weight.npy,
!   transformer_h_{L}_attn_{c_q,c_k,c_v,c_proj}_weight.npy,
!   transformer_h_{L}_mlp_{c_fc,c_proj}_weight.npy   (L = 0-based)

module load_weights_mod
  use fortran_kinds_mod, only: wp
  use stdlib_io_npy, only: load_npy, save_npy
  implicit none
contains

  subroutine load1(path, a)
    character(*), intent(in) :: path
    real(wp), allocatable, intent(out) :: a(:)
    integer :: ios
    character(len=:), allocatable :: msg
    call load_npy(path, a, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', "load failed: ", trim(path), " " // trim(msg)
      call exit(1)
    end if
  end subroutine load1

  subroutine load_into(path, a, at)
    character(*), intent(in) :: path
    real(wp), intent(inout) :: a(:)
    integer, intent(in) :: at
    real(wp), allocatable :: tmp(:)
    call load1(path, tmp)
    a(at:at+size(tmp)-1) = tmp
    deallocate(tmp)
  end subroutine load_into

  subroutine save1(path, a)
    character(*), intent(in) :: path
    real(wp), intent(in) :: a(:)
    integer :: ios
    character(len=:), allocatable :: msg
    call save_npy(path, a, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', "save failed: ", trim(path), " " // trim(msg)
      call exit(1)
    end if
  end subroutine save1

  ! Mirror of load_gpt_weights: split stacked buffers back to per-layer
  ! .npy files (flat float32, exporter convention) for resume.
  subroutine save_gpt_weights(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(wp), intent(in) :: wte(:), lm_head(:)
    real(wp), intent(in) :: c_q(:), c_k(:), c_v(:)
    real(wp), intent(in) :: c_pr(:), c_fc(:), c_pr2(:)
    integer :: ll, qsz, ksz, psz, fcsz, p2sz
    character(len=16) :: lstr
    qsz = n_head*head_dim*d_model
    ksz = n_kv_head*head_dim*d_model
    psz = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model
    call save1(trim(wdir) // "/transformer_wte_weight.npy", wte)
    call save1(trim(wdir) // "/lm_head_weight.npy", lm_head)
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_q_weight.npy", c_q(ll*qsz+1:(ll+1)*qsz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_k_weight.npy", c_k(ll*ksz+1:(ll+1)*ksz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_v_weight.npy", c_v(ll*ksz+1:(ll+1)*ksz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_proj_weight.npy", c_pr(ll*psz+1:(ll+1)*psz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_fc_weight.npy", c_fc(ll*fcsz+1:(ll+1)*fcsz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_proj_weight.npy", c_pr2(ll*p2sz+1:(ll+1)*p2sz))
    end do
  end subroutine save_gpt_weights

  ! Post-save integrity check: every expected file must exist with a real
  ! payload. stdlib save_npy can return ios=0 yet leave 0-byte files when
  ! the disk fills (flush/close errors are not reported) — this silently
  ! destroyed phase-3 step_300/step_400 (2026-09-09, disk full from nix
  ! store). Call after every save; abort LOUDLY on mismatch. Never train
  ! on with garbage recovery points.
  subroutine verify_ckpt_dir(wdir, n_layer, nbad, badpath)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer
    integer, intent(out) :: nbad
    character(len=:), allocatable, intent(out) :: badpath
    character(len=16) :: lstr
    character(len=3) :: mname(8) = ["wte", "lm ", "q  ", "k  ", "v  ", &
                                    "p  ", "fc ", "p2 "]
    integer :: ll, ii
    nbad = 0
    badpath = ""
    call check1(trim(wdir) // "/transformer_wte_weight.npy")
    call check1(trim(wdir) // "/lm_head_weight.npy")
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_q_weight.npy")
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_k_weight.npy")
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_v_weight.npy")
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_proj_weight.npy")
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_fc_weight.npy")
      call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_proj_weight.npy")
    end do
    do ii = 1, 8
      call check1(trim(wdir) // "/adam_m_" // trim(mname(ii)) // ".npy")
      call check1(trim(wdir) // "/adam_v_" // trim(mname(ii)) // ".npy")
    end do
    call check_exists(trim(wdir) // "/template.txt")
  contains
    subroutine check1(path)
      character(*), intent(in) :: path
      logical :: ex
      integer :: sz
      if (nbad /= 0) return
      inquire (file=path, exist=ex, size=sz)
      if (.not. ex .or. sz <= 128) then
        nbad = 1
        badpath = path
      end if
    end subroutine check1
    subroutine check_exists(path)
      character(*), intent(in) :: path
      logical :: ex
      if (nbad /= 0) return
      inquire (file=path, exist=ex)
      if (.not. ex) then
        nbad = 1
        badpath = path
      end if
    end subroutine check_exists
  end subroutine verify_ckpt_dir

  subroutine load_gpt_weights(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(wp), allocatable, intent(out) :: wte(:), lm_head(:)
    real(wp), allocatable, intent(out) :: c_q(:), c_k(:), c_v(:)
    real(wp), allocatable, intent(out) :: c_pr(:), c_fc(:), c_pr2(:)
    integer :: ll
    character(len=16) :: lstr

    call load1(trim(wdir) // "/transformer_wte_weight.npy", wte)
    call load1(trim(wdir) // "/lm_head_weight.npy", lm_head)
    allocate(c_q(n_layer*n_head*head_dim*d_model))
    allocate(c_k(n_layer*n_kv_head*head_dim*d_model))
    allocate(c_v(n_layer*n_kv_head*head_dim*d_model))
    allocate(c_pr(n_layer*d_model*n_head*head_dim))
    allocate(c_fc(n_layer*4*d_model*d_model))
    allocate(c_pr2(n_layer*d_model*4*d_model))
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_q_weight.npy", c_q, ll*n_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_k_weight.npy", c_k, ll*n_kv_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_v_weight.npy", c_v, ll*n_kv_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_proj_weight.npy", c_pr, ll*d_model*n_head*head_dim + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_fc_weight.npy", c_fc, ll*4*d_model*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_proj_weight.npy", c_pr2, ll*d_model*4*d_model + 1)
    end do
  end subroutine load_gpt_weights

end module load_weights_mod
