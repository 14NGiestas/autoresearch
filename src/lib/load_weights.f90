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
  use iso_c_binding
  use stdlib_io_npy, only: load_npy
  implicit none
contains

  subroutine load1(path, a)
    character(*), intent(in) :: path
    real(c_float), allocatable, intent(out) :: a(:)
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
    real(c_float), intent(inout) :: a(*)
    integer, intent(in) :: at
    real(c_float), allocatable :: tmp(:)
    call load1(path, tmp)
    a(at:at+size(tmp)-1) = tmp
    deallocate(tmp)
  end subroutine load_into

  subroutine load_gpt_weights(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(c_float), allocatable, intent(out) :: wte(:), lm_head(:)
    real(c_float), allocatable, intent(out) :: c_q(:), c_k(:), c_v(:)
    real(c_float), allocatable, intent(out) :: c_pr(:), c_fc(:), c_pr2(:)
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
