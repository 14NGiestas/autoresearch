! test_fortran_kernels.f90 — pure Fortran test driver.
!
! Exercises every kernel exposed by the src/ library against an inline
! Fortran reference implementation. No C, no Python, no numpy.
!
! Verifies:
!   - rmsnorm    matches reference
!   - rmsnorm0   matches reference
!   - linear3d   matches reference
!   - linear3dT  matches reference
!   - wte_lookup matches reference
!   - rope_4d    matches reference
!   - relu2      matches reference
!   - causal_attn matches reference
!   - gpt_forward runs to completion, produces finite output, correct shape
!
! Build:
!   gfortran -fopenmp -O3 -ffast-math \
!       test_fortran_kernels.f90 \
!       fortran_rmsnorm.o fortran_linear.o fortran_rope.o fortran_attn.o fortran_gpt.o \
!       -o test_kernels
!   ./test_kernels

program test_kernels
  use iso_c_binding
  use sample_mod, only: sample_token, sample_next, apply_penalties, &
      block_ngram, sort_desc
  use fortran_train_mod
  use fortran_data_mod, only: load_batch, count_rows
  use fortran_sys_mod, only: mkdir_p, dir_exists
  use fortran_linear_mod, only: linear3d, linear3dT, wte_lookup
  use fortran_rmsnorm_mod, only: rmsnorm, rmsnorm0
  use fortran_rope_mod, only: rope_4d
  use fortran_attn_mod, only: causal_attn, relu2, relu2_bwd, attn_bwd, &
      attn_chunk, attn_step, causal_attn_doc
  use fortran_backward_mod, only: linear3d_bwd, rmsnorm0_bwd, rope_4d_bwd, &
      xent_fwd, xent_bwd, wte_bwd
  use fortran_adamw_mod, only: adamw_step
  use fortran_blas_mod, only: linear3d_sgemm
  use fortran_gpt_mod, only: gpt_forward
  use fortran_kv_mod, only: gpt_step, gpt_step_multi
  use fortran_spec_mod, only: accept_prefix, lookup_draft
  use fortran_recurrent_mod, only: recurrent_forward
  implicit none

  integer, parameter :: sp = c_float
  integer :: seed = 42
  integer :: fail_count = 0


  call test_rmsnorm()
  call test_rmsnorm0()
  call test_linear3d()
  call test_linear3dT()
  call test_wte_lookup()
  call test_rope_4d()
  call test_relu2()
  call test_causal_attn()
  call test_causal_attn_gqa()
  call test_causal_attn_doc()
  call test_gpt_forward_shape()
  call test_recurrent_equiv()
  call test_recurrent_loops()
  call test_sample()
  call test_kv_equiv()
  call test_attn_chunk()
  call test_kv_chunk_equiv()
  call test_accept_prefix()
  call test_lookup_draft()
  call test_linear_bwd()
  call test_rmsnorm_bwd()
  call test_wte_bwd()
  call test_rope_bwd()
  call test_xent()
  call test_attn_bwd()
  call test_relu2_bwd()
  call test_adamw()
  call test_full_step()
  call test_mkdir_p()
  call test_decode()
  call test_linear_blas()
  call test_data_batch()
  call test_save_load()

  print '(A,I0,A)', "===", fail_count, " failures ==="
  if (fail_count > 0) call exit(1)

contains

  ! Pseudo-random in [-0.5, 0.5]
  function frand() result(r)
    real(sp) :: r
    seed = mod(seed * 1103515245 + 12345, 2147483647)
    r = real(mod(seed / 65536, 32768), sp) / 32768.0_sp - 0.5_sp
  end function

  subroutine check(cond, label)
    logical, intent(in) :: cond
    character(*), intent(in) :: label
    if (.not. cond) then
      print '(A,A)', "  FAIL: ", label
      fail_count = fail_count + 1
    else
      print '(A)', "  ok"
    end if
  end subroutine

  subroutine fill(x, n, scale)
    real(sp), intent(out) :: x(*)
    integer, intent(in) :: n
    real(sp), intent(in), optional :: scale
    real(sp) :: s
    integer :: i
    s = 1.0_sp
    if (present(scale)) s = scale
    do i = 1, n
      x(i) = frand() * s
    end do
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_rmsnorm()
    integer, parameter :: NN = 8, CC = 16
    real(sp) :: x(NN*CC), w(CC), y(NN*CC), yref(NN*CC)
    real(sp) :: ss, inv, e, max_err
    integer :: i, j

    print '(A)', "=== test_rmsnorm ==="
    call fill(x, NN*CC)
    do i = 1, CC; w(i) = frand() * 0.1_sp + 1.0_sp; end do

    call rmsnorm(x, w, y, NN, CC, 1.0e-6_sp)

    max_err = 0.0_sp
    do i = 1, NN
      ss = 0.0_sp
      do j = 1, CC
        ss = ss + x((i-1)*CC+j) * x((i-1)*CC+j)
      end do
      inv = 1.0_sp / sqrt(ss / real(CC, sp) + 1.0e-6_sp)
      do j = 1, CC
        yref((i-1)*CC+j) = x((i-1)*CC+j) * inv * w(j)
        e = abs(y((i-1)*CC+j) - yref((i-1)*CC+j))
        if (e > max_err) max_err = e
      end do
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-4_sp, "rmsnorm")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_rmsnorm0()
    integer, parameter :: NN = 4, CC = 32
    real(sp) :: x(NN*CC), y(NN*CC), yref(NN*CC)
    real(sp) :: ss, inv, e, max_err
    integer :: i, j

    print '(A)', "=== test_rmsnorm0 ==="
    call fill(x, NN*CC)

    call rmsnorm0(x, y, NN, CC, 1.0e-5_sp)

    max_err = 0.0_sp
    do i = 1, NN
      ss = 0.0_sp
      do j = 1, CC
        ss = ss + x((i-1)*CC+j) * x((i-1)*CC+j)
      end do
      inv = 1.0_sp / sqrt(ss / real(CC, sp) + 1.0e-5_sp)
      do j = 1, CC
        yref((i-1)*CC+j) = x((i-1)*CC+j) * inv
        e = abs(y((i-1)*CC+j) - yref((i-1)*CC+j))
        if (e > max_err) max_err = e
      end do
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-4_sp, "rmsnorm0")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_linear3d()
    integer, parameter :: BR = 2, TC = 5, IF = 8, OF = 6
    real(sp) :: x(BR*TC*IF), w(OF*IF), y(BR*TC*OF)
    real(sp) :: acc, e, max_err
    integer :: bt, oo, i

    print '(A)', "=== test_linear3d (y = x @ w.T) ==="
    call fill(x, BR*TC*IF)
    call fill(w, OF*IF)

    call linear3d(x, w, y, BR, TC, IF, OF)

    max_err = 0.0_sp
    do bt = 1, BR*TC
      do oo = 1, OF
        acc = 0.0_sp
        do i = 1, IF
          acc = acc + x((bt-1)*IF+i) * w((oo-1)*IF+i)
        end do
        e = abs(y((bt-1)*OF+oo) - acc)
        if (e > max_err) max_err = e
      end do
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-3_sp, "linear3d")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_linear3dT()
    integer, parameter :: BR = 2, TC = 5, IF = 8, OF = 6
    real(sp) :: x(BR*TC*IF), w(IF*OF), y(BR*TC*OF)
    real(sp) :: acc, e, max_err
    integer :: bt, oo, i

    print '(A)', "=== test_linear3dT (y = x @ w) ==="
    call fill(x, BR*TC*IF)
    call fill(w, IF*OF)

    call linear3dT(x, w, y, BR, TC, IF, OF)

    max_err = 0.0_sp
    do bt = 1, BR*TC
      do oo = 1, OF
        acc = 0.0_sp
        do i = 1, IF
          acc = acc + x((bt-1)*IF+i) * w((i-1)*OF+oo)
        end do
        e = abs(y((bt-1)*OF+oo) - acc)
        if (e > max_err) max_err = e
      end do
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-3_sp, "linear3dT")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_wte_lookup()
    integer, parameter :: BR = 2, TC = 3, VV = 16, DD = 4
    integer :: idx(BR*TC)
    real(sp) :: wte(VV*DD), out(BR*TC*DD), yref(BR*TC*DD)
    real(sp) :: e, max_err
    integer :: i, j, k, id

    print '(A)', "=== test_wte_lookup ==="
    do i = 1, BR*TC
      idx(i) = int(frand() + 0.5_sp)   ! 0-based: shared[-0.5,0.5]+0.5 -> {0,1}
      if (idx(i) < 0) idx(i) = 0
      if (idx(i) > VV - 1) idx(i) = VV - 1
    end do
    call fill(wte, VV*DD)

    call wte_lookup(idx, wte, out(1:BR*TC*DD), BR, TC, VV, DD)

    do i = 1, BR
      do j = 1, TC
        id = idx((i-1)*TC + j)
        do k = 1, DD
          yref(((i-1)*TC + (j-1))*DD + k) = wte(id*DD + k)
        end do
      end do
    end do

    max_err = 0.0_sp
    do i = 1, BR*TC*DD
      e = abs(out(i) - yref(i))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err == 0.0_sp, "wte_lookup")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_rope_4d()
    integer, parameter :: BR = 1, TC = 4, HH = 2, DD = 8
    integer, parameter :: d2 = DD / 2
    real(sp) :: x(BR*TC*HH*DD), cos_buf(TC*d2), sin_buf(TC*d2)
    real(sp) :: y(BR*TC*HH*DD), yref(BR*TC*HH*DD)
    real(sp) :: e, max_err
    integer :: a, b, c, d, idx1, idx2, cidx

    print '(A)', "=== test_rope_4d ==="
    call fill(x, BR*TC*HH*DD)
    call fill(cos_buf, TC*d2)
    call fill(sin_buf, TC*d2)

    call rope_4d(x, cos_buf, sin_buf, y, BR, TC, HH, DD)

    do a = 1, BR
      do b = 1, TC
        do c = 1, HH
          do d = 1, d2
            idx1 = ((a-1)*TC + (b-1))*HH*DD + (c-1)*DD + d
            idx2 = idx1 + d2
            cidx = (b-1)*d2 + d
            yref(idx1) = x(idx1) * cos_buf(cidx) + x(idx2) * sin_buf(cidx)
            yref(idx2) = x(idx1) * (-sin_buf(cidx)) + x(idx2) * cos_buf(cidx)
          end do
        end do
      end do
    end do

    max_err = 0.0_sp
    do a = 1, BR*TC*HH*DD
      e = abs(y(a) - yref(a))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-5_sp, "rope_4d")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_relu2()
    real(sp) :: x(7), yref(7)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_relu2 ==="
    ! ReLU^2: y = max(0, x)^2
    x(1) = -2.0_sp; yref(1) = 0.0_sp       ! max(0, -2)^2 = 0
    x(2) = -1.0_sp; yref(2) = 0.0_sp
    x(3) = -0.5_sp; yref(3) = 0.0_sp
    x(4) =  0.0_sp; yref(4) = 0.0_sp
    x(5) =  0.5_sp; yref(5) = 0.25_sp
    x(6) =  1.0_sp; yref(6) = 1.0_sp
    x(7) =  2.0_sp; yref(7) = 4.0_sp

    call relu2(x(1:7), 7)

    max_err = 0.0_sp
    do i = 1, 7
      e = abs(x(i) - yref(i))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-6_sp, "relu2")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_causal_attn()
    integer, parameter :: BR = 1, TC = 3, HH = 1, K_H = 1, DD = 4
    real(sp) :: q(BR*TC*HH*DD), k(BR*TC*K_H*DD), v(BR*TC*K_H*DD)
    real(sp) :: y(BR*TC*HH*DD), yref(BR*TC*HH*DD)
    real(sp) :: sc(TC), m, sm, scale, acc, e, max_err
    integer :: a, b, c, s, d, q1, k1, v1

    print '(A)', "=== test_causal_attn ==="
    call fill(q, BR*TC*HH*DD)
    call fill(k, BR*TC*K_H*DD)
    call fill(v, BR*TC*K_H*DD)

    call causal_attn(q, k, v, y, BR, TC, HH, K_H, DD)

    scale = 1.0_sp / sqrt(real(DD, sp))

    do a = 1, BR
      do b = 1, HH
        do c = 1, TC
          m = -1.0e30_sp
          do s = 1, c
            acc = 0.0_sp
            do d = 1, DD
              q1 = ((a-1)*TC + (c-1))*HH*DD + (b-1)*DD + d
              k1 = ((a-1)*TC + (s-1))*K_H*DD + (b-1)*DD + d
              acc = acc + q(q1) * k(k1)
            end do
            sc(s) = acc * scale
            if (sc(s) > m) m = sc(s)
          end do
          sm = 0.0_sp
          do s = 1, c
            sc(s) = exp(sc(s) - m)
            sm = sm + sc(s)
          end do
          do s = 1, c
            sc(s) = sc(s) / sm
          end do
          do d = 1, DD
            acc = 0.0_sp
            do s = 1, c
              v1 = ((a-1)*TC + (s-1))*K_H*DD + (b-1)*DD + d
              acc = acc + sc(s) * v(v1)
            end do
            yref(((a-1)*TC + (c-1))*HH*DD + (b-1)*DD + d) = acc
          end do
        end do
      end do
    end do

    max_err = 0.0_sp
    do a = 1, BR*TC*HH*DD
      e = abs(y(a) - yref(a))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-4_sp, "causal_attn")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_causal_attn_gqa()
    integer, parameter :: BR = 1, TC = 4, HH = 4, K_H = 2, DD = 8
    integer, parameter :: REP = HH / K_H
    real(sp) :: q(BR*TC*HH*DD), k(BR*TC*K_H*DD), v(BR*TC*K_H*DD)
    real(sp) :: y(BR*TC*HH*DD), yref(BR*TC*HH*DD)
    real(sp) :: sc(TC), m, sm, scale, acc, e, max_err
    integer :: a, b, c, s, d, kb, q1, k1, v1

    print '(A)', "=== test_causal_attn_gqa (H=4, K_H=2) ==="
    call fill(q, BR*TC*HH*DD)
    call fill(k, BR*TC*K_H*DD)
    call fill(v, BR*TC*K_H*DD)

    call causal_attn(q, k, v, y, BR, TC, HH, K_H, DD)

    scale = 1.0_sp / sqrt(real(DD, sp))

    do a = 1, BR
      do b = 1, HH
        kb = (b - 1) / REP + 1
        do c = 1, TC
          m = -1.0e30_sp
          do s = 1, c
            acc = 0.0_sp
            do d = 1, DD
              q1 = ((a-1)*TC + (c-1))*HH*DD + (b-1)*DD + d
              k1 = ((a-1)*TC + (s-1))*K_H*DD + (kb-1)*DD + d
              acc = acc + q(q1) * k(k1)
            end do
            sc(s) = acc * scale
            if (sc(s) > m) m = sc(s)
          end do
          sm = 0.0_sp
          do s = 1, c
            sc(s) = exp(sc(s) - m)
            sm = sm + sc(s)
          end do
          do s = 1, c
            sc(s) = sc(s) / sm
          end do
          do d = 1, DD
            acc = 0.0_sp
            do s = 1, c
              v1 = ((a-1)*TC + (s-1))*K_H*DD + (kb-1)*DD + d
              acc = acc + sc(s) * v(v1)
            end do
            yref(((a-1)*TC + (c-1))*HH*DD + (b-1)*DD + d) = acc
          end do
        end do
      end do
    end do

    max_err = 0.0_sp
    do a = 1, BR*TC*HH*DD
      e = abs(y(a) - yref(a))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err < 1.0e-4_sp, "causal_attn_gqa")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_gpt_forward_shape()
    integer, parameter :: BR = 1, TC = 4, VV = 16, DD = 8
    integer, parameter :: n_head = 2, n_kv_head = 2, head_dim = 4, n_layer = 1
    integer :: idx(BR*TC)
    real(sp) :: cos_buf(TC*(head_dim/2)), sin_buf(TC*(head_dim/2))
    real(sp) :: wte(VV*DD)
    real(sp) :: c_q(n_head*head_dim*DD)
    real(sp) :: c_k(n_kv_head*head_dim*DD)
    real(sp) :: c_v(n_kv_head*head_dim*DD)
    real(sp) :: c_proj(DD*n_head*head_dim)
    real(sp) :: c_fc(4*DD*DD)
    real(sp) :: c_proj2(DD*4*DD)
    real(sp) :: lm_head(VV*DD)
    real(sp) :: outp(BR*TC*VV)
    integer :: i
    real(sp) :: mn, mx
    logical :: finite

    print '(A)', "=== test_gpt_forward (shape + finite) ==="

    ! Fill with deterministic random
    do i = 1, BR*TC
      idx(i) = 5 + mod(i, 3)
    end do
    call fill(cos_buf, TC*(head_dim/2))
    call fill(sin_buf, TC*(head_dim/2))
    call fill(wte, VV*DD, 0.1_sp)
    call fill(c_q,  n_head*head_dim*DD, 0.01_sp)
    call fill(c_k,  n_kv_head*head_dim*DD, 0.01_sp)
    call fill(c_v,  n_kv_head*head_dim*DD, 0.01_sp)
    call fill(c_proj, DD*n_head*head_dim, 0.01_sp)
    call fill(c_fc,  4*DD*DD, 0.01_sp)
    call fill(c_proj2, DD*4*DD, 0.01_sp)
    call fill(lm_head, VV*DD, 0.01_sp)

    call gpt_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        outp, BR, TC, VV, DD, n_head, n_kv_head, head_dim, n_layer, 1.0e-5_sp)

    finite = .true.
    mn = 1.0e30_sp; mx = -1.0e30_sp
    do i = 1, BR*TC*VV
      if (.not. (outp(i) == outp(i))) then
        finite = .false.
        exit
      end if
      if (outp(i) < mn) mn = outp(i)
      if (outp(i) > mx) mx = outp(i)
    end do

    print '(A,ES10.3,A,ES10.3)', "  range [", mn, ", ", mx, "]"
    call check(finite, "gpt_forward no NaN")
    call check((mx - mn) > 0.0_sp, "gpt_forward non-trivial output")
  end subroutine

  ! ------------------------------------------------------------------------
  ! recurrent_forward(n_loops=1) must equal gpt_forward(n_layer=1)
  ! bit-exactly: same kernels, same order, same single-layer weights.
  subroutine test_recurrent_equiv()
    integer, parameter :: BR = 1, TC = 4, VV = 16, DD = 8
    integer, parameter :: n_head = 2, n_kv_head = 2, head_dim = 4
    integer :: idx(BR*TC)
    real(sp) :: cos_buf(TC*(head_dim/2)), sin_buf(TC*(head_dim/2))
    real(sp) :: wte(VV*DD)
    real(sp) :: c_q(n_head*head_dim*DD)
    real(sp) :: c_k(n_kv_head*head_dim*DD)
    real(sp) :: c_v(n_kv_head*head_dim*DD)
    real(sp) :: c_proj(DD*n_head*head_dim)
    real(sp) :: c_fc(4*DD*DD)
    real(sp) :: c_proj2(DD*4*DD)
    real(sp) :: lm_head(VV*DD)
    real(sp) :: out_gpt(BR*TC*VV), out_rec(BR*TC*VV)
    integer :: i
    real(sp) :: e, max_err

    print '(A)', "=== test_recurrent_equiv (loops=1 vs gpt layer=1) ==="
    do i = 1, BR*TC
      idx(i) = 3 + mod(i, 5)
    end do
    call fill(cos_buf, TC*(head_dim/2))
    call fill(sin_buf, TC*(head_dim/2))
    call fill(wte, VV*DD, 0.1_sp)
    call fill(c_q, n_head*head_dim*DD, 0.01_sp)
    call fill(c_k, n_kv_head*head_dim*DD, 0.01_sp)
    call fill(c_v, n_kv_head*head_dim*DD, 0.01_sp)
    call fill(c_proj, DD*n_head*head_dim, 0.01_sp)
    call fill(c_fc, 4*DD*DD, 0.01_sp)
    call fill(c_proj2, DD*4*DD, 0.01_sp)
    call fill(lm_head, VV*DD, 0.01_sp)

    call gpt_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out_gpt, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)
    call recurrent_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out_rec, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)

    max_err = 0.0_sp
    do i = 1, BR*TC*VV
      e = abs(out_gpt(i) - out_rec(i))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3)', "  max err = ", max_err
    ! sgemm summation order: exact only pre-BLAS; now honest 1e-6.
    call check(max_err < 1.0e-6_sp, "recurrent(1) == gpt(1)")
  end subroutine

  ! ------------------------------------------------------------------------
  ! More loops must move the logits (iteration does work) and stay finite.
  subroutine test_recurrent_loops()
    integer, parameter :: BR = 1, TC = 4, VV = 16, DD = 8
    integer, parameter :: n_head = 2, n_kv_head = 2, head_dim = 4
    integer :: idx(BR*TC)
    real(sp) :: cos_buf(TC*(head_dim/2)), sin_buf(TC*(head_dim/2))
    real(sp) :: wte(VV*DD)
    real(sp) :: c_q(n_head*head_dim*DD)
    real(sp) :: c_k(n_kv_head*head_dim*DD)
    real(sp) :: c_v(n_kv_head*head_dim*DD)
    real(sp) :: c_proj(DD*n_head*head_dim)
    real(sp) :: c_fc(4*DD*DD)
    real(sp) :: c_proj2(DD*4*DD)
    real(sp) :: lm_head(VV*DD)
    real(sp) :: out1(BR*TC*VV), out4(BR*TC*VV)
    integer :: i
    real(sp) :: e, max_drift
    logical :: finite

    print '(A)', "=== test_recurrent_loops (1 vs 4 loops) ==="
    do i = 1, BR*TC
      idx(i) = 2 + mod(i, 6)
    end do
    call fill(cos_buf, TC*(head_dim/2))
    call fill(sin_buf, TC*(head_dim/2))
    call fill(wte, VV*DD, 0.1_sp)
    call fill(c_q, n_head*head_dim*DD, 0.05_sp)
    call fill(c_k, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_v, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_proj, DD*n_head*head_dim, 0.05_sp)
    call fill(c_fc, 4*DD*DD, 0.05_sp)
    call fill(c_proj2, DD*4*DD, 0.05_sp)
    call fill(lm_head, VV*DD, 0.05_sp)

    call recurrent_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out1, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)
    call recurrent_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out4, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 4, 1.0e-5_sp)

    finite = .true.
    max_drift = 0.0_sp
    do i = 1, BR*TC*VV
      if (.not. (out4(i) == out4(i))) finite = .false.
      e = abs(out4(i) - out1(i))
      if (e > max_drift) max_drift = e
    end do

    print '(A,E10.3)', "  drift(4 vs 1) = ", max_drift
    call check(finite, "recurrent(4) finite")
    call check(max_drift > 0.0_sp, "loops move logits")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_sample()
    real(sp) :: logits(4)
    integer(c_int64_t) :: st
    integer :: i, pos, hits(4)
    logical :: ok_range

    print '(A)', "=== test_sample ==="
    logits = [-1.0_sp, 2.0_sp, 0.5_sp, -0.5_sp]

    ! temp<=0 -> greedy argmax (position 2)
    st = 1_c_int64_t
    pos = sample_token(logits, 4, 0.0_sp, st)
    call check(pos == 2, "greedy argmax")

    ! temp=1: 400 draws stay in range and hit more than one token
    st = 42_c_int64_t
    hits = 0
    ok_range = .true.
    do i = 1, 400
      pos = sample_token(logits, 4, 1.0_sp, st)
      if (pos < 1 .or. pos > 4) ok_range = .false.
      hits(pos) = hits(pos) + 1
    end do
    call check(ok_range, "samples in range")
    call check(count(hits > 0) > 1, "samples spread")
    ! token 2 (logit 2.0) must dominate: expect > 200/400 hits
    print '(A,4I5)', "  hits:", hits
    call check(hits(2) > 200, "argmax dominates")

    ! determinism: same seed -> same first draw
    st = 7_c_int64_t
    pos = sample_token(logits, 4, 1.0_sp, st)
    block
      integer :: pos2
      st = 7_c_int64_t
      pos2 = sample_token(logits, 4, 1.0_sp, st)
      call check(pos == pos2, "seeded determinism")
    end block
  end subroutine

  ! ------------------------------------------------------------------------
  ! 4 sequential cached gpt_step calls must equal one gpt_forward(T=4)
  ! on ALL positions, bit-exactly (same kernels, same op order).
  subroutine test_kv_equiv()
    integer, parameter :: BR = 1, TC = 4, VV = 16, DD = 8
    integer, parameter :: n_head = 2, n_kv_head = 2, head_dim = 4
    integer, parameter :: dkh = n_kv_head*head_dim, MAXT = 4
    integer :: idx(BR*TC)
    real(sp) :: cos_buf(TC*(head_dim/2)), sin_buf(TC*(head_dim/2))
    real(sp) :: wte(VV*DD)
    real(sp) :: c_q(n_head*head_dim*DD)
    real(sp) :: c_k(n_kv_head*head_dim*DD)
    real(sp) :: c_v(n_kv_head*head_dim*DD)
    real(sp) :: c_proj(DD*n_head*head_dim)
    real(sp) :: c_fc(4*DD*DD)
    real(sp) :: c_proj2(DD*4*DD)
    real(sp) :: lm_head(VV*DD)
    real(sp) :: out_full(BR*TC*VV), out_steps(BR*TC*VV)
    real(sp) :: out1(VV)
    real(sp) :: ck(MAXT*dkh), cv(MAXT*dkh)
    integer :: i, t, clen
    real(sp) :: e, max_err
    integer :: d2

    print '(A)', "=== test_kv_equiv (4 steps vs full forward) ==="
    d2 = head_dim / 2
    do i = 1, BR*TC
      idx(i) = 1 + mod(i, 7)
    end do
    call fill(cos_buf, TC*d2)
    call fill(sin_buf, TC*d2)
    call fill(wte, VV*DD, 0.1_sp)
    call fill(c_q, n_head*head_dim*DD, 0.05_sp)
    call fill(c_k, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_v, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_proj, DD*n_head*head_dim, 0.05_sp)
    call fill(c_fc, 4*DD*DD, 0.05_sp)
    call fill(c_proj2, DD*4*DD, 0.05_sp)
    call fill(lm_head, VV*DD, 0.05_sp)

    call gpt_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out_full, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)

    ck = 0.0_sp; cv = 0.0_sp; clen = 0
    do t = 1, TC
      call gpt_step(idx(t:t), cos_buf((t-1)*d2+1:), sin_buf((t-1)*d2+1:), &
          wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
          ck, cv, clen, MAXT, out1, &
          BR, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)
      out_steps((t-1)*VV+1:t*VV) = out1
    end do

    max_err = 0.0_sp
    do i = 1, BR*TC*VV
      e = abs(out_full(i) - out_steps(i))
      if (e > max_err) max_err = e
    end do

    print '(A,E10.3,A,I0)', "  max err = ", max_err, "  cache_len=", clen
    call check(clen == TC, "cache holds all positions")
    call check(max_err < 1.0e-6_sp, "cached steps == full forward")
  end subroutine

  ! ------------------------------------------------------------------------
  ! causal_attn_doc: with docstart == 1 it must equal causal_attn bit-exactly;
  ! with a boundary at position 5, the rows after it must equal running plain
  ! causal attention on those rows alone -- that is what the mask means.
  subroutine test_causal_attn_doc()
    integer, parameter :: B = 1, T = 8, H = 2, KH = 2, D = 4
    integer, parameter :: TB = 4          ! second document: rows 5..8
    real(sp) :: q(B*T*H*D), k(B*T*KH*D), v(B*T*KH*D)
    real(sp) :: yref(B*T*H*D), ydoc(B*T*H*D), ysub(TB*H*D)
    real(sp) :: q2(TB*H*D), k2(TB*KH*D), v2(TB*KH*D)
    integer(c_int) :: ds(B*T)
    real(sp) :: max_err, e
    integer :: i

    print '(A)', "=== test_causal_attn_doc (document mask) ==="
    call fill(q, B*T*H*D)
    call fill(k, B*T*KH*D)
    call fill(v, B*T*KH*D)

    ds = 1
    call causal_attn(q, k, v, yref, B, T, H, KH, D)
    call causal_attn_doc(q, k, v, ydoc, B, T, H, KH, D, ds)
    max_err = 0.0_sp
    do i = 1, B*T*H*D
      e = abs(yref(i) - ydoc(i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err (docstart all 1 vs causal) = ", max_err
    call check(max_err == 0.0_sp, "no-boundary mask is bit-exact vs causal")

    do i = 1, 4
      ds(i) = 1_c_int
    end do
    do i = 5, T
      ds(i) = 5_c_int
    end do
    call causal_attn_doc(q, k, v, ydoc, B, T, H, KH, D, ds)
    q2 = q(4*H*D+1:T*H*D)
    k2 = k(4*KH*D+1:T*KH*D)
    v2 = v(4*KH*D+1:T*KH*D)
    call causal_attn(q2, k2, v2, ysub, 1, TB, H, KH, D)
    max_err = 0.0_sp
    do i = 1, TB*H*D
      e = abs(ysub(i) - ydoc(4*H*D+i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err (doc 2 rows vs its own causal) = ", max_err
    call check(max_err == 0.0_sp, "rows after a boundary ignore the earlier doc")
  end subroutine

  ! ------------------------------------------------------------------------
  ! attn_chunk: chunked cached attention, checked against both references —
  ! two chunks of CH rows must equal causal_attn over the whole sequence,
  ! and the TB=1 case must equal attn_step.
  subroutine test_attn_chunk()
    integer, parameter :: B = 1, H = 2, KH = 2, D = 4, dkh = KH*D
    integer, parameter :: T = 6, CH = 3
    real(sp) :: q(B*T*H*D), k(B*T*KH*D), v(B*T*KH*D)
    real(sp) :: yref(B*T*H*D), ychunk(B*T*H*D)
    real(sp) :: ystep(H*D), y1(CH*H*D), y2(CH*H*D)
    real(sp) :: ck(T*dkh), cv(T*dkh)
    real(sp) :: max_err, e
    integer :: i

    print '(A)', "=== test_attn_chunk (chunks vs causal_attn / attn_step) ==="
    call fill(q, B*T*H*D)
    call fill(k, B*T*KH*D)
    call fill(v, B*T*KH*D)
    call causal_attn(q, k, v, yref, B, T, H, KH, D)

    ! two chunks of 3, cache grown in place (caller appends before the call)
    ck = 0.0_sp
    cv = 0.0_sp
    ck(1:CH*dkh) = k(1:CH*dkh)
    cv(1:CH*dkh) = v(1:CH*dkh)
    call attn_chunk(q, ck, cv, y1, B, H, KH, D, 0, CH)
    ck(CH*dkh+1:T*dkh) = k(CH*dkh+1:T*dkh)
    cv(CH*dkh+1:T*dkh) = v(CH*dkh+1:T*dkh)
    call attn_chunk(q(CH*H*D+1:), ck, cv, y2, B, H, KH, D, CH, CH)
    ychunk(1:CH*H*D) = y1
    ychunk(CH*H*D+1:T*H*D) = y2

    max_err = 0.0_sp
    do i = 1, B*T*H*D
      e = abs(yref(i) - ychunk(i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err (2x3 chunks vs causal) = ", max_err
    call check(max_err < 1.0e-6_sp, "chunked attention == causal_attn")

    ! last position, single query, full cache: must equal last chunk row
    call attn_step(q((T-1)*H*D+1:), ck, cv, ystep, B, H, KH, D, T)
    max_err = 0.0_sp
    do i = 1, H*D
      e = abs(ystep(i) - ychunk((T-1)*H*D+i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err (TB=1 vs attn_step) = ", max_err
    call check(max_err < 1.0e-6_sp, "TB=1 chunk == attn_step")
  end subroutine

  ! ------------------------------------------------------------------------
  ! gpt_step_multi: chunked cached forward (prefill / spec verification)
  ! must reproduce both gpt_forward and per-token gpt_step row by row.
  subroutine test_kv_chunk_equiv()
    integer, parameter :: BR = 1, TC = 6, VV = 16, DD = 8
    integer, parameter :: n_head = 2, n_kv_head = 2, head_dim = 4
    integer, parameter :: dkh = n_kv_head*head_dim, MAXT = 6, CH = 3
    integer :: idx(BR*TC)
    real(sp) :: cos_buf(TC*(head_dim/2)), sin_buf(TC*(head_dim/2))
    real(sp) :: wte(VV*DD)
    real(sp) :: c_q(n_head*head_dim*DD)
    real(sp) :: c_k(n_kv_head*head_dim*DD)
    real(sp) :: c_v(n_kv_head*head_dim*DD)
    real(sp) :: c_proj(DD*n_head*head_dim)
    real(sp) :: c_fc(4*DD*DD)
    real(sp) :: c_proj2(DD*4*DD)
    real(sp) :: lm_head(VV*DD)
    real(sp) :: out_full(BR*TC*VV), out_steps(BR*TC*VV)
    real(sp) :: out_chunk(BR*TC*VV), outc(CH*VV), out1(VV)
    real(sp) :: ck(MAXT*dkh), cv(MAXT*dkh)
    integer :: i, t, clen, srow, tb
    real(sp) :: e, max_err_s, max_err_c
    integer :: d2

    print '(A)', "=== test_kv_chunk_equiv (chunks vs steps vs forward) ==="
    d2 = head_dim / 2
    do i = 1, BR*TC
      idx(i) = 1 + mod(i, 7)
    end do
    call fill(cos_buf, TC*d2)
    call fill(sin_buf, TC*d2)
    call fill(wte, VV*DD, 0.1_sp)
    call fill(c_q, n_head*head_dim*DD, 0.05_sp)
    call fill(c_k, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_v, n_kv_head*head_dim*DD, 0.05_sp)
    call fill(c_proj, DD*n_head*head_dim, 0.05_sp)
    call fill(c_fc, 4*DD*DD, 0.05_sp)
    call fill(c_proj2, DD*4*DD, 0.05_sp)
    call fill(lm_head, VV*DD, 0.05_sp)

    call gpt_forward(idx, cos_buf, sin_buf, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        out_full, BR, TC, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)

    ck = 0.0_sp
    cv = 0.0_sp
    clen = 0
    do t = 1, TC
      call gpt_step(idx(t:t), cos_buf((t-1)*d2+1:), sin_buf((t-1)*d2+1:), &
          wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
          ck, cv, clen, MAXT, out1, &
          BR, VV, DD, n_head, n_kv_head, head_dim, 1, 1.0e-5_sp)
      out_steps((t-1)*VV+1:t*VV) = out1
    end do
    call check(clen == TC, "per-token cache holds all positions")

    ck = 0.0_sp
    cv = 0.0_sp
    clen = 0
    srow = 1
    do while (srow <= TC)
      tb = min(CH, TC - srow + 1)
      call gpt_step_multi(idx(srow:srow+tb-1), &
          cos_buf((srow-1)*d2+1:), sin_buf((srow-1)*d2+1:), &
          wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
          ck, cv, clen, MAXT, outc, &
          BR, VV, DD, n_head, n_kv_head, head_dim, 1, tb, 1.0e-5_sp)
      out_chunk((srow-1)*VV+1:srow*VV) = outc(1:tb*VV)
      srow = srow + tb
    end do

    max_err_s = 0.0_sp
    max_err_c = 0.0_sp
    do i = 1, BR*TC*VV
      e = abs(out_full(i) - out_steps(i))
      if (e > max_err_s) max_err_s = e
      e = abs(out_full(i) - out_chunk(i))
      if (e > max_err_c) max_err_c = e
    end do
    print '(A,E10.3,A,E10.3,A,I0)', "  err steps=", max_err_s, &
        " chunks=", max_err_c, " cache_len=", clen
    call check(clen == TC, "chunked cache holds all positions")
    call check(max_err_s < 1.0e-6_sp, "per-token steps == full forward")
    call check(max_err_c < 1.0e-6_sp, "chunked steps == full forward")
  end subroutine

  ! ------------------------------------------------------------------------
  ! accept_prefix: the greedy speculative commit rule.
  ! (a) table cases; (b) protocol simulation — the committed stream must be
  ! exactly the target's greedy stream and must always advance (>=1 token).
  subroutine test_accept_prefix()
    integer, parameter :: K = 5, NT = 40
    integer :: draft(K), targ(K+1), tg(NT+K+1), buf(NT+K+1)
    integer :: na, corr, p, j, ncomm, npass, ncorr

    print '(A)', "=== test_accept_prefix (greedy spec commit rule) ==="

    ! (a) table cases
    draft(1:4) = [1, 2, 3, 4]
    targ(1:5) = [1, 2, 9, 9, 7]
    call accept_prefix(draft, targ, 4, na, corr)
    call check(na == 2 .and. corr == 9, "partial match: 2 accepted, next from target")

    draft(1:3) = [1, 2, 3]
    targ(1:4) = [1, 2, 3, 5]
    call accept_prefix(draft, targ, 3, na, corr)
    call check(na == 3 .and. corr == 5, "all match: bonus row committed")

    draft(1:2) = [1, 2]
    targ(1:3) = [7, 8, 9]
    call accept_prefix(draft, targ, 2, na, corr)
    call check(na == 0 .and. corr == 7, "first rejected: target token only")

    targ(1) = 4
    call accept_prefix(draft, targ, 0, na, corr)
    call check(na == 0 .and. corr == 4, "k=0: degenerate case is a plain step")

    ! (b) protocol simulation against a deterministic greedy target stream.
    ! A drafter that agrees except at one position every 7th window must
    ! reproduce the target stream exactly, with >=1 token per pass.
    do j = 1, NT + K + 1
      tg(j) = mod(j*7 + 3, 11)
    end do
    p = 1
    ncomm = 0
    npass = 0
    ncorr = 0
    do while (ncomm < NT .and. p <= NT)
      do j = 1, K
        draft(j) = tg(p + j - 1)
      end do
      if (mod(p, 7) == 0) draft(3) = mod(draft(3) + 1, 11)  ! planted mismatch
      do j = 1, K + 1
        targ(j) = tg(p + j - 1)
      end do
      call accept_prefix(draft, targ, K, na, corr)
      if (na + 1 < 1) call check(.false., "pass must commit >= 1 token")
      if (corr /= targ(na + 1)) call check(.false., "correction is target token")
      do j = 1, na
        buf(ncomm + j) = draft(j)
      end do
      buf(ncomm + na + 1) = corr
      if (na < K) ncorr = ncorr + 1
      ncomm = ncomm + na + 1
      p = p + na + 1
      npass = npass + 1
    end do
    call check(ncomm >= NT, "simulation commits the requested span")
    call check(all(buf(1:ncomm) == tg(1:ncomm)), &
        "committed stream == target greedy stream")
    print '(A,I0,A,I0,A,F5.2)', "  passes=", npass, " tokens=", ncomm, &
        " tokens/pass=", real(ncomm, sp) / real(npass, sp)
    call check(ncorr > 0, "planted mismatch actually exercised the reject path")
  end subroutine

  ! ------------------------------------------------------------------------
  ! lookup_draft: prompt-lookup proposals (zero-cost drafter).
  subroutine test_lookup_draft()
    integer :: ids(9), d(6), kk

    print '(A)', "=== test_lookup_draft (prompt-lookup drafter) ==="
    ! suffix ids(6:8)=[1,2,3]; the most recent earlier occurrence is ids(2:4)
    ids = [5, 1, 2, 3, 9, 1, 2, 3, 7]
    call lookup_draft(ids, 8, 3, 6, d, kk)
    call check(kk == 4 .and. all(d(1:4) == [9, 1, 2, 3]), &
        "match found: continuation copied (capped at known ids)")
    call lookup_draft(ids, 8, 3, 2, d, kk)
    call check(kk == 2 .and. all(d(1:2) == [9, 1]), "k truncates proposals")

    ! match length clamps to the available prefix; nothing earlier -> miss
    ids(1:3) = [5, 1, 2]
    call lookup_draft(ids, 3, 5, 4, d, kk)
    call check(kk == 0, "m > p-1 is a miss, not an out-of-bounds read")

    ! no earlier occurrence -> miss (plain step fallback)
    ids(1:6) = [1, 2, 3, 4, 5, 6]
    call lookup_draft(ids, 6, 2, 3, d, kk)
    call check(kk == 0, "no earlier occurrence -> no proposal")

    ! occurrence at the very start of the context
    ids(1:6) = [7, 8, 9, 7, 8, 9]
    call lookup_draft(ids, 6, 3, 5, d, kk)
    call check(kk == 3 .and. all(d(1:3) == [7, 8, 9]), &
        "match at position 1 is usable")
  end subroutine

  ! ------------------------------------------------------------------------
  ! linear3d_bwd vs central finite differences of linear3d.
  subroutine test_linear_bwd()
    integer, parameter :: BR = 2, TC = 3, IF = 4, OF = 5
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: x(BR*TC*IF), w(OF*IF), dy(BR*TC*OF)
    real(sp) :: dx(BR*TC*IF), dw(OF*IF)
    real(sp) :: xp(BR*TC*IF), xm(BR*TC*IF), wp(OF*IF), wm(OF*IF)
    real(sp) :: yp(BR*TC*OF), ym(BR*TC*OF)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_linear_bwd (finite differences) ==="
    call fill(x, BR*TC*IF)
    call fill(w, OF*IF)
    call fill(dy, BR*TC*OF)

    call linear3d_bwd(dy, x, w, dx, dw, BR, TC, IF, OF)

    ! dL/dx via central differences (L = sum(dy*y))
    max_err = 0.0_sp
    do i = 1, BR*TC*IF
      xp = x; xm = x
      xp(i) = xp(i) + H; xm(i) = xm(i) - H
      call linear3d(xp, w, yp, BR, TC, IF, OF)
      call linear3d(xm, w, ym, BR, TC, IF, OF)
      e = abs(dx(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dx = ", max_err
    call check(max_err < 2.0e-3_sp, "linear dx")

    max_err = 0.0_sp
    do i = 1, OF*IF
      wp = w; wm = w
      wp(i) = wp(i) + H; wm(i) = wm(i) - H
      call linear3d(x, wp, yp, BR, TC, IF, OF)
      call linear3d(x, wm, ym, BR, TC, IF, OF)
      e = abs(dw(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dw = ", max_err
    call check(max_err < 2.0e-3_sp, "linear dw")
  end subroutine

  ! ------------------------------------------------------------------------
  ! rmsnorm0_bwd vs central finite differences of rmsnorm0.
  subroutine test_rmsnorm_bwd()
    integer, parameter :: NN = 3, CC = 8
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: x(NN*CC), dy(NN*CC), dx(NN*CC)
    real(sp) :: xp(NN*CC), xm(NN*CC), yp(NN*CC), ym(NN*CC)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_rmsnorm_bwd (finite differences) ==="
    call fill(x, NN*CC)
    call fill(dy, NN*CC)

    call rmsnorm0_bwd(dy, x, dx, NN, CC, 1.0e-5_sp)

    max_err = 0.0_sp
    do i = 1, NN*CC
      xp = x; xm = x
      xp(i) = xp(i) + H; xm(i) = xm(i) - H
      call rmsnorm0(xp, yp, NN, CC, 1.0e-5_sp)
      call rmsnorm0(xm, ym, NN, CC, 1.0e-5_sp)
      e = abs(dx(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dx = ", max_err
    call check(max_err < 2.0e-3_sp, "rmsnorm dx")
  end subroutine

  ! ------------------------------------------------------------------------
  ! wte_bwd: exact scatter-add check, incl. colliding ids (critical sec).
  subroutine test_wte_bwd()
    integer, parameter :: BR = 2, TC = 3, VV = 5, DD = 4
    integer :: idx(BR*TC)
    real(sp) :: dout(BR*TC*DD), dwte(VV*DD), ref(VV*DD)
    real(sp) :: e, max_err
    integer :: i, j, k

    print '(A)', "=== test_wte_bwd (exact scatter) ==="
    idx = [0, 2, 2, 4, 0, 1]   ! collisions on 0 and 2
    call fill(dout, BR*TC*DD)
    dwte = 0.0_sp
    ref = 0.0_sp

    call wte_bwd(idx, dout, dwte, BR, TC, VV, DD)

    do i = 1, BR
      do j = 1, TC
        do k = 1, DD
          ref(idx((i-1)*TC+j)*DD+k) = ref(idx((i-1)*TC+j)*DD+k) + &
              dout(((i-1)*TC+(j-1))*DD+k)
        end do
      end do
    end do

    max_err = 0.0_sp
    do i = 1, VV*DD
      e = abs(dwte(i) - ref(i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err = ", max_err
    call check(max_err == 0.0_sp, "wte scatter exact")
  end subroutine

  ! ------------------------------------------------------------------------
  ! rope_4d_bwd vs central finite differences of rope_4d.
  subroutine test_rope_bwd()
    integer, parameter :: BR = 1, TC = 4, HH = 2, DD = 8
    integer, parameter :: d2 = DD / 2
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: x(BR*TC*HH*DD), dy(BR*TC*HH*DD), dx(BR*TC*HH*DD)
    real(sp) :: cos_buf(TC*d2), sin_buf(TC*d2)
    real(sp) :: xp(BR*TC*HH*DD), xm(BR*TC*HH*DD)
    real(sp) :: yp(BR*TC*HH*DD), ym(BR*TC*HH*DD)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_rope_bwd (finite differences) ==="
    call fill(x, BR*TC*HH*DD)
    call fill(dy, BR*TC*HH*DD)
    call fill(cos_buf, TC*d2)
    call fill(sin_buf, TC*d2)

    call rope_4d_bwd(dy, cos_buf, sin_buf, dx, BR, TC, HH, DD)

    max_err = 0.0_sp
    do i = 1, BR*TC*HH*DD
      xp = x; xm = x
      xp(i) = xp(i) + H; xm(i) = xm(i) - H
      call rope_4d(xp, cos_buf, sin_buf, yp, BR, TC, HH, DD)
      call rope_4d(xm, cos_buf, sin_buf, ym, BR, TC, HH, DD)
      e = abs(dx(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dx = ", max_err
    call check(max_err < 2.0e-3_sp, "rope dx")
  end subroutine

  ! ------------------------------------------------------------------------
  ! xent_bwd vs central finite differences of mean(xent_fwd).
  subroutine test_xent()
    integer, parameter :: BR = 2, TC = 3, VV = 16
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: logits(BR*TC*VV), dlog(BR*TC*VV)
    real(sp) :: lp(BR*TC*VV), lm(BR*TC*VV)
    real(sp) :: np(BR*TC), nm(BR*TC)
    integer :: targets(BR*TC)
    real(sp) :: e, max_err, sc
    integer :: i, k

    print '(A)', "=== test_xent (finite differences) ==="
    call fill(logits, BR*TC*VV)
    do i = 1, BR*TC
      targets(i) = mod(i * 5, VV)
    end do
    sc = 1.0_sp / real(BR*TC, sp)

    call xent_fwd(logits, targets, np, BR, TC, VV)
    call xent_bwd(logits, targets, dlog, BR, TC, VV, sc)

    ! fwd sanity: NLL of argmax-target is small, uniform-target is ~ln V
    call check(all(np >= 0.0_sp), "nll non-negative")

    max_err = 0.0_sp
    do i = 1, BR*TC*VV
      lp = logits; lm = logits
      lp(i) = lp(i) + H; lm(i) = lm(i) - H
      call xent_fwd(lp, targets, np, BR, TC, VV)
      call xent_fwd(lm, targets, nm, BR, TC, VV)
      e = abs(dlog(i) - sc*sum(np-nm) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dlogits = ", max_err
    call check(max_err < 5.0e-3_sp, "xent bwd")
    ! gradient sums to ~zero per row (softmax property)
    do k = 1, BR*TC
      if (abs(sum(dlog((k-1)*VV+1:k*VV))) > 1.0e-5_sp) then
        call check(.false., "xent row-sum zero")
        return
      end if
    end do
    call check(.true., "xent row-sum zero")
  end subroutine

  ! ------------------------------------------------------------------------
  ! attn_bwd (incl. GQA kv sharing) vs central FD of causal_attn.
  subroutine test_attn_bwd()
    integer, parameter :: BR = 1, TC = 3, HH = 2, K_H = 1, DD = 4
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: q(BR*TC*HH*DD), k(BR*TC*K_H*DD), v(BR*TC*K_H*DD)
    real(sp) :: dy(BR*TC*HH*DD)
    real(sp) :: dq(BR*TC*HH*DD), dk(BR*TC*K_H*DD), dv(BR*TC*K_H*DD)
    real(sp) :: qp(BR*TC*HH*DD), qm(BR*TC*HH*DD)
    real(sp) :: kp(BR*TC*K_H*DD), km(BR*TC*K_H*DD)
    real(sp) :: vp(BR*TC*K_H*DD), vm(BR*TC*K_H*DD)
    real(sp) :: yp(BR*TC*HH*DD), ym(BR*TC*HH*DD)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_attn_bwd (finite differences, GQA) ==="
    call fill(q, BR*TC*HH*DD)
    call fill(k, BR*TC*K_H*DD)
    call fill(v, BR*TC*K_H*DD)
    call fill(dy, BR*TC*HH*DD)

    dq = 0.0_sp; dk = 0.0_sp; dv = 0.0_sp
    call attn_bwd(dy, q, k, v, dq, dk, dv, BR, TC, HH, K_H, DD)

    max_err = 0.0_sp
    do i = 1, BR*TC*HH*DD
      qp = q; qm = q
      qp(i) = qp(i) + H; qm(i) = qm(i) - H
      call causal_attn(qp, k, v, yp, BR, TC, HH, K_H, DD)
      call causal_attn(qm, k, v, ym, BR, TC, HH, K_H, DD)
      e = abs(dq(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dq = ", max_err
    call check(max_err < 3.0e-3_sp, "attn dq")

    max_err = 0.0_sp
    do i = 1, BR*TC*K_H*DD
      kp = k; km = k
      kp(i) = kp(i) + H; km(i) = km(i) - H
      call causal_attn(q, kp, v, yp, BR, TC, HH, K_H, DD)
      call causal_attn(q, km, v, ym, BR, TC, HH, K_H, DD)
      e = abs(dk(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dk = ", max_err
    call check(max_err < 3.0e-3_sp, "attn dk")

    max_err = 0.0_sp
    do i = 1, BR*TC*K_H*DD
      vp = v; vm = v
      vp(i) = vp(i) + H; vm(i) = vm(i) - H
      call causal_attn(q, k, vp, yp, BR, TC, HH, K_H, DD)
      call causal_attn(q, k, vm, ym, BR, TC, HH, K_H, DD)
      e = abs(dv(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dv = ", max_err
    call check(max_err < 3.0e-3_sp, "attn dv")
  end subroutine

  ! ------------------------------------------------------------------------
  ! relu2_bwd vs central finite differences of relu2.
  subroutine test_relu2_bwd()
    real(sp), parameter :: H = 1.0e-3_sp
    real(sp) :: x(7), y(7), dy(7), dx(7), yp(7), ym(7)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_relu2_bwd (finite differences) ==="
    x = [-2.0_sp, -1.0_sp, -0.5_sp, 0.0_sp, 0.5_sp, 1.0_sp, 2.0_sp]
    dy = [0.3_sp, -0.2_sp, 0.1_sp, 0.0_sp, 0.4_sp, -0.1_sp, 0.2_sp]
    y = x
    call relu2(y, 7)

    call relu2_bwd(dy, x, dx, 7)

    ! kink at x=0 (i=4): FD undefined there, analytic is one-sided; skip.
    max_err = 0.0_sp
    do i = 1, 7
      if (i == 4) cycle
      yp = x; ym = x
      yp(i) = yp(i) + H; ym(i) = ym(i) - H
      call relu2(yp(1:7), 7)
      call relu2(ym(1:7), 7)
      e = abs(dx(i) - sum(dy*(yp-ym)) / (2.0_sp*H))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err dx = ", max_err
    call check(max_err < 2.0e-3_sp, "relu2 dx")
  end subroutine

  ! ------------------------------------------------------------------------
  subroutine test_adamw()
    real(sp) :: p(4), g(4), m(4), v(4), p0(4)
    real(sp) :: lr, b1, b2, eps, wd
    real(sp) :: e, max_err, loss0, loss1
    integer :: i, k

    print '(A)', "=== test_adamw ==="
    lr = 0.01_sp; b1 = 0.9_sp; b2 = 0.999_sp
    eps = 1.0e-8_sp; wd = 0.1_sp

    ! 1. exact first step from zero state: mhat=g, vhat=g^2
    p = [1.0_sp, -2.0_sp, 0.5_sp, 3.0_sp]
    g = [0.5_sp, 0.25_sp, -1.0_sp, 2.0_sp]
    m = 0.0_sp; v = 0.0_sp
    p0 = p
    call adamw_step(p, g, m, v, 4, lr, b1, b2, eps, wd, 1)
    max_err = 0.0_sp
    do i = 1, 4
      e = abs(p(i) - (p0(i) - lr * (g(i) / (abs(g(i)) + eps) + wd * p0(i))))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3)', "  max err first-step = ", max_err
    call check(max_err < 1.0e-6_sp, "adamw first step exact")
    ! moments match (1-b)*g, (1-b)*g^2
    call check(abs(m(1) - 0.1_sp*0.5_sp) < 1.0e-7_sp, "adamw m state")
    call check(abs(v(2) - 0.001_sp*0.0625_sp) < 1.0e-9_sp, "adamw v state")

    ! 2. decoupled WD with zero grad: geometric decay (1-lr*wd)^t
    p = 2.0_sp; m = 0.0_sp; v = 0.0_sp; g = 0.0_sp
    do k = 1, 3
      call adamw_step(p, g, m, v, 4, lr, b1, b2, eps, wd, k)
    end do
    e = abs(p(1) - 2.0_sp * (1.0_sp - lr*wd)**3)
    print '(A,E10.3)', "  wd decay err = ", e
    call check(e < 1.0e-6_sp, "adamw decoupled wd")

    ! 3. functional: minimize sum(p^2), gradient 2p, 200 steps
    p = [1.0_sp, -1.0_sp, 0.5_sp, -0.5_sp]
    m = 0.0_sp; v = 0.0_sp
    loss0 = sum(p*p)
    do k = 1, 200
      g = 2.0_sp * p
      call adamw_step(p, g, m, v, 4, 0.05_sp, b1, b2, eps, 0.0_sp, k)
    end do
    loss1 = sum(p*p)
    print '(A,E10.3,A,E10.3)', "  loss ", loss0, " -> ", loss1
    call check(loss1 < 0.01_sp * loss0, "adamw converges")
  end subroutine

  ! ------------------------------------------------------------------------
  ! Whole-model: compute_grads vs central FD over EVERY weight + a 5-step
  ! overfit check (single fixed batch NLL must fall) exercising train_step.
  subroutine test_full_step()
    ! H=3e-4: separates FD truncation (~h^2, expect ~10x smaller than
    ! H=1e-3) from ReLU-kink contamination (h-independent). If errors
    ! persist at ~1e-2, the analytic path has a real bug.
    type(dims_t) :: G
    type(params_t) :: M, GR
    type(state_t) :: S
    type(cache_t) :: C
    type(temp_t) :: tmp
    real(sp), parameter :: H = 3.0e-4_sp
    integer :: idx(2), targets(2)
    real(sp) :: cos(4), sin(4)
    real(sp) :: nll, n0, n5
    real(sp) :: e, max_err
    integer :: k

    print '(A)', "=== test_full_step (whole-model FD + overfit) ==="
    G%B = 1; G%T = 2; G%V = 8; G%D = 4
    G%nh = 1; G%nkv = 1; G%hd = 4; G%nl = 1
    G%eps = 1.0e-5_sp
    idx = [3, 5]; targets = [5, 1]
    call fill(cos, 4)
    call fill(sin, 4)

    allocate(M%wte(32), M%lm(32))
    allocate(M%q(16), M%k(16), M%v(16), M%p(16))
    allocate(M%fc(64), M%p2(64))
    call fill(M%wte, 32, 0.3_sp); call fill(M%lm, 32, 0.3_sp)
    call fill(M%q, 16, 0.3_sp); call fill(M%k, 16, 0.3_sp)
    call fill(M%v, 16, 0.3_sp); call fill(M%p, 16, 0.3_sp)
    call fill(M%fc, 64, 0.3_sp); call fill(M%p2, 64, 0.3_sp)
    allocate(GR%wte(32), GR%lm(32))
    allocate(GR%q(16), GR%k(16), GR%v(16), GR%p(16))
    allocate(GR%fc(64), GR%p2(64))
    call init_state(M, S)
    call init_temp(G, tmp)

    call forward_save(idx, targets, cos, sin, M, G, C, tmp, nll)
    call compute_grads(idx, targets, cos, sin, M, G, C, GR, tmp, nll)
    print '(A,F10.5)', "  nll =", nll
    call check(nll == nll .and. nll > 0.0_sp, "nll finite positive")

    max_err = 0.0_sp
    call check_group(M%wte, GR%wte, "wte", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%lm, GR%lm, "lm", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%q, GR%q, "q", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%k, GR%k, "k", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%v, GR%v, "v", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%p, GR%p, "proj", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%fc, GR%fc, "fc", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    call check_group(M%p2, GR%p2, "p2", max_err, M, G, C, tmp, idx, &
        targets, cos, sin, H)
    print '(A,E10.3)', "  max err all grads = ", max_err
    call check(max_err < 5.0e-3_sp, "whole-model grads")

    ! overfit: 5 AdamW steps on this one batch must lower NLL
    n0 = nll
    do k = 1, 5
      call train_step(idx, targets, cos, sin, M, S, G, GR, C, tmp, nll, k, &
          0.02_sp, 0.9_sp, 0.999_sp, 1.0e-8_sp, 0.0_sp)
    end do
    n5 = nll
    print '(A,F10.5,A,F10.5)', "  nll ", n0, " -> ", n5
    call check(n5 < n0, "overfit descends")
  end subroutine test_full_step

  ! ------------------------------------------------------------------------
  ! linear3d_sgemm vs naive linear3d. Different summation order (FMA +
  ! blocking) gives relative ~1e-6 on O(10) values; tolerance is honest
  ! 1e-4 absolute, still 1000x below anything downstream can feel.
  subroutine test_linear_blas()
    integer, parameter :: BR = 4, TC = 16, IF = 128, OF = 128
    real(sp) :: x(BR*TC*IF), w(OF*IF), y1(BR*TC*OF), y2(BR*TC*OF)
    real(sp) :: e, max_err
    integer :: i

    print '(A)', "=== test_linear_blas (sgemm vs naive) ==="
    call fill(x, BR*TC*IF)
    call fill(w, OF*IF)

    call linear3d(x, w, y1, BR, TC, IF, OF)
    call linear3d_sgemm(x, w, y2, BR, TC, IF, OF)

    max_err = 0.0_sp
    do i = 1, BR*TC*OF
      e = abs(y1(i) - y2(i))
      if (e > max_err) max_err = e
    end do
    print '(A,E10.3,A,E10.3)', "  max err = ", max_err, &
        "  ref scale = ", maxval(abs(y1))
    call check(max_err < 1.0e-4_sp, "sgemm parity")
  end subroutine test_linear_blas

  ! ------------------------------------------------------------------------
  subroutine test_decode()
    real(sp) :: lg(5), w(5)
    integer :: gen(4), pos, i, hits(5)
    integer(c_int64_t) :: st
    real(sp) :: pr(4)
    integer :: ox(4)

    print '(A)', "=== test_decode (penalties/blocking/top-p) ==="
    ! penalties exact: gen [1,1,2], pres=1, freq=0.5
    lg = 0.0_sp
    gen(1:3) = [1, 1, 2]
    w = lg
    call apply_penalties(w, 5, gen, 3, 1.0_sp, 0.5_sp)
    call check(abs(w(2) + 2.0_sp) < 1.0e-7_sp, "presence+freq id1")
    call check(abs(w(3) + 1.5_sp) < 1.0e-7_sp, "presence+freq id2")
    call check(abs(w(1)) + abs(w(4)) + abs(w(5)) < 1.0e-7_sp, &
        "untouched rest")

    ! blocking: gen [3,4,4] has bigram (4,4); tail [4] + t=4 would repeat
    ! it -> ban t=4; greedy falls through to id 3 (position 4).
    lg = [0.0_sp, 1.0_sp, 2.0_sp, 3.0_sp, 4.0_sp]
    gen(1:3) = [3, 4, 4]
    st = 9_c_int64_t
    pos = sample_next(lg, 5, 0.0_sp, 1.0_sp, 0.0_sp, 0.0_sp, 1.0_sp, 0, 0.0_sp, gen, 3, 2, st)
    call check(pos == 4, "blocked argmax falls through")
    ! without blocking, greedy takes id 4 (position 5)
    pos = sample_next(lg, 5, 0.0_sp, 1.0_sp, 0.0_sp, 0.0_sp, 1.0_sp, 0, 0.0_sp, gen, 3, 0, st)
    call check(pos == 5, "unblocked argmax")

    ! top-p=0 -> always argmax over 50 draws
    st = 3_c_int64_t
    do i = 1, 50
      pos = sample_next(lg, 5, 1.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 1.0_sp, 0, 0.0_sp, gen, 0, 0, st)
      if (pos /= 5) then
        call check(.false., "top-p 0 argmax")
        exit
      end if
    end do
    call check(.true., "top-p 0 argmax")
    ! top-p=1 spreads (id 4 has ~88% mass, others share rest)
    st = 11_c_int64_t
    hits = 0
    do i = 1, 200
      pos = sample_next(lg, 5, 1.0_sp, 1.0_sp, 0.0_sp, 0.0_sp, 1.0_sp, 0, 0.0_sp, gen, 0, 0, st)
      hits(pos) = hits(pos) + 1
    end do
    call check(hits(5) > 100 .and. sum(hits(1:4)) > 0, "top-p 1 spread")

    ! sort_desc tracks permutation
    pr = [3.0_sp, 1.0_sp, 2.0_sp, 0.5_sp]
    ox = [1, 2, 3, 4]
    call sort_desc(pr, ox, 4)
    call check(all(abs(pr - [3.0_sp, 2.0_sp, 1.0_sp, 0.5_sp]) < 1e-7), &
        "sort values")
    call check(all(ox == [1, 3, 2, 4]), "sort permutation")
  end subroutine test_decode

  ! ------------------------------------------------------------------------
  subroutine test_mkdir_p()
    print '(A)', "=== test_mkdir_p ==="
    call check(mkdir_p("/tmp/fx_mkdir/a/b/c") == 0, "nested create")
    call check(dir_exists("/tmp/fx_mkdir/a/b/c"), "exists after")
    call check(mkdir_p("/tmp/fx_mkdir/a/b/c") == 0, "idempotent EEXIST")
    call check(dir_exists("/tmp/fx_mkdir/nope") .eqv. .false., &
        "missing dir false")
  end subroutine test_mkdir_p

  ! ------------------------------------------------------------------------
  ! Data loader: write a fixture rows file, read back batch + count.
  subroutine test_data_batch()
    integer, parameter :: T = 8
    integer :: idx(2*T), targets(2*T), ngot, n
    integer :: u

    print '(A)', "=== test_data_batch ==="
    open (newunit=u, file="/tmp/fd_rows.txt", status="replace")
    write (u, '(A)') "0 1 2 3 4 5 6 7 8"
    write (u, '(A)') ""
    write (u, '(A)') "8 7 6 5 4 3 2 1 0"
    close (u)

    n = count_rows("/tmp/fd_rows.txt")
    call check(n == 2, "count skips blanks")
    call load_batch("/tmp/fd_rows.txt", 0, 2, T, idx, targets, ngot)
    call check(ngot == 2, "batch got 2")
    call check(all(idx(1:T) == [0, 1, 2, 3, 4, 5, 6, 7]), "row0 ids")
    call check(all(targets(1:T) == [1, 2, 3, 4, 5, 6, 7, 8]), "row0 tgts")
    call check(all(idx(T+1:2*T) == [8, 7, 6, 5, 4, 3, 2, 1]), "row1 ids")
    call load_batch("/tmp/fd_rows.txt", 1, 2, T, idx, targets, ngot)
    call check(ngot == 1, "offset stops at EOF")
    call check(count_rows("/tmp/does_not_exist_xyz") == -1, "missing -> -1")
  end subroutine test_data_batch

  ! ------------------------------------------------------------------------
  ! save_gpt_weights -> load_gpt_weights roundtrip (tiny shapes).
  subroutine test_save_load()
    use load_weights_mod, only: load_gpt_weights, save_gpt_weights
    real(sp), allocatable :: wte(:), lm(:), cq(:), ck(:), cv(:)
    real(sp), allocatable :: cp(:), cf(:), cp2(:)
    real(sp), allocatable :: wte2(:), lm2(:), cq2(:), ck2(:), cv2(:)
    real(sp), allocatable :: cp_(:), cf2(:), cp22(:)
    integer :: u, i
    logical :: ok

    print '(A)', "=== test_save_load ==="
    ! dims V=3 D=2 nh=1 nkv=1 hd=2 nl=1: wte 6, lm 6, q/k/v/p 4,
    ! fc/p2 16.
    allocate(wte(6), lm(6))
    allocate(cq(4), ck(4), cv(4), cp(4), cf(16), cp2(16))
    wte = [(real(i, sp), i = 1, 6)]
    lm = -wte
    cq = 1.0_sp; ck = 2.0_sp; cv = 3.0_sp
    cp = 4.0_sp; cf = 5.0_sp; cp2 = 6.0_sp
    call execute_command_line("mkdir -p /tmp/rt_weights")
    call save_gpt_weights("/tmp/rt_weights", 1, 2, 1, 1, 2, 3, &
        wte, lm, cq, ck, cv, cp, cf, cp2)
    call load_gpt_weights("/tmp/rt_weights", 1, 2, 1, 1, 2, 3, &
        wte2, lm2, cq2, ck2, cv2, cp_, cf2, cp22)
    ok = all(wte2 == wte) .and. all(lm2 == lm) .and. all(cq2 == cq) &
        .and. all(ck2 == ck) .and. all(cv2 == cv) .and. all(cp_ == cp) &
        .and. all(cf2 == cf) .and. all(cp22 == cp2)
    call check(ok, "npy roundtrip exact")
    call check(size(cq2) == 4 .and. size(cf2) == 16, "shapes kept")
  end subroutine test_save_load

  subroutine check_group(w, gr, label, max_err, M, G, C, tmp, idx, targets, &
      cos, sin, Hh)
    ! group-local max printed separately (cumulative max_err hides origin)
    real(sp) :: glocal
    real(sp), intent(inout) :: w(:)
    real(sp), intent(in) :: gr(:)
    character(*), intent(in) :: label
    real(sp), intent(inout) :: max_err
    type(params_t), intent(inout) :: M
    type(dims_t), intent(in) :: G
    type(cache_t), intent(inout) :: C
    type(temp_t), intent(inout) :: tmp
    integer, intent(in) :: idx(:), targets(:)
    real(sp), intent(in) :: cos(:), sin(:)
    real(sp), intent(in) :: Hh
    real(sp) :: w0, np_, nm_, e, fdval, fd_worst, an_worst
    integer :: ii, iworst
    glocal = 0.0_sp
    iworst = 1; fd_worst = 0.0_sp; an_worst = 0.0_sp
    do ii = 1, size(w)
      w0 = w(ii)
      w(ii) = w0 + Hh
      call forward_save(idx, targets, cos, sin, M, G, C, tmp, np_)
      w(ii) = w0 - Hh
      call forward_save(idx, targets, cos, sin, M, G, C, tmp, nm_)
      w(ii) = w0
      fdval = (np_ - nm_) / (2.0_sp * Hh)
      e = abs(gr(ii) - fdval)
      if (e > glocal) then
        glocal = e
        iworst = ii; fd_worst = fdval; an_worst = gr(ii)
      end if
      if (e > max_err) max_err = e
    end do
    print '(A,A,E10.3,A,I0,A,2ES12.4)', "  group ", label, glocal, &
        " @", iworst, " analytic/FD:", an_worst, fd_worst
  end subroutine check_group

end program test_kernels
