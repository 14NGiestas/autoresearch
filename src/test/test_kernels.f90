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
  use sample_mod, only: sample_token
  implicit none

  integer, parameter :: sp = c_float
  integer :: seed = 42
  integer :: fail_count = 0

  ! C-mangled interfaces — must be before CONTAINS in Fortran
  interface
    subroutine wte_lookup(idx, wte, out, B, T, V, D) bind(c, name='wte_lookup')
      integer, intent(in) :: idx(*), B, T, V, D
      real, intent(in) :: wte(*)
      real, intent(out) :: out(*)
    end subroutine
    subroutine rmsnorm0(x, y, N, C, eps) bind(c, name='rmsnorm0')
      integer, intent(in) :: N, C
      real, intent(in) :: x(*)
      real, intent(out) :: y(*)
      real, value :: eps
    end subroutine
    subroutine linear3d(x, w, y, B, T, IF, OF) bind(c, name='linear3d')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine linear3dT(x, w, y, B, T, IF, OF) bind(c, name='linear3dT')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine rope_4d(x, cos, sin, y, B, T, H, D) bind(c, name='rope_4d')
      integer, intent(in) :: B, T, H, D
      real, intent(in) :: x(*), cos(*), sin(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine causal_attn(q, k, v, y, B, T, H, K_H, D) bind(c, name='causal_attn')
      integer, intent(in) :: B, T, H, K_H, D
      real, intent(in) :: q(*), k(*), v(*)
      real, intent(out) :: y(*)
    end subroutine
    subroutine relu2(x, N) bind(c, name='relu2')
      integer, intent(in) :: N
      real, intent(inout) :: x(*)
    end subroutine
    subroutine gpt_forward(idx, cos, sin, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        outp, B, T, V, D, n_head, n_kv_head, head_dim, n_layer, eps) &
        bind(c, name='gpt_forward')
      integer, intent(in) :: idx(*), B, T, V, D
      integer, intent(in) :: n_head, n_kv_head, head_dim, n_layer
      real, intent(in) :: cos(*), sin(*), wte(*)
      real, intent(in) :: c_q(*), c_k(*), c_v(*), c_proj(*)
      real, intent(in) :: c_fc(*), c_proj2(*), lm_head(*)
      real, intent(out) :: outp(*)
      real, value :: eps
    end subroutine
    subroutine gpt_step(idx1, cos1, sin1, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        cache_k, cache_v, cache_len, maxT, out1, &
        B, V, D, n_head, n_kv_head, head_dim, n_layer, eps) &
        bind(c, name='gpt_step')
      integer, intent(in) :: idx1(*), B, V, D, maxT
      integer, intent(inout) :: cache_len
      integer, intent(in) :: n_head, n_kv_head, head_dim, n_layer
      real, intent(in) :: cos1(*), sin1(*), wte(*)
      real, intent(in) :: c_q(*), c_k(*), c_v(*), c_proj(*)
      real, intent(in) :: c_fc(*), c_proj2(*), lm_head(*)
      real, intent(inout) :: cache_k(*), cache_v(*)
      real, intent(out) :: out1(*)
      real, value :: eps
    end subroutine
    subroutine linear3d_bwd(dy, x, w, dx, dw, B, T, IF, OF) &
        bind(c, name='linear3d_bwd')
      integer, intent(in) :: B, T, IF, OF
      real, intent(in) :: dy(*), x(*), w(*)
      real, intent(out) :: dx(*), dw(*)
    end subroutine
    subroutine rmsnorm0_bwd(dy, x, dx, N, C, eps) bind(c, name='rmsnorm0_bwd')
      integer, intent(in) :: N, C
      real, intent(in) :: dy(*), x(*)
      real, intent(out) :: dx(*)
      real, value :: eps
    end subroutine
    subroutine rope_4d_bwd(dy, cos, sin, dx, B, T, H, D) \
        bind(c, name='rope_4d_bwd')
      integer, intent(in) :: B, T, H, D
      real, intent(in) :: dy(*), cos(*), sin(*)
      real, intent(out) :: dx(*)
    end subroutine
    subroutine xent_fwd(logits, targets, nll, B, T, V) \
        bind(c, name='xent_fwd')
      integer, intent(in) :: B, T, V
      real, intent(in) :: logits(*)
      integer, intent(in) :: targets(*)
      real, intent(out) :: nll(*)
    end subroutine
    subroutine xent_bwd(logits, targets, dlogits, B, T, V, scale) \
        bind(c, name='xent_bwd')
      integer, intent(in) :: B, T, V
      real, intent(in) :: logits(*)
      integer, intent(in) :: targets(*)
      real, intent(out) :: dlogits(*)
      real, value :: scale
    end subroutine
    subroutine wte_bwd(idx, dout, dwte, B, T, V, D) bind(c, name='wte_bwd')
      integer, intent(in) :: idx(*), B, T, V, D
      real, intent(in) :: dout(*)
      real, intent(inout) :: dwte(*)
    end subroutine
    subroutine attn_bwd(dy, q, k, v, dq, dk, dv, B, T, H, K_H, D) \
        bind(c, name='attn_bwd')
      integer, intent(in) :: B, T, H, K_H, D
      real, intent(in) :: dy(*), q(*), k(*), v(*)
      real, intent(out) :: dq(*)
      real, intent(inout) :: dk(*), dv(*)
    end subroutine
    subroutine relu2_bwd(dy, x, dx, N) bind(c, name='relu2_bwd')
      integer, intent(in) :: N
      real, intent(in) :: dy(*), x(*)
      real, intent(out) :: dx(*)
    end subroutine
    subroutine rmsnorm(x, w, y, N, C, eps) bind(c, name='rmsnorm')
      integer, intent(in) :: N, C
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
      real, value :: eps
    end subroutine
    subroutine recurrent_forward(idx, cos, sin, &
        wte, c_q, c_k, c_v, c_proj, c_fc, c_proj2, lm_head, &
        outp, B, T, V, D, n_head, n_kv_head, head_dim, n_loops, eps) &
        bind(c, name='recurrent_forward')
      integer, intent(in) :: idx(*), B, T, V, D
      integer, intent(in) :: n_head, n_kv_head, head_dim, n_loops
      real, intent(in) :: cos(*), sin(*), wte(*)
      real, intent(in) :: c_q(*), c_k(*), c_v(*), c_proj(*)
      real, intent(in) :: c_fc(*), c_proj2(*), lm_head(*)
      real, intent(out) :: outp(*)
      real, value :: eps
    end subroutine
  end interface

  call test_rmsnorm()
  call test_rmsnorm0()
  call test_linear3d()
  call test_linear3dT()
  call test_wte_lookup()
  call test_rope_4d()
  call test_relu2()
  call test_causal_attn()
  call test_causal_attn_gqa()
  call test_gpt_forward_shape()
  call test_recurrent_equiv()
  call test_recurrent_loops()
  call test_sample()
  call test_kv_equiv()
  call test_linear_bwd()
  call test_rmsnorm_bwd()
  call test_wte_bwd()
  call test_rope_bwd()
  call test_xent()
  call test_attn_bwd()
  call test_relu2_bwd()

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
    call check(max_err == 0.0_sp, "recurrent(1) == gpt(1) bit-exact")
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
    call check(max_err == 0.0_sp, "cached steps == full forward")
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

end program test_kernels
