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
    subroutine rmsnorm(x, w, y, N, C, eps) bind(c, name='rmsnorm')
      integer, intent(in) :: N, C
      real, intent(in) :: x(*), w(*)
      real, intent(out) :: y(*)
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

end program test_kernels
