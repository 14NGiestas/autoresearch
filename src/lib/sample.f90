! lib/sample.f90 — temperature sampling for generation (greedy + multinomial).
!
! sample_token(logits, temp, state) -> 1-based argmax position on ties/
! temp<=0 (greedy), else a multinomial draw from softmax(logits/temp).
! RNG is xorshift64* with caller-held int64 state (deterministic per seed,
! no global state — safe under OpenMP as long as states are thread-local).

module sample_mod
  use iso_c_binding
  implicit none
contains

  ! uniform in [0,1): xorshift64* (Vigna), state updated in place.
  real(c_float) function rand_u01(state)
    integer(c_int64_t), intent(inout) :: state
    integer(c_int64_t) :: x
    if (state == 0) state = 2685821657736338717_c_int64_t
    x = state
    x = ieor(x, ishft(x, 13))
    x = ieor(x, ishft(x, -7))
    x = ieor(x, ishft(x, 17))
    state = x
    x = x * 2685821657736338717_c_int64_t
    ! top 24 bits -> [0,1)
    rand_u01 = real(iand(ishft(x, -40), 16777215_c_int64_t), c_float) &
        / 16777216.0_c_float
  end function rand_u01

  ! 1-based position of sampled token in logits(1:V).
  integer function sample_token(logits, V, temp, state)
    integer, intent(in) :: V
    real(c_float), intent(in) :: logits(V), temp
    integer(c_int64_t), intent(inout) :: state
    integer :: i
    real(c_float) :: m, s, u, acc
    real(c_float) :: t

    t = temp
    if (t <= 0.0_c_float) then
      ! greedy
      sample_token = 1
      do i = 2, V
        if (logits(i) > logits(sample_token)) sample_token = i
      end do
      return
    end if
    if (t < 1.0e-6_c_float) t = 1.0e-6_c_float

    m = logits(1)
    do i = 2, V
      if (logits(i) > m) m = logits(i)
    end do
    s = 0.0_c_float
    do i = 1, V
      s = s + exp((logits(i) - m) / t)
    end do
    u = rand_u01(state) * s
    acc = 0.0_c_float
    do i = 1, V
      acc = acc + exp((logits(i) - m) / t)
      if (acc >= u) then
        sample_token = i
        return
      end if
    end do
    sample_token = V
  end function sample_token

end module sample_mod
