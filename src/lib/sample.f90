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

  ! Presence + frequency penalties (in-place on caller-owned copy):
  !   logits(id) -= pres + freq * count(id in gen(1:ngen)).
  ! Breaks the rich-get-richer loop feedback directly.
  subroutine apply_penalties(logits, V, gen, ngen, pres, freq)
    integer, intent(in) :: V, ngen
    real(c_float), intent(inout) :: logits(V)
    integer, intent(in) :: gen(ngen)
    real(c_float), intent(in) :: pres, freq
    logical, allocatable :: seen(:)
    integer :: i, j, cnt
    allocate(seen(0:V-1))
    seen = .false.
    do i = 1, ngen
      if (gen(i) < 0 .or. gen(i) >= V) cycle
      if (seen(gen(i))) cycle   ! once per unique id, not per occurrence
      seen(gen(i)) = .true.
      cnt = 0
      do j = 1, ngen
        if (gen(j) == gen(i)) cnt = cnt + 1
      end do
      logits(gen(i)+1) = logits(gen(i)+1) - pres - freq * real(cnt, c_float)
    end do
    deallocate(seen)
  end subroutine apply_penalties

  ! CTRL-style repetition penalty (Keskar et al. 1909.05858):
  !   if token seen in gen, logits(id) /= theta (theta>1 discounts).
  ! Distinct from additive pres/freq; ~15 lines, multiplicative.
  subroutine apply_rep_penalty(logits, V, gen, ngen, theta)
    integer, intent(in) :: V, ngen
    real(c_float), intent(inout) :: logits(V)
    integer, intent(in) :: gen(ngen)
    real(c_float), intent(in) :: theta
    logical, allocatable :: seen(:)
    integer :: i
    if (theta <= 1.0_c_float .or. ngen == 0) return
    allocate(seen(0:V-1))
    seen = .false.
    do i = 1, ngen
      if (gen(i) < 0 .or. gen(i) >= V) cycle
      if (seen(gen(i))) cycle
      seen(gen(i)) = .true.
      logits(gen(i)+1) = logits(gen(i)+1) / theta
    end do
    deallocate(seen)
  end subroutine apply_rep_penalty

  ! No-repeat n-gram blocking: any token completing an n-gram already
  ! seen in gen(1:ngen) gets -inf. Windows [s, s+nn-1] with
  ! s <= ngen-nn+1 (strictly inside history; never self-compare).
  subroutine block_ngram(logits, V, gen, ngen, nn)
    integer, intent(in) :: V, ngen, nn
    real(c_float), intent(inout) :: logits(V)
    integer, intent(in) :: gen(ngen)
    integer :: t, s, k
    logical :: match
    if (nn < 2 .or. ngen < nn) return
    do t = 0, V - 1
      do s = 1, ngen - nn + 1
        match = .true.
        do k = 0, nn - 2
          if (gen(ngen-k) /= gen(s+nn-2-k)) then
            match = .false.
            exit
          end if
        end do
        if (match .and. gen(s+nn-1) == t) then
          logits(t+1) = -huge(1.0_c_float)
          exit
        end if
      end do
    end do
  end subroutine block_ngram

  ! Full pipeline: penalties -> rep -> blocking -> temp/top-p sample.
  ! gen holds previously generated 0-based ids (ngen may be 0).
  integer function sample_next(logits, V, temp, topp, pres, freq, rep, &
      gen, ngen, nblock, state)
    integer, intent(in) :: V, ngen, nblock
    real(c_float), intent(in) :: logits(V), temp, topp, pres, freq, rep
    integer, intent(in) :: gen(ngen)
    integer(c_int64_t), intent(inout) :: state
    real(c_float), allocatable :: work(:)
    allocate(work(V))
    work = logits
    if (pres /= 0.0_c_float .or. freq /= 0.0_c_float) &
        call apply_penalties(work, V, gen, ngen, pres, freq)
    if (rep > 1.0_c_float) call apply_rep_penalty(work, V, gen, ngen, rep)
    if (nblock >= 2) call block_ngram(work, V, gen, ngen, nblock)
    sample_next = sample_top_p(work, V, temp, topp, state)
    deallocate(work)
  end function sample_next

  ! Temperature + nucleus sampling. topp>=1: full distribution.
  integer function sample_top_p(logits, V, temp, topp, state)
    integer, intent(in) :: V
    real(c_float), intent(in) :: logits(V), temp, topp
    integer(c_int64_t), intent(inout) :: state
    real(c_float), allocatable :: sc(:), pr(:)
    integer, allocatable :: ox(:)
    integer :: i, k, cut
    real(c_float) :: m, s, u, acc, t
    t = temp
    if (t <= 0.0_c_float) then
      sample_top_p = 1
      do i = 2, V
        if (logits(i) > logits(sample_top_p)) sample_top_p = i
      end do
      return
    end if
    if (t < 1.0e-6_c_float) t = 1.0e-6_c_float
    allocate(sc(V), pr(V), ox(V))
    m = logits(1)
    do i = 2, V
      if (logits(i) > m) m = logits(i)
    end do
    s = 0.0_c_float
    do i = 1, V
      sc(i) = exp((logits(i) - m) / t)
      s = s + sc(i)
      ox(i) = i
    end do
    do i = 1, V
      pr(i) = sc(i) / s
    end do
    call sort_desc(pr, ox, V)
    if (topp >= 1.0_c_float) then
      cut = V
    else
      acc = 0.0_c_float
      cut = 1
      do k = 1, V
        acc = acc + pr(k)
        cut = k
        if (acc >= topp) exit
      end do
    end if
    u = rand_u01(state) * sum(pr(1:cut))
    acc = 0.0_c_float
    do k = 1, cut
      acc = acc + pr(k)
      if (acc >= u) then
        sample_top_p = ox(k)
        deallocate(sc, pr, ox)
        return
      end if
    end do
    sample_top_p = ox(cut)
    deallocate(sc, pr, ox)
  end function sample_top_p

  ! Quicksort (Lomuto, middle pivot) sorting pr desc, ox alongside.
  recursive subroutine sort_desc(pr, ox, n)
    real(c_float), intent(inout) :: pr(*)
    integer, intent(inout) :: ox(*)
    integer, intent(in) :: n
    integer :: i, j, ti, mid
    real(c_float) :: pv, tp
    if (n < 2) return
    mid = (n + 1) / 2
    tp = pr(mid); pr(mid) = pr(n); pr(n) = tp
    ti = ox(mid); ox(mid) = ox(n); ox(n) = ti
    pv = pr(n)
    i = 0
    do j = 1, n - 1
      if (pr(j) > pv) then
        i = i + 1
        tp = pr(i); pr(i) = pr(j); pr(j) = tp
        ti = ox(i); ox(i) = ox(j); ox(j) = ti
      end if
    end do
    tp = pr(i+1); pr(i+1) = pr(n); pr(n) = tp
    ti = ox(i+1); ox(i+1) = ox(n); ox(n) = ti
    call sort_desc(pr, ox, i)
    if (n - i - 1 > 0) call sort_desc(pr(i+2), ox(i+2), n - i - 1)
  end subroutine sort_desc

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
