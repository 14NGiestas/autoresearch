! lib/sample.f90 — temperature sampling for generation (greedy + multinomial).
!
! sample_token(logits, temp, state) -> 1-based argmax position on ties/
! temp<=0 (greedy), else a multinomial draw from softmax(logits/temp).
! RNG is xorshift64* with caller-held int64 state (deterministic per seed,
! no global state — safe under OpenMP as long as states are thread-local).

module sample_mod
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  implicit none
contains

  ! uniform in [0,1): xorshift64* (Vigna), state updated in place.
  real(wp) function rand_u01(state)
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
    rand_u01 = real(iand(ishft(x, -40), 16777215_c_int64_t), wp) &
        / 16777216.0_wp
  end function rand_u01

  ! Presence + frequency penalties (in-place on caller-owned copy):
  !   logits(id) -= pres + freq * count(id in gen(1:ngen)).
  ! Breaks the rich-get-richer loop feedback directly.
  subroutine apply_penalties(logits, V, gen, ngen, pres, freq)
    integer, intent(in) :: V, ngen
    real(wp), intent(inout) :: logits(:)
    integer, intent(in) :: gen(:)
    real(wp), intent(in) :: pres, freq
    logical :: seen(0:V-1)
    integer :: i, j, cnt
    seen = .false.
    do i = 1, ngen
      if (gen(i) < 0 .or. gen(i) >= V) cycle
      if (seen(gen(i))) cycle   ! once per unique id, not per occurrence
      seen(gen(i)) = .true.
      cnt = 0
      do j = 1, ngen
        if (gen(j) == gen(i)) cnt = cnt + 1
      end do
      logits(gen(i)+1) = logits(gen(i)+1) - pres - freq * real(cnt, wp)
    end do
  end subroutine apply_penalties

  ! CTRL-style repetition penalty (Keskar et al. 1909.05858):
  !   if token seen in gen, logits(id) /= theta (theta>1 discounts).
  ! Distinct from additive pres/freq; ~15 lines, multiplicative.
  subroutine apply_rep_penalty(logits, V, gen, ngen, theta)
    integer, intent(in) :: V, ngen
    real(wp), intent(inout) :: logits(:)
    integer, intent(in) :: gen(:)
    real(wp), intent(in) :: theta
    logical :: seen(0:V-1)
    integer :: i
    if (theta <= 1.0_wp .or. ngen == 0) return
    seen = .false.
    do i = 1, ngen
      if (gen(i) < 0 .or. gen(i) >= V) cycle
      if (seen(gen(i))) cycle
      seen(gen(i)) = .true.
      logits(gen(i)+1) = logits(gen(i)+1) / theta
    end do
  end subroutine apply_rep_penalty

  ! Windowed (forgetting) penalty (Zhu et al. 2310.14971):
  !   pres/freq only on last W tokens, not full history, plus a length
  !   penalty that discourages over-short outputs (logits(EOS) -= plen *
  !   max(0, 1 - ngen/W)). Explains why full-history needs retuning.
  subroutine apply_windowed_penalties(logits, V, gen, ngen, pres, freq, win, plen)
    integer, intent(in) :: V, ngen, win
    real(wp), intent(inout) :: logits(:)
    integer, intent(in) :: gen(:)
    real(wp), intent(in) :: pres, freq, plen
    integer :: i, j, cnt, start
    logical :: seen(0:V-1)
    if (win <= 0 .or. ngen == 0) return
    start = max(1, ngen - win + 1)
    if (pres /= 0.0_wp .or. freq /= 0.0_wp) then
      seen = .false.
      do i = start, ngen
        if (gen(i) < 0 .or. gen(i) >= V) cycle
        if (seen(gen(i))) cycle
        seen(gen(i)) = .true.
        cnt = 0
        do j = start, ngen
          if (gen(j) == gen(i)) cnt = cnt + 1
        end do
        logits(gen(i)+1) = logits(gen(i)+1) - pres - freq * real(cnt, wp)
      end do
    end if
    ! length penalty: if ngen < win, discourage EOS (assume EOS=0) from ending too short
    if (plen /= 0.0_wp .and. ngen < win) then
      logits(1) = logits(1) - plen * real(win - ngen, wp) / real(win, wp)
    end if
  end subroutine apply_windowed_penalties

  ! No-repeat n-gram blocking: any token completing an n-gram already
  ! seen in gen(1:ngen) gets -inf. Windows [s, s+nn-1] with
  ! s <= ngen-nn+1 (strictly inside history; never self-compare).
  subroutine block_ngram(logits, V, gen, ngen, nn)
    integer, intent(in) :: V, ngen, nn
    real(wp), intent(inout) :: logits(:)
    integer, intent(in) :: gen(:)
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
          logits(t+1) = -huge(1.0_wp)
          exit
        end if
      end do
    end do
  end subroutine block_ngram

  ! Full pipeline: penalties (windowed or full) -> rep -> blocking -> temp/top-p.
  ! gen holds previously generated 0-based ids (ngen may be 0).
  integer function sample_next(logits, V, temp, topp, pres, freq, rep, pwin, plen, &
      gen, ngen, nblock, state)
    integer, intent(in) :: V, ngen, nblock, pwin
    real(wp), intent(in) :: logits(:), temp, topp, pres, freq, rep, plen
    integer, intent(in) :: gen(:)
    integer(c_int64_t), intent(inout) :: state
    real(wp) :: work(V)
    work = logits
    if (pwin > 0) then
      call apply_windowed_penalties(work, V, gen, ngen, pres, freq, pwin, plen)
    else
      if (pres /= 0.0_wp .or. freq /= 0.0_wp) &
          call apply_penalties(work, V, gen, ngen, pres, freq)
    end if
    if (rep > 1.0_wp) call apply_rep_penalty(work, V, gen, ngen, rep)
    if (nblock >= 2) call block_ngram(work, V, gen, ngen, nblock)
    sample_next = sample_top_p(work, V, temp, topp, state)
  end function sample_next

  ! Temperature + nucleus sampling. topp>=1: full distribution.
  integer function sample_top_p(logits, V, temp, topp, state)
    integer, intent(in) :: V
    real(wp), intent(in) :: logits(:), temp, topp
    integer(c_int64_t), intent(inout) :: state
    real(wp) :: sc(V), pr(V)
    integer :: ox(V)
    integer :: i, k, cut
    real(wp) :: m, s, u, acc, t
    t = temp
    if (t <= 0.0_wp) then
      sample_top_p = 1
      do i = 2, V
        if (logits(i) > logits(sample_top_p)) sample_top_p = i
      end do
      return
    end if
    if (t < 1.0e-6_wp) t = 1.0e-6_wp
    m = logits(1)
    do i = 2, V
      if (logits(i) > m) m = logits(i)
    end do
    s = 0.0_wp
    do i = 1, V
      sc(i) = exp((logits(i) - m) / t)
      s = s + sc(i)
      ox(i) = i
    end do
    do i = 1, V
      pr(i) = sc(i) / s
    end do
    call sort_desc(pr, ox, V)
    if (topp >= 1.0_wp) then
      cut = V
    else
      acc = 0.0_wp
      cut = 1
      do k = 1, V
        acc = acc + pr(k)
        cut = k
        if (acc >= topp) exit
      end do
    end if
    u = rand_u01(state) * sum(pr(1:cut))
    acc = 0.0_wp
    do k = 1, cut
      acc = acc + pr(k)
      if (acc >= u) then
        sample_top_p = ox(k)
        return
      end if
    end do
    sample_top_p = ox(cut)
  end function sample_top_p

  ! Quicksort (Lomuto, middle pivot) sorting pr desc, ox alongside.
  recursive subroutine sort_desc(pr, ox, n)
    real(wp), intent(inout) :: pr(:)
    integer, intent(inout) :: ox(:)
    integer, intent(in) :: n
    integer :: i, j, ti, mid
    real(wp) :: pv, tp
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
    if (n - i - 1 > 0) call sort_desc(pr(i+2:), ox(i+2:), n - i - 1)
  end subroutine sort_desc

  ! 1-based position of sampled token in logits(1:V).
  integer function sample_token(logits, V, temp, state)
    integer, intent(in) :: V
    real(wp), intent(in) :: logits(:), temp
    integer(c_int64_t), intent(inout) :: state
    integer :: i
    real(wp) :: m, s, u, acc
    real(wp) :: t

    t = temp
    if (t <= 0.0_wp) then
      ! greedy
      sample_token = 1
      do i = 2, V
        if (logits(i) > logits(sample_token)) sample_token = i
      end do
      return
    end if
    if (t < 1.0e-6_wp) t = 1.0e-6_wp

    m = logits(1)
    do i = 2, V
      if (logits(i) > m) m = logits(i)
    end do
    s = 0.0_wp
    do i = 1, V
      s = s + exp((logits(i) - m) / t)
    end do
    u = rand_u01(state) * s
    acc = 0.0_wp
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
