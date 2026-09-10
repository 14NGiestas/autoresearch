! lib/fortran_spec.f90 — greedy speculative decoding: the accept rule.
!
! Draft-and-verify (Leviathan et al. 2023, arXiv:2211.17192; Chen et al.
! 2023, arXiv:2302.01318) spends a cheap drafter's k steps, then verifies
! all k tokens in ONE target pass. Because our decode is memory-bandwidth
! bound at B=1 (the whole weight set streams per token), verifying k tokens
! costs about one token's time, so the win is ~(n_accept+1)x.
!
! Here we implement the TEMPERATURE-0 case only: drafter and target both
! take argmax, the accepted prefix is the longest run where they agree, and
! the token at the first disagreement (or the verification pass's bonus row
! when all k agree) is committed from the TARGET. That makes the committed
! stream EXACTLY the target's greedy stream — a property we can test
! without any weights (src/test/test_kernels.f90, test_accept_prefix).
!
! Sampling mode (temp > 0) needs rejection sampling with p_target/p_draft
! and a residual resample; deliberately left out until the greedy path is
! wired and measured.
!
! Related: our depth-sweep app (recur_sweep) and the tied-layer recurrent
! path suggest candidate drafters we can evaluate on acceptance rate alone;
! the drafter quality question is empirical, the commit rule is not.

module fortran_spec_mod
  use iso_c_binding
  implicit none
contains

  ! Greedy accept rule over one verification window.
  !   draft(1:k)      drafted ids (0-based), positions p+1 .. p+k
  !   targmax(1:k+1)  target argmax at positions p+1 .. p+k+1, all obtained
  !                   from the single verification pass over draft(1:k)
  !   n_accept        out: 0..k leading draft tokens that match the target
  !   correction      out: token committed right after the accepted prefix;
  !                   targmax(n_accept+1) covers both cases — the target's
  !                   own choice at the first disagreement, and the bonus row
  !                   when every draft token was accepted
  ! Committed tokens per verification pass = n_accept + 1 (>= 1 always), so
  ! a pass is never worse than plain decoding: with n_accept = 0 it emits
  ! exactly the token the target would have emitted anyway.
  subroutine accept_prefix(draft, targmax, k, n_accept, correction)
    integer, intent(in)  :: draft(:), targmax(:), k
    integer, intent(out) :: n_accept, correction
    integer :: j

    n_accept = 0
    do j = 1, k
      if (draft(j) /= targmax(j)) exit
      n_accept = n_accept + 1
    end do
    correction = targmax(n_accept + 1)
  end subroutine accept_prefix

  ! Prompt-lookup drafter (LLMA / prompt-lookup decoding). Find the most
  ! recent past occurrence of the last m tokens of ids(1:p) and propose what
  ! followed it. Zero model cost — a miss (kk=0) degrades to a plain step,
  ! so this drafter can never make a pass slower than greedy decoding.
  ! Proposals are capped at already-known ids (position <= p).
  ! Measured on phase-3 weights: 1.6x-4.5x decode speedup on 64-192 token
  ! prompts whose context repeats (math rows, tutorials), 1.0x on an
  ! 8-token prompt with nothing to look up.
  subroutine lookup_draft(ids, p, m, k, d, kk)
    integer, intent(in)  :: ids(:), p, m, k
    integer, intent(out) :: d(:), kk
    integer :: mm, q, s, j
    logical :: hit

    kk = 0
    mm = min(m, p - 1)
    if (mm < 1) return
    q = 0
    ! s = end position of the candidate match; valid range mm..p-mm keeps
    ! the compared window ids(s-mm+1:s) inside the array.
    do s = p - mm, mm, -1
      hit = .true.
      do j = 1, mm
        if (ids(s - mm + j) /= ids(p - mm + j)) then
          hit = .false.
          exit
        end if
      end do
      if (hit) then
        q = s
        exit
      end if
    end do
    if (q == 0) return
    kk = min(k, p - q)
    do j = 1, kk
      d(j) = ids(q + j)
    end do
  end subroutine lookup_draft

end module fortran_spec_mod
