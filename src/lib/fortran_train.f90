! lib/fortran_train.f90 — full training step (hyp_34ea7c endgame).
!
! Forward saves per-layer activations (e, xn_a, q, k, v, ao, e1, f);
! backward runs them in reverse into grads; apply_adamw updates.
! Layouts mirror gpt_forward exactly (same kernels, same order).
! compute_grads is separated from apply_adamw so finite differences
! can verify grads directly (see test_full_step).
! Memory note: v1 saves everything; production would checkpoint
! (recompute norms/linears/ropes from e) — same numerics either way.

module fortran_train_mod
  use iso_c_binding
  use fortran_adamw_mod
  use fortran_backward_mod
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none
  private
  public :: dims_t, params_t, state_t, cache_t
  public :: forward_save, compute_grads, train_step, init_state

  type :: dims_t
    integer :: B, T, V, D, nh, nkv, hd, nl
    real(c_float) :: eps
  end type dims_t

  ! All weights, stacked per-layer exactly like gpt_forward.
  type :: params_t
    real(c_float), allocatable :: wte(:), lm(:)
    real(c_float), allocatable :: q(:), k(:), v(:), p(:), fc(:), p2(:)
  end type params_t

  ! AdamW first/second moments, same shapes as params (fp32 master).
  type :: state_t
    real(c_float), allocatable :: wte(:), lm(:)
    real(c_float), allocatable :: q(:), k(:), v(:), p(:), fc(:), p2(:)
    real(c_float), allocatable :: vwte(:), vlm(:)
    real(c_float), allocatable :: vq(:), vk(:), vv(:), vp(:), vfc(:), vp2(:)
  end type state_t

  ! Saved activations, per-layer stacked (layer l at [l*S+1:], S per size).
  ! Plus ef = final residual stream (for the head backward).
  type :: cache_t
    real(c_float), allocatable :: e(:), xa(:), q(:), k(:), v(:)
    real(c_float), allocatable :: ao(:), e1(:), f(:), ef(:)
    ! qr/kr = POST-RoPE q/k (what causal_attn consumed; attn_bwd must
    ! replay scores from these, not from pre-rope q/k).
    real(c_float), allocatable :: qr(:), kr(:)
  end type cache_t

contains

  subroutine alloc_like(dst, src)
    real(c_float), allocatable, intent(out) :: dst(:)
    real(c_float), intent(in) :: src(:)
    allocate(dst(size(src)))
    dst = 0.0_c_float
  end subroutine alloc_like

  subroutine forward_save(idx, targets, cos, sin, M, G, C, nll)
    integer(c_int), intent(in) :: idx(*), targets(*)
    real(c_float), intent(in) :: cos(*), sin(*)
    type(params_t), intent(in) :: M
    type(dims_t), intent(in) :: G
    type(cache_t), intent(inout) :: C
    real(c_float), intent(out) :: nll
    integer :: BT, DD, d2, hdd, dff, ll, jj, it
    integer :: qsz, ksz, psz, fcsz, p2sz
    real(c_float), allocatable :: emd(:), xn(:), sub(:), qo(:), ko(:), vo(:)
    real(c_float), allocatable :: qrot(:), krot(:), ao(:), mlpd(:), lgt(:)
    real(c_float), allocatable :: nl(:)
    real(c_float) :: sc

    BT = G%B * G%T; DD = G%D; d2 = G%hd / 2; hdd = G%nh * G%hd
    dff = 4 * DD
    qsz = hdd * DD; ksz = G%nkv * G%hd * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    sc = 1.0_c_float / real(BT)

    allocate(emd(BT*DD), xn(BT*DD), sub(BT*DD))
    allocate(qo(BT*hdd), ko(BT*G%nkv*G%hd), vo(BT*G%nkv*G%hd))
    allocate(qrot(BT*hdd), krot(BT*G%nkv*G%hd), ao(BT*DD), mlpd(BT*dff))
    allocate(lgt(BT*G%V), nl(BT))
    ! cache persists across calls (FD perturbs weights hundreds of times);
    ! free the previous one first (all-or-nothing allocation invariant).
    if (allocated(C%e)) deallocate(C%e, C%xa, C%q, C%k, C%v, C%ao, &
        C%e1, C%f, C%ef, C%qr, C%kr)
    allocate(C%e(G%nl*BT*DD), C%xa(G%nl*BT*DD))
    allocate(C%q(G%nl*BT*hdd), C%k(G%nl*BT*G%nkv*G%hd))
    allocate(C%v(G%nl*BT*G%nkv*G%hd), C%ao(G%nl*BT*DD))
    allocate(C%e1(G%nl*BT*DD), C%f(G%nl*BT*dff))
    allocate(C%ef(BT*DD))
    allocate(C%qr(G%nl*BT*hdd), C%kr(G%nl*BT*G%nkv*G%hd))

    call wte_lookup(idx, M%wte, emd, G%B, G%T, G%V, DD)
    call rmsnorm0(emd, xn, BT, DD, G%eps)
    do jj = 1, BT*DD
      emd(jj) = xn(jj)
    end do

    do ll = 0, G%nl - 1
      C%e(ll*BT*DD+1:(ll+1)*BT*DD) = emd
      call rmsnorm0(emd, xn, BT, DD, G%eps)
      C%xa(ll*BT*DD+1:(ll+1)*BT*DD) = xn
      call linear3d(xn, M%q(ll*qsz+1:), qo, G%B, G%T, DD, hdd)
      call linear3d(xn, M%k(ll*ksz+1:), ko, G%B, G%T, DD, G%nkv*G%hd)
      call linear3d(xn, M%v(ll*ksz+1:), vo, G%B, G%T, DD, G%nkv*G%hd)
      C%q(ll*BT*hdd+1:(ll+1)*BT*hdd) = qo
      C%k(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = ko
      C%v(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = vo
      call rope_4d(qo, cos, sin, qrot, G%B, G%T, G%nh, G%hd)
      call rope_4d(ko, cos, sin, krot, G%B, G%T, G%nkv, G%hd)
      C%qr(ll*BT*hdd+1:(ll+1)*BT*hdd) = qrot
      C%kr(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = krot
      call causal_attn(qrot, krot, vo, ao, G%B, G%T, G%nh, G%nkv, G%hd)
      C%ao(ll*BT*DD+1:(ll+1)*BT*DD) = ao
      call linear3d(ao, M%p(ll*psz+1:), sub, G%B, G%T, DD, DD)
      do jj = 1, BT*DD
        emd(jj) = emd(jj) + sub(jj)
      end do
      C%e1(ll*BT*DD+1:(ll+1)*BT*DD) = emd
      call rmsnorm0(emd, xn, BT, DD, G%eps)
      call linear3d(xn, M%fc(ll*fcsz+1:), mlpd, G%B, G%T, DD, dff)
      C%f(ll*BT*dff+1:(ll+1)*BT*dff) = mlpd
      call relu2(mlpd, BT*dff)
      call linear3d(mlpd, M%p2(ll*p2sz+1:), sub, G%B, G%T, dff, DD)
      do jj = 1, BT*DD
        emd(jj) = emd(jj) + sub(jj)
      end do
    end do

    C%ef = emd
    call rmsnorm0(emd, xn, BT, DD, G%eps)
    call linear3d(xn, M%lm, lgt, G%B, G%T, DD, G%V)
    ! mean NLL over all positions (no mask in v1; drivers mask outside)
    nll = 0.0_c_float
    do it = 1, BT
      block
        integer :: j2, tg
        real(c_float) :: mx, sm
        tg = targets(it) + 1
        mx = lgt((it-1)*G%V+1)
        do j2 = 2, G%V
          if (lgt((it-1)*G%V+j2) > mx) mx = lgt((it-1)*G%V+j2)
        end do
        sm = 0.0_c_float
        do j2 = 1, G%V
          sm = sm + exp(lgt((it-1)*G%V+j2) - mx)
        end do
        nl(it) = (mx + log(sm)) - lgt((it-1)*G%V+tg)
        nll = nll + nl(it) * sc
      end block
    end do
    deallocate(emd, xn, sub, qo, ko, vo, qrot, krot, ao, mlpd, lgt, nl)
  end subroutine forward_save

  ! Full reverse pass over saved activations.
  ! Per block (reversed): residual splits, MLP branch, attention branch.
  ! Norm inputs are recomputed from saved pre-norm values (exact, cheap);
  ! r = relu(f) is recomputed from saved f. dk/dv zeroed per layer
  ! (attn_bwd accumulates inout). GR arrays zeroed up front.
  subroutine compute_grads(idx, targets, cos, sin, M, G, C, GR, nll)
    integer(c_int), intent(in) :: idx(*), targets(*)
    real(c_float), intent(in) :: cos(*), sin(*)
    type(params_t), intent(in) :: M
    type(dims_t), intent(in) :: G
    type(cache_t), intent(in) :: C
    type(params_t), intent(inout) :: GR
    real(c_float), intent(out) :: nll
    integer :: BT, DD, hdd, dff, ll, jj, it, j2, tg
    integer :: qsz, ksz, psz, fcsz, p2sz, kvd
    real(c_float) :: sc, ssum
    real(c_float), allocatable :: xn(:), demd(:), xraw(:)
    real(c_float), allocatable :: lgt(:), dlgt(:), dxn(:), rbuf(:)
    real(c_float), allocatable :: dq(:), dk(:), dv(:), dqr(:), dkr(:)
    real(c_float), allocatable :: dao(:), dmo(:), dr(:), df(:), dx1(:)
    real(c_float), allocatable :: dx2(:), dx3(:), dxa(:), mx(:)

    BT = G%B * G%T; DD = G%D; hdd = G%nh * G%hd
    dff = 4 * DD; kvd = G%nkv * G%hd
    qsz = hdd * DD; ksz = kvd * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    sc = 1.0_c_float / real(BT)

    GR%wte = 0.0_c_float; GR%lm = 0.0_c_float
    GR%q = 0.0_c_float; GR%k = 0.0_c_float; GR%v = 0.0_c_float
    GR%p = 0.0_c_float; GR%fc = 0.0_c_float; GR%p2 = 0.0_c_float

    allocate(xn(BT*DD), demd(BT*DD), xraw(BT*DD))
    allocate(lgt(BT*G%V), dlgt(BT*G%V), dxn(BT*DD), rbuf(BT*dff))
    allocate(dq(BT*hdd), dk(BT*kvd), dv(BT*kvd))
    allocate(dqr(BT*hdd), dkr(BT*kvd))
    allocate(dao(BT*DD), dmo(BT*DD), dr(BT*dff), df(BT*dff))
    allocate(dx1(BT*DD), dx2(BT*DD), dx3(BT*DD), dxa(BT*DD), mx(BT))

    ! ---- head: dlogits, dwlm, demd ----
    call rmsnorm0(C%ef, xn, BT, DD, G%eps)
    call linear3d(xn, M%lm, lgt, G%B, G%T, DD, G%V)
    nll = 0.0_c_float
    do it = 1, BT
      tg = targets(it) + 1
      mx(it) = lgt((it-1)*G%V+1)
      do j2 = 2, G%V
        if (lgt((it-1)*G%V+j2) > mx(it)) mx(it) = lgt((it-1)*G%V+j2)
      end do
      ssum = 0.0_c_float
      do j2 = 1, G%V
        ssum = ssum + exp(lgt((it-1)*G%V+j2) - mx(it))
      end do
      nll = nll + ((mx(it) + log(ssum)) - lgt((it-1)*G%V+tg)) * sc
      do j2 = 1, G%V
        dlgt((it-1)*G%V+j2) = exp(lgt((it-1)*G%V+j2) - mx(it)) / ssum * sc
      end do
      dlgt((it-1)*G%V+tg) = dlgt((it-1)*G%V+tg) - sc
    end do
    call linear3d_bwd(dlgt, xn, M%lm, dxn, GR%lm, G%B, G%T, DD, G%V)
    call rmsnorm0_bwd(dxn, C%ef, demd, BT, DD, G%eps)

    ! ---- blocks reversed ----
    ! demd = d(block output). MLP branch: dmo = demd, de1 = demd.
    ! Then attn branch: dao = de1 (with mlp path), de += de1 (residual).
    do ll = G%nl - 1, 0, -1
      rbuf = C%f(ll*BT*dff+1:(ll+1)*BT*dff)
      call relu2(rbuf, BT*dff)   ! r = relu(f), x-input of proj2 bwd
      call linear3d_bwd(demd, rbuf, M%p2(ll*p2sz+1:), dr, &
          GR%p2(ll*p2sz+1:), G%B, G%T, dff, DD)
      call relu2_bwd(dr, C%f(ll*BT*dff+1:), df, BT*dff)
      call rmsnorm0(C%e1(ll*BT*DD+1:), xn, BT, DD, G%eps)
      call linear3d_bwd(df, xn, M%fc(ll*fcsz+1:), dx1, GR%fc(ll*fcsz+1:), &
          G%B, G%T, DD, dff)
      call rmsnorm0_bwd(dx1, C%e1(ll*BT*DD+1:), dx2, BT, DD, G%eps)
      do jj = 1, BT*DD
        demd(jj) = demd(jj) + dx2(jj)   ! de1 = demd + mlp path
      end do
      call linear3d_bwd(demd, C%ao(ll*BT*DD+1:), M%p(ll*psz+1:), dao, &
          GR%p(ll*psz+1:), G%B, G%T, DD, DD)
      dk = 0.0_c_float; dv = 0.0_c_float
      call attn_bwd(dao, C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
          C%v(ll*BT*kvd+1:), dq, dk, dv, &
          G%B, G%T, G%nh, G%nkv, G%hd)
      call rope_4d_bwd(dq, cos, sin, dqr, G%B, G%T, G%nh, G%hd)
      call rope_4d_bwd(dk, cos, sin, dkr, G%B, G%T, G%nkv, G%hd)
      call linear3d_bwd(dqr, C%xa(ll*BT*DD+1:), M%q(ll*qsz+1:), dx1, &
          GR%q(ll*qsz+1:), G%B, G%T, DD, hdd)
      call linear3d_bwd(dkr, C%xa(ll*BT*DD+1:), M%k(ll*ksz+1:), dx2, &
          GR%k(ll*ksz+1:), G%B, G%T, DD, kvd)
      call linear3d_bwd(dv, C%xa(ll*BT*DD+1:), M%v(ll*ksz+1:), dx3, &
          GR%v(ll*ksz+1:), G%B, G%T, DD, kvd)
      do jj = 1, BT*DD
        dxa(jj) = dx1(jj) + dx2(jj) + dx3(jj)   ! d(xa)
      end do
      call rmsnorm0_bwd(dxa, C%e(ll*BT*DD+1:), dx1, BT, DD, G%eps)
      do jj = 1, BT*DD
        demd(jj) = demd(jj) + dx1(jj)   ! de += attn path
      end do
    end do

    ! ---- embeddings: raw lookup recomputed, then norm + scatter ----
    call wte_lookup(idx, M%wte, xraw, G%B, G%T, G%V, DD)
    call rmsnorm0_bwd(demd, xraw, dxn, BT, DD, G%eps)
    call wte_bwd(idx, dxn, GR%wte, G%B, G%T, G%V, DD)
    deallocate(xn, demd, xraw, lgt, dlgt, dxn, rbuf)
    deallocate(dq, dk, dv, dqr, dkr, dao, dmo, dr, df)
    deallocate(dx1, dx2, dx3, dxa, mx)
  end subroutine compute_grads

  ! Allocate + zero AdamW states matching M's shapes.
  subroutine init_state(M, S)
    type(params_t), intent(in) :: M
    type(state_t), intent(out) :: S
    call alloc_like(S%wte, M%wte); call alloc_like(S%lm, M%lm)
    call alloc_like(S%q, M%q); call alloc_like(S%k, M%k)
    call alloc_like(S%v, M%v); call alloc_like(S%p, M%p)
    call alloc_like(S%fc, M%fc); call alloc_like(S%p2, M%p2)
    call alloc_like(S%vwte, M%wte); call alloc_like(S%vlm, M%lm)
    call alloc_like(S%vq, M%q); call alloc_like(S%vk, M%k)
    call alloc_like(S%vv, M%v); call alloc_like(S%vp, M%p)
    call alloc_like(S%vfc, M%fc); call alloc_like(S%vp2, M%p2)
  end subroutine init_state

  subroutine apply_group(p, g, m, v, lr, b1, b2, beps, wd, t)
    real(c_float), intent(inout) :: p(:), m(:), v(:)
    real(c_float), intent(in) :: g(:)
    real(c_float), intent(in) :: lr, b1, b2, beps, wd
    integer, intent(in) :: t
    call adamw_step(p, g, m, v, size(p), lr, b1, b2, beps, wd, t)
  end subroutine apply_group

  ! One full training step: forward + backward + AdamW update.
  subroutine train_step(idx, targets, cos, sin, M, S, G, GR, C, &
      nll, tstep, lr, b1, b2, beps, wd)
    integer(c_int), intent(in) :: idx(*), targets(*)
    real(c_float), intent(in) :: cos(*), sin(*)
    type(params_t), intent(inout) :: M
    type(state_t), intent(inout) :: S
    type(dims_t), intent(in) :: G
    type(params_t), intent(inout) :: GR
    type(cache_t), intent(inout) :: C
    real(c_float), intent(out) :: nll
    integer, intent(in) :: tstep
    real(c_float), intent(in) :: lr, b1, b2, beps, wd
    call forward_save(idx, targets, cos, sin, M, G, C, nll)
    call compute_grads(idx, targets, cos, sin, M, G, C, GR, nll)
    call apply_group(M%wte, GR%wte, S%wte, S%vwte, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%lm, GR%lm, S%lm, S%vlm, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%q, GR%q, S%q, S%vq, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%k, GR%k, S%k, S%vk, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%v, GR%v, S%v, S%vv, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%p, GR%p, S%p, S%vp, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%fc, GR%fc, S%fc, S%vfc, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%p2, GR%p2, S%p2, S%vp2, lr, b1, b2, beps, wd, tstep)
  end subroutine train_step

end module fortran_train_mod
