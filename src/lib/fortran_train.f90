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
  use fortran_kinds_mod, only: wp
  use fortran_adamw_mod
  use fortran_backward_mod
  use fortran_blas_mod
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  implicit none
  private
  public :: dims_t, params_t, state_t, cache_t, temp_t
  public :: forward_save, compute_grads, train_step, init_state, init_temp, free_temp
  public :: wp

  type :: dims_t
    integer :: B, T, V, D, nh, nkv, hd, nl
    real(wp) :: eps
  end type dims_t

  ! All weights, stacked per-layer exactly like gpt_forward.
  type :: params_t
    real(wp), allocatable :: wte(:), lm(:)
    real(wp), allocatable :: q(:), k(:), v(:), p(:), fc(:), p2(:)
  end type params_t

  ! AdamW first/second moments, same shapes as params (fp32 master).
  type :: state_t
    real(wp), allocatable :: wte(:), lm(:)
    real(wp), allocatable :: q(:), k(:), v(:), p(:), fc(:), p2(:)
    real(wp), allocatable :: vwte(:), vlm(:)
    real(wp), allocatable :: vq(:), vk(:), vv(:), vp(:), vfc(:), vp2(:)
  end type state_t

  ! Saved activations, per-layer stacked (layer l at [l*S+1:], S per size).
  ! Plus ef = final residual stream (for the head backward).
  type :: cache_t
    real(wp), allocatable :: e(:), xa(:), q(:), k(:), v(:)
    real(wp), allocatable :: ao(:), e1(:), f(:), ef(:)
    ! qr/kr = POST-RoPE q/k (what causal_attn consumed; attn_bwd must
    ! replay scores from these, not from pre-rope q/k).
    real(wp), allocatable :: qr(:), kr(:)
  end type cache_t

  ! Per-call temp buffers, pre-allocated once and reused forever.
  ! Eliminates ~1GB/step malloc churn. Thread-safe because train_step
  ! runs serially (outer do-loop) and OpenMP only fires inside BLAS.
  ! Per-call temp buffers, pre-allocated once and reused forever.
  ! Eliminates ~1GB/step malloc churn. Thread-safe because train_step
  ! runs serially (outer do-loop) and OpenMP only fires inside BLAS.
  type :: temp_t
    ! forward: POINTERs (val path aliases them, zero copy). Allocated
    ! once in init_temp; pointing at a pointer needs no TARGET anywhere.
    real(wp), pointer :: emd(:) => null(), xn(:) => null()
    real(wp), pointer :: sub(:) => null()
    real(wp), pointer :: qo(:) => null(), ko(:) => null(), vo(:) => null()
    real(wp), pointer :: qrot(:) => null(), krot(:) => null()
    real(wp), pointer :: ao(:) => null()
    real(wp), pointer :: mlpF(:) => null(), lgt(:) => null()   ! mlpF: BT*dff
    real(wp), allocatable :: nl(:)
    ! backward (BT*dff >= BT*DD always since dff=4*DD)
    real(wp), allocatable :: xraw(:), lgt2(:), dlgt(:)
    real(wp), allocatable :: dxn(:), rbuf(:)
    real(wp), allocatable :: dq(:), dk(:), dv(:), dqr(:), dkr(:)
    real(wp), allocatable :: dao(:), dr(:), df(:)
    real(wp), allocatable :: dx1(:), dx2(:), dx3(:), dxa(:)
    real(wp), allocatable :: mx(:), demd(:)   ! mx: BT; demd = d(block_out), BT*dff
    ! BLAS-attention scratch (only touched when attn_blas is on): satt serves
    ! both the forward scores+softmax and the backward dP/dS staging.
    real(wp), allocatable :: satt(:), dPbuf(:), dSbuf(:), dkv(:)
  end type temp_t

contains

  subroutine alloc_like(dst, src)
    real(wp), allocatable, intent(out) :: dst(:)
    real(wp), intent(in) :: src(:)
    allocate(dst(size(src)))
    dst = 0.0_wp
  end subroutine alloc_like

  ! Allocate temp buffers once based on dims. Idempotent.
  subroutine init_temp(G, tmp)
    type(dims_t), intent(in) :: G
    type(temp_t), intent(out) :: tmp
    integer :: BT, DD, hdd, dff, kvd
    BT = G%B * G%T; DD = G%D; hdd = G%nh * G%hd
    dff = 4 * DD; kvd = G%nkv * G%hd
    allocate(tmp%emd(BT*DD), tmp%xn(BT*DD), tmp%sub(BT*DD))
    allocate(tmp%qo(BT*hdd), tmp%ko(BT*kvd), tmp%vo(BT*kvd))
    allocate(tmp%qrot(BT*hdd), tmp%krot(BT*kvd), tmp%ao(BT*DD))
    allocate(tmp%mlpF(BT*dff), tmp%lgt(BT*G%V))
    allocate(tmp%nl(G%B))
    ! compute_grads temps
    allocate(tmp%xraw(BT*DD), tmp%lgt2(BT*G%V), tmp%dlgt(BT*G%V))
    allocate(tmp%dxn(BT*DD), tmp%rbuf(BT*dff))
    allocate(tmp%dq(BT*hdd), tmp%dk(BT*kvd), tmp%dv(BT*kvd))
    allocate(tmp%dqr(BT*hdd), tmp%dkr(BT*kvd))
    allocate(tmp%dao(BT*DD), tmp%dr(BT*dff), tmp%df(BT*dff))
    allocate(tmp%dx1(BT*DD), tmp%dx2(BT*DD), tmp%dx3(BT*DD), tmp%dxa(BT*DD))
    allocate(tmp%mx(BT), tmp%demd(BT*dff))
    ! T*T scratch for the BLAS attention path (3 x T^2 + 2*T*kvd reals; at
    ! T=2048 that is ~50 MB, allocated once and reused every step)
    allocate(tmp%satt(G%T*G%T), tmp%dPbuf(G%T*G%T), tmp%dSbuf(G%T*G%T))
    allocate(tmp%dkv(2*G%T*kvd))
    tmp%satt = 0.0_wp; tmp%dPbuf = 0.0_wp; tmp%dSbuf = 0.0_wp; tmp%dkv = 0.0_wp
    tmp%emd  = 0.0_wp; tmp%xn  = 0.0_wp; tmp%sub = 0.0_wp
    tmp%qo   = 0.0_wp; tmp%ko  = 0.0_wp; tmp%vo  = 0.0_wp
    tmp%qrot = 0.0_wp; tmp%krot= 0.0_wp; tmp%ao  = 0.0_wp
    tmp%mlpF = 0.0_wp; tmp%lgt = 0.0_wp; tmp%nl   = 0.0_wp
  end subroutine init_temp

  subroutine free_temp(tmp)
    type(temp_t), intent(inout) :: tmp
    if (associated(tmp%emd))   deallocate(tmp%emd)
    if (associated(tmp%xn))    deallocate(tmp%xn)
    if (associated(tmp%sub))   deallocate(tmp%sub)
    if (associated(tmp%qo))    deallocate(tmp%qo)
    if (associated(tmp%ko))    deallocate(tmp%ko)
    if (associated(tmp%vo))    deallocate(tmp%vo)
    if (associated(tmp%qrot))  deallocate(tmp%qrot)
    if (associated(tmp%krot))  deallocate(tmp%krot)
    if (associated(tmp%ao))    deallocate(tmp%ao)
    if (associated(tmp%mlpF))  deallocate(tmp%mlpF)
    if (associated(tmp%lgt))   deallocate(tmp%lgt)
    if (allocated(tmp%nl))    deallocate(tmp%nl)
    if (allocated(tmp%xraw))  deallocate(tmp%xraw)
    if (allocated(tmp%lgt2))  deallocate(tmp%lgt2)
    if (allocated(tmp%dlgt))  deallocate(tmp%dlgt)
    if (allocated(tmp%dxn))   deallocate(tmp%dxn)
    if (allocated(tmp%rbuf))  deallocate(tmp%rbuf)
    if (allocated(tmp%dq))    deallocate(tmp%dq)
    if (allocated(tmp%dk))    deallocate(tmp%dk)
    if (allocated(tmp%dv))    deallocate(tmp%dv)
    if (allocated(tmp%dqr))   deallocate(tmp%dqr)
    if (allocated(tmp%dkr))   deallocate(tmp%dkr)
    if (allocated(tmp%dao))   deallocate(tmp%dao)
    if (allocated(tmp%dr))    deallocate(tmp%dr)
    if (allocated(tmp%df))    deallocate(tmp%df)
    if (allocated(tmp%dx1))   deallocate(tmp%dx1)
    if (allocated(tmp%dx2))   deallocate(tmp%dx2)
    if (allocated(tmp%dx3))   deallocate(tmp%dx3)
    if (allocated(tmp%dxa))   deallocate(tmp%dxa)
    if (allocated(tmp%mx))    deallocate(tmp%mx)
    if (allocated(tmp%demd))   deallocate(tmp%demd)
  end subroutine free_temp

  subroutine forward_save(idx, targets, cos, sin, M, G, C, tmp, nll, attn_blas)
    logical, intent(in), optional :: attn_blas
    logical :: useblas
    integer(c_int), intent(in) :: idx(:), targets(:)
    real(wp), intent(in) :: cos(:), sin(:)
    type(params_t), intent(in) :: M
    type(dims_t), intent(in) :: G
    type(cache_t), intent(inout) :: C
    type(temp_t), intent(inout) :: tmp
    real(wp), intent(out) :: nll
    integer :: BT, DD, d2, hdd, dff, ll, jj, it, tg
    integer :: qsz, ksz, psz, fcsz, p2sz
    real(wp) :: sc, mx, sm

    BT = G%B * G%T; DD = G%D; d2 = G%hd / 2; hdd = G%nh * G%hd
    dff = 4 * DD
    qsz = hdd * DD; ksz = G%nkv * G%hd * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    sc = 1.0_wp / real(BT, wp)

    ! cache: allocate once on first call, reuse afterward
    if (.not. allocated(C%e)) then
      allocate(C%e(G%nl*BT*DD), C%xa(G%nl*BT*DD))
      allocate(C%q(G%nl*BT*hdd), C%k(G%nl*BT*G%nkv*G%hd))
      allocate(C%v(G%nl*BT*G%nkv*G%hd), C%ao(G%nl*BT*DD))
      allocate(C%e1(G%nl*BT*DD), C%f(G%nl*BT*dff))
      allocate(C%ef(BT*DD))
      allocate(C%qr(G%nl*BT*hdd), C%kr(G%nl*BT*G%nkv*G%hd))
    end if

    call wte_lookup(idx, M%wte, tmp%emd, G%B, G%T, G%V, DD)
    call rmsnorm0(tmp%emd, tmp%xn, BT, DD, G%eps)
    !$omp parallel do simd
    do jj = 1, BT*DD
      tmp%emd(jj) = tmp%xn(jj)
    end do

    do ll = 0, G%nl - 1
      C%e(ll*BT*DD+1:(ll+1)*BT*DD) = tmp%emd
      call rmsnorm0(tmp%emd, tmp%xn, BT, DD, G%eps)
      C%xa(ll*BT*DD+1:(ll+1)*BT*DD) = tmp%xn
      call linear3d_sgemm(tmp%xn, M%q(ll*qsz+1:), tmp%qo, G%B, G%T, DD, hdd)
      call linear3d_sgemm(tmp%xn, M%k(ll*ksz+1:), tmp%ko, G%B, G%T, DD, G%nkv*G%hd)
      call linear3d_sgemm(tmp%xn, M%v(ll*ksz+1:), tmp%vo, G%B, G%T, DD, G%nkv*G%hd)
      C%q(ll*BT*hdd+1:(ll+1)*BT*hdd) = tmp%qo
      C%k(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = tmp%ko
      C%v(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = tmp%vo
      call rope_4d(tmp%qo, cos, sin, tmp%qrot, G%B, G%T, G%nh, G%hd)
      call rope_4d(tmp%ko, cos, sin, tmp%krot, G%B, G%T, G%nkv, G%hd)
      C%qr(ll*BT*hdd+1:(ll+1)*BT*hdd) = tmp%qrot
      C%kr(ll*BT*G%nkv*G%hd+1:(ll+1)*BT*G%nkv*G%hd) = tmp%krot
      if (present(attn_blas)) then
        useblas = attn_blas
      else
        useblas = .false.
      end if
      if (useblas) then
        call attn_sgemm(tmp%qrot, tmp%krot, tmp%vo, tmp%ao, G%B, G%T, &
            G%nh, G%nkv, G%hd, tmp%satt)
      else
        call causal_attn(tmp%qrot, tmp%krot, tmp%vo, tmp%ao, G%B, G%T, &
            G%nh, G%nkv, G%hd)
      end if
      C%ao(ll*BT*DD+1:(ll+1)*BT*DD) = tmp%ao
      call linear3d_sgemm(tmp%ao, M%p(ll*psz+1:), tmp%sub, G%B, G%T, DD, DD)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%emd(jj) = tmp%emd(jj) + tmp%sub(jj)
      end do
      C%e1(ll*BT*DD+1:(ll+1)*BT*DD) = tmp%emd
      call rmsnorm0(tmp%emd, tmp%xn, BT, DD, G%eps)
      call linear3d_sgemm(tmp%xn, M%fc(ll*fcsz+1:), tmp%mlpF, G%B, G%T, DD, dff)
      C%f(ll*BT*dff+1:(ll+1)*BT*dff) = tmp%mlpF
      call relu2(tmp%mlpF, BT*dff)
      call linear3d_sgemm(tmp%mlpF, M%p2(ll*p2sz+1:), tmp%sub, G%B, G%T, dff, DD)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%emd(jj) = tmp%emd(jj) + tmp%sub(jj)
      end do
    end do

    C%ef = tmp%emd
    call rmsnorm0(tmp%emd, tmp%xn, BT, DD, G%eps)
    call linear3d_sgemm(tmp%xn, M%lm, tmp%lgt, G%B, G%T, DD, G%V)
    ! mean NLL over all positions (no mask in v1; drivers mask outside)
    nll = 0.0_wp
    !$omp parallel do private(tg, jj, mx, sm) reduction(+:nll)
    do it = 1, BT
      tg = targets(it) + 1
      mx = tmp%lgt((it-1)*G%V+1)
      do jj = 2, G%V
        if (tmp%lgt((it-1)*G%V+jj) > mx) mx = tmp%lgt((it-1)*G%V+jj)
      end do
      sm = 0.0_wp
      do jj = 1, G%V
        sm = sm + exp(tmp%lgt((it-1)*G%V+jj) - mx)
      end do
      nll = nll + ((mx + log(sm)) - tmp%lgt((it-1)*G%V+tg)) * sc
    end do
  end subroutine forward_save

  ! Full reverse pass over saved activations.
  ! Per block (reversed): residual splits, MLP branch, attention branch.
  ! Norm inputs are recomputed from saved pre-norm values (exact, cheap);
  ! r = relu(f) is recomputed from saved f. dk/dv zeroed per layer
  ! (attn_bwd accumulates inout). GR arrays zeroed up front.
  subroutine compute_grads(idx, targets, cos, sin, M, G, C, GR, tmp, nll, &
      attn_blas)
    logical, intent(in), optional :: attn_blas
    logical :: useblas
    integer(c_int), intent(in) :: idx(:), targets(:)
    real(wp), intent(in) :: cos(:), sin(:)
    type(params_t), intent(in) :: M
    type(dims_t), intent(in) :: G
    type(cache_t), intent(in) :: C
    type(params_t), intent(inout) :: GR
    type(temp_t), intent(inout) :: tmp
    real(wp), intent(out) :: nll
    integer :: BT, DD, hdd, dff, ll, jj, it, j2, tg
    integer :: qsz, ksz, psz, fcsz, p2sz, kvd
    real(wp) :: sc, ssum

    BT = G%B * G%T; DD = G%D; hdd = G%nh * G%hd
    dff = 4 * DD; kvd = G%nkv * G%hd
    qsz = hdd * DD; ksz = kvd * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    sc = 1.0_wp / real(BT, wp)

    GR%wte = 0.0_wp; GR%lm = 0.0_wp
    GR%q = 0.0_wp; GR%k = 0.0_wp; GR%v = 0.0_wp
    GR%p = 0.0_wp; GR%fc = 0.0_wp; GR%p2 = 0.0_wp

    ! ---- head: dlogits, dwlm, demd ----
    call rmsnorm0(C%ef, tmp%xn, BT, DD, G%eps)
    call linear3d_sgemm(tmp%xn, M%lm, tmp%lgt2, G%B, G%T, DD, G%V)
    nll = 0.0_wp
    !$omp parallel do private(tg, j2, ssum) reduction(+:nll)
    do it = 1, BT
      tg = targets(it) + 1
      tmp%mx(it) = tmp%lgt2((it-1)*G%V+1)
      do j2 = 2, G%V
        if (tmp%lgt2((it-1)*G%V+j2) > tmp%mx(it)) tmp%mx(it) = tmp%lgt2((it-1)*G%V+j2)
      end do
      ssum = 0.0_wp
      do j2 = 1, G%V
        ssum = ssum + exp(tmp%lgt2((it-1)*G%V+j2) - tmp%mx(it))
      end do
      nll = nll + ((tmp%mx(it) + log(ssum)) - tmp%lgt2((it-1)*G%V+tg)) * sc
      do j2 = 1, G%V
        tmp%dlgt((it-1)*G%V+j2) = exp(tmp%lgt2((it-1)*G%V+j2) - tmp%mx(it)) / ssum * sc
      end do
      tmp%dlgt((it-1)*G%V+tg) = tmp%dlgt((it-1)*G%V+tg) - sc
    end do
    call linear3d_bwd_sgemm(tmp%dlgt, tmp%xn, M%lm, tmp%dxn, GR%lm, G%B, G%T, DD, G%V)
    call rmsnorm0_bwd(tmp%dxn, C%ef, tmp%demd, BT, DD, G%eps)

    ! ---- blocks reversed ----
    ! demd = d(block output). MLP branch: dmo = demd, de1 = demd.
    ! Then attn branch: dao = de1 (with mlp path), de += de1 (residual).
    do ll = G%nl - 1, 0, -1
      tmp%rbuf = C%f(ll*BT*dff+1:(ll+1)*BT*dff)
      call relu2(tmp%rbuf, BT*dff)   ! r = relu(f), x-input of proj2 bwd
      call linear3d_bwd_sgemm(tmp%demd, tmp%rbuf, M%p2(ll*p2sz+1:), tmp%dr, &
          GR%p2(ll*p2sz+1:), G%B, G%T, dff, DD)
      call relu2_bwd(tmp%dr, C%f(ll*BT*dff+1:), tmp%df, BT*dff)
      call rmsnorm0(C%e1(ll*BT*DD+1:), tmp%xn, BT, DD, G%eps)
      call linear3d_bwd_sgemm(tmp%df, tmp%xn, M%fc(ll*fcsz+1:), tmp%dx1, GR%fc(ll*fcsz+1:), &
          G%B, G%T, DD, dff)
      call rmsnorm0_bwd(tmp%dx1, C%e1(ll*BT*DD+1:), tmp%dx2, BT, DD, G%eps)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%demd(jj) = tmp%demd(jj) + tmp%dx2(jj)   ! de1 = demd + mlp path
      end do
      call linear3d_bwd_sgemm(tmp%demd, C%ao(ll*BT*DD+1:), M%p(ll*psz+1:), tmp%dao, &
          GR%p(ll*psz+1:), G%B, G%T, DD, DD)
      tmp%dk = 0.0_wp; tmp%dv = 0.0_wp
      if (present(attn_blas)) then
        useblas = attn_blas
      else
        useblas = .false.
      end if
      if (useblas) then
        call attn_bwd_sgemm(tmp%dao, C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%v(ll*BT*kvd+1:), tmp%dq, tmp%dk, tmp%dv, &
            G%B, G%T, G%nh, G%nkv, G%hd, tmp%satt, tmp%dPbuf, tmp%dSbuf, &
            tmp%dkv)
      else
        call attn_bwd(tmp%dao, C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%v(ll*BT*kvd+1:), tmp%dq, tmp%dk, tmp%dv, &
            G%B, G%T, G%nh, G%nkv, G%hd)
      end if
      call rope_4d_bwd(tmp%dq, cos, sin, tmp%dqr, G%B, G%T, G%nh, G%hd)
      call rope_4d_bwd(tmp%dk, cos, sin, tmp%dkr, G%B, G%T, G%nkv, G%hd)
      call linear3d_bwd_sgemm(tmp%dqr, C%xa(ll*BT*DD+1:), M%q(ll*qsz+1:), tmp%dx1, &
          GR%q(ll*qsz+1:), G%B, G%T, DD, hdd)
      call linear3d_bwd_sgemm(tmp%dkr, C%xa(ll*BT*DD+1:), M%k(ll*ksz+1:), tmp%dx2, &
          GR%k(ll*ksz+1:), G%B, G%T, DD, kvd)
      call linear3d_bwd_sgemm(tmp%dv, C%xa(ll*BT*DD+1:), M%v(ll*ksz+1:), tmp%dx3, &
          GR%v(ll*ksz+1:), G%B, G%T, DD, kvd)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%dxa(jj) = tmp%dx1(jj) + tmp%dx2(jj) + tmp%dx3(jj)   ! d(xa)
      end do
      call rmsnorm0_bwd(tmp%dxa, C%e(ll*BT*DD+1:), tmp%dx1, BT, DD, G%eps)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%demd(jj) = tmp%demd(jj) + tmp%dx1(jj)   ! de += attn path
      end do
    end do

    ! ---- embeddings: raw lookup recomputed, then norm + scatter ----
    call wte_lookup(idx, M%wte, tmp%xraw, G%B, G%T, G%V, DD)
    call rmsnorm0_bwd(tmp%demd, tmp%xraw, tmp%dxn, BT, DD, G%eps)
    call wte_bwd(idx, tmp%dxn, GR%wte, G%B, G%T, G%V, DD)
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
    real(wp), intent(inout) :: p(:), m(:), v(:)
    real(wp), intent(in) :: g(:)
    real(wp), intent(in) :: lr, b1, b2, beps, wd
    integer, intent(in) :: t
    call adamw_step(p, g, m, v, size(p), lr, b1, b2, beps, wd, t)
  end subroutine apply_group

  ! One full training step: forward + backward + AdamW update.
  subroutine train_step(idx, targets, cos, sin, M, S, G, GR, C, tmp, &
      nll, tstep, lr, b1, b2, beps, wd, attn_blas)
    logical, intent(in), optional :: attn_blas
    logical :: useblas
    integer(c_int), intent(in) :: idx(:), targets(:)
    real(wp), intent(in) :: cos(:), sin(:)
    type(params_t), intent(inout) :: M
    type(state_t), intent(inout) :: S
    type(dims_t), intent(in) :: G
    type(params_t), intent(inout) :: GR
    type(cache_t), intent(inout) :: C
    type(temp_t), intent(inout) :: tmp
    real(wp), intent(out) :: nll
    integer, intent(in) :: tstep
    real(wp), intent(in) :: lr, b1, b2, beps, wd
    useblas = .false.
    if (present(attn_blas)) useblas = attn_blas
    call forward_save(idx, targets, cos, sin, M, G, C, tmp, nll, useblas)
    call compute_grads(idx, targets, cos, sin, M, G, C, GR, tmp, nll, useblas)
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
