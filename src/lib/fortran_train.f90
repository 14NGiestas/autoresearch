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
  use fortran_muon_mod, only: muon_update_mat
  use fortran_qkhop_mod, only: qkhop_sgemm, qkhop_bwd_sgemm, qkhop_ph_fwd, qkhop_ph_bwd
  implicit none
  private

  ! Muon momentum for hybrid runs (Muon on 2D matrices, AdamW elsewhere).
  ! Fixed per the canonical recipe; the LR (different geometry!) is runtime.
  real(wp), parameter :: MUON_BETA = 0.95_wp
  public :: dims_t, params_t, state_t, cache_t, temp_t
  public :: forward_save, compute_grads, train_step, init_state, init_temp, free_temp
  public :: apply_update
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
  ! mq..mp2: Muon momentum buffers (one per 2D matrix group, no variance).
  type :: state_t
    real(wp), allocatable :: wte(:), lm(:)
    real(wp), allocatable :: q(:), k(:), v(:), p(:), fc(:), p2(:)
    real(wp), allocatable :: vwte(:), vlm(:)
    real(wp), allocatable :: vq(:), vk(:), vv(:), vp(:), vfc(:), vp2(:)
    real(wp), allocatable :: mq(:), mk(:), mv(:), mp(:), mfc(:), mp2(:)
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
    real(wp), allocatable :: sqk(:), sh1(:), sdh1(:), sdsm(:)
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
    ! sdsm e' o buffer do softmax do caminho qkhop/ph: B*T*T floats, NAO
    ! B*T*T*T. Estava dimensionado com um fator G%T a mais (1.073.741.824
    ! floats = 4,29 GB em T=1024), o que respondia por ~4 GB FIXOS de RSS em
    ! qualquer tamanho de modelo (medido: d96 4,4 GB e d216 5,3 GB, e o job 110
    ! mostrou que o RSS e' plano, nao vazamento). Corrigido em 2026-09-18:
    ! d96 4,4 GB -> ~0,15 GB. O sqk ao lado ja' usa B*nh*T*T.
    ! sqk: scores por cabeca (B*nh*T*T, --attn qkhop).
    ! sh1/sdh1/sdsm: workspace do caminho MEAN (qkhop_bwd* exige dimensao
    ! explicita: h1w/dh1w = B*T*D, dSmw = B*T*T). O sdsm estava dimensionado
    ! como BT*G%T*G%T = 1.073.741.824 floats = 4,29 GB em T=1024 por um G%T a
    ! mais -- era ele o custo FIXO de ~4 GB de RSS medido nos jobs 110/120
    ! (d96 4,4 GB, d216 5,3 GB, RSS plano no tempo, logo nao era vazamento).
    allocate(tmp%sqk(G%B*G%nh*G%T*G%T), tmp%sh1(BT*DD), tmp%sdh1(BT*DD))
    allocate(tmp%sdsm(BT*G%T))          ! = B*T*T = 4,2 MB em T=1024
    tmp%sqk = 0.0_wp; tmp%sh1 = 0.0_wp; tmp%sdh1 = 0.0_wp; tmp%sdsm = 0.0_wp
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
    if (allocated(tmp%sqk))  deallocate(tmp%sqk)
    if (allocated(tmp%sh1))  deallocate(tmp%sh1)
    if (allocated(tmp%sdh1)) deallocate(tmp%sdh1)
    if (allocated(tmp%sdsm)) deallocate(tmp%sdsm)
    if (allocated(tmp%mx))    deallocate(tmp%mx)
    if (allocated(tmp%demd))   deallocate(tmp%demd)
  end subroutine free_temp

  subroutine forward_save(idx, targets, cos, sin, M, G, C, tmp, nll, attn_blas, attn_qk, attn_qkph)
    logical, intent(in), optional :: attn_blas, attn_qk, attn_qkph
    logical :: useblas, useqk, useqkph
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

    call wte_lookup(idx, M%wte, tmp%emd, G%B, G%T, DD)
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
      useqk = .false.
      if (present(attn_qk)) useqk = attn_qk
      useqkph = .false.
      if (present(attn_qkph)) useqkph = attn_qkph
      if (useqkph .and. G%nh*G%hd /= G%D) then
        print '(A)', "qkhop-ph exige hdd==D"
        call exit(1)
      end if
      if (useqkph) then
        call qkhop_ph_fwd(tmp%qrot, tmp%krot, tmp%xn, tmp%ao, tmp%sqk, tmp%sh1, G%B, G%T, &
            G%nh, G%nkv, G%hd)
      else if (useqk) then
        call qkhop_sgemm(tmp%qrot, tmp%krot, tmp%xn, tmp%ao, tmp%sqk, tmp%sh1, G%B, G%T, &
            G%nh, G%nkv, G%hd)
      else if (useblas) then
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
      attn_blas, attn_qk, attn_qkph)
    logical, intent(in), optional :: attn_blas, attn_qk, attn_qkph
    logical :: useblas, useqk, useqkph
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
      useqk = .false.
      if (present(attn_qk)) useqk = attn_qk
      useqkph = .false.
      if (present(attn_qkph)) useqkph = attn_qkph
      if (useqkph) then
        call qkhop_ph_fwd(C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%xa(ll*BT*DD+1:), tmp%ao, tmp%sqk, tmp%sh1, G%B, G%T, &
            G%nh, G%nkv, G%hd)
        call qkhop_ph_bwd(tmp%dao, C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%xa(ll*BT*DD+1:), tmp%sqk, tmp%dr(1:BT*DD), tmp%dq, tmp%dk, &
            G%B, G%T, G%nh, G%nkv, G%hd)
      else if (useqk) then
        call qkhop_sgemm(C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%xa(ll*BT*DD+1:), tmp%ao, tmp%sqk, tmp%sh1, G%B, G%T, &
            G%nh, G%nkv, G%hd)
        call qkhop_bwd_sgemm(tmp%dao, C%qr(ll*BT*hdd+1:), C%kr(ll*BT*kvd+1:), &
            C%xa(ll*BT*DD+1:), tmp%sqk, tmp%dr(1:BT*DD), tmp%dq, tmp%dk, &
            tmp%sh1, tmp%sdh1, tmp%sdsm, G%B, G%T, G%nh, G%nkv, G%hd)
      else if (useblas) then
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
        if (useqk .or. useqkph) then
          tmp%dxa(jj) = tmp%dx1(jj) + tmp%dx2(jj) + tmp%dr(jj)   ! d(xa)+direto
        else
          tmp%dxa(jj) = tmp%dx1(jj) + tmp%dx2(jj) + tmp%dx3(jj)   ! d(xa)
        end if
      end do
      call rmsnorm0_bwd(tmp%dxa, C%e(ll*BT*DD+1:), tmp%dx1, BT, DD, G%eps)
      !$omp parallel do simd
      do jj = 1, BT*DD
        tmp%demd(jj) = tmp%demd(jj) + tmp%dx1(jj)   ! de += attn path
      end do
    end do

    ! ---- embeddings: raw lookup recomputed, then norm + scatter ----
    call wte_lookup(idx, M%wte, tmp%xraw, G%B, G%T, DD)
    call rmsnorm0_bwd(tmp%demd, tmp%xraw, tmp%dxn, BT, DD, G%eps)
    call wte_bwd(idx, tmp%dxn, GR%wte, G%B, G%T, DD)
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
    call alloc_like(S%mq, M%q); call alloc_like(S%mk, M%k)
    call alloc_like(S%mv, M%v); call alloc_like(S%mp, M%p)
    call alloc_like(S%mfc, M%fc); call alloc_like(S%mp2, M%p2)
  end subroutine init_state

  subroutine apply_group(p, g, m, v, lr, b1, b2, beps, wd, t)
    real(wp), intent(inout) :: p(:), m(:), v(:)
    real(wp), intent(in) :: g(:)
    real(wp), intent(in) :: lr, b1, b2, beps, wd
    integer, intent(in) :: t
    call adamw_step(p, g, m, v, size(p), lr, b1, b2, beps, wd, t)
  end subroutine apply_group

  ! Muon update for one CONCATENATED multi-layer group (all layers share
  ! the same (OF,IF) geometry): per-layer 2D views + muon_update_mat.
  ! Flat buffers are row-major logical (OF,IF), so the native Fortran view
  ! is the transpose -- scale dims stay logical (OF,IF), never the view.
  subroutine apply_muon_group(p, g, mbuf, lr, beta, rows_l, cols_l, per, nl, wd)
    real(wp), contiguous, target, intent(inout) :: p(:), mbuf(:)
    real(wp), contiguous, target, intent(in) :: g(:)
    real(wp), intent(in) :: lr, beta, wd
    integer, intent(in) :: rows_l, cols_l, per, nl
    real(wp), pointer :: G2(:, :), M2(:, :), U2(:, :)
    real(wp), allocatable, target :: upd(:)
    integer :: ll, s
    allocate (upd(per))
    do ll = 1, nl
      s = (ll - 1)*per
      G2(1:cols_l, 1:rows_l) => g(s + 1:s + per)
      M2(1:cols_l, 1:rows_l) => mbuf(s + 1:s + per)
      U2(1:cols_l, 1:rows_l) => upd
      call muon_update_mat(G2, M2, beta, U2, rows_l, cols_l)
      p(s + 1:s + per) = p(s + 1:s + per)*(1.0_wp - lr*wd) - lr*upd
    end do
    deallocate (upd)
  end subroutine apply_muon_group

  ! Dispatcher: AdamW group (default, historical) or Muon matrix group.
  subroutine apply_opt(p, g, m, v, mbuf, lr, lr_mu, b1, b2, beps, wd, t, &
      rows_l, cols_l, m_opt)
    real(wp), contiguous, intent(inout) :: p(:), m(:), v(:), mbuf(:)
    real(wp), contiguous, intent(in) :: g(:)
    real(wp), intent(in) :: lr, lr_mu, b1, b2, beps, wd
    integer, intent(in) :: t, rows_l, cols_l
    logical, intent(in) :: m_opt
    if (m_opt) then
      call apply_muon_group(p, g, mbuf, lr_mu, MUON_BETA, rows_l, cols_l, rows_l*cols_l, &
          size(p) / (rows_l*cols_l), wd)
    else
      call apply_group(p, g, m, v, lr, b1, b2, beps, wd, t)
    end if
  end subroutine apply_opt

  ! The optimizer half of a training step, on its own so a driver can touch GR
  ! between compute_grads and the update -- the MPI data-parallel app does
  ! exactly that (Allreduce(GR), divide by size, THEN this call), so every rank
  ! applies the same update and stays bit-identical. Pure refactor of the 8
  ! calls that used to be inlined at the end of train_step: same kernels, same
  ! arguments, same order, no math changed.
  !
  ! G carries the per-group (rows,cols) geometry that ONLY Muon reads (AdamW
  ! strides the flat buffers itself), hence optional: an AdamW-only caller can
  ! omit it, train_step and the MPI driver always pass it.
  subroutine apply_update(M, S, GR, tstep, lr, b1, b2, beps, wd, use_muon, lr_muon, G)
    type(params_t), intent(inout) :: M
    type(state_t), intent(inout) :: S
    type(params_t), intent(in) :: GR
    integer, intent(in) :: tstep
    real(wp), intent(in) :: lr, b1, b2, beps, wd
    logical, intent(in) :: use_muon
    real(wp), intent(in) :: lr_muon
    type(dims_t), intent(in), optional :: G
    integer :: gq, gkv, gd, gmlp
    if (present(G)) then
      gq = G%nh*G%hd; gkv = G%nkv*G%hd; gd = G%D; gmlp = 4*G%D
    else
      gq = 0; gkv = 0; gd = 0; gmlp = 0   ! unused on the AdamW path
      if (use_muon) then
        print '(A)', "apply_update: use_muon needs G (Muon geometry)"
        stop 1
      end if
    end if
    call apply_group(M%wte, GR%wte, S%wte, S%vwte, lr, b1, b2, beps, wd, tstep)
    call apply_group(M%lm, GR%lm, S%lm, S%vlm, lr, b1, b2, beps, wd, tstep)
    call apply_opt(M%q, GR%q, S%q, S%vq, S%mq, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gq, gd, use_muon)
    call apply_opt(M%k, GR%k, S%k, S%vk, S%mk, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gkv, gd, use_muon)
    call apply_opt(M%v, GR%v, S%v, S%vv, S%mv, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gkv, gd, use_muon)
    call apply_opt(M%p, GR%p, S%p, S%vp, S%mp, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gd, gq, use_muon)
    call apply_opt(M%fc, GR%fc, S%fc, S%vfc, S%mfc, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gmlp, gd, use_muon)
    call apply_opt(M%p2, GR%p2, S%p2, S%vp2, S%mp2, lr, lr_muon, b1, b2, beps, wd, &
        tstep, gd, gmlp, use_muon)
  end subroutine apply_update

  ! One full training step: forward + backward + optimizer update.
  ! Default AdamW (all groups): previous behavior, bit-identical.
  ! use_muon=.true.: Muon on 2D matrices (q,k,v,p,fc,p2), AdamW stays on
  ! wte/lm/vectors (canonical hybrid recipe). lr_muon lives in Muon
  ! geometry -- never reuse the Adam LR (different units). Optionals so
  ! train_1step/train_loop/tests keep compiling unchanged (Adam default).
  subroutine train_step(idx, targets, cos, sin, M, S, G, GR, C, tmp, &
      nll, tstep, lr, b1, b2, beps, wd, attn_blas, use_muon, lr_muon, attn_qk, attn_qkph)
    logical, intent(in), optional :: attn_blas, use_muon, attn_qk, attn_qkph
    real(wp), intent(in), optional :: lr_muon
    logical :: useblas, m_opt, useqk, useqkph
    real(wp) :: lr_mu
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
    useqk = .false.
    if (present(attn_qk)) useqk = attn_qk
    useqkph = .false.
    if (present(attn_qkph)) useqkph = attn_qkph
    m_opt = .false.
    if (present(use_muon)) m_opt = use_muon
    lr_mu = 0.0_wp
    if (present(lr_muon)) lr_mu = lr_muon
    call forward_save(idx, targets, cos, sin, M, G, C, tmp, nll, useblas, useqk, useqkph)
    call compute_grads(idx, targets, cos, sin, M, G, C, GR, tmp, nll, useblas, useqk, useqkph)
    call apply_update(M, S, GR, tstep, lr, b1, b2, beps, wd, m_opt, lr_mu, G)
  end subroutine train_step

end module fortran_train_mod
