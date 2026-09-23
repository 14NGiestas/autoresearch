! lib/fortran_qkhop.f90 — QK-hop: atencao sem value (relacional).
!
! Forward:  S = softmax(QK'/sqrt(d)) (igual), h = S@X, y = S@h.
! Dois hops sobre o proprio stream: relacao reutilizada, conteudo fora.
! Backward: twin reverso exato (h = S@X e y = S@h, ambos lineares em S e X).
! Usa os mesmos pesos Q/K/P do checkpoint (V alocado mas nao lido).
module fortran_qkhop_mod
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: sgemm
  implicit none
  private
  public :: qkhop_fwd, qkhop_bwd, qkhop_sgemm, qkhop_bwd_sgemm, qkhop_ph_fwd, qkhop_ph_bwd

contains

  ! q,k: (B,T,H,D) pre-RoPE ok (QK-hop nao exige RoPE; recebe ja rotado).
  ! x: (B,T,D) stream de entrada. y: (B,T,D) saida. S: scratch (B,H,T,T).
  subroutine qkhop_fwd(q, k, x, y, S, h1w, B, T, H, K_H, D)
    integer, intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: q(B*T*H*D), k(B*T*K_H*D), x(B*T*D)
    real(wp), intent(out) :: y(B*T*D), S(B*H*T*T), h1w(B*T*D)
    integer :: aa, bb, cc, ss, dd, kb, rep
    real(wp) :: scale, mx, sm, acc
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    !$omp parallel do collapse(2) private(bb,cc,ss,dd,kb,mx,sm,acc) schedule(static)
    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          mx = -huge(1.0_wp)
          do ss = 1, cc
            acc = 0.0_wp
            do dd = 1, D
              acc = acc + q(((aa-1)*T+cc-1)*H*D+(bb-1)*D+dd) * &
                          k(((aa-1)*T+ss-1)*K_H*D+(kb-1)*D+dd)
            end do
            acc = acc * scale
            S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) = acc
            if (acc > mx) mx = acc
          end do
          sm = 0.0_wp
          do ss = 1, cc
            sm = sm + exp(S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) - mx)
          end do
          do ss = 1, cc
            S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) = &
              exp(S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) - mx) / sm
          end do
          do ss = cc + 1, T
            S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) = 0.0_wp
          end do
        end do
      end do
    end do
    ! hop1: h1 = mean_heads(S) @ x ; hop2: y = mean_heads(S) @ h1.
    ! Media sobre heads (barata, sem params): relacao compartilhada.
    !$omp parallel do private(aa,bb,cc,ss,dd,acc) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do dd = 1, D
          acc = 0.0_wp
          do bb = 1, H
            do ss = 1, cc
              acc = acc + S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                          x(((aa-1)*T+ss-1)*D+dd)
            end do
          end do
          h1w(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
        do dd = 1, D
          acc = 0.0_wp
          do bb = 1, H
            do ss = 1, cc
              acc = acc + S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                          h1w(((aa-1)*T+ss-1)*D+dd)
            end do
          end do
          y(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
      end do
    end do
  end subroutine qkhop_fwd

  ! dy: d(y). Retorna dx, dq, dk (dk soma por grupo GQA).
  ! Sm = media_b(S_b); y = Sm@h1, h1 = Sm@x; dSm = dSm2+dSm1; jac. softmax/head.
  subroutine qkhop_bwd(dy, q, k, x, S, dx, dq, dk, h1w, dh1w, dSmw, B, T, H, K_H, D)
    integer, intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: dy(B*T*D), q(B*T*H*D), k(B*T*K_H*D)
    real(wp), intent(in) :: x(B*T*D), S(B*H*T*T)
    real(wp), intent(out) :: dx(B*T*D), dq(B*T*H*D), dk(B*T*K_H*D)
    integer :: aa, bb, cc, rr, ss, dd, kb, rep
    real(wp) :: scale, acc, sdot, dl
    real(wp), intent(out) :: h1w(B*T*D), dh1w(B*T*D), dSmw(B*T*T)
    ! (workspace via h1w/dh1w/dSmw do caller; sem arrays automaticos)
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    dx = 0.0_wp; dq = 0.0_wp; dk = 0.0_wp
    dh1w = 0.0_wp
    h1w = 0.0_wp; dh1w = 0.0_wp; dSmw = 0.0_wp
    !$omp parallel do private(aa,bb,cc,ss,dd,acc) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do dd = 1, D
          acc = 0.0_wp
          do bb = 1, H
            do ss = 1, cc
              acc = acc + S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                          x(((aa-1)*T+ss-1)*D+dd)
            end do
          end do
          h1w(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
      end do
    end do
    !$omp parallel do private(aa,bb,cc,rr,dd) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do bb = 1, H
          do rr = 1, cc
            do dd = 1, D
              dh1w(((aa-1)*T+rr-1)*D+dd) = dh1w(((aa-1)*T+rr-1)*D+dd) + &
                S((((aa-1)*H+bb-1)*T+cc-1)*T+rr) * &
                dy(((aa-1)*T+cc-1)*D+dd) / real(H, wp)
            end do
          end do
        end do
      end do
    end do
    !$omp parallel do private(aa,bb,cc,rr,dd) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do bb = 1, H
          do rr = 1, cc
            do dd = 1, D
              dx(((aa-1)*T+rr-1)*D+dd) = dx(((aa-1)*T+rr-1)*D+dd) + &
                S((((aa-1)*H+bb-1)*T+cc-1)*T+rr) * &
                dh1w(((aa-1)*T+cc-1)*D+dd) / real(H, wp)
            end do
          end do
        end do
      end do
    end do
    !$omp parallel do private(aa,cc,rr,dd,acc) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do rr = 1, cc
          acc = 0.0_wp
          do dd = 1, D
            acc = acc + dy(((aa-1)*T+cc-1)*D+dd) * h1w(((aa-1)*T+rr-1)*D+dd) + &
                        dh1w(((aa-1)*T+cc-1)*D+dd) * x(((aa-1)*T+rr-1)*D+dd)
          end do
          dSmw(((aa-1)*T+cc-1)*T+rr) = acc / real(H, wp)
        end do
        do rr = cc + 1, T
          dSmw(((aa-1)*T+cc-1)*T+rr) = 0.0_wp
        end do
      end do
    end do
    !$omp parallel do private(aa,bb,cc,ss,dd,kb,sdot,dl) schedule(static)
    do aa = 1, B
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          sdot = 0.0_wp
          do ss = 1, cc
            sdot = sdot + S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                            dSmw(((aa-1)*T+cc-1)*T+ss)
          end do
          do ss = 1, cc
            dl = S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                 (dSmw(((aa-1)*T+cc-1)*T+ss) - sdot) * scale
            do dd = 1, D
              dq(((aa-1)*T+cc-1)*H*D+(bb-1)*D+dd) = &
                dq(((aa-1)*T+cc-1)*H*D+(bb-1)*D+dd) + dl * &
                k(((aa-1)*T+ss-1)*K_H*D+(kb-1)*D+dd)
              dk(((aa-1)*T+ss-1)*K_H*D+(kb-1)*D+dd) = &
                dk(((aa-1)*T+ss-1)*K_H*D+(kb-1)*D+dd) + dl * &
                q(((aa-1)*T+cc-1)*H*D+(bb-1)*D+dd)
            end do
          end do
        end do
      end do
    end do
  end subroutine qkhop_bwd


  ! BLAS twin (mesma matematica do naive; copias explicitas so onde stride mente).
  ! S: (B,H,T,T) sections contiguas T*T. X/h1/y: blocos (T,D) row-major.
  subroutine qkhop_sgemm(q, k, x, y, S, h1w, B, T, H, K_H, D)
    use iso_c_binding, only: c_int64_t, c_int
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: q(:), k(:), x(:)
    real(wp), intent(out) :: y(:), S(:), h1w(:)
    integer :: aa, bb, kb, rep, ii, jj
    integer(c_int64_t) :: m, n, kk
    real(wp) :: scale, mx, smx
    real(wp), allocatable :: Sm(:,:), SmT(:,:), XT(:,:), h1T(:,:), yT(:,:)
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    allocate (Sm(T,T), SmT(T,T), XT(D,T), h1T(D,T), yT(D,T))
    do aa = 1, B
      Sm = 0.0_wp
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
        call sgemm('T', 'N', m, n, kk, scale, &
             k((aa-1)*T*K_H*D + (kb-1)*D + 1:), int(K_H*D, c_int64_t), &
             q((aa-1)*T*H*D + (bb-1)*D + 1:), int(H*D, c_int64_t), &
             0.0_wp, S(((aa-1)*H+bb-1)*T*T+1:), int(T, c_int64_t))
        !$omp parallel do private(ii,jj,mx,smx) schedule(static)
        do ii = 1, T
          mx = -huge(1.0_wp)
          do jj = 1, ii
            if (S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) > mx) &
              mx = S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj)
          end do
          smx = 0.0_wp
          do jj = 1, ii
            smx = smx + exp(S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) - mx)
          end do
          do jj = 1, ii
            S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) = &
              exp(S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) - mx) / smx
          end do
          do jj = ii + 1, T
            S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) = 0.0_wp
          end do
        end do
      end do
      !$omp parallel do collapse(2) private(ii,jj,bb) schedule(static)
      do jj = 1, T
        do ii = 1, T
          do bb = 1, H
            Sm(ii, jj) = Sm(ii, jj) + S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj)
          end do
        end do
      end do
      Sm = Sm / real(H, wp)
      do jj = 1, T
        do ii = 1, D
          XT(ii, jj) = x((aa-1)*T*D + (jj-1)*D + ii)
        end do
        do ii = 1, T
          SmT(ii, jj) = Sm(jj, ii)
        end do
      end do
      m = int(D, c_int64_t); n = int(T, c_int64_t); kk = int(T, c_int64_t)
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           XT, int(D, c_int64_t), SmT, int(T, c_int64_t), &
           0.0_wp, h1T, int(D, c_int64_t))
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           h1T, int(D, c_int64_t), SmT, int(T, c_int64_t), &
           0.0_wp, yT, int(D, c_int64_t))
      do jj = 1, T
        do ii = 1, D
          h1w((aa-1)*T*D + (jj-1)*D + ii) = h1T(ii, jj)
          y((aa-1)*T*D + (jj-1)*D + ii) = yT(ii, jj)
        end do
      end do
    end do
    deallocate (Sm, SmT, XT, h1T, yT)
  end subroutine qkhop_sgemm



  subroutine qkhop_bwd_sgemm(dy, q, k, x, S, dx, dq, dk, h1w, dh1w, dSmw, &
      B, T, H, K_H, D)
    use iso_c_binding, only: c_int64_t, c_int
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: dy(:), q(:), k(:), x(:), S(:)
    real(wp), intent(out) :: dx(:), dq(:), dk(:)
    real(wp), intent(out) :: h1w(:), dh1w(:), dSmw(:)
    integer :: aa, bb, cc, ss, kb, rep, ii, jj
    integer(c_int64_t) :: m, n, kk
    real(wp) :: sdot, dlx, scale
    real(wp), allocatable :: Sm(:,:), dL(:,:)
    real(wp), allocatable :: dyC(:,:), h1C(:,:), xC(:,:), dh1C(:,:)
    real(wp), allocatable :: Kc(:,:), Qc(:,:), dSmC(:,:), SmT(:,:)
    real(wp), allocatable :: dxC(:,:)
    rep = H / K_H
    scale = 1.0_wp / sqrt(real(D, wp))
    allocate (Sm(T,T), dL(T,T), dyC(T,D), h1C(T,D), xC(T,D), dh1C(T,D))
    allocate (Kc(T,D), Qc(T,D), dSmC(T,T), SmT(T,T), dxC(T,D))
    dx = 0.0_wp; dq = 0.0_wp; dk = 0.0_wp
    do aa = 1, B
      Sm = 0.0_wp
      !$omp parallel do collapse(2) private(ii,jj,bb) schedule(static)
      do jj = 1, T
        do ii = 1, T
          do bb = 1, H
            Sm(ii, jj) = Sm(ii, jj) + S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj)
          end do
        end do
      end do
      Sm = Sm / real(H, wp)
      !$omp parallel do private(jj,ii) schedule(static)
      do jj = 1, T
        do ii = 1, D
          dyC(jj, ii) = dy((aa-1)*T*D + (jj-1)*D + ii)
          xC(jj, ii) = x((aa-1)*T*D + (jj-1)*D + ii)
        end do
      end do
      !$omp parallel do collapse(2) private(ii,jj) schedule(static)
      do jj = 1, T
        do ii = 1, T
          SmT(ii, jj) = Sm(jj, ii)
        end do
      end do
      m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           Sm, int(T, c_int64_t), xC, int(T, c_int64_t), &
           0.0_wp, h1C, int(T, c_int64_t))
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           SmT, int(T, c_int64_t), dyC, int(T, c_int64_t), &
           0.0_wp, dh1C, int(T, c_int64_t))
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           SmT, int(T, c_int64_t), dh1C, int(T, c_int64_t), &
           0.0_wp, dxC, int(T, c_int64_t))
      m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
      call sgemm('N', 'N', m, n, kk, 1.0_wp, &
           Sm, int(T, c_int64_t), xC, int(T, c_int64_t), &
           0.0_wp, h1C, int(T, c_int64_t))
      do jj = 1, T
        do ii = 1, D
          dh1w((aa-1)*T*D + (jj-1)*D + ii) = dh1C(jj, ii)
          dx((aa-1)*T*D + (jj-1)*D + ii) = dxC(jj, ii)
          h1w((aa-1)*T*D + (jj-1)*D + ii) = h1C(jj, ii)
        end do
      end do
      m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
      call sgemm('N', 'T', m, n, kk, 1.0_wp, &
           dyC, int(T, c_int64_t), h1C, int(T, c_int64_t), &
           0.0_wp, dSmC, int(T, c_int64_t))
      call sgemm('N', 'T', m, n, kk, 1.0_wp, &
           dh1C, int(T, c_int64_t), xC, int(T, c_int64_t), &
           1.0_wp, dSmC, int(T, c_int64_t))
      do jj = 1, T
        do ii = 1, T
          dSmw(((aa-1)*T+ii-1)*T+jj) = dSmC(ii, jj) / real(H, wp)
        end do
      end do
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do cc = 1, T
          sdot = 0.0_wp
          do ss = 1, cc
            sdot = sdot + S(((aa-1)*H+bb-1)*T*T+(cc-1)*T+ss) * &
                            dSmw(((aa-1)*T+cc-1)*T+ss)
          end do
          do ss = 1, cc
            dlx = S(((aa-1)*H+bb-1)*T*T+(cc-1)*T+ss) * &
                 (dSmw(((aa-1)*T+cc-1)*T+ss) - sdot) * scale
            dL(cc, ss) = dlx
          end do
          do ss = cc + 1, T
            dL(cc, ss) = 0.0_wp
          end do
        end do
        do jj = 1, T
          do ii = 1, D
            Kc(jj, ii) = k((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii)
            Qc(jj, ii) = q((aa-1)*T*H*D + (jj-1)*H*D + (bb-1)*D + ii)
          end do
        end do
        do jj = 1, T
          do ii = 1, D
            dh1C(jj, ii) = sum(dL(jj, 1:T) * Kc(1:T, ii))
            xC(jj, ii) = sum(dL(1:T, jj) * Qc(1:T, ii))
          end do
        end do
        do jj = 1, T
          do ii = 1, D
            dq((aa-1)*T*H*D + (jj-1)*H*D + (bb-1)*D + ii) = dh1C(jj, ii)
            dk((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii) = &
              dk((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii) + xC(jj, ii)
          end do
        end do
      end do
    end do
    deallocate (Sm, dL, dyC, h1C, xC, dh1C, Kc, Qc, dSmC, SmT, dxC)
  end subroutine qkhop_bwd_sgemm



  ! Per-head (sem media): y_b = S_b@(S_b@X), concat heads. Mesma tese, sem
  ! colapsar especializacao. y: (B,T,H,D) p/ o-proj (hdd==D nos nossos archs).
  subroutine qkhop_ph_fwd(q, k, x, y, S, h1w, B, T, H, K_H, D)
    use iso_c_binding, only: c_int64_t, c_int
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: q(:), k(:), x(:)
    real(wp), intent(out) :: y(:), S(:), h1w(:)
    integer :: aa, bb, kb, rep, ii, jj
    integer(c_int64_t) :: m, n, kk
    real(wp) :: scale, mx, smx
    real(wp), allocatable :: Sb(:,:), Xb(:,:), H1(:,:), Yb(:,:)
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    allocate (Sb(T,T), Xb(T,D), H1(T,D), Yb(T,D))
    do aa = 1, B
      do jj = 1, T
        do ii = 1, D
          Xb(jj, ii) = x((aa-1)*T*D + (jj-1)*D + ii)
        end do
      end do
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
        call sgemm('T', 'N', m, n, kk, scale, &
             k((aa-1)*T*K_H*D + (kb-1)*D + 1:), int(K_H*D, c_int64_t), &
             q((aa-1)*T*H*D + (bb-1)*D + 1:), int(H*D, c_int64_t), &
             0.0_wp, Sb, int(T, c_int64_t))
        do ii = 1, T
          do jj = 1, ii - 1
            mx = Sb(ii, jj)
            Sb(ii, jj) = Sb(jj, ii)
            Sb(jj, ii) = mx
          end do
        end do
        do ii = 1, T
          mx = -huge(1.0_wp)
          do jj = 1, ii
            if (Sb(ii, jj) > mx) mx = Sb(ii, jj)
          end do
          smx = 0.0_wp
          do jj = 1, ii
            smx = smx + exp(Sb(ii, jj) - mx)
          end do
          do jj = 1, ii
            Sb(ii, jj) = exp(Sb(ii, jj) - mx) / smx
          end do
          do jj = ii + 1, T
            Sb(ii, jj) = 0.0_wp
          end do
        end do
        do jj = 1, T
          do ii = 1, T
            S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj) = Sb(ii, jj)
          end do
        end do
        m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             Sb, int(T, c_int64_t), Xb, int(T, c_int64_t), &
             0.0_wp, H1, int(T, c_int64_t))
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             Sb, int(T, c_int64_t), H1, int(T, c_int64_t), &
             0.0_wp, Yb, int(T, c_int64_t))
        do jj = 1, T
          do ii = 1, D
            h1w((aa-1)*T*H*D + (bb-1)*T*D + (jj-1)*D + ii) = H1(jj, ii)
            y((aa-1)*T*H*D + (bb-1)*T*D + (jj-1)*D + ii) = Yb(jj, ii)
          end do
        end do
      end do
    end do
    deallocate (Sb, Xb, H1, Yb)
  end subroutine qkhop_ph_fwd



  ! Sem workspace do caller: os tres scratches antigos (w1/w2/w3) eram
  ! intent(out) e SO' zerados -- vestigio de uma versao anterior. O corpo usa
  ! alocaveis locais. Removidos em 2026-09-18 (o buffer sdsm sozinho custava
  ! 4,29 GB de RSS em T=1024 por causa de um G%T a mais na dimensao).
  subroutine qkhop_ph_bwd(dy, q, k, x, S, dx, dq, dk, &
      B, T, H, K_H, D)
    use iso_c_binding, only: c_int64_t, c_int
    integer(c_int), intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: dy(:), q(:), k(:), x(:), S(:)
    real(wp), intent(out) :: dx(:), dq(:), dk(:)
    integer :: aa, bb, kb, rep, cc, ss, ii, jj
    integer(c_int64_t) :: m, n, kk
    real(wp) :: scale, sdot, dlx
    real(wp), allocatable :: Sb(:,:), Xb(:,:), H1(:,:), Yb(:,:)
    real(wp), allocatable :: DYc(:,:), dH1(:,:), dSm(:,:), dLb(:,:)
    real(wp), allocatable :: KBc(:,:), Qb(:,:)
    real(wp), allocatable :: SbT(:,:)
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    allocate (Sb(T,T), Xb(T,D), H1(T,D), Yb(T,D))
    allocate (DYc(T,D), dH1(T,D), dSm(T,T), dLb(T,T), KBc(T,D), Qb(T,D))
    allocate (SbT(T,T))
    dx = 0.0_wp; dq = 0.0_wp; dk = 0.0_wp
    do aa = 1, B
      do jj = 1, T
        do ii = 1, D
          Xb(jj, ii) = x((aa-1)*T*D + (jj-1)*D + ii)
        end do
      end do
      do bb = 1, H
        kb = (bb - 1) / rep + 1
        do jj = 1, T
          do ii = 1, T
            Sb(ii, jj) = S(((aa-1)*H+bb-1)*T*T+(ii-1)*T+jj)
          end do
          do ii = 1, D
            DYc(jj, ii) = dy((aa-1)*T*H*D + (bb-1)*T*D + (jj-1)*D + ii)
            KBc(jj, ii) = k((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii)
            Qb(jj, ii) = q((aa-1)*T*H*D + (jj-1)*H*D + (bb-1)*D + ii)
          end do
        end do
        m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             Sb, int(T, c_int64_t), Xb, int(T, c_int64_t), &
             0.0_wp, H1, int(T, c_int64_t))
        do jj = 1, T
          do ii = 1, T
            SbT(ii, jj) = Sb(jj, ii)
          end do
        end do
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             SbT, int(T, c_int64_t), DYc, int(T, c_int64_t), &
             0.0_wp, dH1, int(T, c_int64_t))
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             SbT, int(T, c_int64_t), dH1, int(T, c_int64_t), &
             0.0_wp, Yb, int(T, c_int64_t))
        !$omp parallel do collapse(2) private(ii,jj) schedule(static)
        do jj = 1, T
          do ii = 1, D
            dx((aa-1)*T*D + (jj-1)*D + ii) = dx((aa-1)*T*D + (jj-1)*D + ii) + &
              Yb(jj, ii)
          end do
        end do
        m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             Sb, int(T, c_int64_t), Xb, int(T, c_int64_t), &
             0.0_wp, H1, int(T, c_int64_t))
        m = int(T, c_int64_t); n = int(T, c_int64_t); kk = int(D, c_int64_t)
        call sgemm('N', 'T', m, n, kk, 1.0_wp, &
             DYc, int(T, c_int64_t), H1, int(T, c_int64_t), &
             0.0_wp, dSm, int(T, c_int64_t))
        call sgemm('N', 'T', m, n, kk, 1.0_wp, &
             dH1, int(T, c_int64_t), Xb, int(T, c_int64_t), &
             1.0_wp, dSm, int(T, c_int64_t))
        do cc = 1, T
          sdot = 0.0_wp
          do ss = 1, cc
            sdot = sdot + Sb(cc, ss) * dSm(cc, ss)
          end do
          do ss = 1, cc
            dlx = Sb(cc, ss) * (dSm(cc, ss) - sdot) * scale
            dLb(cc, ss) = dlx
          end do
          do ss = cc + 1, T
            dLb(cc, ss) = 0.0_wp
          end do
        end do
        m = int(T, c_int64_t); n = int(D, c_int64_t); kk = int(T, c_int64_t)
        call sgemm('N', 'N', m, n, kk, 1.0_wp, &
             dLb, int(T, c_int64_t), KBc, int(T, c_int64_t), &
             0.0_wp, H1, int(T, c_int64_t))
        call sgemm('T', 'N', m, n, kk, 1.0_wp, &
             dLb, int(T, c_int64_t), Qb, int(T, c_int64_t), &
             0.0_wp, Yb, int(T, c_int64_t))
        do jj = 1, T
          do ii = 1, D
            dq((aa-1)*T*H*D + (jj-1)*H*D + (bb-1)*D + ii) = H1(jj, ii)
            dk((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii) = &
              dk((aa-1)*T*K_H*D + (jj-1)*K_H*D + (kb-1)*D + ii) + Yb(jj, ii)
          end do
        end do
      end do
    end do
    deallocate (Sb, Xb, H1, Yb, DYc, dH1, dSm, dLb, KBc, Qb, SbT)
  end subroutine qkhop_ph_bwd


end module fortran_qkhop_mod
