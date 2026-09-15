! lib/fortran_qkhop.f90 — QK-hop: atencao sem value (relacional).
!
! Forward:  S = softmax(QK'/sqrt(d)) (igual), h = S@X, y = S@h.
! Dois hops sobre o proprio stream: relacao reutilizada, conteudo fora.
! Backward: twin reverso exato (h = S@X e y = S@h, ambos lineares em S e X).
! Usa os mesmos pesos Q/K/P do checkpoint (V alocado mas nao lido).
module fortran_qkhop_mod
  use fortran_kinds_mod, only: wp
  implicit none
  private
  public :: qkhop_fwd, qkhop_bwd

contains

  ! q,k: (B,T,H,D) pre-RoPE ok (QK-hop nao exige RoPE; recebe ja rotado).
  ! x: (B,T,D) stream de entrada. y: (B,T,D) saida. S: scratch (B,H,T,T).
  subroutine qkhop_fwd(q, k, x, y, S, B, T, H, K_H, D)
    integer, intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: q(B*T*H*D), k(B*T*K_H*D), x(B*T*D)
    real(wp), intent(out) :: y(B*T*D), S(B*H*T*T)
    integer :: aa, bb, cc, ss, dd, kb, rep
    real(wp) :: scale, mx, sm, acc
    real(wp) :: h1(B*T*D)
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
          h1(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
        do dd = 1, D
          acc = 0.0_wp
          do bb = 1, H
            do ss = 1, cc
              acc = acc + S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                          h1(((aa-1)*T+ss-1)*D+dd)
            end do
          end do
          y(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
      end do
    end do
  end subroutine qkhop_fwd

  ! dy: d(y). Retorna dx, dq, dk (dk soma por grupo GQA).
  ! Sm = media_b(S_b); y = Sm@h1, h1 = Sm@x; dSm = dSm2+dSm1; jac. softmax/head.
  subroutine qkhop_bwd(dy, q, k, x, S, dx, dq, dk, B, T, H, K_H, D)
    integer, intent(in) :: B, T, H, K_H, D
    real(wp), intent(in) :: dy(B*T*D), q(B*T*H*D), k(B*T*K_H*D)
    real(wp), intent(in) :: x(B*T*D), S(B*H*T*T)
    real(wp), intent(out) :: dx(B*T*D), dq(B*T*H*D), dk(B*T*K_H*D)
    integer :: aa, bb, cc, rr, ss, dd, kb, rep
    real(wp) :: scale, acc, sdot, dl
    real(wp) :: dh1(B*T*D), h1(B*T*D), dSm(B*T*T)
    scale = 1.0_wp / sqrt(real(D, wp))
    rep = H / K_H
    dx = 0.0_wp; dq = 0.0_wp; dk = 0.0_wp
    dh1 = 0.0_wp
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
          h1(((aa-1)*T+cc-1)*D+dd) = acc / real(H, wp)
        end do
      end do
    end do
    !$omp parallel do private(aa,bb,cc,rr,dd) schedule(static)
    do aa = 1, B
      do cc = 1, T
        do bb = 1, H
          do rr = 1, cc
            do dd = 1, D
              dh1(((aa-1)*T+rr-1)*D+dd) = dh1(((aa-1)*T+rr-1)*D+dd) + &
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
                dh1(((aa-1)*T+cc-1)*D+dd) / real(H, wp)
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
            acc = acc + dy(((aa-1)*T+cc-1)*D+dd) * h1(((aa-1)*T+rr-1)*D+dd) + &
                        dh1(((aa-1)*T+cc-1)*D+dd) * x(((aa-1)*T+rr-1)*D+dd)
          end do
          dSm(((aa-1)*T+cc-1)*T+rr) = acc / real(H, wp)
        end do
        do rr = cc + 1, T
          dSm(((aa-1)*T+cc-1)*T+rr) = 0.0_wp
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
                            dSm(((aa-1)*T+cc-1)*T+ss)
          end do
          do ss = 1, cc
            dl = S((((aa-1)*H+bb-1)*T+cc-1)*T+ss) * &
                 (dSm(((aa-1)*T+cc-1)*T+ss) - sdot) * scale
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

end module fortran_qkhop_mod
