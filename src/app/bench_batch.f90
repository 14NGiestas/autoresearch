! app/bench_batch.f90 — quanto o lote (B sequências por passo) compra?
!
! O treinador hoje processa UMA sequência de T tokens por passo, então cada
! GEMM das camadas lineares tem M = T = 2048 linhas. Com lote B, M = B*T e o
! mesmo sgemm recebe matrizes B vezes mais altas: mais intensidade aritmética,
! menos tempo por FLOP. A atenção não muda de forma (uma chamada por sequência),
! então o ganho vem das lineares -- que são a maior parte dos FLOPs. Este bench
! mede isso nas formas do modelo (d=768, f=4) ANTES de reescrever o treinador.
program bench_batch
  use fortran_kinds_mod, only: wp
  use fortran_blas_mod, only: sgemm
  use M_CLI2, only: set_args, iget
  use iso_c_binding, only: c_int64_t
  implicit none

  integer, parameter :: D = 768, F = 4, T = 2048, NB = 4
  integer, parameter :: bs(NB) = [1, 2, 4, 8]
  integer :: ish, ib, reps, rr
  ! o wrapper de sgemm do projeto usa dimensões c_int64_t (como o BLAS real)
  integer(c_int64_t) :: M, K, N
  real(wp), allocatable :: a(:), bmat(:), c(:)
  real(wp) :: t0, t1, flop, gf, gf_ref(2), sp

  call set_args('--reps REPS 3')
  reps = iget('reps')

  print '(A,I0,A,I0,A,I0,A)', 'modelo: d=', D, ' f=', F, ' T=', T, &
      ' (hoje: uma sequencia por passo)'
  print '(A)', ''

  do ish = 1, 2
    if (ish == 1) then
      K = D; N = D
      print '(A)', '--- attn/proj   (M,768)x(768,768)'
    else
      K = D; N = F * D
      print '(A)', '--- mlp fc      (M,768)x(768,3072)'
    end if
    do ib = 1, NB
      M = bs(ib) * T
      if (allocated(a)) deallocate (a, bmat, c)
      allocate (a(M * K), bmat(K * N), c(M * N))
      call random_number(a)
      call random_number(bmat)
      c = 0.0_wp
      ! aquecimento: deixa o BLAS escolher o kernel e tocar a memória
      call sgemm('N', 'N', M, N, K, 1.0_wp, a, M, bmat, K, 0.0_wp, c, M)
      call cpu_time(t0)
      do rr = 1, reps
        call sgemm('N', 'N', M, N, K, 1.0_wp, a, M, bmat, K, 0.0_wp, c, M)
      end do
      call cpu_time(t1)
      flop = 2.0_wp * real(M, wp) * real(N, wp) * real(K, wp) * real(reps, wp)
      gf = flop / max(t1 - t0, 1.0e-9_wp) / 1.0e9_wp
      if (bs(ib) == 1) gf_ref(ish) = gf
      sp = gf / max(gf_ref(ish), 1.0e-9_wp)
      print '(A,I2,A,I6,A,F8.1,A,F6.2,A)', '  B=', bs(ib), '  M=', int(M), &
          '  GFLOP/s=', gf, '  x', sp, ' vs B=1'
      deallocate (a, bmat, c)
    end do
  end do

  block
    real(wp) :: lin, att
    ! por sequência de T=2048 tokens: lineares = 6*N_params*T;
    ! atenção = 12 camadas * (QK^T + PV) * 2 (fwd) * 3 (fwd+bwd)
    lin = 6.0_wp * 97.5e6_wp * real(T, wp)
    att = 12.0_wp * 2.0_wp * (real(T, wp)**2 * real(D, wp) * 2.0_wp) * 2.0_wp * 3.0_wp
    print '(A)', ''
    print '(A,F7.1,A)', '  FLOPs por sequencia: lineares = ', lin / 1e9_wp, ' GFLOP'
    print '(A,F7.1,A)', '                       atencao  = ', att / 1e9_wp, ' GFLOP'
    print '(A,F5.1,A)', '  fracao na atencao: ', 100.0_wp * att / (att + lin), '%'
  end block
end program bench_batch
