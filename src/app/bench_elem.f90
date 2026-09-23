! app/bench_elem.f90 — quanto do PASSO e' kernel ELEMENTWISE (nao-GEMM)?
!
! Por que existe: medimos que agrupar (B) NAO compra throughput (858/863/849
! tokens/s em B=1/4/16, tempo por passo exatamente linear em B) e que estamos a
! ~14 GFLOP/s = 2-4% do pico. Isso descarta "a forma do GEMM e' o gargalo" e
! aponta para trabalho ESCALAR POR TOKEN. Este bench mede os tres kernels
! elementwise do forward nas formas do modelo (rmsnorm, rope_4d, relu^2) e,
! com --step-ms, imprime a FATIA do passo que eles explicam.
!
! Uso: bench_elem [--T 1024] [--reps 3] [--step-ms 1193]
program bench_elem
  use, intrinsic :: iso_fortran_env, only: real64
  use fortran_kinds_mod, only: wp
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER
  use fortran_rmsnorm_mod, only: rmsnorm
  use fortran_rope_mod, only: rope_4d
  use fortran_attn_mod, only: relu2
  use fortran_energy_mod, only: energy_seconds
  use M_CLI2, only: set_args, iget, rget
  implicit none

  integer :: T, reps, r, l, d_kv, d_ff
  real(real64) :: t_rms, t_rope, t_relu, t_sum, step_ms, t0r, t1r
  ! TODOS os tres kernels trabalham com buffers PLANOS (rank-1) + (NN, CC), como
  ! o forward do modelo os usa (o workspace do train e' flat).
  real(wp), allocatable :: x(:), xn(:), w(:), q(:), k(:), qr(:), kr(:), &
      cs(:), sn(:), f(:)

  call set_args('--T 1024 --reps 3 --step-ms 0', help_text=[character(len=80) :: &
      'NAME', '  bench_elem - elementwise kernels (rmsnorm, rope, relu^2) at model shapes', &
      '', 'SYNOPSIS', '  bench_elem [--T 1024] [--reps 3] [--step-ms MS]', &
      '', 'DESCRIPTION', &
      '  Mede os kernels elementwise do forward somados sobre as camadas, ou', &
      '  seja, o que eles custam num passo. Com --step-ms (tempo medido de um', &
      '  passo de treino), imprime a FATIA do passo explicada por eles.'])
  T = iget('T'); reps = iget('reps'); step_ms = rget('step-ms')

  d_kv = A_KV*A_HD
  d_ff = 4*A_D
  allocate (x(T*A_D), xn(T*A_D), w(A_D), q(T*A_HEAD*A_HD), k(T*d_kv), &
      qr(T*A_HEAD*A_HD), kr(T*d_kv), cs(T*A_HD/2), sn(T*A_HD/2), f(T*d_ff))
  call random_number(x); call random_number(w); call random_number(q)
  call random_number(k); call random_number(f); call random_number(cs); call random_number(sn)

  print '(A,I0,A,I0,A,I0,A,I0,A,I0)', '# arch: d_model=', A_D, ' n_head=', A_HEAD, &
      ' n_kv=', A_KV, ' n_layer=', A_LAYER, ' T=', T
  print '(A,I0,A,I0,A,I0)', '# formas por camada: rmsnorm ', T, 'x', A_D, &
      '  relu2 ', T, 'x', d_ff

  t0r = energy_seconds()
  do r = 1, reps
    do l = 1, A_LAYER
      call rmsnorm(x, w, xn, T, A_D, 1.0e-5_wp)
    end do
  end do
  t1r = energy_seconds()
  t_rms = (t1r - t0r)/real(reps, real64)

  t0r = energy_seconds()
  do r = 1, reps
    do l = 1, A_LAYER
      call rope_4d(q, cs, sn, qr, 1, T, A_HEAD, A_HD)
      call rope_4d(k, cs, sn, kr, 1, T, A_KV, A_HD)
    end do
  end do
  t1r = energy_seconds()
  t_rope = (t1r - t0r)/real(reps, real64)

  t0r = energy_seconds()
  do r = 1, reps
    do l = 1, A_LAYER
      call relu2(f, T*d_ff)
    end do
  end do
  t1r = energy_seconds()
  t_relu = (t1r - t0r)/real(reps, real64)

  t_sum = t_rms + t_rope + t_relu
  print '(A)'
  print '(A)', '# kernel     camadas   ms/passo   GB/s efetivo   % do passo'
  call row('# rmsnorm', t_rms, real(A_LAYER, real64)*real(T*A_D, real64)*4.0_real64*2.0_real64)
  call row('# rope_4d', t_rope, real(A_LAYER, real64)*real(T*(A_HEAD*A_HD + d_kv), real64)*4.0_real64*2.0_real64)
  call row('# relu2  ', t_relu, real(A_LAYER, real64)*real(T*d_ff, real64)*4.0_real64*2.0_real64)
  call row('# SOMA   ', t_sum, 0.0_real64)
  print '(A)'
  print '(A)', '# O que NAO esta aqui: GEMMs (use bench_gemm) e atencao (bench_attn).'
  print '(A)', '# Se a soma destes tres ja for grande, o ganho esta em vetorizar elementwise.'

contains

  ! uma linha da tabela: ms/passo, banda efetiva (se bytes>0) e fatia do passo
  subroutine row(name, t, bytes)
    character(*), intent(in) :: name
    real(real64), intent(in) :: t, bytes
    real(real64) :: gbs
    gbs = -1.0_real64
    if (t > 0.0_real64 .and. bytes > 0.0_real64) gbs = bytes/t/1.0e9_real64
    if (bytes > 0.0_real64) then
      write (*, '(A,3X,I6,3X,F10.3,3X,F10.2,3X,F7.1)') name, A_LAYER, 1.0e3_real64*t, gbs, pct(t)
    else
      write (*, '(A,3X,I6,3X,F10.3,3X,A,3X,F7.1)') name, A_LAYER, 1.0e3_real64*t, '     --', pct(t)
    end if
  end subroutine row

  function pct(t) result(p)
    real(real64), intent(in) :: t
    real(real64) :: p
    if (step_ms <= 0.0_real64) then
      p = -1.0_real64
    else
      p = 100.0_real64*(1.0e3_real64*t)/step_ms
    end if
  end function pct

end program bench_elem
