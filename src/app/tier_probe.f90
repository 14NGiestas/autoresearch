! app/tier_probe.f90 — LEDGER de memoria: recomputar vs guardar/ler.
!
! Dois modos, MESMO instrumento e MESMA lei (compute:IO), constantes diferentes:
!
!   --mode infer   cenario de DECODE: precisa do KV de uma janela de N tokens.
!                  Alternativa a ler: recomputar as projecoes K/V + atencao.
!                  Unidade: bytes de KV por token.
!
!   --mode train   cenario de TREINO: guardar ativacoes para o backward ou
!                  recomputar no backward (checkpointing). Unidade: bytes de
!                  ativacao por token, DERIVADOS do que o proprio fortran_train
!                  aloca (C%e..C%kr = 10*d_model + 3*d_kv por camada/token).
!                  Reporta tambem o estado do otimizador (m,v = 2P): ler uma vez
!                  por passo de cada nivel e' o custo de fazer offload.
!
! Fontes de constante: a MAQUINA vem de lib/fortran_probe_mod (banda, latencia,
! J por nivel); o MODELO vem do arch.txt do checkpoint (--ckpt), nao de
! fortran_arch_mod -- essa constante ja' esteve configurada para outra arch que
! nao a dos experimentos (um instrumento nao herda essa ambiguidade). Energia/
! tempo/CPU/IO: fortran_energy (RAPL + /proc/self), medidos por ESTE processo;
! o app AVISA quando o sensor legivel nao e' de CPU (ver docs/kv_tier.md).
!
! Uso: tier_probe [--mode infer|train] [--ckpt DIR] [--tokens N] [--reps R]
!                 [--mb M] [--nreads K] [--path F] [--json F] [--sensor S]
program tier_probe
  use, intrinsic :: iso_fortran_env, only: int64, real32, real64
  use fortran_kinds_mod, only: wp
  use fortran_energy_mod, only: energy_init, energy_mark, energy_sensor, &
      energy_joules, energy_interval_t
  use fortran_probe_mod, only: probe_tier_t, probe_file, probe_cell, probe_report, &
      probe_json, probe_trust, probe_verbose
  use load_weights_mod, only: read_arch_any
  use M_CLI2, only: set_args, iget, sget
  implicit none

  integer :: dm, nh, nkv, hd, nl, vv, ctx, bos   ! arch do checkpoint (runtime)
  character(len=256) :: arch_src
  integer :: dkv, ff, rv
  integer :: n_tok, reps, file_mb, nreads, r, l, u, mode_id
  integer(int64) :: n_tk_rep, nbytes_file
  real(wp), allocatable :: x(:,:), wq(:,:), wk(:,:), wv(:,:), wo(:,:), w1(:,:), &
      w2(:,:), wlm(:,:), y(:,:), z(:,:), yk(:,:), yv(:,:), s(:,:)
  real(real64) :: acc, us_kv, us_pr, us_at, us_lm, us_win, nstar, j_win
  real(real64) :: per_tok_read(4), batched(2), j_read_tok
  ! train mode
  real(real64) :: act_floats, act_bytes_tok, act_bytes_lay, params, moments_mb
  real(real64) :: us_pr_layer, tstar(2), off_ms(2), step_ms
  type(energy_interval_t) :: iv_kv, iv_pr, iv_at, iv_lm
  type(probe_tier_t) :: c_ram_seq, c_ram_rand, c_disk_seq, c_disk_rand
  character(len=256) :: path, jpath, ckpt, sens_arg, sens, mode
  character(len=256) :: trust_msg
  logical :: trusted, cpu_energy

  call set_args('--mode infer --ckpt /tmp/mix/init3m --tokens 512 --reps 3 ' // &
      '--mb 256 --nreads 4096 --json /dev/null --path /tmp/tier_probe.bin --sensor none', &
      help_text=[character(len=96) :: &
      'NAME', '  tier_probe - recompute vs store/read: the memory-tier ledger', &
      '', 'SYNOPSIS', '  tier_probe [--mode infer|train] [--ckpt DIR] [--tokens N] [--reps R]', &
      '             [--mb M] [--nreads K] [--path F] [--json F] [--sensor S]', &
      '', 'DESCRIPTION', &
      '  --mode    infer = KV window (decode); train = activations + optimizer state', &
      '  --ckpt    checkpoint dir to read arch.txt from', &
      '  --tokens  window in tokens (infer) / tokens per batch element (train)', &
      '  --reps    repetitions per probe cell (median is reported)', &
      '  --mb      scratch KV/activation file size', &
      '  --nreads  random-offset single-item reads (latency probe)', &
      '  --json    append one JSON object per probe cell', &
      '  --sensor  force an energy sensor path (see the GPU warning in the output)'])
  mode = sget('mode'); ckpt = sget('ckpt'); n_tok = iget('tokens'); reps = iget('reps')
  file_mb = iget('mb'); nreads = iget('nreads')
  path = sget('path'); jpath = sget('json'); sens_arg = sget('sensor')

  ! P1: a arch vem do checkpoint pelo leitor UNICO (__metadata__ do safetensors
  ! primeiro, arch.txt como fallback) -- nao de um parser local. Ver
  ! docs/tier_probe.md e src/lib/load_weights.f90.
  call read_arch_any(trim(ckpt), dm, nh, nkv, hd, nl, vv, ctx, bos, arch_src)
  dkv = nkv*hd
  ff = 4*dm
  rv = storage_size(0.0_wp)/8
  n_tk_rep = int(n_tok, int64)*int(reps, int64)
  acc = 0.0_real64
  mode_id = 0
  if (trim(mode) == 'train') mode_id = 1

  if (len_trim(sens_arg) > 0 .and. trim(sens_arg) /= 'none') then
    call energy_init(sensor=trim(sens_arg))
  else
    call energy_init()
  end if
  sens = energy_sensor()
  cpu_energy = index(sens, 'powercap') > 0 .or. index(sens, 'rapl') > 0 .or. &
      index(sens, 'amd_energy') > 0 .or. index(sens, 'zenpower') > 0
  if (.not. cpu_energy) then
    write (*, '(A)') '#'
    write (*, '(A,A,A)') '# ATENCAO: sensor = ', trim(sens), '  NAO e energia de CPU.'
    write (*, '(A)') '#   Colunas/razoes de J medem ESSE dispositivo, nao o trabalho deste processo.'
    write (*, '(A)') '#   TEMPO, BANDA e LATENCIA continuam validos: nao dependem do sensor.'
    write (*, '(A)') '#'
  end if

  write (*, '(A,A)') '# modo: ', trim(mode)
  write (*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') '# arch (de '//trim(ckpt)//'/arch.txt): d_model=', &
      dm, ' n_head=', nh, ' n_kv=', nkv, ' head_dim=', hd, ' n_layer=', nl, ' vocab=', vv
  write (*, '(A,I0,A,I0,A,I0,A,I0)') '# janela_tokens=', n_tok, ' reps=', reps, &
      ' arquivo_MB=', file_mb, ' nreads=', nreads

  ! ---- 1) constantes da MAQUINA (probe generico) -------------------------
  nbytes_file = probe_file(path, file_mb, item_floats())
  write (*, '(A,I0,A,I0,A)') '# arquivo de trabalho: ', file_mb, ' MB (', nbytes_file, ' B)'
  call probe_cell(path, 'ram', 'seq', reps, item_floats(), nreads, c_ram_seq)
  call probe_cell(path, 'ram', 'rand', reps, item_floats(), nreads, c_ram_rand)
  call probe_cell(path, 'disk', 'seq', reps, item_floats(), nreads, c_disk_seq)
  call probe_cell(path, 'disk', 'rand', reps, item_floats(), nreads, c_disk_rand)
  call probe_report(c_ram_seq); call probe_report(c_ram_rand)
  call probe_report(c_disk_seq); call probe_report(c_disk_rand)
  trusted = probe_trust(c_ram_seq, c_disk_seq, trust_msg)
  write (*, '(A,L1,A,A)') '# probe confiavel: ', trusted, '  (', trim(trust_msg), ')'
  per_tok_read(1) = c_ram_seq%us_per_op
  per_tok_read(2) = c_ram_rand%us_per_op
  per_tok_read(3) = c_disk_seq%us_per_op
  per_tok_read(4) = c_disk_rand%us_per_op

  ! ---- 2) constantes do MODELO: recompute (medido) -----------------------
  allocate (x(n_tok, dm), wq(dm, dm), wk(dm, dkv), wv(dm, dkv), wo(dm, dm), &
      w1(dm, ff), w2(ff, dm), wlm(dm, vv), y(n_tok, dm), z(n_tok, ff), &
      yk(n_tok, dkv), yv(n_tok, dkv), s(n_tok, n_tok))
  call random_number(x);  call random_number(wq); call random_number(wk)
  call random_number(wv); call random_number(wo); call random_number(w1)
  call random_number(w2); call random_number(wlm)
  yk = matmul(x, wk); yv = matmul(x, wv); y = matmul(x, wq)

  do r = 1, reps                      ! K/V so' (x camada impede hoisting)
    do l = 1, nl
      yk = matmul(x, wk)*real(l, wp)
      yv = matmul(x, wv)*real(l, wp)
    end do
    acc = acc + real(sum(yk), real64) + real(sum(yv), real64)
  end do
  call energy_mark('recompute_kv', tokens=n_tk_rep, iv=iv_kv)

  do r = 1, reps                      ! projecoes encadeadas (1 bloco = 1 camada)
    y = x
    do l = 1, nl
      yk = matmul(y, wk)
      yv = matmul(y, wv)
      y  = matmul(y, wq)
      y  = matmul(y, wo)
      z  = matmul(y, w1)
      y  = matmul(z, w2)
    end do
    acc = acc + real(sum(y), real64)
  end do
  call energy_mark('recompute_proj', tokens=n_tk_rep, iv=iv_pr)

  do r = 1, reps                      ! atencao da janela (NxN)
    y = matmul(x, wq)
    do l = 1, nl
      s = matmul(y, transpose(y))
      y = matmul(s, y)*0.01_wp
    end do
    acc = acc + real(sum(y), real64)
  end do
  call energy_mark('recompute_attn', tokens=n_tk_rep, iv=iv_at)

  do r = 1, reps                      ! lm_head: 1x por token novo
    y = matmul(x, wlm)
    acc = acc + real(sum(y), real64)
  end do
  call energy_mark('recompute_lmhead', tokens=int(reps, int64), iv=iv_lm)

  us_pr = 1.0e6_real64*iv_pr%wall_s/max(1, n_tok*reps)
  us_at = 1.0e6_real64*iv_at%wall_s/max(1, n_tok*reps)
  us_lm = 1.0e6_real64*iv_lm%wall_s/max(1.0_real64, real(reps*n_tok, real64))

  if (mode_id == 0) then
    call report_infer()
  else
    call report_train()
  end if

  write (*, '(A,ES12.3)') '# checksum (impede DCE): ', acc
  write (*, '(A,ES12.3)') '# j_total do processo: ', energy_joules()

  if (trim(jpath) /= '/dev/null' .and. len_trim(jpath) > 0) then
    open (newunit=u, file=jpath, status='unknown', position='append', action='write')
    call probe_json(c_ram_seq, u, 'tier_probe'); call probe_json(c_ram_rand, u, 'tier_probe')
    call probe_json(c_disk_seq, u, 'tier_probe'); call probe_json(c_disk_rand, u, 'tier_probe')
    write (u, '(A,A,A,I0,A,F12.3,A,L1,A,A,A)') '{"level":"model","pattern":"', trim(mode), &
        '","unit_floats":', item_floats(), ',"us_recompute_unit_x1000":', &
        1000.0_real64*us_pr, ',"trusted":', trusted, ',"ckpt":"'//trim(ckpt)//'"}'
    close (u)
    write (*, '(A,A)') '# json anexado em: ', trim(jpath)
  end if
  write (*, '(A,A)') '# arquivo de trabalho deixado em: ', trim(path)

contains

  ! ---- unidade do probe: em infer = 1 token de KV; em train = 1 token de
  ! ativacao NA camada (o que se guardaria/leria por camada).
  function item_floats() result(nf)
    integer :: nf
    if (mode_id == 0) then
      nf = nl*2*dkv                     ! KV de 1 token (todas as camadas)
    else
      nf = act_floats_layer()           ! ativacoes de 1 token EM 1 camada
    end if
  end function item_floats

  ! Ativacoes guardadas por camada/token: vem do allocate de fortran_train.f90
  ! (C%e, C%xa, C%q, C%ao, C%e1, C%qr = 6*d_model; C%f = dff = 4*d_model;
  !  C%k, C%v, C%kr = 3*d_kv).
  function act_floats_layer() result(nf)
    integer :: nf
    nf = 10*dm + 3*dkv
  end function act_floats_layer

  ! ---- modo infer: janela de KV -----------------------------------------
  subroutine report_infer()
    real(real64) :: kv_bytes
    kv_bytes = real(nl*2*dkv*rv, real64)
    us_kv = 1.0e6_real64*iv_kv%wall_s/max(1, n_tok*reps)
    us_win = us_pr + us_at
    batched(1) = per_tok_read(2) + real(n_tok - 1, real64)*per_tok_read(1)
    batched(2) = per_tok_read(4) + real(n_tok - 1, real64)*per_tok_read(3)
    nstar = (per_tok_read(4) - us_pr)*real(n_tok, real64)/max(1.0e-9_real64, us_at)
    j_win = (iv_pr%j + iv_at%j)/max(1, n_tok*reps)
    j_read_tok = c_disk_rand%j/max(1.0_real64, real(nreads, real64))
    write (*, '(A,I0,A)') '# KV por token: ', int(kv_bytes), ' B'
    write (*, '(A,F9.2,A,F9.2,A,F9.2,A,F9.2)') '# us/token do MODELO: kv=', us_kv, &
        '  proj=', us_pr, '  attn=', us_at, '  janela(proj+attn)=', us_win
    write (*, '(A,F9.2,A)') '# lm_head (1x por token novo): ', us_lm
    write (*, '(A,4F12.3)') '# MAQUINA us/token lido (seq_ram rand_ram seq_disk rand_disk): ', per_tok_read
    write (*, '(A,F5.0,A,F11.1,A,F11.1,A)') '# janela de ', real(n_tok, real64), &
        ' tokens: recomputar=', us_win*n_tok, ' us   ler(1 IO/token, frio)=', per_tok_read(4)*n_tok, ' us'
    write (*, '(A,F11.1,A,F11.1,A,F6.1,A)') '# em LOTE: ', batched(2), ' us (frio)  ', &
        batched(1), ' us (RAM)  -> recomputar/ler = ', us_win*n_tok/max(1.0e-9_real64, batched(2)), 'x'
    write (*, '(A,F8.1,A)') '# janela N* (empata com leitura fria token-a-token): ', nstar, ' tokens'
    write (*, '(A,ES11.3,A,ES11.3,A,F7.1,A)') '# energia por token: recomputar ', j_win, &
        ' J  ler(disco,rand) ', j_read_tok, ' J  -> razao ', j_read_tok/max(1.0e-12_real64, j_win), 'x'
  end subroutine report_infer

  ! ---- modo train: ativacoes + estado do otimizador ----------------------
  subroutine report_train()
    real(real64) :: act_mb, per_lay(4), off_bytes
    act_floats = real(act_floats_layer(), real64)
    act_bytes_lay = act_floats*real(rv, real64)                  ! por token, por camada
    act_bytes_tok = act_bytes_lay*real(nl, real64)               ! por token, todas as camadas
    act_mb = act_bytes_tok*real(n_tok, real64)/1.0e6_real64      ! lote de n_tok tokens
    ! parametros do modelo a partir da arch (mesma conta do checkpoint: ver
    ! docs/fortran_gpt.md e os shapes reais): wte + lm_head + por camada
    ! q,k,v,o,fc,proj
    params = 2.0_real64*real(vv, real64)*real(dm, real64) + &
        real(nl, real64)*real(2*dm*dm + 2*dm*dkv + 8*dm*dm, real64)
    moments_mb = 2.0_real64*params*real(rv, real64)/1.0e6_real64
    us_pr_layer = us_pr/real(nl, real64)                         ! 1 camada = 1 bloco
    ! T* do checkpointing: guardar/ler a ativacao de UMA camada vs recomputar
    per_lay(1) = act_bytes_lay/1.0e6_real64/max(1.0e-9_real64, c_ram_seq%bw_mbps)*1.0e6_real64
    per_lay(2) = act_bytes_lay/1.0e6_real64/max(1.0e-9_real64, c_disk_seq%bw_mbps)*1.0e6_real64
    ! N* = latencia por IO / (recomputar a camada - ler a ativacao dela). Vale
    ! para leitura DISPERSA (1 IO por token da janela); leitura sequencial de
    ! ativacao (o caso real do backward, em pilha) nao paga latencia por token.
    if (us_pr_layer > per_lay(1)) then
      tstar(1) = per_tok_read(2)/max(1.0e-9_real64, us_pr_layer - per_lay(1))
    else
      tstar(1) = -1.0_real64
    end if
    if (us_pr_layer > per_lay(2)) then
      tstar(2) = per_tok_read(4)/max(1.0e-9_real64, us_pr_layer - per_lay(2))
    else
      tstar(2) = -1.0_real64
    end if
    ! offload do estado do otimizador: 2P bytes lidos 1x por passo
    off_bytes = 2.0_real64*params*real(rv, real64)
    off_ms(1) = 1.0e3_real64*off_bytes/1.0e6_real64/max(1.0e-9_real64, c_ram_seq%bw_mbps)
    off_ms(2) = 1.0e3_real64*off_bytes/1.0e6_real64/max(1.0e-9_real64, c_disk_seq%bw_mbps)
    step_ms = 1.0e-3_real64*us_pr*real(n_tok, real64)            ! forward do lote (so' proj)
    write (*, '(A,I0,A,I0,A,I0,A)') '# ativacoes (do allocate de fortran_train): ', &
        act_floats_layer(), ' floats/camada/token (', int(act_bytes_lay), &
        ' B) e ', int(act_bytes_tok), ' B/token no modelo todo'
    write (*, '(A,F10.3,A,F10.3,A)') '# lote de ', real(n_tok, real64), &
        ' tokens guarda ', act_mb, ' MB de ativacao (fwd)'
    write (*, '(A,F12.1,A)') '# parametros (da arch): ', params, ' (m,v = 2P)'
    write (*, '(A,F10.2,A)') '# estado do otimizador: ', moments_mb, ' MB (fp32) -- 1x por passo se offload'
    write (*, '(A,F10.3,A,F9.3,A,F9.3,A)') '# recompute de 1 camada/token: ', us_pr_layer, &
        ' us   ler a ativacao dela (seq): RAM ', per_lay(1), ' us  disco ', per_lay(2), ' us'
    write (*, '(A,F10.1,A,F10.1,A)') '# T* (guardar vs recomputar 1 camada, tokens): RAM ', tstar(1), &
        '  disco ', tstar(2), '  (>: recomputar vence)'
    write (*, '(A,F9.2,A,F9.2,A,F9.2,A)') '# offload do estado do otimizador, 1x/passo: RAM ', off_ms(1), &
        ' ms  disco ', off_ms(2), ' ms   (forward do lote = ', step_ms, ' ms)'
    if (.not. trusted) then
      write (*, '(A)') '# SEM VEREDITO: o probe nao confia nas celulas acima (maquina ocupada?'
      write (*, '(A)') '#   celula instavel?). Os numeros derivados ACIMA estao contaminados --'
      write (*, '(A)') '#   repita com a maquina ociosa (ou use --reps maior) antes de decidir.'
      return
    end if
    if (off_ms(2) > 0.1_real64*step_ms) then
      write (*, '(A)') '# VEREDITO: offload de m,v para DISCO custa >10% de um passo -> manter em'
      write (*, '(A)') '#   RAM, ou encolher (bf16/fp16 -> metade dos bytes), ou nao carregar entre syncs.'
    else
      write (*, '(A)') '# VEREDITO: offload de m,v para disco custa <10% de um passo NESTA maquina.'
    end if
    if (tstar(2) > real(n_tok, real64)) then
      write (*, '(A)') '# VEREDITO: nestes T, recomputar a camada no backward e'' MAIS CARO que ler'
      write (*, '(A)') '#   a ativacao do disco nesta maquina -> guardar (RAM, se couber).'
    end if
  end subroutine report_train

end program tier_probe
