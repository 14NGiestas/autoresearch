! mpi/app/train_ddp.f90 — treinador DATA-PARALLEL de verdade (MPI), tau = --sync_every.
!
! Por que existe: a emulacao de varios workers que ja temos troca checkpoint por
! scp e so testa tau GRANDE. A matematica de local SGD so promete imposto ZERO
! em tau = 1 (a media dos gradientes de K lotes e o gradiente de um lote K vezes
! maior). MPI e o que destrava medir tau = 1 com K processos de verdade.
!
! --sync grad   DDP classico. A cada H passos: MPI_Allreduce(MPI_SUM) sobre CADA
!               array de GR, divide por size e SO ENTAO apply_update. Todos os
!               ranks aplicam o MESMO update sobre os MESMOS pesos -> ficam
!               BIT-IDENTICOS (e o invariante que o teste checa). Com H>1 os
!               gradientes de H micro-lotes locais sao somados antes do reduce
!               (gradient accumulation); H=1 e o DDP puro.
! --sync delta  local SGD com media de pesos, agora via MPI: guarda uma copia dos
!               pesos no inicio do bloco, roda H passos locais (cada rank com o
!               seu lote, cada um aplicando o proprio update) e no fim faz
!               Allreduce do delta (M - M0), divide por size e M = M0 + delta/size.
!               Os momentos do Adam NAO sao promediados (so os pesos): ver
!               docs/mpi_ddp.md, "o que NAO foi verificado".
!
! --data slice  rank r le a fatia [r*chunk, (r+1)*chunk) do pool, em ciclo:
!               offset = mod(rank*chunk + mod(k-1, chunk), ntrain). Com
!               --chunk ntrain TODOS os ranks leem os MESMOS dados — e assim que
!               o teste de invariante compara 1 rank x 2 ranks.
! --data rotate rank r le a linha (passo*size + r) mod ntrain: lotes disjuntos a
!               cada passo, pool inteiro coberto a cada passo (data-parallel puro).
!
! Checkpoint: apenas no rank 0 (--save_all 1 grava tambem out/rank_<r>/...).
! Sem fork e sem /tmp: mkdir_p (nada de execute_command_line).
!
! Build/uso: docs/mpi_ddp.md. Este e um projeto fpm SEPARADO (mpi/fpm.toml) com
! --build-dir proprio: nada aqui toca src/build/arch_* (jobs na fila dependem
! daqueles binarios).

program train_ddp
  use mpi
  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, real64
  use fortran_train_mod, only: wp, dims_t, params_t, state_t, cache_t, temp_t, &
      forward_save, compute_grads, apply_update, init_state, init_temp
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights, save_gpt_weights_st
  use fortran_adam_state_mod, only: load_adam_state, save_adam_state, &
      load_muon_state, save_muon_state
  use fortran_data_mod, only: load_batch
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER, A_VOCAB => VV, A_CTX => TT, A_BOS => BOS, &
      write_arch_txt, require_arch
  use M_CLI2, only: set_args, sget, rget, iget, specified
  use fortran_sys_mod, only: mkdir_p
  use fortran_linear_mod, only: wte_lookup
  use fortran_blas_mod, only: linear3d_sgemm
  use fortran_rmsnorm_mod, only: rmsnorm0
  use fortran_rope_mod, only: rope_4d
  use fortran_attn_mod, only: causal_attn, attn_sgemm, relu2
  implicit none

  integer, parameter :: B = 1
  integer, parameter :: TT = A_CTX, D = A_D
  integer, parameter :: N_HEAD = A_HEAD, N_KV = A_KV, HD = A_HD
  integer, parameter :: N_LAYER = A_LAYER, VV = A_VOCAB
  ! --sync-fp16: 0 = DESLIGADO (caminho fp32 bit-exato, o default e o que a
  ! secao 4 do docs/mpi_ddp.md prova byte a byte). 1 = bf16 (topo do fp32).
  ! 2 = ponto fixo de 16 bits com escala dinamica (mais preciso que fp16 para
  ! gradiente). Nenhum dos dois e' IEEE half: Fortran/MPICH nao tem real16.
  integer, parameter :: QP_NONE = 0, QP_BF16 = 1, QP_FIX = 2

  type(dims_t) :: G
  type(params_t) :: M, GR, GA, MB
  type(state_t) :: S
  type(cache_t) :: C
  type(temp_t) :: tmp
  integer(c_int) :: idx(B*TT), targets(B*TT)
  real(wp) :: ct(TT*(HD/2)), st(TT*(HD/2))
  real(wp), allocatable :: scratch(:)
  ! --sync-fp16 (nice-to-have, DEFAULT OFF): buffers do coletivo de 16 bits.
  ! qbuf = payload quantizado (int16), qall = ida/volta do coletivo, qtmp/qacc =
  ! decodificacao/soma no root (bf16) ou destino do SUM exato (fix).
  integer(int16), allocatable :: qbuf(:), qall(:)
  real(wp), allocatable :: qtmp(:), qacc(:)
  real(real64) :: qmax(8), qmaxall(8)
  integer, allocatable :: tbytes(:)
  character(len=512) :: wdir, rowsfile, outdir, bytesfile, rdir
  character(len=MPI_MAX_PROCESSOR_NAME) :: procname
  character(len=16) :: smode, dmode
  character(len=8) :: ckfmt, qmode_s
  integer :: ierr, rank, nranks, nsteps, t0, k, tstep, ngot, row, u, ios, i, j
  integer :: namelen, start_row, ntrain, save_every, log_every, sync_every
  integer :: nval, val_every, chunk, chunk_opt, mpi_wp, qmode, nq
  integer(int64) :: qbytes
  integer :: sync_mode, data_mode   ! 1 = grad|slice, 2 = delta|rotate
  logical :: am_root, save_all, use_muon, attn_blas, muon_found, adam_found
  logical :: sync_due, sync_log
  real(wp) :: lr, lr_eff, muon_lr, nll, vnll, theta, ang
  real(real64) :: t_start, t_end, comm_s, wall
  integer(int64) :: fp, fpmin, fpmax, tokens, n_el

  ! ---- MPI primeiro: rank/size decidem o que cada processo le e escreve ----
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  call MPI_Get_processor_name(procname, namelen, ierr)
  am_root = (rank == 0)
  mpi_wp = merge(MPI_REAL4, MPI_REAL8, wp == real32)

  ! ---- flags ----
  nsteps = 10; lr = 0.0003_wp; t0 = 1; start_row = 0; ntrain = 40
  save_every = 10; log_every = 1; sync_every = 1; chunk_opt = 0
  nval = 0; val_every = 0; bytesfile = "BYTES"
  call set_args('--weights WEIGHTS --rows ROWS --out OUT --nsteps 10' // &
      ' --lr 0.0003 --t0 1 --ntrain 40 --start_row 0 --bytes BYTES' // &
      ' --save_every 10 --log_every 1 --sync_every 1 --sync grad' // &
      ' --data slice --chunk 0 --save_all 0 --attn blas' // &
      ' --opt adam --muon-lr 0.02 --ckpt-format npy' // &
      ' --nval 0 --val_every 0 --sync_log 0 --sync-fp16 none', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  train_ddp - data-parallel MPI trainer (sync grad | sync delta)', &
      'SYNOPSIS', &
      '  mpirun -n K train_ddp --weights DIR --rows FILE --out DIR --nsteps N', &
      'OPTIONS', &
      '  --attn blas|naive     kernel de atencao. Default blas (attn_sgemm, o', &
      '                        mesmo que o train_run usa nos runs registrados:', &
      '                        ~13x mais rapido que o naive em T=1024 e', &
      '                        bit-identico entre ranks, entao a invariante do', &
      '                        DDP nao muda). naive = causal_attn, mantido como', &
      '                        opcao para reproduzir medidas antigas.', &
      '  --sync grad|delta     grad: Allreduce(GR) then apply_update, so every', &
      '                        rank applies the same update (bit-identical).', &
      '                        delta: H local steps, then Allreduce(mean of the', &
      '                        weight delta) = local SGD weight averaging.', &
      '  --sync_every H        steps between synchronizations (tau). H=1 is the', &
      '                        only tau where the math says the tax is zero.', &
      '  --data slice|rotate   slice: rank r reads rows [r*chunk,(r+1)*chunk) of', &
      '                        the pool, cycling; rotate: row (step*size+r) mod ntrain.', &
      '  --chunk N             rows per rank in --data slice. 0 (default) =', &
      '                        ntrain/size. N=ntrain makes every rank read the', &
      '                        WHOLE pool: the 1-rank x K-rank invariant test.', &
      '  --save_all 1          every rank writes out/rank_<r>/... (default: rank 0).', &
      '  --sync_log 1          imprime o tempo de CADA MPI_Allreduce (separar o', &
      '                        custo do coletivo da contencao de CPU). Default 0.', &
      '  --sync-fp16 none|bf16|fix   DEFAULT none = payload fp32 (38,04 MB por', &
      '                        sync) e TRAJETORIA BIT-EXATA -- e o ativo de', &
      '                        verificacao; nao mexa nele. bf16/fix cortam o', &
      '                        payload para 16 bits (19,02 MB, 2x menos trafego', &
      '                        de rede): bf16 = 8 bits de mantissa (RNE no topo do', &
      '                        fp32, Gather+soma no root+Bcast); fix = ponto fixo', &
      '                        com escala dinamica (MAX por grupo) num unico', &
      '                        MPI_Allreduce(MPI_INT16_T, SUM) exato. Nos dois,', &
      '                        TODOS os ranks terminam com o MESMO buffer', &
      '                        (seguem bit-identicos entre si); o que muda e'' a', &
      '                        trajetoria contra o run fp32 (desvio medido em', &
      '                        mpi/tests_quant.sh). nao existe IEEE half: o modo', &
      '                        fix e'' o "fp16" pratico (15 bits efetivos).', &
      '  --ckpt-format npy|st|both   npy (historico) | st (model.safetensors) | both.', &
      '                        arch.txt viaja nos tres.'], &
      version_text=[character(len=80) :: 'train_ddp 1.0'])
  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  outdir = trim(sget('out'))
  bytesfile = trim(sget('bytes'))
  nsteps = iget('nsteps'); lr = rget('lr'); t0 = iget('t0')
  start_row = iget('start_row'); ntrain = iget('ntrain')
  save_every = iget('save_every'); log_every = iget('log_every')
  sync_every = iget('sync_every'); chunk_opt = iget('chunk')
  nval = iget('nval'); val_every = iget('val_every')
  save_all = iget('save_all') /= 0
  sync_log = iget('sync_log') /= 0
  attn_blas = trim(sget('attn')) == 'blas'
  use_muon = trim(sget('opt')) == 'muon'
  muon_lr = rget('muon-lr')
  ckfmt = trim(sget('ckpt-format'))
  smode = trim(sget('sync')); dmode = trim(sget('data'))
  qmode_s = trim(sget('sync-fp16'))
  select case (qmode_s)
  case ('none', 'off', '0')
    qmode = QP_NONE
  case ('bf16')
    qmode = QP_BF16
  case ('fix', 'fp16')   ! fp16 = alias do ponto fixo (nao existe real16 aqui)
    qmode = QP_FIX
  case default
    call die("require --sync-fp16 none|bf16|fix")
  end select
  muon_found = .false.; adam_found = .false.
  if (.not. specified('weights')) call die("require --weights")
  if (.not. specified('rows')) call die("require --rows")
  if (.not. specified('out')) call die("require --out")
  if (nsteps < 1) call die("require --nsteps>=1")
  if (smode /= 'grad' .and. smode /= 'delta') call die("require --sync grad|delta")
  if (dmode /= 'slice' .and. dmode /= 'rotate') call die("require --data slice|rotate")
  if (ckfmt /= 'npy' .and. ckfmt /= 'st' .and. ckfmt /= 'both') &
      call die("require --ckpt-format npy|st|both")
  sync_mode = merge(1, 2, smode == 'grad')
  data_mode = merge(1, 2, dmode == 'slice')
  if (sync_every < 1) sync_every = 1
  if (ntrain < 1) call die("--ntrain must be >= 1")
  if (chunk_opt > 0) then
    chunk = chunk_opt
  else
    chunk = ntrain / nranks
    if (chunk < 1) call die("ntrain < nranks: use --chunk to overlap slices")
  end if
  if (bytesfile == 'BYTES') &
      bytesfile = trim(wdir) // '/../tok_tables/token_bytes.txt'
  if (sync_mode == 2 .and. mod(save_every, sync_every) /= 0 .and. am_root) then
    print '(A)', "warning: --sync delta com save_every nao multiplo de sync_every:" // &
        " um checkpoint no meio do bloco NAO tem os pesos promediados"
  end if

  ! ---- run auto-descritivo no log ----
  print '(A,I0,A,I0,A,A)', "rank ", rank, "/", nranks - 1, " host ", trim(procname)
  if (am_root) then
    print '(A,I0)', "nranks      ", nranks
    print '(A,A,A,I0)', "sync        ", trim(smode), "  tau ", sync_every
    print '(A,A)', "sync-fp16   ", trim(qmode_s)
    print '(A,A,A,I0)', "data        ", trim(dmode), "  chunk/rank ", chunk
    print '(A,A)', "weights     ", trim(wdir)
    print '(A,A)', "rows        ", trim(rowsfile)
    print '(A,A)', "out         ", trim(outdir)
    print '(A,A)', "ckpt-format ", trim(ckfmt)
    print '(A,I0,A,I0,A,I0)', "pool        ", ntrain, " rows from ", start_row, &
        "  slice/rank ", chunk
  end if
  if (sync_mode == 1) then
    print '(A,I0,A,I0,A,I0)', "rank ", rank, ": slice starts at pool row ", &
        mod(rank*chunk, ntrain), " len ", chunk
  else
    print '(A,I0,A,I0)', "rank ", rank, ": rotate offset ", rank, " of ", nranks
  end if
  flush (6)

  ! ---- setup ----
  G%B = B; G%T = TT; G%V = VV; G%D = D
  G%nh = N_HEAD; G%nkv = N_KV; G%hd = HD; G%nl = N_LAYER
  G%eps = 1.0e-5_wp

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  call require_arch(trim(wdir))
  call init_state(M, S)
  call load_adam_state(trim(wdir), S, adam_found)
  if (use_muon) call load_muon_state(trim(wdir), S, muon_found)
  if (am_root) then
    print '(A,L1,A,L1)', "resumed Adam moments=", adam_found, "  muon=", muon_found
  end if
  call init_temp(G, tmp)
  call alloc_params(GR, M); call alloc_params(GA, M); call alloc_params(MB, M)
  allocate (scratch(max(size(M%wte), size(M%lm), size(M%q), size(M%fc))))
  call copy_params(MB, M)
  if (qmode /= QP_NONE) then
    nq = max(size(M%wte), size(M%lm), size(M%q), size(M%k), size(M%v), &
        size(M%p), size(M%fc), size(M%p2))
    allocate (qbuf(nq), qall(nq*nranks), qtmp(nq), qacc(nq))
    qbuf = 0_int16; qall = 0_int16; qtmp = 0.0_wp; qacc = 0.0_wp
    qbytes = 2_int64*int(size(M%wte) + size(M%lm) + size(M%q) + size(M%k) + &
        size(M%v) + size(M%p) + size(M%fc) + size(M%p2), int64)
    ! Sanidade do datatype de 16 bits ANTES de usar (1 elemento): se este MPI nao
    ! souber somar MPI_INT16_T, queremos saber aqui, com mensagem, e nao num
    ! silencio/exit 1 no meio de um run (foi o que custou os jobs 114/116).
    qbuf(1) = 3_int16
    call MPI_Allreduce(qbuf(1:1), qall(1:1), 1, MPI_INT16_T, MPI_SUM, &
        MPI_COMM_WORLD, ierr)
    if (ierr /= MPI_SUCCESS .or. int(qall(1)) /= 3*nranks) then
      print '(A,I0,A,I0)', "rank ", rank, ": MPI_INT16_T/SUM falhou (ierr=", &
          ierr, ", soma=", int(qall(1))
      call die("--sync-fp16: este MPI nao soma MPI_INT16_T")
    end if
    if (am_root) print '(A,I0,A,I0,A)', "payload 16b ", qbytes, &
        " bytes por sync (vs 38043648 fp32)"
  end if

  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_wp ** (-2.0_wp * real(j - 1, wp) / real(HD, wp))
      ang = real(i - 1, wp) * theta
      ct((i-1)*(HD/2)+j) = cos(ang)
      st((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do

  if (nval > 0 .or. val_every > 0) then
    allocate (tbytes(0:VV-1))
    open (newunit=u, file=trim(bytesfile), status="old", iostat=ios)
    if (ios /= 0) call die("cannot open bytes file: "//trim(bytesfile))
    do i = 0, VV - 1
      read (u, *, iostat=ios) tbytes(i)
      if (ios /= 0) call die("short bytes file")
    end do
    close (u)
  end if

  ! Diretorios de checkpoint pre-criados com os pools de OpenMP ainda frios:
  ! mkdir_p no meio do run so bate no guard "ja existe" (nada de fork).
  if (am_root) then
    if (mkdir_p(trim(outdir)) /= 0) call die("cannot create out dir: "//trim(outdir))
  end if
  if (save_all .and. .not. am_root) then
    write (rdir, '(A,A,I0)') trim(outdir), '/rank_', rank
    if (mkdir_p(trim(rdir)) /= 0) call die("cannot create rank dir")
  end if
  do k = 1, nsteps
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      if (am_root .or. save_all) then
        if (mkdir_p(trim(ckdir_for(t0 + k - 1))) /= 0) call die("cannot create ckpt dir")
      end if
    end if
  end do

  ! ---- loop ----
  tokens = 0_int64; comm_s = 0.0_real64
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t_start = MPI_Wtime()
  do k = 1, nsteps
    tstep = t0 + k - 1
    ! Warmup linear de 2 passos, igual ao train_run (k relativo ao run): sem ele o
    ! primeiro passo queima o estado e 1-rank e K-rank nao comparam.
    lr_eff = lr * min(1.0_wp, real(k, wp) / 2.0_wp)
    sync_due = (mod(k, sync_every) == 0 .or. k == nsteps)
    ! delta: fotografia dos pesos no INICIO do bloco (antes do 1o passo local)
    if (sync_mode == 2 .and. mod(k - 1, sync_every) == 0) call copy_params(MB, M)

    call pick_row(k)
    call load_batch(trim(rowsfile), row, B, TT, idx, targets, ngot)
    if (ngot < B) then
      print '(A,I0,A,I0)', "rank ", rank, ": rows file too short at row ", row
      call die("rows file too short")
    end if

    call forward_save(idx, targets, ct, st, M, G, C, tmp, nll, attn_blas=attn_blas)
    call compute_grads(idx, targets, ct, st, M, G, C, GR, tmp, nll, attn_blas=attn_blas)
    if (sync_mode == 1) then
      ! DDP: acumula os micro-lotes locais, media entre ranks, aplica UMA vez.
      call accum_params(GA, GR)
      if (sync_due) then
        if (qmode == QP_NONE) then
          call allreduce_params(GA, comm_s)
          call scale_params(GA, 1.0_wp / real(nranks, wp))
        else
          call allreduce_quant_params(GA, comm_s)   ! ja' devolve a MEDIA
        end if
        call apply_update(M, S, GA, tstep, lr_eff, 0.9_wp, 0.999_wp, 1.0e-8_wp, &
            0.0_wp, use_muon, muon_lr, G)
        call zero_params(GA)
      end if
    else
      ! local SGD: o update do passo e LOCAL; a media entra so no delta.
      call apply_update(M, S, GR, tstep, lr_eff, 0.9_wp, 0.999_wp, 1.0e-8_wp, &
          0.0_wp, use_muon, muon_lr, G)
      if (sync_due) then
        call sub_params(GR, M, MB)          ! GR = M - M0 (scratch reutilizada)
        if (qmode == QP_NONE) then
          call allreduce_params(GR, comm_s)
          call scale_params(GR, 1.0_wp / real(nranks, wp))
        else
          call allreduce_quant_params(GR, comm_s)   ! ja' devolve a MEDIA
        end if
        call add3_params(M, MB, GR)         ! M = M0 + mean(delta)
      end if
    end if
    tokens = tokens + int(B*TT, int64)

    if (mod(k, log_every) == 0 .or. k == nsteps) then
      fp = fp_params(M)
      print '(A,I0,A,I0,A,I0,A,F10.5,A,F10.5,A,Z16.16,A,I0)', "rank ", rank, &
          " step ", tstep, " row ", row, " nll ", nll, " lr ", lr_eff, &
          " fp ", fp, " sync ", merge(1, 0, sync_due)
      flush (6)
    end if
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      if (am_root .or. save_all) call save_ckpt(tstep, lr_eff)
    end if
    if (val_every > 0 .and. nval > 0 .and. (mod(k, val_every) == 0 .or. k == nsteps)) then
      if (am_root) then
        vnll = val_bpb(trim(rowsfile), start_row + ntrain, nval)
        print '(A,I0,A,F10.5)', "val @", tstep, " bpb ", vnll
        flush (6)
      end if
    end if
  end do
  t_end = MPI_Wtime()
  wall = t_end - t_start

  ! ---- invariante de rank: a ultima coisa que o loop faz e' sincronizar, entao
  ! TODOS os ranks tem de terminar com o MESMO peso, bit a bit. O MIN/MAX do
  ! fingerprint e a afirmacao do proprio app (a prova dura e' `cmp` nos .npy). ----
  fp = fp_params(M)
  call MPI_Allreduce(fp, fpmin, 1, MPI_INTEGER8, MPI_MIN, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(fp, fpmax, 1, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, ierr)
  n_el = int(size(M%wte), int64) + int(size(M%lm), int64) + int(size(M%q), int64) &
      + int(size(M%k), int64) + int(size(M%v), int64) + int(size(M%p), int64) &
      + int(size(M%fc), int64) + int(size(M%p2), int64)
  print '(A,I0,A,Z16.16)', "rank ", rank, " final fp ", fp
  if (am_root) then
    print '(A,I0)', "weight elements ", n_el
    print '(A,Z16.16,A,Z16.16,A,L1)', "bit-identical across ranks: min 0x", &
        fpmin, " max 0x", fpmax, " -> ", (fpmin == fpmax)
    print '(A,I0,A,I0,A,F10.3,A,F10.3,A,F8.5)', "tokens/rank ", tokens, &
        " total ", tokens*int(nranks, int64), " wall_s ", wall, &
        " allreduce_s ", comm_s, " allreduce_frac ", comm_s/max(wall, 1.0d-9)
    print '(A,F10.3,A,F10.2)', "s/step (global) ", wall/real(nsteps, real64), &
        "  tokens/s global ", real(tokens*int(nranks, int64), real64)/max(wall, 1.0d-9)
    print '(A,F10.4)', "sync tax (s/step/rank) ", comm_s/real(nsteps, real64)
    flush (6)
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call MPI_Finalize(ierr)

contains

  ! ---- params_t: alocar / copiar / somar / escalar (8 grupos, sempre na mesma ordem) ----
  subroutine alloc_params(P, R)
    type(params_t), intent(out) :: P
    type(params_t), intent(in) :: R
    call like_params(P%wte, R%wte); call like_params(P%lm, R%lm)
    call like_params(P%q, R%q); call like_params(P%k, R%k); call like_params(P%v, R%v)
    call like_params(P%p, R%p); call like_params(P%fc, R%fc); call like_params(P%p2, R%p2)
  end subroutine alloc_params

  subroutine like_params(d, s)
    real(wp), allocatable, intent(out) :: d(:)
    real(wp), intent(in) :: s(:)
    allocate (d(size(s))); d = 0.0_wp
  end subroutine like_params

  subroutine copy_params(D, R)
    type(params_t), intent(inout) :: D
    type(params_t), intent(in) :: R
    D%wte = R%wte; D%lm = R%lm; D%q = R%q; D%k = R%k; D%v = R%v
    D%p = R%p; D%fc = R%fc; D%p2 = R%p2
  end subroutine copy_params

  subroutine zero_params(P)
    type(params_t), intent(inout) :: P
    P%wte = 0.0_wp; P%lm = 0.0_wp; P%q = 0.0_wp; P%k = 0.0_wp; P%v = 0.0_wp
    P%p = 0.0_wp; P%fc = 0.0_wp; P%p2 = 0.0_wp
  end subroutine zero_params

  subroutine accum_params(D, R)
    type(params_t), intent(inout) :: D
    type(params_t), intent(in) :: R
    D%wte = D%wte + R%wte; D%lm = D%lm + R%lm; D%q = D%q + R%q; D%k = D%k + R%k
    D%v = D%v + R%v; D%p = D%p + R%p; D%fc = D%fc + R%fc; D%p2 = D%p2 + R%p2
  end subroutine accum_params

  subroutine sub_params(D, A, Bp)
    type(params_t), intent(inout) :: D
    type(params_t), intent(in) :: A, Bp
    D%wte = A%wte - Bp%wte; D%lm = A%lm - Bp%lm; D%q = A%q - Bp%q
    D%k = A%k - Bp%k; D%v = A%v - Bp%v; D%p = A%p - Bp%p
    D%fc = A%fc - Bp%fc; D%p2 = A%p2 - Bp%p2
  end subroutine sub_params

  subroutine add3_params(D, A, Bp)
    type(params_t), intent(inout) :: D
    type(params_t), intent(in) :: A, Bp
    D%wte = A%wte + Bp%wte; D%lm = A%lm + Bp%lm; D%q = A%q + Bp%q
    D%k = A%k + Bp%k; D%v = A%v + Bp%v; D%p = A%p + Bp%p
    D%fc = A%fc + Bp%fc; D%p2 = A%p2 + Bp%p2
  end subroutine add3_params

  subroutine scale_params(P, s)
    type(params_t), intent(inout) :: P
    real(wp), intent(in) :: s
    P%wte = P%wte*s; P%lm = P%lm*s; P%q = P%q*s; P%k = P%k*s; P%v = P%v*s
    P%p = P%p*s; P%fc = P%fc*s; P%p2 = P%p2*s
  end subroutine scale_params

  ! Soma elemento a elemento entre ranks (in place), cronometrando o coletivo:
  ! e esse numero que responde "quanto custa tau" (o imposto de sincronizacao).
  subroutine allreduce_params(P, acc)
    type(params_t), intent(inout) :: P
    real(real64), intent(inout) :: acc
    real(real64) :: w0
    w0 = MPI_Wtime()
    call ar_params(P%wte); call ar_params(P%lm)
    call ar_params(P%q); call ar_params(P%k); call ar_params(P%v)
    call ar_params(P%p); call ar_params(P%fc); call ar_params(P%p2)
    acc = acc + (MPI_Wtime() - w0)
    ! --sync_log 1: um numero por COLETIVO, para separar custo fixo (setup do
    ! canal shm/primeiro toque) do custo marginal por sync.
    if (sync_log) then
      print '(A,I0,A,I0,A,F12.5)', "  sync step ", tstep, " rank ", rank, &
          " allreduce_s ", MPI_Wtime() - w0
      flush (6)
    end if
  end subroutine allreduce_params

  subroutine ar_params(a)
    real(wp), intent(inout) :: a(:)
    integer :: n
    n = size(a)
    call MPI_Allreduce(a, scratch(1:n), n, mpi_wp, MPI_SUM, MPI_COMM_WORLD, ierr)
    a = scratch(1:n)
  end subroutine ar_params

  ! ---------------------------------------------------------------------------
  ! Coletivo de 16 bits (--sync-fp16 bf16|fix). NAO e' o caminho default: o
  ! default (qmode = QP_NONE) segue por allreduce_params, fp32, bit-exato.
  ! Devolve P = MEDIA entre ranks do buffer de entrada (o caller NAO divide de
  ! novo) e TODOS os ranks saem com o MESMO conteudo quantizado. Payload 19,02 MB
  ! em vez de 38,04 MB.
  ! ---------------------------------------------------------------------------
  subroutine allreduce_quant_params(P, acc)
    type(params_t), intent(inout) :: P
    real(real64), intent(inout) :: acc
    real(real64) :: w0
    w0 = MPI_Wtime()
    if (qmode == QP_FIX) then
      ! Escala dinamica: MAX de |g| por grupo, em UM coletivo de 8 real64 (64 B).
      ! Sem isso a escala teria de vir do passo anterior (stale) e poderia clipar.
      qmax(1) = maxval(abs(P%wte)); qmax(2) = maxval(abs(P%lm))
      qmax(3) = maxval(abs(P%q));   qmax(4) = maxval(abs(P%k))
      qmax(5) = maxval(abs(P%v));   qmax(6) = maxval(abs(P%p))
      qmax(7) = maxval(abs(P%fc));  qmax(8) = maxval(abs(P%p2))
      call MPI_Allreduce(qmax, qmaxall, 8, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
    end if
    call q_grp(P%wte, 1); call q_grp(P%lm, 2)
    call q_grp(P%q, 3);   call q_grp(P%k, 4);   call q_grp(P%v, 5)
    call q_grp(P%p, 6);   call q_grp(P%fc, 7);  call q_grp(P%p2, 8)
    acc = acc + (MPI_Wtime() - w0)
  end subroutine allreduce_quant_params

  ! Um grupo de GR/delta. bf16: Gather(0) + soma em fp32 no root + Bcast do
  ! resultado requantizado (2 coletivos de 19 MB = 38 MB de link, contra 76 MB
  ! do allreduce fp32). fix: UM Allreduce(MPI_INT16_T, SUM) — a pre-escala por
  ! 1/nranks garante que a soma INTEIRA caiba no int16, entao ela e' exata.
  subroutine q_grp(a, ig)
    real(wp), intent(inout) :: a(:)
    integer, intent(in) :: ig
    real(wp) :: step
    integer :: n, r
    n = size(a)
    if (qmode == QP_BF16) then
      call bf16_enc(a, qbuf(1:n))
      call MPI_Gather(qbuf, n, MPI_INT16_T, qall, n, MPI_INT16_T, 0, &
          MPI_COMM_WORLD, ierr)
      if (am_root) then
        qacc(1:n) = 0.0_wp
        do r = 1, nranks
          call bf16_dec(qall((r-1)*n+1:r*n), qtmp(1:n))
          qacc(1:n) = qacc(1:n) + qtmp(1:n)
        end do
        qacc(1:n) = qacc(1:n) / real(nranks, wp)
        call bf16_enc(qacc(1:n), qbuf(1:n))
      end if
      call MPI_Bcast(qbuf, n, MPI_INT16_T, 0, MPI_COMM_WORLD, ierr)
      call bf16_dec(qbuf(1:n), a)
    else
      if (qmaxall(ig) <= 0.0_real64) then
        a = 0.0_wp
        return
      end if
      ! passo de quantizacao: |q_i| <= 32000/nranks e |sum q_i| <= 32000 < 32767
      step = real(qmaxall(ig)/32000.0_real64, wp)
      qbuf(1:n) = int(nint(a/ (step*real(nranks, wp))), int16)
      call MPI_Allreduce(qbuf, qall, n, MPI_INT16_T, MPI_SUM, MPI_COMM_WORLD, ierr)
      a = real(qall(1:n), wp)*step
    end if
  end subroutine q_grp

  ! bf16 = os 16 bits de cima do fp32, com round-to-nearest-even no bit 16.
  ! Mesmo expoente do fp32 (sem overflow/underflow novo); 8 bits de mantissa.
  subroutine bf16_enc(a, q)
    real(wp), intent(in) :: a(:)
    integer(int16), intent(out) :: q(:)
    integer(int32) :: i32
    integer :: n
    do n = 1, size(a)
      i32 = transfer(a(n), i32)
      i32 = i32 + 32767_int32 + iand(ishft(i32, -16), 1_int32)
      q(n) = int(ishft(i32, -16), int16)
    end do
  end subroutine bf16_enc

  subroutine bf16_dec(q, a)
    integer(int16), intent(in) :: q(:)
    real(wp), intent(out) :: a(:)
    integer(int32) :: i32
    integer :: n
    do n = 1, size(a)
      i32 = ishft(int(q(n), int32), 16)
      a(n) = transfer(i32, a(n))
    end do
  end subroutine bf16_dec

  ! ---- fingerprint: XOR-rotate dos bits de cada peso, na ordem dos arrays.
  ! Nao e' cripto: responde "os bits sao iguais?" entre ranks e entre runs, e a
  ! prova final e' `cmp` nos .npy. Sem multiplicacao (nada de overflow assinado).
  integer(int64) function fp_params(P) result(h)
    type(params_t), intent(in) :: P
    h = 1469598103934665603_int64
    h = mix(h, P%wte); h = mix(h, P%lm); h = mix(h, P%q); h = mix(h, P%k)
    h = mix(h, P%v); h = mix(h, P%p); h = mix(h, P%fc); h = mix(h, P%p2)
  end function fp_params

  integer(int64) function mix(h, a) result(r)
    integer(int64), intent(in) :: h
    real(wp), intent(in) :: a(:)
    integer(int32) :: bits
    integer :: n
    r = h
    do n = 1, size(a)
      bits = transfer(a(n), bits)
      r = ieor(r, int(bits, int64))
      r = ishftc(r, 13)
    end do
  end function mix

  ! ---- qual linha cada rank le no passo k (1-based) ----
  subroutine pick_row(k)
    integer, intent(in) :: k
    if (data_mode == 2) then
      row = start_row + mod((k - 1)*nranks + rank, ntrain)
    else
      row = start_row + mod(rank*chunk + mod(k - 1, chunk), ntrain)
    end if
  end subroutine pick_row

  function ckdir_for(tstep) result(s)
    integer, intent(in) :: tstep
    character(len=512) :: s
    if (save_all) then
      write (s, '(A,A,I0,A,I0)') trim(outdir), '/rank_', rank, '/step_', tstep
    else
      write (s, '(A,I0)') trim(outdir) // '/step_', tstep
    end if
  end function ckdir_for

  ! ---- checkpoint (rank 0, ou todos com --save_all) ----
  subroutine save_ckpt(tstep, lr_eff)
    integer, intent(in) :: tstep
    real(wp), intent(in) :: lr_eff
    character(len=512) :: dir
    dir = ckdir_for(tstep)
    if (mkdir_p(trim(dir)) /= 0) call die("cannot create ckpt: "//trim(dir))
    if (ckfmt == 'npy' .or. ckfmt == 'both') then
      call save_gpt_weights(trim(dir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
          M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
    end if
    if (ckfmt == 'st' .or. ckfmt == 'both') then
      call save_gpt_weights_st(trim(dir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
          TT, A_BOS, tstep, lr_eff, int(tstep, int64)*int(B*TT, int64), &
          trim(rowsfile), M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
    end if
    call save_adam_state(trim(dir), S)
    if (use_muon) call save_muon_state(trim(dir), S)
    call write_arch_txt(trim(dir))
    print '(A,I0,A,A)', "rank ", rank, " saved ", trim(dir)
    flush (6)
  end subroutine save_ckpt

  ! ---- bpb exata (byte-weighted) sobre nval linhas: a metrica fixa do projeto ----
  real(wp) function val_bpb(rowsfile, first, nval)
    character(*), intent(in) :: rowsfile
    integer, intent(in) :: first, nval
    integer(c_int) :: v_idx(B*TT), v_tgt(B*TT)
    integer :: n, jj, tid, b2
    real(wp) :: tn, tb
    real(wp) :: nlls(B*TT)
    tn = 0.0_wp; tb = 0.0_wp
    do b2 = 0, nval - 1
      call load_batch(rowsfile, first + b2, B, TT, v_idx, v_tgt, n)
      if (n < B) exit
      call forward_nlls(v_idx, v_tgt, nlls)
      do jj = 1, B*TT
        tid = v_tgt(jj)
        if (tbytes(tid) > 0) then
          tn = tn + nlls(jj)
          tb = tb + real(tbytes(tid), wp)
        end if
      end do
    end do
    val_bpb = tn / (log(2.0_wp) * tb)
  end function val_bpb

  ! forward only, NLL por posicao (mesmas temps do treino, ociosas entre passos)
  subroutine forward_nlls(idx, targets, nlls)
    integer(c_int), intent(in) :: idx(:), targets(:)
    real(wp), intent(out) :: nlls(:)
    real(wp), pointer :: emd(:), xn(:), sub(:), qo(:), ko(:), vo(:)
    real(wp), pointer :: qrot(:), krot(:), ao(:), mlpd(:), lgt(:)
    integer :: BT, DD, hdd, dff, ll, it, j2, tg
    integer :: qsz, ksz, psz, fcsz, p2sz
    real(wp) :: mx, sm
    BT = B*TT; DD = D; hdd = N_HEAD*HD
    dff = 4*DD
    qsz = hdd*DD; ksz = N_KV*HD*DD; psz = DD*hdd
    fcsz = dff*DD; p2sz = DD*dff
    emd => tmp%emd; xn => tmp%xn; sub => tmp%sub
    qo => tmp%qo; ko => tmp%ko; vo => tmp%vo
    qrot => tmp%qrot; krot => tmp%krot; ao => tmp%ao; mlpd => tmp%mlpF
    lgt => tmp%lgt
    call wte_lookup(idx, M%wte, emd, B, TT, VV, DD)
    call rmsnorm0(emd, xn, BT, DD, G%eps)
    emd = xn
    do ll = 0, N_LAYER - 1
      call rmsnorm0(emd, xn, BT, DD, G%eps)
      call linear3d_sgemm(xn, M%q(ll*qsz+1:), qo, B, TT, DD, hdd)
      call linear3d_sgemm(xn, M%k(ll*ksz+1:), ko, B, TT, DD, N_KV*HD)
      call linear3d_sgemm(xn, M%v(ll*ksz+1:), vo, B, TT, DD, N_KV*HD)
      call rope_4d(qo, ct, st, qrot, B, TT, N_HEAD, HD)
      call rope_4d(ko, ct, st, krot, B, TT, N_KV, HD)
      if (attn_blas) then
        call attn_sgemm(qrot, krot, vo, ao, B, TT, N_HEAD, N_KV, HD, tmp%satt)
      else
        call causal_attn(qrot, krot, vo, ao, B, TT, N_HEAD, N_KV, HD)
      end if
      call linear3d_sgemm(ao, M%p(ll*psz+1:), sub, B, TT, DD, DD)
      emd = emd + sub
      call rmsnorm0(emd, xn, BT, DD, G%eps)
      call linear3d_sgemm(xn, M%fc(ll*fcsz+1:), mlpd, B, TT, DD, dff)
      call relu2(mlpd, BT*dff)
      call linear3d_sgemm(mlpd, M%p2(ll*p2sz+1:), sub, B, TT, dff, DD)
      emd = emd + sub
    end do
    call rmsnorm0(emd, xn, BT, DD, G%eps)
    call linear3d_sgemm(xn, M%lm, lgt, B, TT, DD, VV)
    do it = 1, BT
      tg = targets(it) + 1
      mx = lgt((it-1)*VV+1)
      do j2 = 2, VV
        if (lgt((it-1)*VV+j2) > mx) mx = lgt((it-1)*VV+j2)
      end do
      sm = 0.0_wp
      do j2 = 1, VV
        sm = sm + exp(lgt((it-1)*VV+j2) - mx)
      end do
      nlls(it) = (mx + log(sm)) - lgt((it-1)*VV+tg)
    end do
  end subroutine forward_nlls

  ! Aborta TODOS os ranks (um rank sozinho esperando num coletivo = job pendurado)
  subroutine die(msg)
    character(*), intent(in) :: msg
    print '(A,I0,A,A)', "rank ", rank, ": ", trim(msg)
    flush (6)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    error stop 1
  end subroutine die

end program train_ddp
