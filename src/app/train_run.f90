! app/train_run.f90 — multi-batch trainer (slice 3, hyp_fa2388).
!
! Usage:
!   fortran-fpm run train_run -- <weights> <rows> <outdir> <nsteps> \
!       [lr=0.0003] [t0=1] [log_every=1] [save_every=10] [start_row=0] \
!       [ntrain=40] [val_every=5] [nval=8] [keep_last=2] [bytes_file]
! Cycles train rows [start_row, start_row+ntrain); every val_every steps
! scores rows [start_row+ntrain, +nval) as exact val-bpb (token byte
! lengths from bytes_file, default tok_tables/token_bytes.txt).
! --ckpt-format st|npy|both (default ST): um model.safetensors por checkpoint,
! com nomes canonicos e __metadata__ (arch + card) -- o arquivo autodescritivo
! que viaja com os pesos. `npy` reproduz o layout historico (checkpoints legados
! e o tooling Python ainda nao migrado); `both` grava as duas representacoes do
! MESMO byte. Enquanto os consumidores nao migrarem para scripts/st_read.py,
! scripts/ckio.py faz as ferramentas que leem .npy ABORTAREM num checkpoint st.
! Ver src/lib/load_weights.f90 e docs/safetensors_backend.md.
! 2-step linear warmup in code (lesson of the overfit run): lr * min(1,
! k/2) for run-relative step k. Best-val snapshot to outdir/best/,
! rotation keeps last keep_last step_N/ dirs.
! Dims: depth-12 (D=768 NH=6 NKV=6 HD=128 NL=12 VV=8192, T=2048, B=1).

program train_run
  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only: int64, real64
  use fortran_energy_mod
  use fortran_train_mod
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights, &
      save_gpt_weights_st, verify_ckpt_dir
  use fortran_chat_mod, only: write_template_txt
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER, A_VOCAB => VV, A_CTX => TT, A_BOS => BOS, &
      write_arch_txt, read_arch_txt, arch_report, require_arch
  use fortran_adam_state_mod, only: load_adam_state, save_adam_state, &
      load_muon_state, save_muon_state
  use fortran_data_mod, only: load_batch
  use M_CLI2, only: set_args, sget, rget, iget, specified
  use fortran_sys_mod, only: mkdir_p
  use fortran_linear_mod
  use fortran_rmsnorm_mod
  use fortran_rope_mod
  use fortran_attn_mod
  use fortran_blas_mod, only: linear3d_sgemm
  implicit none

  integer, parameter :: sp = c_float
  ! --batch: quantas sequencias por passo. RUNTIME (nao constante de
  ! compilacao): as formas do modelo saem de G%B, entao so' os arrays
  ! locais deste app dependiam de B -- e eles agora sao alocados.
  integer :: B = 1
  integer, parameter :: TT = A_CTX, D = A_D
  integer, parameter :: N_HEAD = A_HEAD, N_KV = A_KV, HD = A_HD
  integer, parameter :: N_LAYER = A_LAYER, VV = A_VOCAB

  type(dims_t) :: G
  type(params_t) :: M, GR
  type(state_t) :: S
  type(cache_t) :: C
  type(temp_t) :: tmp
  character(len=512) :: wdir, rowsfile, outdir, ckdir, bytesfile
  character(len=:), allocatable :: badpath
  integer, allocatable :: idx(:), targets(:)
  integer :: ngot
  integer, allocatable :: tbytes(:)
  real(sp) :: ct(TT*(HD/2)), st(TT*(HD/2))
  real(sp) :: nll, vnll, tnll, lr, lr_eff, best
  logical :: adam_found
  logical :: use_muon, muon_found
  real(sp) :: muon_lr
  integer :: nsteps, t0, log_every, save_every, start_row
  integer :: ntrain, val_every, nval, keep_last, nprobe, nprobe_opt
  integer :: k, i, j, tstep, u, ios, r, nbad
  logical :: attn_blas, attn_qk, attn_qkph, anneal
  character(len=8) :: ckfmt
  integer(int64) :: ck_tokens
  ! Energia auto-medida (fortran_energy): o card de cada save leva o delta exato
  ! desde o save anterior; a trilha leva a deriva de potencia/cores.
  type(energy_interval_t) :: epre, ecard
  integer(int64) :: e_dtok, e_ptok
  integer :: last_save, etrace_u, eios
  logical :: etrace_ok
  real(real64) :: e_w
  real(sp) :: theta, ang

  lr = 0.0003_sp; t0 = 1; log_every = 1; save_every = 10; start_row = 0
  ntrain = 40; val_every = 5; nval = 8; keep_last = 2
  nprobe = 0
  bytesfile = ""
  nsteps = -1
  call set_args('--weights WEIGHTS --rows ROWS --out OUT --nsteps 20' // &
      ' --lr 0.0003 --t0 1 --log_every 1 --save_every 10' // &
      ' --start_row 0 --ntrain 40 --val_every 5 --nval 8 --keep_last 2' // &
      ' --trn_probe 0 --attn naive --bytes BYTES' // &
      ' --opt adam --muon-lr 0.02 --ckpt-format st --anneal 0 --batch 1', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  train_run - multi-batch trainer (slice 3)', &
      'SYNOPSIS', &
      '  train_run --weights DIR --rows FILE --out DIR --nsteps N', &
      'OPTIONS', &
      '  --attn naive  attention kernel: naive (default, the arithmetic every', &
      '                recorded run used), blas (attn_sgemm/attn_bwd_sgemm) or qkhop:', &
      '                ~13x faster at T=2048, agrees to ~1e-6, different', &
      '                summation order). Applies to training AND to the val', &
      '                probes so both sides use the same kernel.', &
      '  --anneal 1    cosine LR to zero over the run (local-SGD reconvergence', &
      '                before a merge). Default 0 = constant LR.', &
      '  --ckpt-format st|npy|both   st (default) writes ONE model.safetensors', &
      '                per checkpoint, with canonical names + arch/card in', &
      '                __metadata__; npy keeps the historical transformer_*.npy', &
      '                layout (legacy checkpoints and the Python tooling that has', &
      '                not migrated yet); both writes the two representations of', &
      '                the SAME bytes (so they can be compared).', &
      '                The Adam/Muon state and arch.txt/template.txt are written', &
      '                in every mode: only the weights change representation.'], &
      version_text=[character(len=80) :: 'train_run 1.0'])
  wdir = trim(sget('weights'))
  rowsfile = trim(sget('rows'))
  outdir = trim(sget('out'))
  nsteps = iget('nsteps')
  lr = rget('lr')
  t0 = iget('t0')
  log_every = iget('log_every')
  save_every = iget('save_every')
  start_row = iget('start_row')
  ntrain = iget('ntrain')
  nprobe_opt = iget('trn_probe')
  attn_blas = trim(sget('attn')) == 'blas'
  attn_qk = trim(sget('attn')) == 'qkhop'
  attn_qkph = trim(sget('attn')) == 'qkhop-ph'
  val_every = iget('val_every')
  nval = iget('nval')
  keep_last = iget('keep_last')
  bytesfile = trim(sget('bytes'))
  B = iget('batch')
  if (B < 1) then
    print '(A)', 'require --batch >= 1'
    call exit(1)
  end if
  allocate (idx(B*TT), targets(B*TT))
  ckfmt = trim(sget('ckpt-format'))
  if (ckfmt /= 'npy' .and. ckfmt /= 'st' .and. ckfmt /= 'both') then
    print '(A)', 'require --ckpt-format npy|st|both'
    call exit(2)
  end if
  use_muon = trim(sget('opt')) == 'muon'
  ! --anneal: cosine ate zero no fim do run. E o que a teoria de local SGD
  ! exige para os descendentes reconvergirem antes do merge (composicao).
  anneal = trim(sget('anneal')) == '1' .or. trim(sget('anneal')) == 'sim'
  muon_lr = rget('muon-lr')
  if (use_muon) print '(A,F8.5)', "opt=muon (hybrid: Muon 2D + Adam resto), muon_lr=", muon_lr
  print '(A)', 'ckpt-format: '//trim(ckfmt)
  if (.not. specified('weights') .or. .not. specified('rows') &
      .or. .not. specified('out') .or. nsteps < 1) then
    print '(A)', 'require --weights --rows --out --nsteps>=1 (--help)'
    call exit(2)
  end if
  if (bytesfile == 'BYTES') &
      bytesfile = trim(wdir) // '/../tok_tables/token_bytes.txt'

  ! Auto-medicao (depois do parse: os argumentos acima ja decidiram o run). Sem
  ! sensor nada quebra: a energia fica 0.0 e cpu_s/cores_busy/IO continuam sendo
  ! medidos -- ver energy-fortran/README.md.
  call energy_init()
  print '(2A)', 'energy sensor: ', energy_sensor()
  print '(2A)', 'energy scope : ', energy_scope()

  ! Pre-create every checkpoint dir NOW (pools cold -> fork-safe).
  ! Mid-run mkdir_p then only ever hits dir_exists -> no fork, no bind(C).
  if (mkdir_p(trim(outdir)) /= 0) then
    print '(2A)', 'cannot create out dir: ', trim(outdir)
    call exit(1)
  end if
  if (mkdir_p(trim(outdir) // '/best') /= 0) then
    print '(2A)', 'cannot create best dir: ', trim(outdir)
    call exit(1)
  end if
  ! Trilha de energia (tstep,tokens,J_phase,watts,cores_busy) a cada log_every:
  ! potencia e cores ao longo do run, para ver deriva termica/throttling. O
  ! J_phase e o acumulado DESDE O ULTIMO SAVE (energy_peek nao consome o
  ! intervalo), entao a trilha nao mexe no delta que o checkpoint reporta.
  etrace_u = -1
  etrace_ok = .false.
  open (newunit=etrace_u, file=trim(outdir) // '/energy_trace.csv', &
      status='replace', action='write', iostat=eios)
  if (eios == 0) then
    etrace_ok = .true.
    write (etrace_u, '(A)') 'tstep,tokens,J_phase,watts,cores_busy'
    flush (etrace_u)
  else
    print '(2A)', 'warning: cannot write the energy trace: ', &
        trim(outdir) // '/energy_trace.csv'
  end if
  do k = 1, nsteps
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      write (ckdir, '(A,I0)') trim(outdir) // '/step_', t0 + k - 1
      if (mkdir_p(trim(ckdir)) /= 0) then
        print '(2A)', 'cannot create ckpt dir: ', trim(ckdir)
        call exit(1)
      end if
    end if
  end do

  G%B = B; G%T = TT; G%V = VV; G%D = D
  G%nh = N_HEAD; G%nkv = N_KV; G%hd = HD; G%nl = N_LAYER
  G%eps = 1.0e-5_sp

  call load_gpt_weights(trim(wdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
  call require_arch(trim(wdir))

  call init_state(M, S)
  ! Optimizer carry across phases (Cognivolve): old checkpoints without
  ! adam_*.npy keep zeros = previous behavior.
  call load_adam_state(trim(wdir), S, adam_found)
  if (adam_found) then
    print '(A)', "resumed Adam moments (optimizer carry across phases)"
  else
    print '(A)', "fresh Adam moments (no adam_*.npy in weights dir)"
  end if
  if (use_muon) then
    call load_muon_state(trim(wdir), S, muon_found)
    if (muon_found) then
      print '(A)', "resumed Muon momentum buffers"
    else
      print '(A)', "fresh Muon momentum (no muon_moment_*.npy in weights dir)"
    end if
  end if
  call init_temp(G, tmp)
  allocate(GR%wte(size(M%wte)), GR%lm(size(M%lm)))
  allocate(GR%q(size(M%q)), GR%k(size(M%k)), GR%v(size(M%v)))
  allocate(GR%p(size(M%p)), GR%fc(size(M%fc)), GR%p2(size(M%p2)))
  allocate(tbytes(0:VV-1))
  open (newunit=u, file=trim(bytesfile), status="old", iostat=ios)
  if (ios /= 0) then
    print '(2A)', "cannot open bytes file: ", trim(bytesfile)
    call exit(1)
  end if
  do i = 0, VV - 1
    read (u, *, iostat=ios) tbytes(i)
    if (ios /= 0) then
      print '(A)', "short bytes file"
      call exit(1)
    end if
  end do
  close (u)

  do i = 1, TT
    do j = 1, HD/2
      theta = 10000.0_sp ** (-2.0_sp * real(j-1, sp) / real(HD, sp))
      ang = real(i-1, sp) * theta
      ct((i-1)*(HD/2)+j) = cos(ang)
      st((i-1)*(HD/2)+j) = sin(ang)
    end do
  end do

  best = huge(1.0_sp)
  ! Fecha a fase de setup (load de pesos/estado + tabelas): daqui para frente o
  ! intervalo aberto e o treino que ainda nao foi salvo.
  call energy_mark('setup')
  last_save = t0 - 1
  do k = 1, nsteps
    tstep = t0 + k - 1
    ! 2-step linear warmup on run-relative k (overfit-run lesson)
    lr_eff = lr * min(1.0_sp, real(k, sp) / 2.0_sp)
    if (anneal .and. nsteps > 1) then
      lr_eff = lr_eff * 0.5_sp * (1.0_sp + cos(3.14159265358979_sp * &
          real(k - 1, sp) / real(nsteps - 1, sp)))
    end if
    ! cycle within [start_row, start_row+ntrain): r is 0-based offset
    r = mod(k - 1, ntrain)
    call load_batch(trim(rowsfile), start_row + r, B, TT, idx, targets, ngot)
    if (ngot < B) then
      print '(A)', "rows file too short"
      call exit(1)
    end if
    call train_step(idx, targets, ct, st, M, S, G, GR, C, tmp, nll, tstep, &
        lr_eff, 0.9_sp, 0.999_sp, 1.0e-8_sp, 0.0_sp, attn_blas=attn_blas, attn_qk=attn_qk, attn_qkph=attn_qkph, &
        use_muon=use_muon, lr_muon=muon_lr)
    if (mod(k, log_every) == 0 .or. k == nsteps) then
      print '(A,I0,A,F10.5,A,F8.5)', "step ", tstep, " nll ", nll, &
        " lr ", lr_eff
      flush (6)
      if (etrace_ok) then
        ! watts = potencia media DESDE A LINHA ANTERIOR (energy_watts mantem o
        ! proprio estado e nao toca no intervalo do checkpoint).
        e_w = energy_watts()
        e_ptok = int(tstep - last_save, int64)*int(B*TT, int64)
        call energy_peek(epre, e_ptok)
        write (etrace_u, '(I0,A,I0,A,F0.4,A,F0.4,A,F0.4)') tstep, ',', e_ptok, &
            ',', epre%j, ',', e_w, ',', epre%cores_busy
        flush (etrace_u)
      end if
    end if
    if (mod(k, save_every) == 0 .or. k == nsteps) then
      write (ckdir, '(A,I0)') trim(outdir) // "/step_", tstep
      if (mkdir_p(trim(ckdir)) /= 0) then
        print '(2A)', "cannot create ", trim(ckdir)
        call exit(1)
      end if
      ck_tokens = int(tstep, int64)*int(B*TT, int64)
      ! O delta de energia DESTE checkpoint sai da propria API: o mark fecha o
      ! intervalo aberto no save anterior (tokens do intervalo = B*TT*passos).
      e_dtok = int(tstep - last_save, int64)*int(B*TT, int64)
      call energy_mark('ate_save', tokens=e_dtok, iv=ecard)
      print '(A,I0,A,F0.4,A,F0.4,A,F0.5,A,F0.4)', 'energy ckpt ', tstep, &
          ' J ', ecard%j, ' J/tok ', ecard%j_per_token, ' W_mean ', ecard%w_mean, &
          ' cpu_s ', ecard%cpu_s
      flush (6)
      if (ckfmt == 'npy' .or. ckfmt == 'both') then
        call save_gpt_weights(trim(ckdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
            M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2)
      end if
      if (ckfmt == 'st' .or. ckfmt == 'both') then
        call save_gpt_weights_st(trim(ckdir), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
            TT, A_BOS, tstep, lr_eff, ck_tokens, trim(rowsfile), &
            M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2, energy=ecard)
      end if
      call save_adam_state(trim(ckdir), S)
      if (use_muon) call save_muon_state(trim(ckdir), S)
      call write_template_txt(trim(ckdir))
      call write_arch_txt(trim(ckdir))
      call verify_ckpt_dir(trim(ckdir), N_LAYER, nbad, badpath, &
          st_only=(ckfmt == 'st'))
      if (nbad /= 0) then
        print '(2A)', "checkpoint verify failed (disk full?): ", trim(badpath)
        call exit(1)
      end if
      ! Sem rotacao de checkpoints: `rm -rf` via execute_command_line forkaria com
      ! pools OpenMP vivos e mataria os filhos (smash observado nos steps
      ! 110/120). Best-effort: steps antigos ficam no disco, limpeza offline.
      ! (Aqui existia uma subroutine rotate_ckpts vazia so' para documentar isso.)
      ! Custo do proprio checkpoint (estado do otimizador + verify + rotate) fica
      ! numa fase separada: e ele que responde "quanto custa salvar".
      call energy_mark('ckpt')
      last_save = tstep
    end if
    if (mod(k, val_every) == 0 .or. k == nsteps) then
      vnll = val_bpb(trim(rowsfile), start_row + ntrain, nval)
      ! train-bpb on a probe slice of the train pool. Default (0) keeps the
      ! original behaviour: the whole pool after the val slice, which is what
      ! "same scale" meant at 120-row corpora. Big row files need a cap: the
      ! probe is a full forward of T=2048 per row, so ntrain - nval rows would
      ! cost hours per validation (a 59k-row prose pool is ~65 h). Capping it
      ! samples the TAIL of the pool (start_row+ntrain-nprobe), i.e. rows this
      ! phase does not train on, which is the number we actually want.
      ! O probe de treino e' um forward COMPLETO por linha: com nprobe = ntrain
      ! (o default antigo) um pool de 11.797 linhas custa ~1,8 h POR VALIDACAO --
      ! foi o que travou o job 143 (parecia spin, era conta). O default agora e'
      ! um teto pequeno; quem quiser o pool inteiro passa --trn_probe -1.
      nprobe = ntrain - nval
      if (nprobe_opt == 0) nprobe = min(64, ntrain - nval)          ! default: teto
      if (nprobe_opt > 0) nprobe = min(nprobe_opt, ntrain - nval)   ! explicito
      if (nprobe_opt < 0) nprobe = ntrain - nval                    ! -1 = pool todo
      tnll = val_bpb(trim(rowsfile), start_row + ntrain - nprobe, nprobe)
      print '(A,I0,A,F10.5)', "val @", tstep, " bpb    ", vnll
      print '(A,I0,A,F10.5)', "trn @", tstep, " bpb    ", tnll
      print '(A,I0,A,F10.5)', "gap @", tstep, " bpb    ", vnll - tnll
      if (vnll < best) then
        best = vnll
        if (mkdir_p(trim(outdir) // "/best") /= 0) then
          print '(A)', "cannot create best/"
          call exit(1)
        end if
        ! Delta proprio do best/ (nao reusa o do save do passo): quando os dois
        ! caem no MESMO tstep o intervalo ja foi fechado pelo mark do step, entao
        ! aqui sobram a validacao e o snapshot com tokens=0 e J_per_tok=0 -- que e
        ! o honesto, este save nao treinou nada de novo.
        e_dtok = int(tstep - last_save, int64)*int(B*TT, int64)
        call energy_mark('ate_save', tokens=e_dtok, iv=ecard)
        if (ckfmt == 'npy' .or. ckfmt == 'both') then
          call save_gpt_weights(trim(outdir) // "/best", N_LAYER, D, &
              N_HEAD, N_KV, HD, VV, M%wte, M%lm, M%q, M%k, M%v, M%p, &
              M%fc, M%p2)
        end if
        if (ckfmt == 'st' .or. ckfmt == 'both') then
          call save_gpt_weights_st(trim(outdir) // "/best", N_LAYER, D, N_HEAD, &
              N_KV, HD, VV, TT, A_BOS, tstep, lr_eff, ck_tokens, trim(rowsfile), &
              M%wte, M%lm, M%q, M%k, M%v, M%p, M%fc, M%p2, energy=ecard)
        end if
        call save_adam_state(trim(outdir) // "/best", S)
        if (use_muon) call save_muon_state(trim(outdir) // "/best", S)
        call write_template_txt(trim(outdir) // "/best")
      call write_arch_txt(trim(outdir) // "/best")
        call verify_ckpt_dir(trim(outdir) // "/best", N_LAYER, nbad, badpath, &
            st_only=(ckfmt == 'st'))
        if (nbad /= 0) then
          print '(2A)', "best verify failed (disk full?): ", trim(badpath)
          call exit(1)
        end if
        call energy_mark('ckpt')
        last_save = tstep
        print '(A)', "new best snapshot"
      end if
      flush (6)
    end if
  end do

  ! Energia do run: as mesmas fases que foram para os cards, mais o total. Sem
  ! sensor a linha sai com J=0 e os campos de CPU/IO/threads preenchidos.
  call energy_mark('final')
  if (etrace_ok) close (etrace_u)
  print '(A)', ''
  call energy_report()
  print '(2A)', 'energy_json ', energy_report_json()
  flush (6)
contains
  ! exact val-bpb over nval rows (byte-masked NLL, train.py metric)
  real(sp) function val_bpb(rowsfile, first, nval)
    character(*), intent(in) :: rowsfile
    integer, intent(in) :: first, nval
    integer, allocatable :: v_idx(:), v_tgt(:)
    integer :: n, jj, tid, b2
    real(sp) :: tn, tb
    real(sp), allocatable :: nlls(:)
    allocate (v_idx(B*TT), v_tgt(B*TT), nlls(B*TT))
    tn = 0.0_sp; tb = 0
    do b2 = 0, nval - 1
      call load_batch(rowsfile, first + b2, B, TT, v_idx, v_tgt, n)
      if (n < B) exit
      call forward_nlls(v_idx, v_tgt, nlls)
      do jj = 1, B*TT
        tid = v_tgt(jj)
        if (tbytes(tid) > 0) then
          tn = tn + nlls(jj)
          tb = tb + tbytes(tid)
        end if
      end do
    end do
    val_bpb = tn / (0.69314718056_sp * real(tb, sp))
  end function val_bpb

  ! per-position NLLs for one batch (forward only)
  subroutine forward_nlls(idx, targets, nlls)
    integer, intent(in) :: idx(:), targets(:)
    real(sp), intent(out) :: nlls(:)
    ! Val scratch = train temps (idle between steps): pointer-associated,
    ! zero malloc. Shapes match tmp exactly (same B,T,D,V,hdd,dff).
    real(sp), pointer :: emd(:), xn(:), sub(:), qo(:), ko(:), vo(:)
    real(sp), pointer :: qrot(:), krot(:), ao(:), mlpd(:), lgt(:)
    integer :: BT, DD, hdd, dff, ll, it, j2, tg
    integer :: qsz, ksz, psz, fcsz, p2sz
    real(sp) :: mx, sm
    BT = B * TT; DD = D; hdd = N_HEAD * HD
    dff = 4 * DD
    qsz = hdd * DD; ksz = N_KV * HD * DD; psz = DD * hdd
    fcsz = dff * DD; p2sz = DD * dff
    emd => tmp%emd; xn => tmp%xn; sub => tmp%sub
    qo => tmp%qo; ko => tmp%ko; vo => tmp%vo
    qrot => tmp%qrot; krot => tmp%krot; ao => tmp%ao; mlpd => tmp%mlpF
    lgt => tmp%lgt
    call wte_lookup(idx, M%wte, emd, B, TT, DD)
    call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
    emd = xn
    do ll = 0, N_LAYER - 1
      call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
      call linear3d_sgemm(xn, M%q(ll*qsz+1:), qo, B, TT, DD, hdd)
      call linear3d_sgemm(xn, M%k(ll*ksz+1:), ko, B, TT, DD, N_KV*HD)
      call linear3d_sgemm(xn, M%v(ll*ksz+1:), vo, B, TT, DD, N_KV*HD)
      call rope_4d(qo, ct, st, qrot, B, TT, N_HEAD, HD)
      call rope_4d(ko, ct, st, krot, B, TT, N_KV, HD)
      if (attn_blas) then
        call attn_sgemm(qrot, krot, vo, ao, B, TT, N_HEAD, N_KV, HD, &
            tmp%satt)
      else
        call causal_attn(qrot, krot, vo, ao, B, TT, N_HEAD, N_KV, HD)
      end if
      call linear3d_sgemm(ao, M%p(ll*psz+1:), sub, B, TT, DD, DD)
      emd = emd + sub
      call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
      call linear3d_sgemm(xn, M%fc(ll*fcsz+1:), mlpd, B, TT, DD, dff)
      call relu2(mlpd, BT*dff)
      call linear3d_sgemm(mlpd, M%p2(ll*p2sz+1:), sub, B, TT, dff, DD)
      emd = emd + sub
    end do
    call rmsnorm0(emd, xn, BT, DD, 1.0e-5_sp)
    call linear3d_sgemm(xn, M%lm, lgt, B, TT, DD, VV)
    !$omp parallel do private(tg, j2, mx, sm)
    do it = 1, BT
      tg = targets(it) + 1
      mx = lgt((it-1)*VV+1)
      do j2 = 2, VV
        if (lgt((it-1)*VV+j2) > mx) mx = lgt((it-1)*VV+j2)
      end do
      sm = 0.0_sp
      do j2 = 1, VV
        sm = sm + exp(lgt((it-1)*VV+j2) - mx)
      end do
      nlls(it) = (mx + log(sm)) - lgt((it-1)*VV+tg)
    end do

  end subroutine forward_nlls

end program train_run
