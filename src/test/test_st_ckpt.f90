! test/test_st_ckpt.f90 — backend safetensors do trainer: os DOIS formatos, os MESMOS bytes.
!
! O que este teste prova, e por que importa:
!   1. escrever o mesmo modelo como .npy e como model.safetensors produz payloads
!      BYTE A BYTE iguais (comparados em bytes crus, nao em valores: valores iguais
!      podem esconder NaN/-0.0/dtype errado; bytes iguais nao);
!   2. a leitura auto-detecta o formato e reconstroi os mesmos buffers empilhados;
!   3. um diretorio que so tem model.safetensors carrega (o caso do eval_bpb/repl
!      apontando para um checkpoint novo);
!   4. a arquitetura declarada no __metadata__ (arch.*) e conferida: arquivo de
!      outra arquitetura ABORTA, como o arch.txt sempre fez;
!   5. metadata com acento e com chave VAZIA sobrevive, e os bytes UTF-8 vao crus
!      para o arquivo (nada de \u nem de transliteracao);
!   6. verify_ckpt_dir continua pegando checkpoint corrompido -- inclusive o caso
!      real que motivou o verify (disco cheio = payload truncado);
!   7. o MESMO arquivo e lido por Python (oraculo pure-Python da biblioteca), com
!      os hashes de payload batendo com os .npy -- prova cross-language;
!   8. o caminho DEFAULT do trainer (train_run SEM --ckpt-format, que hoje e st) e
!      o --ckpt-format npy: dois passos de treino REAIS de 1 step em scratch,
!      conferindo que o default escreve SO model.safetensors (+ adam/arch/
!      template), que o legado continua escrevendo os .npy e que os payloads dos
!      dois sao byte a byte iguais (o que o job 91 (fermi_stsmoke) faz a mao);
!      e as FERRAMENTAS Python (scripts/test_ckio.py) lendo esse checkpoint
!      st-only real -- no minimo duas delas.
!
! Nada aqui treina: sao ~4 KiB de pesos sinteticos e uma arquitetura minuscula.
!
! O binario tem um modo interno (`test_st_ckpt arch-probe DIR d_model`) usado pelo
! proprio teste para observar o aborto por arquitetura divergente num subprocesso:
! a falha e `call exit(1)` por desenho, entao nao da para chama-la in-process.

program test_st_ckpt
  use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32
  use fortran_kinds_mod, only: wp
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER, A_VOCAB => VV, A_CTX => TT, A_BOS => BOS, &
      write_arch_txt
  use fortran_sys_mod, only: mkdir_p
  use fortran_chat_mod, only: write_template_txt
  use load_weights_mod, only: save_gpt_weights, save_gpt_weights_st, &
      load_gpt_weights, verify_ckpt_dir, ST_CKPT
  use safetensors, only: st_writer, st_reader, st_ok
  use stdlib_io_npy, only: save_npy
  implicit none

  ! Arquitetura minuscula: o teste e sobre BYTES, nao sobre tamanho. Os mesmos
  ! caminhos de codigo (fatias empilhadas, nomes l0.q..) sao exercitados.
  integer, parameter :: NL = 2, D = 8, NH = 2, NKV = 1, HD = 4, VV = 64
  integer, parameter :: CTX = 16, BOS = 63
  integer, parameter :: QSZ = NH*HD*D, KSZ = NKV*HD*D, PSZ = D*NH*HD
  integer, parameter :: FCSZ = 4*D*D, P2SZ = D*4*D

  character(len=*), parameter :: ROOT = 'build/st_ckpt_test'
  character(len=:), allocatable :: dir_npy, dir_st, dir_so, dir_bad

  real(wp), allocatable :: wte(:), lm(:), c_q(:), c_k(:), c_v(:), c_pr(:), c_fc(:), c_p2(:)

  integer :: fail_count = 0
  integer(int32) :: seed = 20260917_int32
  character(len=512) :: argv0, arg1, arg2, arg3

  ! ---- modo interno: prova que arch.* divergente aborta (ver test_arch_mismatch)
  call get_command_argument(0, argv0)
  call get_command_argument(1, arg1)
  if (trim(arg1) == 'arch-probe') then
    block
    real(wp), allocatable :: aw(:), al(:), aq(:), ak(:), av(:), ap(:), af(:), ap2(:)
    integer :: bad_d, ios
    call get_command_argument(2, arg2)
    call get_command_argument(3, arg3)
    read (arg3, *, iostat=ios) bad_d
    if (ios /= 0) call exit(4)
    ! metadata diz D, pedimos bad_d: check_declared_arch tem de abortar (exit 1)
    call load_gpt_weights(trim(arg2), NL, bad_d, NH, NKV, HD, VV, &
        aw, al, aq, ak, av, ap, af, ap2)
    print '(A)', 'arch-probe: NAO abortou (bug)'
    call exit(3)
    end block
  end if

  dir_npy = ROOT//'/npy'
  dir_st = ROOT//'/st'
  dir_so = ROOT//'/st_only'
  dir_bad = ROOT//'/truncated'

  ! ---- pesos sinteticos (com NaN/Inf/-0.0/denormal de proposito)
  allocate(wte(VV*D), lm(VV*D), c_q(NL*QSZ), c_k(NL*KSZ), c_v(NL*KSZ))
  allocate(c_pr(NL*PSZ), c_fc(NL*FCSZ), c_p2(NL*P2SZ))
  call fill(wte, 1.0_wp)
  call fill(lm, 1.0_wp)
  call fill(c_q, 1.0_wp)
  call fill(c_k, 1.0_wp)
  call fill(c_v, 1.0_wp)
  call fill(c_pr, 1.0_wp)
  call fill(c_fc, 1.0_wp)
  call fill(c_p2, 1.0_wp)
  ! Padroes que um round-trip por VALOR estragaria (NaN /= NaN, -0.0 == 0.0) e
  ! que um round-trip por BYTES nao pode estragar.
  wte(1) = transfer(int(z'7FC00000', int32), 1.0_wp)     ! NaN quieto
  wte(2) = transfer(int(z'7F800000', int32), 1.0_wp)     ! +Inf
  wte(3) = transfer(int(z'FF800000', int32), 1.0_wp)     ! -Inf
  wte(4) = transfer(int(z'80000000', int32), 1.0_wp)     ! -0.0
  lm(5) = transfer(int(z'00000001', int32), 1.0_wp)      ! denormal
  c_p2(7) = transfer(int(z'FFFFFFFF', int32), 1.0_wp)    ! NaN negativo

  print '(A)', '== safetensors backend: round-trip byte a byte =='

  call test_write_and_compare()
  call test_autodetect_read()
  call test_st_only_dir()
  call test_verify()
  call test_arch_mismatch()
  call test_meta_utf8_empty()
  call test_python_reader()
  call test_train_run_default_path()

  print '(A,I0,A)', '===', fail_count, ' failures ==='
  if (fail_count > 0) call exit(1)

contains

  subroutine check(cond, label)
    logical, intent(in) :: cond
    character(*), intent(in) :: label
    if (cond) then
      print '(A,A)', '  ok    ', label
    else
      print '(A,A)', '  FAIL: ', label
      fail_count = fail_count + 1
    end if
  end subroutine check

  ! Pseudo-aleatorio determinista (= test_kernels) + um termo dependente do indice,
  ! para o payload nao ser trivial e um erro de offset aparecer como diff.
  subroutine fill(x, scale)
    real(wp), intent(out) :: x(:)
    real(wp), intent(in) :: scale
    integer :: i
    do i = 1, size(x)
      seed = mod(seed*1103515245_int32 + 12345_int32, 2147483647_int32)
      x(i) = (real(mod(seed/65536_int32, 32768_int32), wp)/32768.0_wp - 0.5_wp)*scale
      x(i) = x(i) + real(mod(i, 7), wp)*0.125_wp
    end do
  end subroutine fill

  ! ------------------------------------------------------------------ 1
  subroutine test_write_and_compare()
    integer :: stat, ll
    character(len=:), allocatable :: msg, val
    type(st_reader) :: r
    logical :: found
    character(len=16) :: lstr
    character(len=24) :: nm

    print '(A)', '-- 1. escrita nos dois formatos + comparacao BYTE A BYTE'
    call wipe(dir_npy)
    call wipe(dir_st)
    call check(mkdir_p(dir_npy) == 0, 'mkdir '//dir_npy)
    call check(mkdir_p(dir_st) == 0, 'mkdir '//dir_st)

    call save_gpt_weights(dir_npy, NL, D, NH, NKV, HD, VV, &
        wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_p2)
    call write_template_txt(dir_npy)
    call mk_dummy_adam(dir_npy)

    call save_gpt_weights_st(dir_st, NL, D, NH, NKV, HD, VV, CTX, BOS, &
        42, 0.0003_wp, 7168_int64, 'rows_f0.txt', &
        wte, lm, c_q, c_k, c_v, c_pr, c_fc, c_p2)
    call write_template_txt(dir_st)
    call mk_dummy_adam(dir_st)
    call check(file_exists(dir_st//'/'//ST_CKPT), 'model.safetensors existe')

    call r%open(dir_st//'/'//ST_CKPT, stat, msg)
    call check(stat == st_ok, 'header valida (open)')
    if (stat /= st_ok) return
    call check(r%n_tensors() == 2 + 6*NL, 'n_tensors = 2 + 6*n_layer')

    ! wte e lm direto; por camada, a fatia correspondente do buffer empilhado.
    call cmp_slice(r, 'wte', dir_npy//'/transformer_wte_weight.npy', wte)
    call cmp_slice(r, 'lm', dir_npy//'/lm_head_weight.npy', lm)
    do ll = 0, NL - 1
      write (lstr, '(I0)') ll
      nm = 'l'//trim(lstr)//'.q'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_attn_c_q_weight.npy', c_q(ll*QSZ+1:(ll+1)*QSZ))
      nm = 'l'//trim(lstr)//'.k'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_attn_c_k_weight.npy', c_k(ll*KSZ+1:(ll+1)*KSZ))
      nm = 'l'//trim(lstr)//'.v'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_attn_c_v_weight.npy', c_v(ll*KSZ+1:(ll+1)*KSZ))
      nm = 'l'//trim(lstr)//'.p'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_attn_c_proj_weight.npy', c_pr(ll*PSZ+1:(ll+1)*PSZ))
      nm = 'l'//trim(lstr)//'.fc'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_mlp_c_fc_weight.npy', c_fc(ll*FCSZ+1:(ll+1)*FCSZ))
      nm = 'l'//trim(lstr)//'.p2'
      call cmp_slice(r, trim(nm), dir_npy//'/transformer_h_'//trim(lstr)// &
          '_mlp_c_proj_weight.npy', c_p2(ll*P2SZ+1:(ll+1)*P2SZ))
    end do

    ! dtype F32 (nao F64) e shape 1-D flat, igual ao .npy exportado pelo lab
    call r%dtype('wte', val, stat, msg)
    call check(val == 'F32', 'dtype de wte e F32 (nao F64)')
    block
      integer(int64), allocatable :: shp(:)
      call r%shape('wte', shp, stat, msg)
      call check(size(shp) == 1 .and. shp(1) == int(VV*D, int64), &
          'shape de wte e 1-D [vocab*d_model]')
    end block

    ! metadata: procedencia, arquitetura e card
    call r%meta('format_version', val, found)
    call check(found .and. trim(val) == '1', 'metadata format_version = 1')
    call r%meta('producer', val, found)
    call check(found .and. len_trim(val) > 0, 'metadata producer presente')
    call r%meta('arch.d_model', val, found)
    call check(found .and. trim(val) == '8', 'metadata arch.d_model = 8')
    call r%meta('arch.n_layer', val, found)
    call check(found .and. trim(val) == '2', 'metadata arch.n_layer = 2')
    call r%meta('arch.head_dim', val, found)
    call check(found .and. trim(val) == '4', 'metadata arch.head_dim = 4')
    call r%meta('n_tensors', val, found)
    call check(found .and. trim(val) == '14', 'metadata n_tensors = 14')
    call r%meta('card', val, found)
    call check(found .and. index(val, '"steps":42') > 0 .and. &
        index(val, '"rows_file":"rows_f0.txt"') > 0 .and. &
        index(val, '"metrics":{}') > 0, 'card tem steps/rows_file/metrics vazio')
    call r%close()
  end subroutine test_write_and_compare

  ! Payload do tensor `name` no .st vs payload do .npy correspondente.
  subroutine cmp_slice(r, name, npypath, slice)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: name, npypath
    real(wp), intent(in) :: slice(:)
    integer(int8), pointer :: raw(:) => null()
    integer(int8), allocatable :: want(:)
    integer :: stat
    character(len=:), allocatable :: msg
    call npy_payload(npypath, want)
    call r%get_raw(name, raw, stat, msg)
    if (stat /= st_ok) then
      call check(.false., trim(name)//': get_raw falhou: '//trim(msg))
      return
    end if
    call check(size(raw) == size(want), trim(name)//': payload de '//trim(i2s(size(want)))// &
        ' bytes (== '//trim(i2s(size(slice)))//' floats)')
    if (size(raw) == size(want)) then
      call check(all(raw == want), trim(name)//': payload BYTE A BYTE igual ao .npy')
    end if
    deallocate(raw)
  end subroutine cmp_slice

  ! ------------------------------------------------------------------ 2
  subroutine test_autodetect_read()
    real(wp), allocatable :: a_wte(:), a_lm(:), a_q(:), a_k(:), a_v(:), a_pr(:), a_fc(:), a_p2(:)
    print '(A)', '-- 2. leitura auto-detectada (npy e st) reconstroi os mesmos buffers'

    call load_gpt_weights(dir_npy, NL, D, NH, NKV, HD, VV, &
        a_wte, a_lm, a_q, a_k, a_v, a_pr, a_fc, a_p2)
    call check_same(a_wte, wte, 'npy: wte')
    call check_same(a_lm, lm, 'npy: lm')
    call check_same(a_q, c_q, 'npy: q empilhado')
    call check_same(a_p2, c_p2, 'npy: p2 empilhado')

    call load_gpt_weights(dir_st, NL, D, NH, NKV, HD, VV, &
        a_wte, a_lm, a_q, a_k, a_v, a_pr, a_fc, a_p2)
    call check_same(a_wte, wte, 'st: wte')
    call check_same(a_lm, lm, 'st: lm')
    call check_same(a_q, c_q, 'st: q empilhado')
    call check_same(a_k, c_k, 'st: k empilhado')
    call check_same(a_v, c_v, 'st: v empilhado')
    call check_same(a_pr, c_pr, 'st: p empilhado')
    call check_same(a_fc, c_fc, 'st: fc empilhado')
    call check_same(a_p2, c_p2, 'st: p2 empilhado')
  end subroutine test_autodetect_read

  subroutine check_same(a, b, label)
    real(wp), intent(in) :: a(:), b(:)
    character(*), intent(in) :: label
    integer(int8), allocatable :: ba(:), bb(:)
    integer(int64) :: nb
    if (size(a) /= size(b)) then
      call check(.false., label//' (tamanhos diferentes)')
      return
    end if
    nb = int(size(a), int64)*int(storage_size(a(1))/8, int64)
    ba = transfer(a, 0_int8, nb)
    bb = transfer(b, 0_int8, nb)
    call check(all(ba == bb), label//' (bytes identicos)')
  end subroutine check_same

  ! ------------------------------------------------------------------ 3
  subroutine test_st_only_dir()
    real(wp), allocatable :: a_wte(:), a_lm(:), a_q(:), a_k(:), a_v(:), a_pr(:), a_fc(:), a_p2(:)
    print '(A)', '-- 3. diretorio com SO model.safetensors carrega'
    call wipe(dir_so)
    call check(mkdir_p(dir_so) == 0, 'mkdir '//dir_so)
    call copy_file(dir_st//'/'//ST_CKPT, dir_so//'/'//ST_CKPT)
    call check(file_exists(dir_so//'/'//ST_CKPT), 'model.safetensors copiado sozinho')
    call check(.not. file_exists(dir_so//'/transformer_wte_weight.npy'), &
        'nenhum .npy no diretorio')
    call check(.not. file_exists(dir_so//'/arch.txt'), 'nenhum arch.txt no diretorio')
    call load_gpt_weights(dir_so, NL, D, NH, NKV, HD, VV, &
        a_wte, a_lm, a_q, a_k, a_v, a_pr, a_fc, a_p2)
    call check_same(a_wte, wte, 'st-only: wte')
    call check_same(a_p2, c_p2, 'st-only: p2')
  end subroutine test_st_only_dir

  ! ------------------------------------------------------------------ 4
  subroutine test_verify()
    integer :: nbad
    character(len=:), allocatable :: badpath
    integer(int8), allocatable :: bytes(:)
    integer(int64) :: n
    print '(A)', '-- 4. verify_ckpt_dir (e o caso real: payload truncado)'

    call verify_ckpt_dir(dir_npy, NL, nbad, badpath)
    call check(nbad == 0, 'verify modo npy passa (badpath="'//trim(badpath)//'")')
    call verify_ckpt_dir(dir_st, NL, nbad, badpath, st_only=.true.)
    call check(nbad == 0, 'verify modo st passa (badpath="'//trim(badpath)//'")')

    ! Disco cheio = arquivo completo no header e curto no payload. E o acidente
    ! que motivou o verify (save_npy devolvia ios=0 num arquivo de 0 byte); aqui o
    ! proprio header valida a cobertura e derruba o checkpoint.
    call wipe(dir_bad)
    call check(mkdir_p(dir_bad) == 0, 'mkdir '//dir_bad)
    call read_file(dir_st//'/'//ST_CKPT, bytes, n)
    call write_file(dir_bad//'/'//ST_CKPT, bytes(1:int(n) - 4))
    call write_template_txt(dir_bad)
    call mk_dummy_adam(dir_bad)
    call verify_ckpt_dir(dir_bad, NL, nbad, badpath, st_only=.true.)
    call check(nbad /= 0, 'verify PEGA payload truncado: '//trim(badpath))
  end subroutine test_verify

  ! ------------------------------------------------------------------ 5
  subroutine test_arch_mismatch()
    character(len=:), allocatable :: cmd
    real(wp), allocatable :: a_wte(:), a_lm(:), a_q(:), a_k(:), a_v(:), a_pr(:), a_fc(:), a_p2(:)
    integer :: cstat, est
    print '(A)', '-- 5. arch.* divergente no metadata ABORTA (subprocesso)'
    ! arquitetura certa: carrega sem abortar
    call load_gpt_weights(dir_st, NL, D, NH, NKV, HD, VV, &
        a_wte, a_lm, a_q, a_k, a_v, a_pr, a_fc, a_p2)
    call check(.true., 'arch.* correto carrega')
    ! arquitetura errada (d_model=7 contra 8 no arquivo): tem de sair com exit 1
    cmd = trim(argv0)//' arch-probe '//dir_st//' 7 >/dev/null 2>&1'
    call execute_command_line(trim(cmd), exitstat=est, cmdstat=cstat)
    if (cstat /= 0) then
      call check(.false., 'nao consegui rodar o subprocesso arch-probe')
      return
    end if
    call check(est == 1, 'arch.d_model errado aborta com exit=1 (obtido '//trim(i2s(est))//')')
  end subroutine test_arch_mismatch

  ! ------------------------------------------------------------------ 6
  subroutine test_meta_utf8_empty()
    type(st_writer) :: w
    type(st_reader) :: r
    real(real32) :: x(3) = [1.0, 2.0, 3.0]
    character(len=:), allocatable :: msg, val
    integer(int8), allocatable :: bytes(:), needle(:)
    integer(int64) :: n
    integer :: stat
    logical :: found
    character(len=*), parameter :: path = ROOT//'/meta.safetensors'

    print '(A)', '-- 6. metadata com acento e chave VAZIA'
    call w%init()
    call w%set_meta('caracterização', 'acentuação e ção')
    call w%set_meta('', 'empty-key')
    call w%set_meta('plain', 'ascii')
    call w%set('x', x)
    call w%write(path, stat, msg)
    call check(stat == st_ok, 'write com metadata UTF-8')

    call r%open(path, stat, msg)
    call check(stat == st_ok, 'open do arquivo com metadata UTF-8')
    call r%meta('caracterização', val, found)
    call check(found .and. val == 'acentuação e ção', 'valor com acento volta igual')
    call r%meta('', val, found)
    call check(found .and. val == 'empty-key', 'chave vazia volta')
    call check(r%n_meta() == 3, 'tres chaves de metadata')
    call r%close()

    ! Os bytes UTF-8 tem de estar CRUS no arquivo (nada de \u, nada de '?'): e o
    ! que faz o metadado ser legivel por Python/HF sem combinacao previa.
    call read_file(path, bytes, n)
    needle = utf8_bytes('caracterização')
    call check(contains_seq(bytes(1:int(n)), needle), &
        'bytes UTF-8 crus da chave acentuada no arquivo')
    needle = utf8_bytes('acentuação e ção')
    call check(contains_seq(bytes(1:int(n)), needle), &
        'bytes UTF-8 crus do valor acentuado no arquivo')
    ! Chave vazia aparece como "" no JSON (e nao como chave ausente).
    needle = utf8_bytes('"":"empty-key"')
    call check(contains_seq(bytes(1:int(n)), needle), 'chave vazia serializada como ""')
  end subroutine test_meta_utf8_empty

  ! ------------------------------------------------------------------ 7
  subroutine test_python_reader()
    character(len=:), allocatable :: script, cmd, py
    character(len=256) :: pybuf
    integer :: cstat, est
    print '(A)', '-- 7. Python (oraculo da biblioteca) le o mesmo arquivo'
    script = find_script([character(len=48) :: 'scripts/st_read.py', '../scripts/st_read.py', &
                          '../../scripts/st_read.py'])
    if (len(script) == 0) then
      print '(A)', '  skip  scripts/st_read.py nao encontrado'
      return
    end if
    pybuf = ''
    call get_environment_variable('ST_PYTHON', pybuf)
    py = trim(pybuf)
    if (len(py) == 0) py = 'python3'
    cmd = trim(py)//' '//script//' --ckpt-dir '//dir_st//' --npy-dir '//dir_npy// &
        ' --compare-npy'
    call execute_command_line(trim(cmd), exitstat=est, cmdstat=cstat)
    if (cstat /= 0 .or. est == 127) then
      print '(A)', '  skip  python3 indisponivel ('//trim(py)//')'
      return
    end if
    call check(est == 0, 'python: payloads .st == .npy, card e arch.* validos')
  end subroutine test_python_reader

  ! ==================================================================== 8
  ! Caminho DEFAULT do trainer (sem --ckpt-format) e o legado (--ckpt-format npy),
  ! com um passo de treino DE VERDADE (1 step, arch compilada, ~4 s cada). O que
  ! este bloco prova e o que o job 91 faz a mao:
  !   (a) sem flag o checkpoint sai st-only (model.safetensors, zero peso .npy);
  !   (b) com --ckpt-format npy sai tudo .npy como sempre;
  !   (c) os payloads dos dois sao byte a byte iguais (mesma init/rows/lr);
  !   (d) as ferramentas Python leem o checkpoint st-only real (>=2 delas).
  subroutine test_train_run_default_path()
    integer, parameter :: DD = A_D, HH = A_HEAD, KV = A_KV, HDL = A_HD
    integer, parameter :: NL2 = A_LAYER, VVV = A_VOCAB
    real(wp), allocatable :: wte(:), lm(:), q(:), k(:), v(:), p(:), fc(:), p2(:)
    character(len=:), allocatable :: trun, evdir, nydir, ddir, py, script, cmd
    logical :: ran
    integer :: stat, ios

    print '(A)', '-- 8. train_run: default (st) x --ckpt-format npy, 1 step real'
    trun = find_binary('train_run')
    if (len(trun) == 0) then
      print '(A)', '  skip  binario train_run nao encontrado (rode fpm build antes do test)'
      return
    end if
    if (A_D < 16) then
      print '(A)', '  skip  arch minuscula demais para um passo de treino'
      return
    end if

    ! --- init legado em .npy (o trainer carrega e o --ckpt-format npy reescreve)
    ddir = ROOT//'/train_init'
    call wipe(ddir)
    if (mkdir_p(ddir) /= 0) then
      call check(.false., 'mkdir '//ddir)
      return
    end if
    allocate (wte(VVV*DD), lm(VVV*DD))
    allocate (q(NL2*HH*HDL*DD), k(NL2*KV*HDL*DD), v(NL2*KV*HDL*DD))
    allocate (p(NL2*DD*HH*HDL), fc(NL2*4*DD*DD), p2(NL2*DD*4*DD))
    ! 0.02 constante: init valida e sem NaN (make_init usa N(0, 0.02^2))
    wte = 0.02_wp; lm = 0.02_wp; q = 0.02_wp; k = 0.02_wp
    v = 0.02_wp; p = 0.02_wp; fc = 0.02_wp; p2 = 0.02_wp
    call save_gpt_weights(ddir, NL2, DD, HH, KV, HDL, VVV, &
        wte, lm, q, k, v, p, fc, p2)
    call write_arch_txt(ddir)
    call write_template_txt(ddir)
    call write_rows_file(ROOT//'/train_rows.txt', 2, A_CTX)
    call write_token_bytes(ROOT//'/train_tokbytes.txt', VVV)

    evdir = ROOT//'/run_default'
    nydir = ROOT//'/run_npy'
    call wipe(evdir)
    call wipe(nydir)
    call run_train(trun, ddir, evdir, '', ran)
    if (.not. ran) return
    call run_train(trun, ddir, nydir, '--ckpt-format npy', ran)
    if (.not. ran) return

    ! (a) default = st-only
    call check(file_exists(evdir//'/step_1/model.safetensors'), &
        'default (sem flag): model.safetensors existe')
    call check(n_weight_npy(evdir//'/step_1') == 0, &
        'default (sem flag): zero transformer_*.npy de peso')
    call check(count_state_npy(evdir//'/step_1') == 16, &
        'default (sem flag): 16 adam_*.npy de estado')
    call check(file_exists(evdir//'/step_1/arch.txt') .and. &
        file_exists(evdir//'/step_1/template.txt'), &
        'default (sem flag): arch.txt + template.txt')

    ! (b) --ckpt-format npy continua como sempre
    call check(.not. file_exists(nydir//'/step_1/model.safetensors'), &
        '--ckpt-format npy: nenhum model.safetensors')
    call check(n_weight_npy(nydir//'/step_1') == 2 + 6*NL2, &
        '--ckpt-format npy: 2+6*n_layer .npy de peso')

    ! (c) mesmos bytes nos dois caminhos (o check manual do job 91, automatizado)
    call cmp_st_vs_npy(evdir//'/step_1', nydir//'/step_1', NL2)

    ! (d) st_read.py valida o checkpoint que o DEFAULT escreveu
    script = find_script([character(len=48) :: '../scripts/st_read.py', &
                          'scripts/st_read.py', '../../scripts/st_read.py'])
    py = pick_python()
    if (len(script) > 0) then
      cmd = trim(py)//' '//script//' --ckpt-dir '//abs_path(evdir//'/step_1')//' --quiet'
      call execute_command_line(trim(cmd), exitstat=stat, cmdstat=ios)
      call check(ios == 0 .and. stat == 0, &
          'st_read.py valida o checkpoint escrito pelo DEFAULT')
    else
      print '(A)', '  skip  scripts/st_read.py nao encontrado'
    end if

    ! (e) as ferramentas Python sobre o checkpoint st-only real (>=2 ferramentas)
    script = find_script([character(len=48) :: '../scripts/test_ckio.py', &
                          'scripts/test_ckio.py', '../../scripts/test_ckio.py'])
    if (len(script) > 0) then
      if (numpy_disponivel(py)) then
        cmd = trim(py)//' '//script//' --ckpt-st '//abs_path(evdir//'/step_1')// &
              ' --ckpt-npy '//abs_path(nydir//'/step_1')// &
              ' --work '//abs_path(ROOT//'/ckio_tools')
        call execute_command_line(trim(cmd), exitstat=stat, cmdstat=ios)
        call check(ios == 0 .and. stat == 0, &
            'test_ckio.py: ferramentas Python leem o checkpoint st-only real '// &
            '(rescale, merge, cirurgia, compose_metrics, fedavg)')
      else
        print '(A)', '  skip  test_ckio.py: '//trim(py)//' nao tem numpy '// &
            '(defina ST_PYTHON=/caminho/python-com-numpy para rodar)'
      end if
    else
      print '(A)', '  skip  scripts/test_ckio.py nao encontrado'
    end if

    ! (f) o CLI: default aceita 'st' e recusa valor invalido
    cmd = trim(trun)//' --weights '//ddir//' --rows '//ROOT//'/train_rows.txt'// &
          ' --out '//ROOT//'/cli_probe --nsteps 1 --ckpt-format bogus'// &
          ' --bytes '//ROOT//'/train_tokbytes.txt >/dev/null 2>&1'
    call execute_command_line(trim(cmd), exitstat=stat, cmdstat=ios)
    call check(ios == 0 .and. stat /= 0, '--ckpt-format invalido: aborta (exit != 0)')

    deallocate (wte, lm, q, k, v, p, fc, p2)
  end subroutine test_train_run_default_path

  ! Roda o train_run 1 step em scratch. `extra` = flags adicionais (string).
  ! --nval 0 de proposito: o val_bpb com 0 linhas da NaN, o teste do "best
  ! snapshot" (vnll < best) nao dispara, e assim o passo nao escreve um SEGUNDO
  ! checkpoint inteiro so para o teste ficar mais lento.
  subroutine run_train(trun, wdir, outdir, extra, ok)
    character(*), intent(in) :: trun, wdir, outdir, extra
    logical, intent(out) :: ok
    character(len=:), allocatable :: cmd
    integer :: stat, ios
    ! OMP_NUM_THREADS explicito: sem isso o OpenBLAS pega os 16 cores e o passo de
    ! 1 s vira 23 s de thrash (medido); 4 threads mantem o teste curto e a mesma
    ! ordem de reducao nos dois runs (e o que faz a comparacao byte a byte valer).
    cmd = 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE '// &
          trim(trun)//' --weights '//trim(wdir)//' --rows '//ROOT//'/train_rows.txt'// &
          ' --out '//trim(outdir)//' --nsteps 1 --lr 0.0003 --ntrain 1 --nval 0'// &
          ' --val_every 9999999 --trn_probe 0 --save_every 1 --attn blas'// &
          ' --anneal 0 --bytes '//ROOT//'/train_tokbytes.txt '//trim(extra)// &
          ' >'//ROOT//'/train_last.log 2>&1'
    call execute_command_line(trim(cmd), exitstat=stat, cmdstat=ios)
    ok = (ios == 0 .and. stat == 0)
    call check(ok, 'train_run 1 step '//trim(extra)//' (exit '//trim(i2s(stat))//')')
    if (.not. ok) then
      print '(A)', '  (log em '//ROOT//'/train_last.log)'
    end if
  end subroutine run_train

  integer function n_weight_npy(d)
    character(*), intent(in) :: d
    n_weight_npy = count_npy(d, skip_state=.true.)
  end function n_weight_npy

  integer function count_state_npy(d)
    character(*), intent(in) :: d
    count_state_npy = count_npy(d, skip_state=.false.)
  end function count_state_npy

  ! Conta .npy no diretorio (em Fortran puro: le o diretorio via ls)
  integer function count_npy(d, skip_state)
    character(*), intent(in) :: d
    logical, intent(in) :: skip_state
    character(len=:), allocatable :: cmd
    character(len=512) :: buf
    integer :: u, ios, cnt
    logical :: ex
    if (skip_state) then
      cmd = 'ls '//trim(d)//'/*.npy 2>/dev/null | grep -vc "/adam_[mv]_\\|/muon_"'
    else
      cmd = 'ls '//trim(d)//'/*.npy 2>/dev/null | grep -c "/adam_[mv]_\\|/muon_"'
    end if
    call execute_command_line(trim(cmd)//' >'//ROOT//'/count.txt', exitstat=ios)
    cnt = 0
    inquire (file=ROOT//'/count.txt', exist=ex)
    if (ex) then
      open (newunit=u, file=ROOT//'/count.txt', status='old', iostat=ios)
      if (ios == 0) then
        read (u, '(A)', iostat=ios) buf
        if (ios == 0) read (buf, *, iostat=ios) cnt
        close (u)
      end if
    end if
    count_npy = cnt
  end function count_npy

  ! Compara o payload de cada peso canonico no .st com o .npy correspondente.
  subroutine cmp_st_vs_npy(dir_st, dir_npy, n_layer)
    character(*), intent(in) :: dir_st, dir_npy
    integer, intent(in) :: n_layer
    type(st_reader) :: r
    character(len=:), allocatable :: msg
    character(len=16) :: lstr
    integer :: stat, ll, nbad2, ncmp
    call r%open(trim(dir_st)//'/'//ST_CKPT, stat, msg)
    if (stat /= st_ok) then
      call check(.false., 'cmp st x npy: open '//trim(msg))
      return
    end if
    nbad2 = 0
    ncmp = 0
    call cmp_one(r, 'wte', trim(dir_npy)//'/transformer_wte_weight.npy', nbad2, ncmp)
    call cmp_one(r, 'lm', trim(dir_npy)//'/lm_head_weight.npy', nbad2, ncmp)
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call cmp_one(r, 'l'//trim(lstr)//'.q', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_attn_c_q_weight.npy', nbad2, ncmp)
      call cmp_one(r, 'l'//trim(lstr)//'.k', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_attn_c_k_weight.npy', nbad2, ncmp)
      call cmp_one(r, 'l'//trim(lstr)//'.v', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_attn_c_v_weight.npy', nbad2, ncmp)
      call cmp_one(r, 'l'//trim(lstr)//'.p', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_attn_c_proj_weight.npy', nbad2, ncmp)
      call cmp_one(r, 'l'//trim(lstr)//'.fc', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_mlp_c_fc_weight.npy', nbad2, ncmp)
      call cmp_one(r, 'l'//trim(lstr)//'.p2', trim(dir_npy)//'/transformer_h_'// &
          trim(lstr)//'_mlp_c_proj_weight.npy', nbad2, ncmp)
    end do
    call check(nbad2 == 0, 'st x npy: '//trim(i2s(ncmp))// &
        ' payloads byte a byte iguais ('//trim(i2s(nbad2))//' diferentes)')
    call r%close()
  end subroutine cmp_st_vs_npy

  ! Um par (tensor canonico no .st, payload do .npy) -- incrementa os contadores.
  subroutine cmp_one(r, name, npypath, nbad, ncmp)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: name, npypath
    integer, intent(inout) :: nbad, ncmp
    integer(int8), pointer :: raw(:) => null()
    integer(int8), allocatable :: want(:)
    character(len=:), allocatable :: msg
    integer :: stat
    call npy_payload(npypath, want)
    call r%get_raw(name, raw, stat, msg)
    ncmp = ncmp + 1
    if (stat /= st_ok .or. size(raw) /= size(want) .or. .not. all(raw == want)) then
      nbad = nbad + 1
      if (nbad <= 3) print '(A)', '    difere: '//trim(name)
    end if
    if (associated(raw)) deallocate (raw)
  end subroutine cmp_one

  ! Rows file texto: n_rows linhas de ctx+1 ids 0-based (formato do loader).
  subroutine write_rows_file(path, n_rows, ctx)
    character(*), intent(in) :: path
    integer, intent(in) :: n_rows, ctx
    integer :: u, r, i, ios
    open (newunit=u, file=path, status='replace', iostat=ios)
    if (ios /= 0) then
      call check(.false., 'write_rows_file '//path)
      return
    end if
    do r = 1, n_rows
      do i = 1, ctx + 1
        if (i > 1) write (u, '(A)', advance='no') ' '
        ! determinista, dentro do vocab (BOS-1)
        write (u, '(I0)', advance='no') mod((r*7919 + i*104729), A_BOS)
      end do
      write (u, '(A)') ''
    end do
    close (u)
  end subroutine write_rows_file

  ! token_bytes.txt: usa o do laboratorio se existir; senao sintetiza.
  subroutine write_token_bytes(path, nvocab)
    character(*), intent(in) :: path
    integer, intent(in) :: nvocab
    character(len=512) :: home, cached
    character(len=:), allocatable :: cmd
    integer :: u, i, ios
    home = ''
    call get_environment_variable('HOME', home)
    cached = trim(home)//'/.cache/autoresearch/tok_tables/token_bytes.txt'
    if (file_exists(trim(cached))) then
      cmd = 'cp '//trim(cached)//' '//trim(path)
      call execute_command_line(trim(cmd), exitstat=ios)
      if (ios == 0) return
    end if
    open (newunit=u, file=path, status='replace', iostat=ios)
    do i = 1, nvocab
      write (u, '(A)') '4'
    end do
    close (u)
  end subroutine write_token_bytes

  ! Caminho absoluto (o fpm roda o teste com cwd=src/, e os scripts Python rodam
  ! com cwd=raiz do repo: caminho relativo nao serve para os dois).
  function abs_path(rel) result(p)
    character(*), intent(in) :: rel
    character(len=:), allocatable :: p
    character(len=1024) :: buf
    character(len=1024), save :: cwd = ''
    integer :: u, ios
    if (len_trim(cwd) == 0) then
      call execute_command_line('pwd > '//ROOT//'/pwd.txt', exitstat=ios)
      open (newunit=u, file=ROOT//'/pwd.txt', status='old', iostat=ios)
      if (ios == 0) then
        read (u, '(A)', iostat=ios) buf
        close (u)
        if (ios == 0) cwd = trim(buf)
      end if
    end if
    p = trim(cwd)//'/'//rel
  end function abs_path

  ! Python para os scripts: $ST_PYTHON, senao o venv-numpy do repo, senao python3.
  ! (o script st_read.py nao precisa de numpy; o tooling Python precisa.)
  function pick_python() result(py)
    character(len=:), allocatable :: py
    character(len=256) :: buf
    integer :: ios
    buf = ''
    call get_environment_variable('ST_PYTHON', buf)
    if (len_trim(buf) > 0) then
      py = trim(buf)
      return
    end if
    call execute_command_line('../.venv-numpy/bin/python3 -c "import numpy" '// &
        '2>/dev/null', exitstat=ios)
    if (ios == 0) then
      py = '../.venv-numpy/bin/python3'
      return
    end if
    py = 'python3'
  end function pick_python

  logical function numpy_disponivel(py)
    character(*), intent(in) :: py
    integer :: ios
    call execute_command_line(trim(py)//' -c "import numpy" 2>/dev/null', exitstat=ios)
    numpy_disponivel = (ios == 0)
  end function numpy_disponivel

  ! Binario de app do MESMO build dir do teste (irmao de app/), com fallback glob.
  function find_binary(app) result(path)
    character(*), intent(in) :: app
    character(len=:), allocatable :: path
    character(len=512) :: cand, buf
    integer :: p, u, ios
    logical :: ex
    path = ''
    ! argv0 = .../<build-dir>/<gfortran_hash>/test/test_st_ckpt
    p = index(argv0, '/test/')
    if (p > 0) then
      cand = argv0(1:p - 1)//'/app/'//app
      inquire (file=trim(cand), exist=ex)
      if (ex) then
        path = trim(cand)
        return
      end if
    end if
    call execute_command_line('ls -t build/*/app/'//app//' 2>/dev/null | head -1 >'// &
        ROOT//'/bin.txt', exitstat=ios)
    buf = ''
    open (newunit=u, file=ROOT//'/bin.txt', status='old', iostat=ios)
    if (ios == 0) then
      read (u, '(A)', iostat=ios) buf
      close (u)
      if (ios == 0 .and. len_trim(buf) > 0) path = trim(buf)
    end if
  end function find_binary

  ! ------------------------------------------------------------- utilidades
  subroutine wipe(dir)
    character(*), intent(in) :: dir
    call execute_command_line('rm -rf '//trim(dir))
  end subroutine wipe

  logical function file_exists(path)
    character(*), intent(in) :: path
    inquire (file=path, exist=file_exists)
  end function file_exists

  function find_script(cands) result(s)
    character(*), intent(in) :: cands(:)
    character(len=:), allocatable :: s
    integer :: i
    s = ''
    do i = 1, size(cands)
      if (file_exists(trim(cands(i)))) then
        s = trim(cands(i))
        return
      end if
    end do
  end function find_script

  subroutine read_file(path, b, n)
    character(*), intent(in) :: path
    integer(int8), allocatable, intent(out) :: b(:)
    integer(int64), intent(out) :: n
    integer :: u, sz, ios
    inquire (file=path, size=sz, iostat=ios)
    if (ios /= 0 .or. sz < 0) then
      allocate (b(0))
      n = 0_int64
      return
    end if
    allocate (b(sz))
    n = int(sz, int64)
    open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=ios)
    if (ios == 0 .and. sz > 0) read (u, iostat=ios) b
    if (ios == 0) close (u)
  end subroutine read_file

  subroutine write_file(path, b)
    character(*), intent(in) :: path
    integer(int8), intent(in) :: b(:)
    integer :: u, ios
    open (newunit=u, file=path, access='stream', form='unformatted', status='replace', iostat=ios)
    if (ios /= 0) then
      call check(.false., 'write_file '//path)
      return
    end if
    if (size(b) > 0) write (u, iostat=ios) b
    close (u)
  end subroutine write_file

  subroutine copy_file(src, dst)
    character(*), intent(in) :: src, dst
    integer(int8), allocatable :: b(:)
    integer(int64) :: n
    call read_file(src, b, n)
    call write_file(dst, b)
  end subroutine copy_file

  ! Payload cru de um .npy: pula magic+versao+header (v1: uint16, v2+: uint32).
  ! O tamanho comparado (e nao o valor) ja denuncia dtype errado: um .npy float64
  ! teria o dobro dos bytes.
  subroutine npy_payload(path, pay)
    character(*), intent(in) :: path
    integer(int8), allocatable, intent(out) :: pay(:)
    integer(int8), allocatable :: b(:)
    integer(int64) :: n
    integer :: major, hlen, off
    call read_file(path, b, n)
    if (n < 12 .or. iand(int(b(1)), 255) /= 147) then
      allocate (pay(0))
      call check(.false., 'nao parece um .npy: '//path)
      return
    end if
    major = iand(int(b(7)), 255)
    hlen = iand(int(b(9)), 255) + 256*iand(int(b(10)), 255)
    off = 10 + hlen
    if (major >= 2) then
      hlen = iand(int(b(9)), 255) + 256*iand(int(b(10)), 255) + &
             65536*iand(int(b(11)), 255)
      off = 12 + hlen
    end if
    off = off + 1
    if (off > int(n)) then
      allocate (pay(0))
      return
    end if
    pay = b(off:int(n))
  end subroutine npy_payload

  ! Metadados que verify_ckpt_dir exige (16 arquivos adam). O teste nao treina,
  ! entao cria arquivos com tamanho plausivel -- que e o que a checagem olha.
  subroutine mk_dummy_adam(dir)
    character(*), intent(in) :: dir
    real(wp) :: dummy(64)
    character(len=3) :: mname(8) = ['wte', 'lm ', 'q  ', 'k  ', 'v  ', 'p  ', 'fc ', 'p2 ']
    character(len=512) :: path
    integer :: ii, ios
    dummy = 0.0_wp
    do ii = 1, 8
      write (path, '(5A)') trim(dir), '/adam_m_', trim(mname(ii)), '.npy'
      call save_npy(trim(path), dummy, iostat=ios)
      write (path, '(5A)') trim(dir), '/adam_v_', trim(mname(ii)), '.npy'
      call save_npy(trim(path), dummy, iostat=ios)
    end do
  end subroutine mk_dummy_adam

  pure function i2s(v) result(s)
    integer, intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i2s

  ! Bytes UTF-8 de um literal (o fonte e UTF-8, entao basta copiar os bytes).
  pure function utf8_bytes(s) result(b)
    character(*), intent(in) :: s
    integer(int8) :: b(len(s))
    integer :: i
    do i = 1, len(s)
      b(i) = int(iachar(s(i:i)), int8)
    end do
  end function utf8_bytes

  pure function contains_seq(hay, needle) result(found)
    integer(int8), intent(in) :: hay(:), needle(:)
    logical :: found
    integer :: i, j
    found = .false.
    if (size(needle) == 0 .or. size(needle) > size(hay)) return
    do i = 1, size(hay) - size(needle) + 1
      do j = 1, size(needle)
        if (hay(i + j - 1) /= needle(j)) exit
        if (j == size(needle)) then
          found = .true.
          return
        end if
      end do
    end do
  end function contains_seq

end program test_st_ckpt
