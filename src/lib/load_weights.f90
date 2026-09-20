! lib/load_weights.f90 — shared checkpoint weight loader.
!
! Loads .npy flats exported by scripts/export_weights.py via
! stdlib_io_npy::load_npy. Per-layer weights are stacked: layer ll occupies
! [(ll-1)*per+1 : ll*per], matching fortran_gpt_mod's expectation.
!
! Filenames: transformer_wte_weight.npy, lm_head_weight.npy,
!   transformer_h_{L}_attn_{c_q,c_k,c_v,c_proj}_weight.npy,
!   transformer_h_{L}_mlp_{c_fc,c_proj}_weight.npy   (L = 0-based)
!
! DUAS REPRESENTACOES, MESMOS BYTES (fase 1 do backend safetensors): o mesmo peso
! pode viver num diretorio de .npy (formato historico) ou num unico
! model.safetensors com nomes canonicos (wte, lm, l{ll}.q/k/v/p/fc/p2). A leitura
! AUTO-DETECTA pela presenca do model.safetensors -- nenhum app precisa saber qual
! e; a escrita e escolhida por --ckpt-format no train_run. O objetivo declarado e
! comparar BYTES entre os dois (test_safetensors_ckpt), nao valores: se os dois
! caminhos nao produzem o mesmo payload, um deles esta mentindo.
!
! A arquitetura viaja no __metadata__ (arch.*) do .safetensors: um diretorio que
! so tem model.safetensors continua sendo recusado se nao casar com este binario
! (mesma politica do arch.txt, ver check_declared_arch).

module load_weights_mod
  use, intrinsic :: iso_fortran_env, only: int64, real32, real64
  use fortran_kinds_mod, only: wp
  use stdlib_io_npy, only: load_npy, save_npy
  ! O tipo MEDIDO que viaja para o card. Nao existe um tipo espelho aqui: o
  ! consumidor (train_run) passa o mesmo energy_interval_t que o energy_mark
  ! devolveu, ja com total/sensor/escopo/kind preenchidos pela API do pacote.
  use fortran_energy_mod, only: energy_interval_t
  use safetensors, only: st_writer, st_reader, st_ok
  use fortran_arch_mod, only: arch_schema, arch_canonical, arch_canonical_of, &
      arch_id, arch_id_of
  use safetensors_json, only: json_escape, json_num
  implicit none

  ! Nome do arquivo unico e convencao de nomes dos tensores.
  character(*), parameter :: ST_CKPT = 'model.safetensors'

contains

  ! Falha alta e unica do modulo (o resto do arquivo ja fazia isso com print+exit).
  subroutine die(what)
    character(*), intent(in) :: what
    print '(A)', trim(what)
    call exit(1)
  end subroutine die

  ! Formata um real para um numero JSON valido (o card e um STRING no metadata,
  ! mas quem le faz json.loads nele; "3.00000000E-04" e JSON valido).
  ! A real value for the card. Delegate to the function of the package. That
  ! function turns NaN and Inf into null, because JSON holds no such value. A
  ! card with NaN is not readable, and the card must stay readable when a run
  ! diverges.
  function json_real(x) result(s)
    real(wp), intent(in) :: x
    character(len=:), allocatable :: s
    s = json_num(real(x, real64))
  end function json_real

  ! Um real64 como numero JSON/texto de metadata. UM formatador para os dois
  ! lugares (o objeto `energy` dentro do card e as chaves planas energy.*): assim
  ! as duas representacoes do mesmo numero NAO podem divergir.
  function rnum(x) result(s)
    real(real64), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=40) :: b
    write (b, '(ES15.8E2)') x
    s = trim(adjustl(b))
  end function rnum

  ! `,"energy":{...}` do card (string vazia quando ninguem mediu). Os textos vem do
  ! fortran_energy (paths de /sys e descricao do escopo), sem aspas nem quebra de
  ! linha, entao nao precisam de escape.
  function energy_card_json(energy) result(s)
    type(energy_interval_t), intent(in), optional :: energy
    character(len=:), allocatable :: s
    if (.not. present(energy)) then
      s = ''
      return
    end if
    s = ',"energy":{"J":'//rnum(energy%j)//',"J_total":'//rnum(energy%j_total)// &
        ',"J_per_tok":'//rnum(energy%j_per_token)//',"W_mean":'//rnum(energy%w_mean)// &
        ',"cpu_s":'//rnum(energy%cpu_s)//',"cpu_pct":'//rnum(energy%cpu_pct)// &
        ',"cores_busy":'//rnum(energy%cores_busy)//',"wall_s":'//rnum(energy%wall_s)// &
        ',"tokens":'//rnum(energy%tokens)//',"sensor":"'//trim(energy%sensor)// &
        '","scope":"'//trim(energy%scope)//'","kind":"'//trim(energy%kind)// &
        '","self_measured":"'//self_flag(energy)//'"}'
  end function energy_card_json

  ! '1'/'0' -- string porque no __metadata__ do safetensors tudo e string, e o
  ! card_annotate.py (e qualquer leitor) compara com '1' dos dois lados.
  function self_flag(energy) result(s)
    type(energy_interval_t), intent(in) :: energy
    character(len=1) :: s
    s = '0'
    if (energy%self_measured) s = '1'
  end function self_flag

  ! As MESMAS quantidades como chaves planas do __metadata__ (a convencao que o
  ! card_annotate.py e os jobs ja leem) + a flag que diz que a energia foi medida
  ! pelo processo: com energy.self_measured=1 o card_annotate NAO sobrescreve nada.
  subroutine set_meta_energy(w, energy)
    type(st_writer), intent(inout) :: w
    type(energy_interval_t), intent(in), optional :: energy
    if (.not. present(energy)) return
    call w%set_meta('energy.self_measured', self_flag(energy))
    call w%set_meta('energy.J', trim(rnum(energy%j)))
    call w%set_meta('energy.J_total', trim(rnum(energy%j_total)))
    call w%set_meta('energy.J_per_tok', trim(rnum(energy%j_per_token)))
    call w%set_meta('energy.W_mean', trim(rnum(energy%w_mean)))
    call w%set_meta('energy.cpu_s', trim(rnum(energy%cpu_s)))
    call w%set_meta('energy.cpu_pct', trim(rnum(energy%cpu_pct)))
    call w%set_meta('energy.cores_busy', trim(rnum(energy%cores_busy)))
    call w%set_meta('energy.wall_s', trim(rnum(energy%wall_s)))
    call w%set_meta('energy.tokens', trim(rnum(energy%tokens)))
    call w%set_meta('energy.sensor', trim(energy%sensor))
    call w%set_meta('energy.scope', trim(energy%scope))
    call w%set_meta('energy.kind', trim(energy%kind))
  end subroutine set_meta_energy

  ! Chave de metadata com valor inteiro (arch.* etc).
  subroutine set_meta_i(w, key, v)
    type(st_writer), intent(inout) :: w
    character(*), intent(in) :: key
    integer, intent(in) :: v
    call w%set_meta_int(key, int(v, int64))
  end subroutine set_meta_i

  subroutine load1(path, a)
    character(*), intent(in) :: path
    real(wp), allocatable, intent(out) :: a(:)
    integer :: ios
    character(len=:), allocatable :: msg
    logical :: ex
    integer :: sz
    ! stdlib load_npy segfaults (instead of iostat) on a missing file in some
    ! versions, and chokes on 0-byte files -- check first so every app fails
    ! loud, never silent/SIGSEGV.
    inquire (file=path, exist=ex, size=sz)
    if (.not. ex .or. sz <= 0) then
      print '(2A)', "load failed (missing or empty file): ", trim(path)
      call exit(1)
    end if
    call load_npy(path, a, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', "load failed: ", trim(path), " " // trim(msg)
      call exit(1)
    end if
  end subroutine load1

  subroutine load_into(path, a, at)
    character(*), intent(in) :: path
    real(wp), intent(inout) :: a(:)
    integer, intent(in) :: at
    real(wp), allocatable :: tmp(:)
    call load1(path, tmp)
    a(at:at+size(tmp)-1) = tmp
    deallocate(tmp)
  end subroutine load_into

  subroutine save1(path, a)
    character(*), intent(in) :: path
    real(wp), intent(in) :: a(:)
    integer :: ios
    character(len=:), allocatable :: msg
    call save_npy(path, a, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', "save failed: ", trim(path), " " // trim(msg)
      call exit(1)
    end if
  end subroutine save1

  ! Mirror of load_gpt_weights: split stacked buffers back to per-layer
  ! .npy files (flat float32, exporter convention) for resume.
  subroutine save_gpt_weights(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(wp), intent(in) :: wte(:), lm_head(:)
    real(wp), intent(in) :: c_q(:), c_k(:), c_v(:)
    real(wp), intent(in) :: c_pr(:), c_fc(:), c_pr2(:)
    integer :: ll, qsz, ksz, psz, fcsz, p2sz
    character(len=16) :: lstr
    ! vocab_size nao aparece em nome de arquivo, mas aparece no SHAPE: um
    ! wte/lm_head com tamanho errado gera checkpoint que so' falha depois, no
    ! load -- exatamente o "lixo silencioso" que o verify_ckpt_dir existe para
    ! pegar. Falha alto aqui e' mais barato que descobrir no proximo job.
    if (size(wte) /= vocab_size*d_model .or. size(lm_head) /= vocab_size*d_model) then
      print '(A,2I10)', 'save_gpt_weights: wte/lm_head != vocab*d_model: ', &
          size(wte), vocab_size*d_model
      call exit(1)
    end if
    qsz = n_head*head_dim*d_model
    ksz = n_kv_head*head_dim*d_model
    psz = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model
    call save1(trim(wdir) // "/transformer_wte_weight.npy", wte)
    call save1(trim(wdir) // "/lm_head_weight.npy", lm_head)
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_q_weight.npy", c_q(ll*qsz+1:(ll+1)*qsz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_k_weight.npy", c_k(ll*ksz+1:(ll+1)*ksz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_v_weight.npy", c_v(ll*ksz+1:(ll+1)*ksz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_proj_weight.npy", c_pr(ll*psz+1:(ll+1)*psz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_fc_weight.npy", c_fc(ll*fcsz+1:(ll+1)*fcsz))
      call save1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_proj_weight.npy", c_pr2(ll*p2sz+1:(ll+1)*p2sz))
    end do
  end subroutine save_gpt_weights

  ! Post-save integrity check: every expected file must exist with a real
  ! payload. stdlib save_npy can return ios=0 yet leave 0-byte files when
  ! the disk fills (flush/close errors are not reported) — this silently
  ! destroyed phase-3 step_300/step_400 (2026-09-09, disk full from nix
  ! store). Call after every save; abort LOUDLY on mismatch. Never train
  ! on with garbage recovery points.
  subroutine verify_ckpt_dir(wdir, n_layer, nbad, badpath, st_only)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer
    integer, intent(out) :: nbad
    character(len=:), allocatable, intent(out) :: badpath
    logical, intent(in), optional :: st_only
    character(len=16) :: lstr
    character(len=3) :: mname(8) = ["wte", "lm ", "q  ", "k  ", "v  ", &
                                    "p  ", "fc ", "p2 "]
    integer :: ll, ii
    logical :: st_mode
    st_mode = .false.
    if (present(st_only)) st_mode = st_only
    nbad = 0
    badpath = ""
    if (st_mode) then
      ! Checagem mais FORTE que a do .npy: nao basta o arquivo existir com
      ! tamanho plausivel -- o header e revalidado por inteiro (offsets, dtypes,
      ! cobertura do buffer). Um arquivo truncado por disco cheio falha aqui,
      ! que e exatamente o caso que o verify existe para pegar.
      call check_st(trim(wdir) // "/" // ST_CKPT, 2 + 6*n_layer)
    else
      call check1(trim(wdir) // "/transformer_wte_weight.npy")
      call check1(trim(wdir) // "/lm_head_weight.npy")
      do ll = 0, n_layer - 1
        write (lstr, '(I0)') ll
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_attn_c_q_weight.npy")
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_attn_c_k_weight.npy")
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_attn_c_v_weight.npy")
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_attn_c_proj_weight.npy")
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_mlp_c_fc_weight.npy")
        call check1(trim(wdir) // "/transformer_h_" // trim(lstr) // &
            "_mlp_c_proj_weight.npy")
      end do
    end if
    do ii = 1, 8
      call check1(trim(wdir) // "/adam_m_" // trim(mname(ii)) // ".npy")
      call check1(trim(wdir) // "/adam_v_" // trim(mname(ii)) // ".npy")
    end do
    call check_exists(trim(wdir) // "/template.txt")
  contains
    ! Header inteiro do .safetensors revalidado + contagem de tensores esperada.
    subroutine check_st(path, want_tensors)
      character(*), intent(in) :: path
      integer, intent(in) :: want_tensors
      type(st_reader) :: r
      character(len=:), allocatable :: msg
      integer :: stat
      if (nbad /= 0) return
      call r%open(path, stat, msg)
      if (stat /= st_ok) then
        nbad = 1
        badpath = path // ' (' // trim(msg) // ')'
        return
      end if
      if (r%n_tensors() /= want_tensors) then
        nbad = 1
        write (badpath, '(A,A,I0,A,I0)') trim(path), ' has ', r%n_tensors(), &
            ' tensors, expected ', want_tensors
      end if
      call r%close()
    end subroutine check_st

    subroutine check1(path)
      character(*), intent(in) :: path
      logical :: ex
      integer :: sz
      if (nbad /= 0) return
      inquire (file=path, exist=ex, size=sz)
      if (.not. ex .or. sz <= 128) then
        nbad = 1
        badpath = path
      end if
    end subroutine check1
    subroutine check_exists(path)
      character(*), intent(in) :: path
      logical :: ex
      if (nbad /= 0) return
      inquire (file=path, exist=ex)
      if (.not. ex) then
        nbad = 1
        badpath = path
      end if
    end subroutine check_exists
  end subroutine verify_ckpt_dir

  ! ---------------------------------------------------------------------------
  ! Escrita safetensors: UM arquivo, nomes canonicos (wte, lm, l{ll}.q/k/v/p/fc/p2),
  ! arquitetura e procedencia no __metadata__. Os mesmos bytes do caminho .npy --
  ! e isso que o teste compara (payload byte a byte), nao os valores.
  !
  ! `step`, `lr`, `tokens` e `rowsfile` vao no campo `card` (JSON): o checkpoint
  ! passa a se descrever sozinho. As metricas ficam VAZIAS de proposito -- quem
  ! treina preenche depois; um numero inventado aqui viraria verdade no experimento.
  subroutine save_gpt_weights_st(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, ctx, bos, step, lr, tokens, rowsfile, &
      wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2, energy)
    character(*), intent(in) :: wdir, rowsfile
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size, ctx, bos, step
    real(wp), intent(in) :: lr
    integer(int64), intent(in) :: tokens
    real(wp), intent(in) :: wte(:), lm_head(:)
    real(wp), intent(in) :: c_q(:), c_k(:), c_v(:)
    real(wp), intent(in) :: c_pr(:), c_fc(:), c_pr2(:)
    type(energy_interval_t), intent(in), optional :: energy
    type(st_writer) :: w
    character(len=:), allocatable :: msg, card, tname
    character(len=16) :: lstr
    integer :: ll, qsz, ksz, psz, fcsz, p2sz, stat

    qsz = n_head*head_dim*d_model
    ksz = n_kv_head*head_dim*d_model
    psz = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model

    ! Sanidade barata: se as fatias nao fecham, o arquivo sairia certo na
    ! contagem e errado no conteudo -- o pior caso possivel. Melhor morrer aqui.
    if (size(wte) /= vocab_size*d_model .or. size(lm_head) /= vocab_size*d_model .or. &
        size(c_q) /= n_layer*qsz .or. size(c_k) /= n_layer*ksz .or. &
        size(c_v) /= n_layer*ksz .or. size(c_pr) /= n_layer*psz .or. &
        size(c_fc) /= n_layer*fcsz .or. size(c_pr2) /= n_layer*p2sz) then
      call die('save_gpt_weights_st: buffer sizes do not match the arch (refusing to write)')
    end if

    call w%init()
    call w%set_meta('format_version', '1')
    call w%set_meta('producer', 'autoresearch/src train_run')
    call w%set_meta('arch.source', 'fortran_arch_mod')
    call set_meta_i(w, 'arch.d_model', d_model)
    call set_meta_i(w, 'arch.n_head', n_head)
    call set_meta_i(w, 'arch.n_kv', n_kv_head)
    call set_meta_i(w, 'arch.n_layer', n_layer)
    call set_meta_i(w, 'arch.vocab', vocab_size)
    call set_meta_i(w, 'arch.ctx', ctx)
    call set_meta_i(w, 'arch.bos', bos)
    call set_meta_i(w, 'arch.head_dim', head_dim)
    ! IDENTIDADE, nao so' campos: uma string canonica e um id derivado dos
    ! parametros compilados. E' contra isto que o binario se confere (e o
    ! que substitui, aos poucos, o vinculo por NOME de pasta de build).
    call w%set_meta('arch.schema', arch_schema())
    call w%set_meta('arch.canonical', arch_canonical_of(d_model, n_head, &
        n_kv_head, head_dim, n_layer, vocab_size, ctx, bos))
    call w%set_meta('arch.id', arch_id_of(arch_canonical_of(d_model, n_head, &
        n_kv_head, head_dim, n_layer, vocab_size, ctx, bos)))
    call set_meta_i(w, 'n_tensors', 2 + 6*n_layer)
    card = '{"steps":'//i2c(step)//',"lr":'//json_real(lr)// &
           ',"tokens":'//i8c(tokens)//',"rows_file":"'//json_escape(trim(rowsfile))// &
           '","metrics":{}'//energy_card_json(energy)//'}'
    call w%set_meta('card', card)
    call set_meta_energy(w, energy)

    call w%set('wte', wte)
    call w%set('lm', lm_head)
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      tname = 'l'//trim(lstr)//'.q'
      call w%set(tname, c_q(ll*qsz+1:(ll+1)*qsz))
      tname = 'l'//trim(lstr)//'.k'
      call w%set(tname, c_k(ll*ksz+1:(ll+1)*ksz))
      tname = 'l'//trim(lstr)//'.v'
      call w%set(tname, c_v(ll*ksz+1:(ll+1)*ksz))
      tname = 'l'//trim(lstr)//'.p'
      call w%set(tname, c_pr(ll*psz+1:(ll+1)*psz))
      tname = 'l'//trim(lstr)//'.fc'
      call w%set(tname, c_fc(ll*fcsz+1:(ll+1)*fcsz))
      tname = 'l'//trim(lstr)//'.p2'
      call w%set(tname, c_pr2(ll*p2sz+1:(ll+1)*p2sz))
    end do
    call w%write(trim(wdir) // '/' // ST_CKPT, stat, msg)
    if (stat /= st_ok) then
      call die('safetensors save failed for '//trim(wdir)//': '//trim(msg))
    end if
  end subroutine save_gpt_weights_st

  ! Inteiros -> texto (o card e montado a mao; nao vale arrastar um emissor JSON).
  function i2c(v) result(s)
    integer, intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i2c

  function i8c(v) result(s)
    integer(int64), intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i8c

  ! ---------------------------------------------------------------------------
  ! Leitura safetensors. Mesma politica do caminho .npy: qualquer divergencia
  ! entre o que o arquivo declara e o que este binario espera ABORTA (nao existe
  ! "ler o que der"). Aqui isso vale duas vezes: primeiro a arquitetura declarada
  ! no __metadata__ (arch.*), depois o tamanho de cada tensor.
  subroutine load_gpt_weights_st(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(wp), allocatable, intent(out) :: wte(:), lm_head(:)
    real(wp), allocatable, intent(out) :: c_q(:), c_k(:), c_v(:)
    real(wp), allocatable, intent(out) :: c_pr(:), c_fc(:), c_pr2(:)
    type(st_reader) :: r
    character(len=:), allocatable :: msg, tname
    character(len=16) :: lstr
    integer :: ll, qsz, ksz, psz, fcsz, p2sz, stat

    qsz = n_head*head_dim*d_model
    ksz = n_kv_head*head_dim*d_model
    psz = d_model*n_head*head_dim
    fcsz = 4*d_model*d_model
    p2sz = d_model*4*d_model

    call r%open(trim(wdir) // '/' // ST_CKPT, stat, msg)
    if (stat /= st_ok) then
      call die('cannot open '//trim(wdir)//'/'//ST_CKPT//': '//trim(msg))
    end if
    call check_declared_arch(r, wdir, n_layer, d_model, n_head, n_kv_head, &
        head_dim, vocab_size)

    call st_load(r, 'wte', wdir, wte)
    call st_load(r, 'lm', wdir, lm_head)
    allocate(c_q(n_layer*qsz), c_k(n_layer*ksz), c_v(n_layer*ksz))
    allocate(c_pr(n_layer*psz), c_fc(n_layer*fcsz), c_pr2(n_layer*p2sz))
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      tname = 'l'//trim(lstr)//'.q'
      call st_load_into(r, tname, wdir, c_q, ll*qsz + 1)
      tname = 'l'//trim(lstr)//'.k'
      call st_load_into(r, tname, wdir, c_k, ll*ksz + 1)
      tname = 'l'//trim(lstr)//'.v'
      call st_load_into(r, tname, wdir, c_v, ll*ksz + 1)
      tname = 'l'//trim(lstr)//'.p'
      call st_load_into(r, tname, wdir, c_pr, ll*psz + 1)
      tname = 'l'//trim(lstr)//'.fc'
      call st_load_into(r, tname, wdir, c_fc, ll*fcsz + 1)
      tname = 'l'//trim(lstr)//'.p2'
      call st_load_into(r, tname, wdir, c_pr2, ll*p2sz + 1)
    end do
    call r%close()
  end subroutine load_gpt_weights_st

  ! Um tensor para um buffer novo. p32 (e nao real(wp)) de proposito: o get
  ! tipado da lib resolve pelo TIPO do ponteiro, entao fixar F32 aqui mantem o
  ! caminho valido mesmo se `wp` virar real64 um dia (a atribuicao converte).
  subroutine st_load(r, name, wdir, dest)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: name, wdir
    real(wp), allocatable, intent(out) :: dest(:)
    real(real32), pointer :: p32(:)
    integer :: stat
    character(len=:), allocatable :: msg
    p32 => null()
    call r%get(name, p32, stat, msg)
    if (stat /= st_ok) then
      call die('tensor '//trim(name)//' in '//trim(wdir)//'/'//ST_CKPT//': '//trim(msg))
    end if
    dest = p32
    deallocate(p32)
  end subroutine st_load

  ! Um tensor para dentro de um buffer empilhado (fatia ja existente).
  subroutine st_load_into(r, name, wdir, dest, at)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: name, wdir
    real(wp), intent(inout) :: dest(:)
    integer, intent(in) :: at
    real(wp), allocatable :: tmp(:)
    call st_load(r, name, wdir, tmp)
    if (at < 1 .or. at + size(tmp) - 1 > size(dest)) then
      call die('tensor '//trim(name)//' in '//trim(wdir)//'/'//ST_CKPT// &
          ': does not fit the expected per-layer slot (wrong arch?)')
    end if
    dest(at:at+size(tmp)-1) = tmp
    deallocate(tmp)
  end subroutine st_load_into

  ! A arquitetura declarada no __metadata__ (arch.*) tem de bater com a deste
  ! binario. Chave AUSENTE e ignorada (arquivo de outra ferramenta); chave
  ! presente e divergente aborta -- mesma politica do arch.txt, pelo mesmo
  ! motivo: checkpoint de outra arquitetura produz lixo silencioso.
  subroutine check_declared_arch(r, wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim, vocab_size
    call cmp_meta(r, wdir, 'arch.d_model', d_model)
    call cmp_meta(r, wdir, 'arch.n_head', n_head)
    call cmp_meta(r, wdir, 'arch.n_kv', n_kv_head)
    call cmp_meta(r, wdir, 'arch.n_layer', n_layer)
    call cmp_meta(r, wdir, 'arch.vocab', vocab_size)
    call cmp_meta(r, wdir, 'arch.head_dim', head_dim)
    ! ctx/bos nao entram na checagem: nao afetam o layout dos pesos (So o
    ! require_arch do arch.txt os confere, e so quando o arch.txt existe).
    ! Identidade: o checkpoint tem de ser CONSISTENTE CONSIGO -- os campos
    ! arch.*, a canonica declarada (arch.canonical) e o id declarado (arch.id)
    ! tem de contar a mesma historia. Isto pega metadata corrompida ou editada a
    ! mao, e funciona para archs sinteticas (testes). A identidade contra ESTE
    ! binario e' a checagem campo-a-campo acima; a SELECAO de binario e' feita
    ! por `arch_id --from DIR` / bin/arch_check.sh.
    call check_arch_selfconsistent(r, wdir)
  end subroutine check_declared_arch

  ! Autoconsistencia da identidade: campos <-> canônica <-> id.
  subroutine check_arch_selfconsistent(r, wdir)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: wdir
    integer :: d, nh, nkv, hd, nl, vv, ctx, bos
    character(len=:), allocatable :: val, c1
    logical :: found, any_missing
    d = 0; nh = 0; nkv = 0; hd = 0; nl = 0; vv = 0; ctx = 0; bos = 0
    call read_meta_i(r, 'arch.d_model', d)
    call read_meta_i(r, 'arch.n_head', nh)
    call read_meta_i(r, 'arch.n_kv', nkv)
    call read_meta_i(r, 'arch.head_dim', hd)
    call read_meta_i(r, 'arch.n_layer', nl)
    call read_meta_i(r, 'arch.vocab', vv)
    call read_meta_i(r, 'arch.ctx', ctx)
    call read_meta_i(r, 'arch.bos', bos)
    any_missing = min(d, nh, nkv, hd, nl, vv, ctx) <= 0
    if (any_missing) return                    ! checkpoint antigo: nada a checar
    c1 = arch_canonical_of(d, nh, nkv, hd, nl, vv, ctx, bos)
    call r%meta('arch.canonical', val, found)
    if (found .and. trim(val) /= trim(c1)) then
      print '(A)', 'FATAL: metadata de arch inconsistente (arch.canonical != campos)'
      print '(2A)', '  arquivo declara: ', trim(val)
      print '(2A)', '  campos somam   : ', trim(c1)
      print '(2A)', '  checkpoint: ', trim(wdir)
      call exit(1)
    end if
    call r%meta('arch.id', val, found)
    if (found .and. trim(val) /= arch_id_of(c1)) then
      print '(A)', 'FATAL: metadata de arch inconsistente (arch.id != hash dos campos)'
      print '(2A)', '  arquivo declara: ', trim(val)
      print '(2A)', '  campos somam   : ', arch_id_of(c1)
      print '(2A)', '  checkpoint: ', trim(wdir)
      call exit(1)
    end if
  end subroutine check_arch_selfconsistent

  ! Arch de um checkpoint: __metadata__ do safetensors PRIMEIRO (padrao, viaja com
  ! os pesos), arch.txt como FALLBACK (init em .npy). Um so' leitor para todo
  ! mundo -- app e teste -- em vez de cada um parsear o sidecar a sua maneira.
  subroutine read_arch_any(dir, d_model, n_head, n_kv_head, head_dim, n_layer, &
      vocab_size, ctx, bos, source)
    character(*), intent(in) :: dir
    integer, intent(out) :: d_model, n_head, n_kv_head, head_dim, n_layer, &
        vocab_size, ctx, bos
    character(*), intent(out) :: source
    type(st_reader) :: r
    character(len=:), allocatable :: msg
    integer :: stat, u, ios
    character(len=512) :: line
    logical :: have_all

    d_model = 0; n_head = 0; n_kv_head = 0; head_dim = 0; n_layer = 0
    vocab_size = 0; ctx = 0; bos = 0
    source = ''
    have_all = .false.
    call r%open(trim(dir)//'/'//ST_CKPT, stat, msg)
    if (stat == st_ok) then
      call read_meta_i(r, 'arch.d_model', d_model)
      call read_meta_i(r, 'arch.n_head', n_head)
      call read_meta_i(r, 'arch.n_kv', n_kv_head)
      call read_meta_i(r, 'arch.head_dim', head_dim)
      call read_meta_i(r, 'arch.n_layer', n_layer)
      call read_meta_i(r, 'arch.vocab', vocab_size)
      call read_meta_i(r, 'arch.ctx', ctx)
      call read_meta_i(r, 'arch.bos', bos)
      call r%close()
      have_all = min(d_model, n_head, n_kv_head, head_dim, n_layer, &
          vocab_size, ctx) > 0
      if (have_all) source = trim(dir)//'/'//ST_CKPT//' (__metadata__)'
    end if
    if (.not. have_all) then
      open (newunit=u, file=trim(dir)//'/arch.txt', status='old', action='read', iostat=ios)
      if (ios /= 0) then
        write (*, '(2A)') 'read_arch_any: sem metadata de arch em ', trim(dir)
        call exit(1)
      end if
      do
        read (u, '(A)', iostat=ios) line
        if (ios /= 0) exit
        call take_kv(line, 'd_model', d_model)
        call take_kv(line, 'n_head', n_head)
        call take_kv(line, 'n_kv', n_kv_head)
        call take_kv(line, 'head_dim', head_dim)
        call take_kv(line, 'n_layer', n_layer)
        call take_kv(line, 'vocab', vocab_size)
        call take_kv(line, 'ctx', ctx)
        call take_kv(line, 'bos', bos)
      end do
      close (u)
      if (min(d_model, n_head, n_kv_head, head_dim, n_layer, vocab_size, ctx) <= 0) then
        write (*, '(2A)') 'read_arch_any: arch.txt incompleto em ', trim(dir)
        call exit(1)
      end if
      source = trim(dir)//'/arch.txt'
    end if
    write (*, '(2A)') '# arch lida de: ', trim(source)
  end subroutine read_arch_any

  subroutine read_meta_i(r, key, val)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: key
    integer, intent(inout) :: val
    character(len=:), allocatable :: s
    integer :: got, ios
    logical :: found
    call r%meta(key, s, found)
    if (.not. found) return
    read (s, *, iostat=ios) got
    if (ios == 0) val = got
  end subroutine read_meta_i

  subroutine take_kv(line, key, val)
    character(*), intent(in) :: line, key
    integer, intent(inout) :: val
    integer :: p, q
    p = index(line, trim(key)//' =')
    if (p /= 1) return
    q = index(line, '=')
    read (line(q+1:), *) val
  end subroutine take_kv

  subroutine cmp_meta(r, wdir, key, want)
    type(st_reader), intent(in) :: r
    character(*), intent(in) :: wdir, key
    integer, intent(in) :: want
    character(len=:), allocatable :: val
    integer :: got, ios
    logical :: found
    call r%meta(key, val, found)
    if (.not. found) return
    read (val, *, iostat=ios) got
    if (ios /= 0) then
      call die('metadata '//trim(key)//' in '//trim(wdir)//'/'//ST_CKPT// &
          ' is not an integer: "'//trim(val)//'"')
    end if
    if (got /= want) then
      print '(A,I0,A,I0)', 'FATAL: architecture mismatch ('//trim(key)// &
          '): this binary expects ', want, ' | file declares ', got
      print '(2A)', '  checkpoint: ', trim(wdir)
      print '(A)', '  este binario foi compilado para outro tamanho de modelo;'
      print '(A)', '  aponte para o checkpoint certo (ou use set_arch.sh e rebuild)'
      call exit(1)
    end if
  end subroutine cmp_meta

  ! Ponto de entrada unico dos apps: escolhe a representacao pela PRESENCA do
  ! model.safetensors no diretorio. Nenhum app precisa saber qual e -- e por isso
  ! que a auto-deteccao vive aqui e nao em cada leitor.
  !
  ! O aviso de qual caminho foi lido vai para o STDERR (unit 0), nunca para o
  ! stdout: eval_bpb tem o stdout parseado por scripts/eval_driver.py, que espera
  ! linhas de NLL. Um "loading..." no stdout quebraria o driver em silencio.
  subroutine load_gpt_weights(wdir, n_layer, d_model, n_head, n_kv_head, &
      head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
    character(*), intent(in) :: wdir
    integer, intent(in) :: n_layer, d_model, n_head, n_kv_head, head_dim
    integer, intent(in) :: vocab_size
    real(wp), allocatable, intent(out) :: wte(:), lm_head(:)
    real(wp), allocatable, intent(out) :: c_q(:), c_k(:), c_v(:)
    real(wp), allocatable, intent(out) :: c_pr(:), c_fc(:), c_pr2(:)
    integer :: ll
    character(len=16) :: lstr
    logical :: has_st

    inquire (file=trim(wdir) // '/' // ST_CKPT, exist=has_st)
    if (has_st) then
      write (0, '(3A)') 'weights: ', trim(wdir), '/'//ST_CKPT//' (safetensors, canonical names)'
      call load_gpt_weights_st(wdir, n_layer, d_model, n_head, n_kv_head, &
          head_dim, vocab_size, wte, lm_head, c_q, c_k, c_v, c_pr, c_fc, c_pr2)
      return
    end if
    write (0, '(3A)') 'weights: ', trim(wdir), '/transformer_*.npy (legacy layout)'

    call load1(trim(wdir) // "/transformer_wte_weight.npy", wte)
    call load1(trim(wdir) // "/lm_head_weight.npy", lm_head)
    allocate(c_q(n_layer*n_head*head_dim*d_model))
    allocate(c_k(n_layer*n_kv_head*head_dim*d_model))
    allocate(c_v(n_layer*n_kv_head*head_dim*d_model))
    allocate(c_pr(n_layer*d_model*n_head*head_dim))
    allocate(c_fc(n_layer*4*d_model*d_model))
    allocate(c_pr2(n_layer*d_model*4*d_model))
    do ll = 0, n_layer - 1
      write (lstr, '(I0)') ll
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_q_weight.npy", c_q, ll*n_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_k_weight.npy", c_k, ll*n_kv_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_v_weight.npy", c_v, ll*n_kv_head*head_dim*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_attn_c_proj_weight.npy", c_pr, ll*d_model*n_head*head_dim + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_fc_weight.npy", c_fc, ll*4*d_model*d_model + 1)
      call load_into(trim(wdir) // "/transformer_h_" // trim(lstr) // &
          "_mlp_c_proj_weight.npy", c_pr2, ll*d_model*4*d_model + 1)
    end do
  end subroutine load_gpt_weights

end module load_weights_mod
