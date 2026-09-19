! lib/fortran_probe.f90 — PROBE da hierarquia de memoria, medido de dentro do
! processo. Modulo GENERICO: nao conhece o modelo, o KV, nem o fortran_gpt --
! so' mede a maquina. Quem traduz isso em decisao (recomputar ou ler?) e' o app
! que usa este modulo (src/app/kv_tier.f90), juntando estas constantes com as
! constantes do modelo que vem de fortran_arch_mod.
!
! O que ele mede, para um arquivo de trabalho:
!
!   ram  x seq    leitura sequencial quente (page cache)         [banda]
!   ram  x rand   leitura de N bytes em offset ALEATORIO, quente  [latencia/IO]
!   disk x seq    idem sequencial, depois de fsync+FADV_DONTNEED  [banda do device]
!   disk x rand   idem aleatorio, frio                            [latencia do device]
!
! Regras de honestidade embutidas:
!   * "frio" e' verificado pelo contador read_bytes de /proc/self/io (o quente
!     nao pode ter ido ao disco; o frio tem que ter ido) -- ver probe_trust();
!   * tiro unico nao vale: cada celula roda --reps vezes (com nova eviccao entre
!     repeticoes no caso frio) e reporta a MEDIANA e o espalhamento;
!   * o contexto de contencao fica registrado (cores_busy/cpu_pct), porque medir
!     com a maquina ocupada muda a banda (medido: 9.4 a 22 GB/s na mesma maquina);
!   * energia: J vem do fortran_energy (RAPL). LIMITACAO: o contador do package
!     NAO ve o SSD -- no nivel disco o J medido e' custo de ESPERA do processo.
!
! TODO(energia): este modulo e' candidato natural a mudar para o pacote
! energy-fortran (probe generico de maquina, sem nada de GPT). Fica aqui por
! enquanto porque o energy-fortran esta em publicacao por outro agente.
module fortran_probe_mod
  use, intrinsic :: iso_fortran_env, only: int64, real32, real64
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t
  use fortran_energy_mod, only: energy_mark, energy_seconds, energy_interval_t
  implicit none
  private

  public :: probe_tier_t, probe_file, probe_cell, probe_report, probe_json, &
      probe_median, probe_trust, probe_verbose

  ! Uma celula medida: um nivel (ram|disk) x um padrao (seq|rand).
  type :: probe_tier_t
    character(len=16) :: level = ''      ! 'ram' | 'disk'
    character(len=16) :: pattern = ''    ! 'seq' | 'rand'
    real(real64) :: bytes = 0.0_real64   ! bytes movidos na medicao
    real(real64) :: wall_s = 0.0_real64  ! mediana das repeticoes
    real(real64) :: spread_s = 0.0_real64 ! (max-min)/mediana das repeticoes
    real(real64) :: bw_mbps = 0.0_real64 ! MB/s da mediana
    real(real64) :: us_per_op = 0.0_real64 ! us por read() (rand) ou por item (seq)
    real(real64) :: j = 0.0_real64       ! J do RAPL na mediana
    real(real64) :: j_per_gb = 0.0_real64
    real(real64) :: rd_mb = 0.0_real64   ! bytes que vieram do disco (read_bytes)
    real(real64) :: cores_busy = 0.0_real64
    real(real64) :: cpu_pct = 0.0_real64
    integer :: reps = 0
    logical :: ok = .false.
  end type probe_tier_t

  interface
    function c_fadvise(fd, off, ln, advice) bind(C, name="posix_fadvise") result(rc)
      import :: c_int, c_int64_t
      integer(c_int), value :: fd
      integer(c_int64_t), value :: off, ln
      integer(c_int), value :: advice
      integer(c_int) :: rc
    end function c_fadvise
    function c_fsync(fd) bind(C, name="fsync") result(rc)
      import :: c_int
      integer(c_int), value :: fd
      integer(c_int) :: rc
    end function c_fsync
  end interface

  integer, parameter :: ADV_DONTNEED = 4
  integer, parameter :: MAX_REPS = 32
  integer, parameter :: ITEM_REALS = 768      ! 3072 B por item (1 token de KV 3M)
  integer, parameter :: PROBE_RV = storage_size(0.0_real32)/8   ! bytes por real do item

contains

  ! ---- cria o arquivo de trabalho (mb MB) e devolve o caminho usado -------
  function probe_file(path, mb, item_reals) result(bytes)
    character(*), intent(in) :: path
    integer, intent(in) :: mb, item_reals
    integer(int64) :: bytes
    integer :: u, nrec, i
    real(real32), allocatable :: buf(:), rec(:)
    nrec = max(1, int(real(mb, real64)*1.0e6_real64)/max(1, item_reals*PROBE_RV))
    allocate (buf(item_reals*256), rec(item_reals))
    call random_number(buf); call random_number(rec)
    open (newunit=u, file=path, form='unformatted', access='stream', &
        status='replace', action='readwrite')
    do i = 1, nrec/256
      write (u) buf
    end do
    do i = 1, mod(nrec, 256)
      write (u) rec
    end do
    bytes = int(nrec, int64)*int(item_reals, int64)*int(PROBE_RV, int64)
    ! fsync + 1 s de quiescencia ANTES de qualquer medida. Sem isso, o writeback
    ! das paginas sujas ainda esta drenando quando a primeira celula comeca e o
    ! spread passa de 1000% (medido 2026-09-18). Quem mede nao pode medir a
    ! propria escrita.
    i = c_fsync(fnum(u))
    close (u)
    call sleep(1)
  end function probe_file

  ! ---- mede UMA celula (level x pattern), com `reps` repeticoes -------------
  ! level='ram'  : le' o arquivo (page cache), sem eviccao.
  ! level='disk' : fsync + FADV_DONTNEED antes de CADA repeticao.
  ! pattern='seq': percorre o arquivo inteiro (bytes/item_reals operacoes).
  ! pattern='rand': faz n_ops leituras de item_reals reais em offset sorteado.
  subroutine probe_cell(path, level, pattern, reps, item_reals, n_ops, t)
    character(*), intent(in) :: path, level, pattern
    integer, intent(in) :: reps, item_reals, n_ops
    type(probe_tier_t), intent(out) :: t
    integer :: u, fd, i, k, r, nrec, nr
    integer(int64) :: state
    integer(int64), allocatable :: offs(:)
    real(real32), allocatable :: rec(:), buf(:)
    real(real64) :: wall(MAX_REPS), jj(MAX_REPS), rd(MAX_REPS), cb(MAX_REPS), cp(MAX_REPS)
    type(energy_interval_t) :: iv
    real(real64) :: med
    logical :: evict

    t%level = level; t%pattern = pattern; t%reps = reps; t%ok = .false.
    evict = level == 'disk'
    nrec = file_items(path, item_reals)
    if (nrec <= 0) return
    if (pattern == 'rand') then
      nr = min(nrec, max(1, n_ops))
    else
      nr = nrec
    end if
    allocate (rec(item_reals), buf(item_reals*256))
    call random_number(rec); call random_number(buf)
    allocate (offs(nr))
    state = 12345_int64
    do k = 1, nr
      state = mod(state*1103515245_int64 + 12345_int64, 2147483648_int64)
      offs(k) = mod(state, int(nrec, int64))*int(item_reals, int64)*int(PROBE_RV, int64) + 1
    end do

    ! Aquecimento: para o nivel ram, uma passada FORA da medida (senao a pagina
    ! pode estar suja do write e o numero sai contaminado).
    if (.not. evict) call probe_pass(path, pattern, item_reals, nr, offs, nrec)

    do r = 1, reps
      open (newunit=u, file=path, form='unformatted', access='stream', status='old')
      fd = fnum(u)
      if (evict) then
        i = c_fsync(fd)
        i = c_fadvise(fd, 0_c_int64_t, int(nrec, c_int64_t)*int(item_reals, c_int64_t)* &
                      int(PROBE_RV, c_int64_t), &
                      ADV_DONTNEED)
      end if
      if (pattern == 'rand') then
        do k = 1, nr
          read (u, pos=offs(k)) rec
        end do
      else
        do i = 1, nrec/256
          read (u) buf
        end do
        do i = 1, mod(nrec, 256)
          read (u) rec
        end do
      end if
      call energy_mark('probe_' // trim(level) // '_' // trim(pattern), &
          tokens=int(nr, int64), iv=iv)
      close (u)
      wall(r) = iv%wall_s; jj(r) = iv%j; rd(r) = iv%rd_mb
      cb(r) = iv%cores_busy; cp(r) = iv%cpu_pct
      if (probe_verbose()) write (*, '(A,A,A,A,I0,A,F10.6,A,ES10.3)') &
          '#   rep ', trim(level), ' x ', trim(pattern), r, ' wall=', iv%wall_s, ' J=', iv%j
    end do

    t%bytes = real(nr, real64)*real(item_reals, real64)*real(PROBE_RV, real64)
    med = probe_median(wall(1:reps))
    t%wall_s = med
    t%spread_s = (maxval(wall(1:reps)) - minval(wall(1:reps)))/max(1.0e-12_real64, med)
    t%bw_mbps = (t%bytes/1.0e6_real64)/max(1.0e-12_real64, med)
    t%us_per_op = 1.0e6_real64*med/real(max(1, nr), real64)
    t%j = probe_median(jj(1:reps))
    t%j_per_gb = t%j/(t%bytes/1.0e9_real64 + 1.0e-12_real64)
    t%rd_mb = probe_median(rd(1:reps))
    t%cores_busy = probe_median(cb(1:reps))
    t%cpu_pct = probe_median(cp(1:reps))
    t%ok = .true.
  end subroutine probe_cell

  ! ---- uma passada de leitura (usada no aquecimento e nas medidas) --------
  subroutine probe_pass(path, pattern, item_reals, nr, offs, nrec)
    character(*), intent(in) :: path, pattern
    integer, intent(in) :: item_reals, nr, nrec
    integer(int64), intent(in) :: offs(:)
    integer :: u, i, k
    real(real32), allocatable :: rec(:), buf(:)
    allocate (rec(item_reals), buf(item_reals*256))
    call random_number(rec); call random_number(buf)
    open (newunit=u, file=path, form='unformatted', access='stream', status='old')
    if (pattern == 'rand') then
      do k = 1, nr
        read (u, pos=offs(k)) rec
      end do
    else
      do i = 1, nrec/256
        read (u) buf
      end do
      do i = 1, mod(nrec, 256)
        read (u) rec
      end do
    end if
    close (u)
  end subroutine probe_pass

  function probe_verbose() result(v)
    logical :: v
    character(len=8) :: e
    call get_environment_variable('PROBE_VERBOSE', e)
    v = len_trim(e) > 0 .and. trim(e) /= '0'
  end function probe_verbose

  ! ---- mediana (copia pequena; sem dependencia) ---------------------------
  function probe_median(v) result(m)
    real(real64), intent(in) :: v(:)
    real(real64) :: m, w(size(v))
    integer :: i, k
    w = v
    do i = 2, size(w)
      do k = i, 2, -1
        if (w(k) < w(k - 1)) then
          m = w(k); w(k) = w(k - 1); w(k - 1) = m
        else
          exit
        end if
      end do
    end do
    k = size(w)
    if (mod(k, 2) == 1) then
      m = w((k + 1)/2)
    else
      m = 0.5_real64*(w(k/2) + w(k/2 + 1))
    end if
  end function probe_median

  ! ---- o numero e' confiavel? (mecanismo + contencao) ---------------------
  function probe_trust(ram, disk, msg) result(ok)
    type(probe_tier_t), intent(in) :: ram, disk
    character(*), intent(out) :: msg
    logical :: ok
    ok = .false.
    if (.not. ram%ok .or. .not. disk%ok) then
      msg = 'celula nao medida'
      return
    end if
    if (disk%rd_mb < 0.5_real64*disk%bytes/1.0e6_real64) then
      msg = 'frio nao veio do disco (FADV_DONTNEED sem efeito?)'
      return
    end if
    if (ram%rd_mb > 0.05_real64*disk%rd_mb) then
      msg = 'quente leu do disco (page cache nao estava quente?)'
      return
    end if
    if (ram%bw_mbps < 0.5_real64*disk%bw_mbps) then
      msg = 'banda quente < 0.5x fria (RAM mais lenta que o device?)'
      return
    end if
    if (ram%spread_s > 0.5_real64 .or. disk%spread_s > 0.5_real64) then
      write (msg, '(A,F6.0,A,F6.0,A)') 'celula instavel (spread ram ', 100.0_real64*ram%spread_s, &
          '%, disco ', 100.0_real64*disk%spread_s, '%): repita com a maquina ociosa'
      return
    end if
    ok = .true.
    msg = 'ok'
    ! cpu_s do /proc/self/stat tem granularidade de USER_HZ (10 ms): em celula
    ! curta o cores_busy oscila por quantizacao, entao so' vale se wall for
    ! grande o bastante para a medida significar algo.
    if (ram%wall_s > 0.2_real64 .and. ram%cores_busy > 1.5_real64) then
      msg = 'ok, mas a maquina estava ocupada (cores_busy>1.5): trate grosseiro'
    end if
  end function probe_trust

  ! ---- relatorio legivel --------------------------------------------------
  subroutine probe_report(t, unit)
    type(probe_tier_t), intent(in) :: t
    integer, intent(in), optional :: unit
    integer :: u
    u = 6
    if (present(unit)) u = unit
    write (u, '(A,A,A,A,A,F9.2,A,F10.1,A,F9.3,A,F9.2,A,ES10.3,A,ES10.3,A,F6.2,A,F6.1,A,F6.1,A)') &
        '# probe ', trim(t%level), ' x ', trim(t%pattern), ': mediana ', t%wall_s, &
        ' s  spread ', 100.0_real64*t%spread_s, ' %  banda ', t%bw_mbps, &
        ' MB/s  us/op ', t%us_per_op, '  J ', t%j, '  J/GB ', t%j_per_gb, &
        '  rd_MB ', t%rd_mb, '  cores ', t%cores_busy, '  cpu% ', t%cpu_pct
  end subroutine probe_report

  ! ---- JSON (uma linha por celula; para citar o numero num run) -----------
  subroutine probe_json(t, unit, tag)
    type(probe_tier_t), intent(in) :: t
    integer, intent(in) :: unit
    character(*), intent(in), optional :: tag
    character(len=64) :: tg
    tg = ''
    if (present(tag)) tg = trim(tag)
    write (unit, '(A)', advance='no') '{"level":"'//trim(t%level)//'","pattern":"'// &
        trim(t%pattern)//'","tag":"'//trim(tg)//'"'
    write (unit, '(A,F14.6)', advance='no') ',"wall_s":', t%wall_s
    write (unit, '(A,F12.6)', advance='no') ',"spread":', t%spread_s
    write (unit, '(A,F14.3)', advance='no') ',"bw_mbps":', t%bw_mbps
    write (unit, '(A,F12.4)', advance='no') ',"us_per_op":', t%us_per_op
    write (unit, '(A,ES14.6)', advance='no') ',"j":', t%j
    write (unit, '(A,ES14.6)', advance='no') ',"j_per_gb":', t%j_per_gb
    write (unit, '(A,F12.3)', advance='no') ',"rd_mb":', t%rd_mb
    write (unit, '(A,F8.3)', advance='no') ',"cores_busy":', t%cores_busy
    write (unit, '(A,F8.2)', advance='no') ',"cpu_pct":', t%cpu_pct
    write (unit, '(A,I0,A)') ',"reps":', t%reps, '}'
  end subroutine probe_json

  ! ---- helpers ------------------------------------------------------------
  function file_items(path, item_reals) result(n)
    character(*), intent(in) :: path
    integer, intent(in) :: item_reals
    integer :: n
    integer :: u
    integer(int64) :: sz
    open (newunit=u, file=path, form='unformatted', access='stream', status='old', &
        action='read')
    inquire (u, size=sz)
    close (u)
    n = int(sz/max(1, item_reals*PROBE_RV))
  end function file_items

end module fortran_probe_mod
