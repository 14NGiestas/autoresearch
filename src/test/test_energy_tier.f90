! test_energy_tier.f90 — invariantes do instrumento de energia/IO e do tier de KV.
!
! Nao testa DESEMPENHO (numeros absolutos dependem da maquina e do barulho da
! fila). Testa o que TEM que ser verdade em qualquer maquina, e que quebra se o
! mecanismo quebrar:
!
!   1. a interface do fortran_energy responde: sensor nao-vazio, wall/cpu > 0,
!      J finito e nao-negativo;
!   2. page cache nao toca o disco: na leitura QUENTE, read_bytes ~ 0;
!   3. fsync + POSIX_FADV_DONTNEED funcionam: na leitura FRIA, read_bytes ~= bytes;
!   4. RAM nao e' mais lenta que o dispositivo (banda quente >= 0.5 * banda fria);
!   5. latencia por IO aparece quando se le' 1 token por read() em offset
!      aleatorio (custo aleatorio >= 2x o custo sequencial por token);
!   6. a contabilidade de tokens/`us por unidade` do energy_mark fecha.
!
! Tudo isso com arquivo de 24 MB e < 1 s, para caber no `fortran-fpm test`.
program test_energy_tier
  use, intrinsic :: iso_fortran_env, only: int64, real64
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t
  use fortran_kinds_mod, only: wp
  use fortran_arch_mod, only: A_D => D_MODEL, A_LAYER => N_LAYER, A_KV => N_KV, &
      A_HD => HD
  use fortran_energy_mod
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none

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
  integer, parameter :: RV = storage_size(0.0_wp)/8
  integer, parameter :: D_KV = A_KV*A_HD
  integer, parameter :: KV_BYTES = A_LAYER*2*D_KV*RV
  integer, parameter :: FILE_MB = 24, NREADS = 512, NTOK = 64
  ! Em real, e' multiplo de 256: as contas de bloco abaixo ficam exatas
  ! (divisao inteira que trunca e' aviso no nosso gate, e com razao).
  integer, parameter :: NREC = int(real(FILE_MB, real64)*1.0e6_real64 / &
      real(KV_BYTES, real64) / 256.0_real64)*256

  integer :: nfail, u, fd, i, k, nrep
  real(real64) :: t0
  integer(int64) :: state
  integer(int64), allocatable :: offs(:)
  real(wp), allocatable :: a(:,:), b(:,:), c(:,:), rec(:), buf(:)
  real(real64) :: acc, mb, bw_warm, bw_cold, per_seq, per_rand, mb_read_warm, mb_read_cold
  type(energy_interval_t) :: iv_c, iv_w, iv_ws, iv_wr, iv_ev, iv_cs, iv_cr
  type(energy_interval_t) :: iv_tot
  character(len=*), parameter :: path = '/tmp/test_energy_tier.bin'

  nfail = 0
  acc = 0.0_real64
  mb = real(NREC, real64)*real(KV_BYTES, real64)/1.0e6_real64

  call energy_init()
  write (*, '(A,A)') 'energy sensor: ', trim(energy_sensor())
  write (*, '(A,A)') 'energy scope : ', trim(energy_scope())

  ! ---- 1. compute: energia/CPU contabilizados ---------------------------------
  allocate(a(NTOK, A_D), b(A_D, D_KV), c(NTOK, D_KV))
  call random_number(a); call random_number(b)
  ! Duracao minima de ~50 ms: /proc/self/stat tem granularidade de USER_HZ
  ! (10 ms), entao uma fase de microssegundos pode reportar cpu_s = 0 e o
  ! invariante "cpu_s > 0" nao diz nada sobre o instrumento.
  nrep = 0
  t0 = energy_seconds()
  do
    c = matmul(a, b)
    nrep = nrep + 1
    if (energy_seconds() - t0 >= 0.05_real64) exit
  end do
  acc = acc + real(sum(c), real64)
  call energy_mark('compute', tokens=int(NTOK, int64)*int(nrep, int64), iv=iv_c)
  call check(iv_c%wall_s > 0.0_real64, 'compute: wall_s > 0')
  call check(iv_c%cpu_s > 0.0_real64, 'compute: cpu_s > 0 (fase com >= 50 ms)')
  call check(nrep > 0, 'compute: iteracoes contadas para o orcamento de tokens')
  call check(iv_c%j >= 0.0_real64 .and. iv_c%j < 1.0e9_real64, 'compute: J finito e >= 0')
  call check(int(iv_c%tokens) == NTOK*nrep, 'compute: tokens atribuidos')
  call check(iv_c%j_per_token >= 0.0_real64, 'compute: j_per_token definido')

  ! ---- escreve o arquivo de KV ------------------------------------------------
  allocate(rec(KV_BYTES/RV), buf(KV_BYTES/RV*256))
  call random_number(buf); call random_number(rec)
  open (newunit=u, file=path, form='unformatted', access='stream', status='replace', &
      action='readwrite')
  do i = 1, NREC/256
    write (u) buf
  end do
  call energy_mark('write', tokens=int(NREC, int64), iv=iv_w)
  call check(iv_w%wr_mb > 0.9_real64*mb, 'write: wr_mb ~= tamanho do arquivo')
  fd = fnum(u)

  ! ---- 2/4. leitura QUENTE (page cache) ---------------------------------------
  rewind (u)
  do i = 1, NREC/256
    read (u) buf
  end do
  call energy_mark('read_warm_seq', tokens=int(NREC, int64), iv=iv_ws)
  mb_read_warm = iv_ws%rd_mb

  ! posicoes aleatorias (mesmas nas duas leituras _rand)
  allocate(offs(NREADS))
  state = 999_int64
  do k = 1, NREADS
    state = mod(state*1103515245_int64 + 12345_int64, 2147483648_int64)
    offs(k) = mod(state, int(NREC, int64))*KV_BYTES + 1
  end do
  do k = 1, NREADS
    read (u, pos=offs(k)) rec
  end do
  call energy_mark('read_warm_rand', tokens=int(NREADS, int64), iv=iv_wr)

  ! ---- 3. leitura FRIA (fsync + DONTNEED) -------------------------------------
  i = c_fsync(fd)
  i = c_fadvise(fd, 0_c_int64_t, int(NREC, c_int64_t)*KV_BYTES, ADV_DONTNEED)
  call energy_mark('evict', iv=iv_ev)
  rewind (u)
  do i = 1, NREC/256
    read (u) buf
  end do
  call energy_mark('read_cold_seq', tokens=int(NREC, int64), iv=iv_cs)
  mb_read_cold = iv_cs%rd_mb
  i = c_fsync(fd)
  i = c_fadvise(fd, 0_c_int64_t, int(NREC, c_int64_t)*KV_BYTES, ADV_DONTNEED)
  do k = 1, NREADS
    read (u, pos=offs(k)) rec
  end do
  call energy_mark('read_cold_rand', tokens=int(NREADS, int64), iv=iv_cr)
  close (u, status='delete')

  call check(mb_read_warm < 0.1_real64*mb, 'page cache nao toca o disco (quente: rd_mb ~ 0)')
  call check(mb_read_cold > 0.9_real64*mb, 'FADV_DONTNEED funciona (frio: rd_mb ~= bytes)')

  ! ---- 4. RAM >= dispositivo --------------------------------------------------
  bw_warm = mb/max(1.0e-9_real64, iv_ws%wall_s)
  bw_cold = mb/max(1.0e-9_real64, iv_cs%wall_s)
  call check(bw_warm >= 0.5_real64*bw_cold, 'banda quente >= 0.5x banda fria')
  per_seq = 1.0e6_real64*iv_cs%wall_s/real(NREC, real64)
  per_rand = 1.0e6_real64*iv_cr%wall_s/real(NREADS, real64)
  call check(per_rand >= 2.0_real64*per_seq, 'latencia por IO (aleatorio) >= 2x sequencial')

  ! ---- 6. contexto do run -----------------------------------------------------
  call energy_peek(iv_tot)
  call check(iv_tot%j_total >= 0.0_real64, 'j_total >= 0')
  call check(len_trim(iv_tot%kind) > 0, 'kind do sensor preenchido')
  call check(.not. ieee_is_nan(acc), 'checksum nao-NaN')

  write (*, '(A,F8.2,A,F8.2,A,F10.2,A,F10.2,A,I0)') 'bandas MB/s: quente ', bw_warm, &
      ' frio ', bw_cold, ' | us/token seq ', per_seq, ' rand ', per_rand, &
      ' | falhas ', nfail
  if (nfail /= 0) error stop 'test_energy_tier: FALHOU'
  write (*, '(A)') 'test_energy_tier: OK'

contains

  subroutine check(ok, what)
    logical, intent(in) :: ok
    character(*), intent(in) :: what
    if (ok) then
      write (*, '(A,A)') '  ok   ', what
    else
      write (*, '(A,A)') '  FAIL ', what
      nfail = nfail + 1
    end if
  end subroutine check

end program test_energy_tier
