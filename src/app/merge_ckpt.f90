! app/merge_ckpt.f90 — model soup em Fortran: média ponderada de dois checkpoints.
!
! Mecanismo (sim, é só álgebra linear — a parte não-trivial é o PORQUÊ funciona):
!   θ_soup = α·θ_A + (1−α)·θ_B,  elemento a elemento, sobre os 74 arquivos de peso.
!
! Por que isso funciona em vez de destruir o modelo: os dois ramos partiram do
! MESMO init, com os MESMOS hiperparâmetros, no MESMO objetivo, em dados da mesma
! distribuição. Com isso os pesos ficam na mesma bacia de perda (conectividade
! linear por modo): drift total medido nos nossos ramos = 0.000373 relativo.
! Nessa condição, a média preserva a estrutura compartilhada e tira a média dos
! erros independentes aprendidos em dados disjuntos — é bagging no espaço de
! pesos em vez de na saída. É por isso que a soup (2.08738) bateu os dois pais
! (2.11527, 2.12831): cada ramo viu 2.05M tokens diferentes, e a média herdou
! conhecimento dos DOIS blocos sem treinar um passo sequer.
!
! Regras (as mesmas da sessão):
!   - require_arch em A e B: valida cada um contra a arquitetura deste binário.
!     Transitividade faz o resto (A==binário e B==binário => A==B).
!     Mismatches abortam ALTO, nunca geram lixo silencioso.
!   - momentos do Adam NÃO têm média: são histórico do otimizador, não estado do
!     modelo. Escrevemos ZEROS EXPLÍCITOS (não ausência): o diretório fica
!     estruturalmente idêntico a um checkpoint normal, qualquer ferramenta o lê,
!     e retomar dele é fresh start documentado. verify_ckpt_dir EXIGE os 16
!     arquivos adam (e template.txt) -- e nunca vamos relaxar esse guarda, que
!     pegou um desastre real.
!   - verify_ckpt_dir depois do save: stdlib save_npy pode retornar ios=0 e deixar
!     arquivo de 0 bytes em disco cheio (destruiu phase-3 step_300/step_400) —
!     treinar em cima disso nunca mais.
!
! O scripts/merge_checkpoints.py continua existindo como REFERÊNCIA (benchmark):
! a suite de pares compara os dois byte a byte (esperado: diff máx ~1e-7, que é
! o arredondamento fp32 entre acumulação em float64 do Python e wp do Fortran).
program merge_ckpt
  use iso_c_binding
  use fortran_kinds_mod, only: wp
  use load_weights_mod, only: load_gpt_weights, save_gpt_weights, verify_ckpt_dir
  use stdlib_io_npy, only: save_npy
  use fortran_arch_mod, only: A_D => D_MODEL, A_HEAD => N_HEAD, A_KV => N_KV, &
      A_HD => HD, A_LAYER => N_LAYER, A_VOCAB => VV, A_CTX => TT, A_BOS => BOS, &
      write_arch_txt, require_arch
  use fortran_chat_mod, only: write_template_txt
  use M_CLI2, only: set_args, sget, rget, specified
  implicit none

  integer, parameter :: D = A_D, N_HEAD = A_HEAD, N_KV = A_KV, HD = A_HD
  integer, parameter :: N_LAYER = A_LAYER, VV = A_VOCAB

  character(len=512) :: dirA, dirB, dirout
  real(wp) :: alpha
  logical :: ok, archok
  integer :: nbad, i
  character(len=:), allocatable :: badpath
  real(wp), allocatable :: a_wte(:), a_lm(:), a_q(:), a_k(:), a_v(:)
  real(wp), allocatable :: a_pr(:), a_fc(:), a_p2(:)
  real(wp), allocatable :: b_wte(:), b_lm(:), b_q(:), b_k(:), b_v(:)
  real(wp), allocatable :: b_pr(:), b_fc(:), b_p2(:)
  real(wp), allocatable :: s_wte(:), s_lm(:), s_q(:), s_k(:), s_v(:)
  real(wp), allocatable :: s_pr(:), s_fc(:), s_p2(:)
  real(kind=8) :: sq, dr, rel
  integer :: u, ios

  call set_args('--a A --b B --out OUT --alpha ALPHA', &
      help_text=[character(len=80) :: &
      'NAME', &
      '  merge_ckpt - soup: media ponderada de dois checkpoints (mesmo pai,', &
      '  mesmos hiperparametros, dados disjuntos)', &
      'SYNOPSIS', &
      '  merge_ckpt --a DIR --b DIR --out DIR [--alpha 0.5]', &
      'OPTIONS', &
      '  --alpha X   peso de A (0..1; default 0.5). Fora de [0,1] aborta.', &
      'NOTES', &
      '  Momentos do Adam nao tem media e nao sao copiados: zeros ao retomar.', &
      '  arch.txt de A e B sao validados; divergencia aborta.'], &
      version_text=[character(len=80) :: 'merge_ckpt 1.0'])
  dirA = trim(sget('a')); dirB = trim(sget('b')); dirout = trim(sget('out'))
  alpha = 0.5_wp
  if (specified('alpha')) alpha = real(rget('alpha'), wp)
  if (.not. specified('a') .or. .not. specified('b') .or. .not. specified('out')) then
    print '(A)', 'require --a DIR --b DIR --out DIR [--alpha X] (--help for all)'
    call exit(2)
  end if
  if (alpha < 0.0_wp .or. alpha > 1.0_wp) then
    print '(A,F6.3)', 'alpha fora de [0,1]: ', alpha
    call exit(2)
  end if
  if (trim(dirA) == trim(dirout) .or. trim(dirB) == trim(dirout)) then
    print '(A)', 'out tem de ser um diretorio novo, diferente de --a e --b'
    call exit(2)
  end if

  ! Valida OS DOIS contra a arquitetura deste binário (transitividade: A==bin e
  ! B==bin => A==B). Sem isso, misturar vocabulários diferentes dá lixo mudo.
  call require_arch(trim(dirA))
  call require_arch(trim(dirB))

  call load_gpt_weights(trim(dirA), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      a_wte, a_lm, a_q, a_k, a_v, a_pr, a_fc, a_p2)
  call load_gpt_weights(trim(dirB), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      b_wte, b_lm, b_q, b_k, b_v, b_pr, b_fc, b_p2)

  call execute_command_line('mkdir -p ' // trim(dirout))

  ! Média em wp nativo (rápido, exato o bastante: a referência em float64 do
  ! Python difere por ~1e-7, que é o arredondamento fp32 — e o bpb não se move).
  s_wte = alpha*a_wte + (1.0_wp - alpha)*b_wte
  s_lm  = alpha*a_lm  + (1.0_wp - alpha)*b_lm
  s_q   = alpha*a_q   + (1.0_wp - alpha)*b_q
  s_k   = alpha*a_k   + (1.0_wp - alpha)*b_k
  s_v   = alpha*a_v   + (1.0_wp - alpha)*b_v
  s_pr  = alpha*a_pr  + (1.0_wp - alpha)*b_pr
  s_fc  = alpha*a_fc  + (1.0_wp - alpha)*b_fc
  s_p2  = alpha*a_p2  + (1.0_wp - alpha)*b_p2

  ! Drift total: ||a-b||/||a|| sobre TODOS os pesos (acumulador em real*8, porque
  ! a soma ingênua em real32 de ~1e8 termos perderia os decimais que importam).
  sq = 0.0d0; dr = 0.0d0
  call acc(a_wte, b_wte); call acc(a_lm, b_lm); call acc(a_q, b_q)
  call acc(a_k, b_k); call acc(a_v, b_v); call acc(a_pr, b_pr)
  call acc(a_fc, b_fc); call acc(a_p2, b_p2)
  rel = sqrt(sq / max(dr, tiny(1.0d0)))
  print '(A,F8.6)', 'drift total (||a-b||/||a||): ', rel
  print '(A)', '  (pequeno = mesma bacia, a média tende a ajudar;'
  print '(A)', '   grande = bacias diferentes, a média pode piorar)'

  call save_gpt_weights(trim(dirout), N_LAYER, D, N_HEAD, N_KV, HD, VV, &
      s_wte, s_lm, s_q, s_k, s_v, s_pr, s_fc, s_p2)
  ! Zeros explícitos do Adam: retomar daqui = fresh start (momentos não têm
  ! média). Sem eles o verify abaixo falha -- e o verify não se relaxa.
  call save_zero(trim(dirout)//'/adam_m_wte.npy', s_wte)
  call save_zero(trim(dirout)//'/adam_v_wte.npy', s_wte)
  call save_zero(trim(dirout)//'/adam_m_lm.npy', s_lm)
  call save_zero(trim(dirout)//'/adam_v_lm.npy', s_lm)
  call save_zero(trim(dirout)//'/adam_m_q.npy', s_q)
  call save_zero(trim(dirout)//'/adam_v_q.npy', s_q)
  call save_zero(trim(dirout)//'/adam_m_k.npy', s_k)
  call save_zero(trim(dirout)//'/adam_v_k.npy', s_k)
  call save_zero(trim(dirout)//'/adam_m_v.npy', s_v)
  call save_zero(trim(dirout)//'/adam_v_v.npy', s_v)
  call save_zero(trim(dirout)//'/adam_m_p.npy', s_pr)
  call save_zero(trim(dirout)//'/adam_v_p.npy', s_pr)
  call save_zero(trim(dirout)//'/adam_m_fc.npy', s_fc)
  call save_zero(trim(dirout)//'/adam_v_fc.npy', s_fc)
  call save_zero(trim(dirout)//'/adam_m_p2.npy', s_p2)
  call save_zero(trim(dirout)//'/adam_v_p2.npy', s_p2)
  call write_arch_txt(trim(dirout))
  call write_template_txt(trim(dirout))
  ! verify é o ÚLTIMO passo: ele exige pesos + adam + arch + template.
  ! (Antes chamávamos verify antes de escrever arch/template e ele falhava
  !  no próprio diretório que acabávamos de criar -- ordem importa.)
  call verify_ckpt_dir(trim(dirout), N_LAYER, nbad, badpath)
  if (nbad /= 0) then
    print '(2A)', 'save falhou (disco cheio?): ', trim(badpath)
    call exit(1)
  end if

  open (newunit=u, file=trim(dirout)//'/soup.txt', status='replace', &
      action='write', iostat=ios)
  if (ios == 0) then
    write (u, '(A)') '# merge_ckpt: procedencia da soup'
    write (u, '(2A)') 'parent_A = ', trim(dirA)
    write (u, '(2A)') 'parent_B = ', trim(dirB)
    write (u, '(A,F6.4)') 'alpha = ', alpha
    write (u, '(A,F12.8)') 'drift_total = ', rel
    write (u, '(A)') 'adam = zeros EXPLICITOS (momentos nao tem media; retomar = fresh start)'
    close (u)
  end if
  print '(3A)', 'soup em ', trim(dirout), ' (pesos + arch.txt + soup.txt)'

contains

  ! Zeros no formato exato do saver do projeto (mesma rotina save_npy,
  ! mesmo dtype fp32), para o checkpoint ficar estruturalmente completo.
  subroutine save_zero(path, like)
    character(*), intent(in) :: path
    real(wp), intent(in) :: like(:)
    real(wp), allocatable :: z(:)
    integer :: ios
    character(len=:), allocatable :: msg
    allocate(z(size(like)))
    z = 0.0_wp
    call save_npy(path, z, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      print '(3A)', 'save zero falhou: ', trim(path), ' ' // trim(msg)
      call exit(1)
    end if
  end subroutine save_zero

  subroutine acc(aa, bb)
    real(wp), intent(in) :: aa(:), bb(:)
    integer :: i
    do i = 1, size(aa)
      sq = sq + (real(aa(i), kind=8) - real(bb(i), kind=8))**2
      dr = dr + real(aa(i), kind=8)**2
    end do
  end subroutine acc

end program merge_ckpt
