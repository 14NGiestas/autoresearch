! test_texture.f90 — invariantes do painel de textura (nao valores dourados).
!
! O que tem de valer em qualquer maquina:
!   1. determinismo: mesmo texto -> mesmo painel;
!   2. PROSA bate SALADA nos eixos que importam (palavra_plausivel para cima,
!      byte_alto e distinct_2 para baixo) -- e o que faz a metrica servir;
!   3. a escada classifica cada um no degrau certo;
!   4. bordas: vazio, 1 byte, e texto so' de bytes altos nao quebram;
!   5. o JSON tem as chaves e fecha.
program test_texture
  use, intrinsic :: iso_fortran_env, only: real64
  use fortran_texture_mod, only: texture_t, texture_panel, texture_json
  implicit none
  integer :: nfail, i
  type(texture_t) :: tp, ts, t1, t2, te
  character(len=:), allocatable :: prose, salad, jline

  nfail = 0
  ! prosa (ASCII, com palavras e pontuacao) -- 3 blocos concatenados
  prose = "A computacao e a ciencia que estuda processos que podem ser descritos por " // &
      "algoritmos. Um algoritmo e uma sequencia finita de instrucoes bem definidas, " // &
      "tipicamente usada para resolver uma classe de problemas. O modelo aprende com " // &
      "dados e ajusta seus pesos; o objetivo desta linha de trabalho e medir a " // &
      "eficiencia do treinamento e a textura do que o modelo gera, sem enganar a " // &
      "gente com numeros bonitos que nao significam nada na pratica do dia a dia."
  ! salada: bytes altos, sem estrutura de palavra
  salad = ""
  do i = 1, 120
    salad = salad // achar(160 + mod(i*7, 90)) // achar(190 + mod(i*11, 60))
  end do

  call texture_panel(prose, tp)
  call texture_panel(salad, ts)
  call check(tp%palavra_plausivel > ts%palavra_plausivel, 'prosa > salada em palavra_plausivel')
  call check(tp%byte_alto < ts%byte_alto, 'prosa < salada em byte_alto')
  call check(tp%distinct2 < ts%distinct2, 'prosa < salada em distinct_2 (salada nao tem redundancia)')
  call check(index(tp%estagio, '1)') == 0, 'prosa NAO cai no degrau 1')
  call check(index(ts%estagio, '1)') > 0, 'salada cai no degrau 1')
  call check(tp%palavra_plausivel > 0.8_real64, 'prosa tem palavra_plausivel > 0.8')

  call texture_panel(prose, t1); call texture_panel(prose, t2)
  call check(t1%n == t2%n .and. abs(t1%palavra_plausivel - t2%palavra_plausivel) <= 0.0_real64, &
      'determinismo')

  call texture_panel("", te)
  call check(te%n == 0, 'vazio nao quebra')
  call texture_panel("x", te)
  call check(te%n == 1, '1 byte nao quebra')

  jline = texture_json(tp)
  call check(index(jline, '"palavra_plausivel"') > 0, 'json tem palavra_plausivel')
  call check(index(jline, '"estagio"') > 0, 'json tem estagio')
  call check(jline(len_trim(jline):len_trim(jline)) == '}', 'json fecha')

  write (*, '(A,F7.4,A,F7.4,A,A)') 'prosa: palavra_plausivel=', tp%palavra_plausivel, &
      ' byte_alto=', tp%byte_alto, ' -> ', trim(tp%estagio)
  write (*, '(A,F7.4,A,F7.4,A,A)') 'salada: palavra_plausivel=', ts%palavra_plausivel, &
      ' byte_alto=', ts%byte_alto, ' -> ', trim(ts%estagio)
  if (nfail /= 0) error stop 'test_texture: FALHOU'
  write (*, '(A)') 'test_texture: OK'

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
end program test_texture
