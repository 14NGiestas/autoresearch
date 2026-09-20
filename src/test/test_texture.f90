! test_texture.f90 — the invariants of the texture module.
!
! The test holds no fixed value. It holds these rules:
!   1. The module gives the same result for the same text.
!   2. Real prose wins against a salad on the three important values. Prose has
!      a higher palavra_plausivel, and a lower byte_alto and distinct_2.
!   3. Each text falls on the correct step of the scale.
!   4. An empty text, a one-byte text, and a text of high bytes do not stop the
!      program.
!   5. The JSON line holds the keys and ends with a brace.
program test_texture
  use, intrinsic :: iso_fortran_env, only: real64
  use fortran_texture_mod, only: texture_t, texture_panel, texture_json
  implicit none
  integer :: nfail, i
  type(texture_t) :: tp, ts, t1, t2, te
  character(len=:), allocatable :: prose, salad, jline

  nfail = 0
  ! Real prose with ASCII words and punctuation. Three blocks follow.
  prose = "A computacao e a ciencia que estuda processos que podem ser descritos por " // &
      "algoritmos. Um algoritmo e uma sequencia finita de instrucoes bem definidas, " // &
      "tipicamente usada para resolver uma classe de problemas. O modelo aprende com " // &
      "dados e ajusta seus pesos; o objetivo desta linha de trabalho e medir a " // &
      "eficiencia do treinamento e a textura do que o modelo gera, sem enganar a " // &
      "gente com numeros bonitos que nao significam nada na pratica do dia a dia."
  ! A salad of bytes. It holds no word structure.
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

  write (*, '(A,F7.4,A,F7.4,A,A)') 'prose: plausible_word=', tp%palavra_plausivel, &
      ' byte_alto=', tp%byte_alto, ' -> ', trim(tp%estagio)
  write (*, '(A,F7.4,A,F7.4,A,A)') 'salad: plausible_word=', ts%palavra_plausivel, &
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
