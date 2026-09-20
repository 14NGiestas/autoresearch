! lib/fortran_texture.f90 — measure the texture of a text.
!
! The module measures bytes. It does not decode the text first. A decode step
! replaces a bad byte with a replacement character, and that changes the result.
! One character is one byte. The function iachar gives the byte value.
!
! The scale has three steps:
!   1. Byte floor. The text is a salad of bytes.
!   2. Usable generator. The words and the punctuation are correct, and no loop
!      repeats.
!   3. Reasoner. The model corrects itself. The R1 "aha" moment is on this
!      step. A 3M model with ctx 1024 cannot reach step 3. Step 2 is the target.
!
! The module reports these values:
!   alpha              The part of the bytes that are ASCII letters.
!   byte_alto          The part of the bytes with a value of 128 or more.
!   palavra_plausivel  The part of the letter runs with 2 to 12 bytes.
!   distinct_1/2/3     The type/token ratio of byte n-grams. A salad gives a
!                      high value. A language repeats common pairs.
!   maior_laco         The longest run of equal 8-byte n-grams.
!   rep_frac           The part of the 8-byte n-grams that appear two times.
! The last two values count only n-grams with a non-blank byte.
!
! The test test_texture.f90 holds the calibration. Real prose gives a
! palavra_plausivel near 0.95 and a byte_alto near 0.005. A trained 3M model
! gives 0.20 and 0.146. That is step 1.
module fortran_texture_mod
  use, intrinsic :: iso_fortran_env, only: int64, real64
  implicit none
  private

  public :: texture_t, texture_panel, texture_json, texture_stage

  integer, parameter :: LACO_K = 8
  integer, parameter :: MAX_GRAMS = 200000

  type :: texture_t
    integer :: n = 0
    real(real64) :: alpha = 0.0_real64
    real(real64) :: byte_alto = 0.0_real64
    real(real64) :: palavra_plausivel = 0.0_real64
    real(real64) :: distinct1 = 0.0_real64
    real(real64) :: distinct2 = 0.0_real64
    real(real64) :: distinct3 = 0.0_real64
    real(real64) :: rep_frac = 0.0_real64
    integer :: palavras = 0
    integer :: maior_laco = 0
    character(len=48) :: estagio = ''
  end type texture_t

contains

  logical pure function is_alpha(b) result(r)
    integer, intent(in) :: b
    r = (b >= 65 .and. b <= 90) .or. (b >= 97 .and. b <= 122)
  end function is_alpha

  logical pure function is_blank(b) result(r)
    integer, intent(in) :: b
    r = b == 32 .or. b == 9 .or. b == 10 .or. b == 13
  end function is_blank

  ! ---- painel completo ---------------------------------------------------
  subroutine texture_panel(text, t)
    character(*), intent(in) :: text
    type(texture_t), intent(out) :: t
    integer :: i, b, j, n, na, nh, runs, plaus, cur
    integer :: d1, d2, d3
    integer :: best, curn, rep
    integer(int64) :: s1, s2, s3
    integer :: grams(MAX_GRAMS)
    integer :: ng, k
    logical :: has_nonspace

    n = len(text)
    t%n = n
    if (n <= 0) then
      t%estagio = 'empty or too short'
      return
    end if

    ! ---- contagens por byte ---------------------------------------------
    na = 0; nh = 0
    do i = 1, n
      b = iachar(text(i:i))
      if (is_alpha(b)) na = na + 1
      if (b >= 128) nh = nh + 1
    end do
    t%alpha = real(na, real64)/real(n, real64)
    t%byte_alto = real(nh, real64)/real(n, real64)

    ! ---- runs de letras ASCII (as "palavras") ---------------------------
    runs = 0; plaus = 0; cur = 0
    do i = 1, n + 1
      b = -1
      if (i <= n) b = iachar(text(i:i))
      if (i <= n .and. is_alpha(b)) then
        cur = cur + 1
      else
        if (cur > 0) then
          runs = runs + 1
          if (cur >= 2 .and. cur <= 12) plaus = plaus + 1
        end if
        cur = 0
      end if
    end do
    t%palavras = runs
    t%palavra_plausivel = 0.0_real64
    if (runs > 0) t%palavra_plausivel = real(plaus, real64)/real(runs, real64)

    ! ---- distinct-n sobre BYTES (hash simples + tabela de vistos) -------
    ! Use hashes. Do not store the n-grams. Memory stays constant. A hash
    ! collision is not a problem here, because this value is a metric and not an
    ! identifier.
    call distinctn(text, 1, d1, s1)
    call distinctn(text, 2, d2, s2)
    call distinctn(text, 3, d3, s3)
    t%distinct1 = ratiod(d1, s1)
    t%distinct2 = ratiod(d2, s2)
    t%distinct3 = ratiod(d3, s3)

    ! ---- laco: 8-gramas consecutivos iguais (so' nao-brancos) -----------
    ng = 0
    do i = 1, n - LACO_K + 1
      has_nonspace = .false.
      do j = 0, LACO_K - 1
        if (.not. is_blank(iachar(text(i + j:i + j)))) has_nonspace = .true.
      end do
      if (has_nonspace .and. ng < MAX_GRAMS) then
        ng = ng + 1
        grams(ng) = hashgram(text, i, LACO_K)
      end if
    end do
    ! Mesma licao do crash do repl: nada de guarda dentro de .and. (Fortran nao
    ! garante curto-circuito). Fronteira ESTRUTURAL.
    ! Start the loop at 2. Then the index k-1 always exists. No guard is
    ! necessary, and no guard sits inside an .and. Fortran does not promise a
    ! short circuit in .and. That fault stopped the repl.
    best = 0
    if (ng >= 1) then
      best = 1; curn = 1
      do k = 2, ng
        if (grams(k) == grams(k - 1)) then
          curn = curn + 1
        else
          curn = 1
        end if
        if (curn > best) best = curn
      end do
    end if
    t%maior_laco = best
    rep = 0
    do k = 1, ng
      do j = k + 1, ng
        if (grams(j) == grams(k)) then
          rep = rep + 1
          exit
        end if
      end do
    end do
    t%rep_frac = 0.0_real64
    if (ng > 0) t%rep_frac = real(rep, real64)/real(ng, real64)

    t%estagio = texture_stage(t)
  end subroutine texture_panel

  ! type/token de n-gramas de n bytes
  subroutine distinctn(text, k, distinct, total)
    character(*), intent(in) :: text
    integer, intent(in) :: k
    integer, intent(out) :: distinct
    integer(int64), intent(out) :: total
    integer, allocatable :: hs(:)
    integer :: i, j, m
    integer(int64) :: tmp
    m = max(0, len(text) - k + 1)
    total = int(m, int64)
    distinct = 0
    if (m <= 0) return
    allocate (hs(m))
    do i = 1, m
      hs(i) = hashgram(text, i, k)
    end do
    ! insertion sort (n pequeno: centenas a milhares de bytes)
    do i = 2, m
      tmp = int(hs(i), int64)
      j = i
      do while (j > 1)
        if (int(hs(j - 1), int64) <= tmp) exit
        hs(j) = hs(j - 1)
        j = j - 1
      end do
      hs(j) = int(tmp)
    end do
    distinct = 1
    do i = 2, m
      if (hs(i) /= hs(i - 1)) distinct = distinct + 1
    end do
    deallocate (hs)
  end subroutine distinctn

  integer function hashgram(text, start, k) result(h)
    character(*), intent(in) :: text
    integer, intent(in) :: start, k
    integer :: i, b
    integer(int64) :: acc
    acc = int(z'0123456789abcdef', int64)
    do i = 0, k - 1
      b = iachar(text(start + i:start + i))
      acc = ieor(acc, int(b, int64))
      acc = ior(shiftl(acc, 17), iand(shiftr(acc, 47), int(z'1ffff', int64)))
      acc = ieor(acc, shiftr(acc, 29))
    end do
    h = int(iand(acc, int(z'7fffffff', int64)))     ! positivo, 31 bits
  end function hashgram

  real(real64) function ratiod(a, b) result(r)
    integer, intent(in) :: a
    integer(int64), intent(in) :: b
    if (b <= 0_int64) then
      r = 0.0_real64
    else
      r = real(a, real64)/real(b, real64)
    end if
  end function ratiod

  ! ---- a escada, calibrada nos controles deste repo ---------------------
  function texture_stage(t) result(s)
    type(texture_t), intent(in) :: t
    character(len=48) :: s
    if (t%n < 20) then
      s = 'empty or too short'
    else if (t%palavra_plausivel < 0.50_real64 .or. t%byte_alto > 0.10_real64) then
      s = '1) byte floor (salad)'
    else if (t%palavra_plausivel < 0.80_real64 .or. t%maior_laco > 3 .or. &
             t%rep_frac > 0.30_real64) then
      s = 'transition: structure appears'
    else
      s = '2) usable generator'
    end if
  end function texture_stage

  ! ---- one JSON line, for the card and the metadata ----------------------
  ! The function returns an allocatable string. It does not write into a
  ! character(*) dummy. A write into an unallocated deferred-length string stops
  ! the program with "End of record". The first version of the test hit this
  ! fault.
  function texture_json(t, key) result(s)
    type(texture_t), intent(in) :: t
    character(*), intent(in), optional :: key
    character(len=:), allocatable :: s
    character(len=32) :: k
    character(len=512) :: buf
    k = 'texture'
    if (present(key)) then
      if (len_trim(key) > 0) k = trim(key)
    end if
    write (buf, '(A,A,A,I0,A,F7.4,A,F7.4,A,F7.4,A,F7.4,A,F7.4,A,F7.4,A,F7.4,A,I0,A,I0,A,A,A)') &
        '{"', trim(k), '":{"n":', t%n, &
        ',"alpha":', t%alpha, ',"byte_alto":', t%byte_alto, &
        ',"palavra_plausivel":', t%palavra_plausivel, &
        ',"distinct1":', t%distinct1, ',"distinct2":', t%distinct2, &
        ',"distinct3":', t%distinct3, ',"rep_frac":', t%rep_frac, &
        ',"palavras":', t%palavras, ',"maior_laco":', t%maior_laco, &
        ',"estagio":"', trim(t%estagio), '"}}'
    s = trim(buf)
  end function texture_json

end module fortran_texture_mod
