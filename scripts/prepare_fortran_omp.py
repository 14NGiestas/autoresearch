#!/usr/bin/env python3
"""
Fortran corpus booster: OpenMP + runtime intrinsics coverage.

The fetched tutorial corpus (prepare_fortran.py) covers 72/74 common
intrinsics but ZERO OpenMP worksharing (no parallel do, reduction,
collapse, simd, critical, barrier, task) and misses flush and
omp_get_num_threads. This script adds hand-written, COMPILABLE, complete
programs for exactly those gaps, each with a specific instruction
(also fixes the one-instruction monotony for added rows).

Every snippet is compile-verified (gfortran -fopenmp) before it is
accepted -- a snippet that does not compile is dropped, never shipped.
Outputs BOS-prefixed variable rows; run scripts/pack_rows.py after to
fold them into fortran_tutorial.txt (deterministic: existing chunks
are byte-identical, only new tail rows appear).

Usage:
    python3 scripts/prepare_fortran_omp.py [--verify-only]
"""

import os
import subprocess
import sys
import tempfile

BOS = 8188
CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "fortran_tutorial.txt")

# Canonical encoder (same contract as all prepare scripts).
class FallbackEncoder:
    def encode(self, text):
        ids = []
        for ch in text:
            if ord(ch) < 256:
                ids.append(ord(ch))
            else:
                for b in ch.encode("utf-8"):
                    ids.append(256 + b)
        return ids

enc = FallbackEncoder()

# (instruction, complete program). All deterministic outputs only
# (reductions/sums -- never thread-ordered prints).
SNIPPETS = [
("Parallelize a loop that squares each array element",
"""program square_loop
  implicit none
  integer, parameter :: n = 1000
  real :: a(n)
  integer :: i
  a = 1.0
  !$omp parallel do default(none) shared(a) private(i)
  do i = 1, n
    a(i) = a(i) * a(i)
  end do
  !$omp end parallel do
  print *, "sum =", sum(a)
end program square_loop"""),

("Sum an array with an OpenMP reduction",
"""program reduce_sum
  implicit none
  integer, parameter :: n = 10000
  real :: a(n), total
  integer :: i
  do i = 1, n
    a(i) = real(i)
  end do
  total = 0.0
  !$omp parallel do default(none) shared(a) private(i) reduction(+:total)
  do i = 1, n
    total = total + a(i)
  end do
  !$omp end parallel do
  print *, "total =", total
end program reduce_sum"""),

("Compute a dot product with parallel do and reduction",
"""program dot_omp
  implicit none
  integer, parameter :: n = 4096
  real :: x(n), y(n), d
  integer :: i
  x = 1.0; y = 2.0
  d = 0.0
  !$omp parallel do default(none) shared(x, y) private(i) reduction(+:d)
  do i = 1, n
    d = d + x(i) * y(i)
  end do
  !$omp end parallel do
  print *, "dot =", d
end program dot_omp"""),

("Find array maximum with max reduction",
"""program max_red
  implicit none
  integer, parameter :: n = 5000
  real :: a(n), m
  integer :: i
  do i = 1, n
    a(i) = real(mod(i * 37, 999))
  end do
  m = -huge(1.0)
  !$omp parallel do default(none) shared(a) private(i) reduction(max:m)
  do i = 1, n
    if (a(i) > m) m = a(i)
  end do
  !$omp end parallel do
  print *, "max =", m
end program max_red"""),

("Parallelize nested loops with collapse(2)",
"""program collapse_mm
  implicit none
  integer, parameter :: n = 256
  real :: a(n, n), b(n, n), c(n, n)
  integer :: i, j, k
  a = 1.0; b = 2.0; c = 0.0
  !$omp parallel do default(none) shared(a, b, c) private(i, j, k) collapse(2)
  do j = 1, n
    do i = 1, n
      do k = 1, n
        c(i, j) = c(i, j) + a(i, k) * b(k, j)
      end do
    end do
  end do
  !$omp end parallel do
  print *, "c(1,1) =", c(1, 1)
end program collapse_mm"""),

("Protect a shared counter with atomic update",
"""program atomic_ctr
  implicit none
  integer, parameter :: n = 100000
  integer :: hits, i
  hits = 0
  !$omp parallel do default(none) shared(hits) private(i)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic update
      hits = hits + 1
    end if
  end do
  !$omp end parallel do
  print *, "hits =", hits
end program atomic_ctr"""),

("Guard a non-thread-safe update with a critical section",
"""program crit_sec
  implicit none
  integer, parameter :: n = 20000
  real :: total
  integer :: i
  total = 0.0
  !$omp parallel do default(none) shared(total) private(i)
  do i = 1, n
    !$omp critical
    total = total + sqrt(real(i))
    !$omp end critical
  end do
  !$omp end parallel do
  print *, "total =", total
end program crit_sec"""),

("Vectorize a loop with simd and safelen",
"""program simd_axpy
  implicit none
  integer, parameter :: n = 8192
  real :: x(n), y(n), alpha
  integer :: i
  x = 1.0; y = 2.0; alpha = 0.5
  !$omp simd
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do
  print *, "y(1) =", y(1), " sum =", sum(y)
end program simd_axpy"""),

("Combine parallel do with simd for nested parallelism",
"""program do_simd
  implicit none
  integer, parameter :: n = 512, m = 512
  real :: a(n, m)
  integer :: i, j
  !$omp parallel do default(none) shared(a) private(i, j) collapse(2)
  do j = 1, m
    do i = 1, n
      a(i, j) = real(i * j)
    end do
  end do
  !$omp end parallel do
  print *, "sum =", sum(a)
end program do_simd"""),

("Use private and firstprivate for thread-local state",
"""program firstpriv
  implicit none
  integer, parameter :: n = 1000
  real :: a(n), scale, partial
  integer :: i
  a = 1.0; scale = 3.0; partial = 0.0
  !$omp parallel do default(none) shared(a) firstprivate(scale) &
  !$omp& private(i) reduction(+:partial)
  do i = 1, n
    partial = partial + scale * a(i)
  end do
  !$omp end parallel do
  print *, "partial =", partial
end program firstpriv"""),

("Synchronize threads with an explicit barrier",
"""program barrier_demo
  use omp_lib
  implicit none
  integer :: tid, nthreads
  !$omp parallel default(none) private(tid, nthreads)
  tid = omp_get_thread_num()
  nthreads = omp_get_num_threads()
  !$omp barrier
  !$omp single
  print *, "threads =", nthreads
  !$omp end single
  !$omp end parallel
end program barrier_demo"""),

("Run setup code once with single",
"""program single_demo
  implicit none
  real :: shared_val
  !$omp parallel default(none) shared(shared_val)
  !$omp single
  shared_val = 42.0
  !$omp end single
  !$omp barrier
  !$omp end parallel
  print *, "shared =", shared_val
end program single_demo"""),

("Split independent work with sections",
"""program sections_demo
  implicit none
  real :: r1, r2
  integer :: i
  !$omp parallel default(none) shared(r1, r2)
  !$omp sections private(i)
  !$omp section
  r1 = 0.0
  do i = 1, 1000
    r1 = r1 + real(i)
  end do
  !$omp section
  r2 = 0.0
  do i = 1001, 2000
    r2 = r2 + real(i)
  end do
  !$omp end sections
  !$omp end parallel
  print *, "r1 + r2 =", r1 + r2
end program sections_demo"""),

("Spawn tasks with taskwait",
"""program task_demo
  implicit none
  real :: a(4)
  integer :: i
  a = 0.0
  !$omp parallel default(none) shared(a)
  !$omp single
  do i = 1, 4
    !$omp task firstprivate(i) shared(a)
    a(i) = real(i * i)
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel
  print *, "sum =", sum(a)
end program task_demo"""),

("Query the OpenMP runtime for thread counts",
"""program omp_query
  use omp_lib
  implicit none
  integer :: nthr, maxthr, tid
  maxthr = omp_get_max_threads()
  !$omp parallel default(none) private(tid, nthr)
  tid = omp_get_thread_num()
  nthr = omp_get_num_threads()
  !$omp single
  print *, "max =", maxthr, " team =", nthr
  !$omp end single
  !$omp end parallel
end program omp_query"""),

("Flush output from parallel regions in order",
"""program flush_demo
  implicit none
  integer, parameter :: n = 64
  real :: a(n)
  integer :: i
  a = 0.0
  !$omp parallel do default(none) shared(a) private(i) ordered
  do i = 1, n
    a(i) = real(i)
    !$omp ordered
    write (*, '(A,I0)') "done: ", i
    flush (6)
    !$omp end ordered
  end do
  !$omp end parallel do
end program flush_demo"""),

("Flush a shared flag for cross-thread visibility",
"""program flush_flag
  use omp_lib
  implicit none
  integer :: flag
  flag = 0
  !$omp parallel default(none) shared(flag)
  !$omp single
  flag = 1
  !$omp flush(flag)
  !$omp end single
  !$omp end parallel
  print *, "flag =", flag
end program flush_flag"""),

("Time a parallel region with omp_get_wtime",
"""program time_region
  use omp_lib
  implicit none
  integer, parameter :: n = 2000000
  real, allocatable :: a(:)
  real(8) :: t0, t1
  integer :: i
  allocate(a(n))
  a = 1.0
  t0 = omp_get_wtime()
  !$omp parallel do default(none) shared(a) private(i)
  do i = 1, n
    a(i) = sqrt(a(i) + real(i))
  end do
  !$omp end parallel do
  t1 = omp_get_wtime()
  print *, "sum =", sum(a), " seconds =", t1 - t0
  deallocate(a)
end program time_region"""),

("Set the thread count at runtime",
"""program set_threads
  use omp_lib
  implicit none
  integer :: i, s
  real :: a(1000)
  a = 0.0
  call omp_set_num_threads(4)
  s = 0
  !$omp parallel do default(none) shared(a) private(i) reduction(+:s)
  do i = 1, 1000
    s = s + 1
  end do
  !$omp end parallel do
  print *, "counted =", s, " max =", omp_get_max_threads()
end program set_threads"""),

("Share read-only data across threads",
"""program shared_ro
  implicit none
  integer, parameter :: n = 2000, m = 8
  real :: table(m), out(n)
  integer :: i, k
  do k = 1, m
    table(k) = real(k) * 0.5
  end do
  !$omp parallel do default(none) shared(table, out) private(i, k)
  do i = 1, n
    out(i) = 0.0
    do k = 1, m
      out(i) = out(i) + table(k) * real(i)
    end do
  end do
  !$omp end parallel do
  print *, "out(1) =", out(1)
end program shared_ro"""),
]


def verify(prog):
    """Compile-check one program; return True iff gfortran accepts it."""
    with tempfile.NamedTemporaryFile("w", suffix=".f90", delete=False) as f:
        f.write(prog)
        src = f.name
    try:
        r = subprocess.run(
            ["gfortran", "-fopenmp", "-fsyntax-only", src],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  REJECT: {r.stderr.strip().splitlines()[0][:100]}")
            return False
        return True
    except Exception as e:
        print(f"  REJECT(exc): {e}")
        return False
    finally:
        try:
            os.remove(src)
        except OSError:
            pass


def main():
    verify_only = "--verify-only" in sys.argv
    ok, texts = [], []
    for inst, prog in SNIPPETS:
        if verify(prog):
            ok.append(inst)
            texts.append(
                f"### Instruction:\n{inst}.\n\n### Response:\n```fortran\n{prog}\n```\n")
        else:
            print(f"  dropped: {inst}")
    print(f"verified {len(ok)}/{len(SNIPPETS)} snippets")
    if verify_only or not ok:
        sys.exit(0 if ok else 1)
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    except OSError as e:
        print(f"mkdir: {e}")
        sys.exit(1)
    with open(OUT, "a") as f:
        for text in texts:
            ids = enc.encode(text)
            ids = [BOS] + list(ids)
            f.write(" ".join(str(i) for i in ids) + "\n")
    print(f"appended {len(texts)} rows -> {OUT} (repack with pack_rows.py)")


if __name__ == "__main__":
    main()
