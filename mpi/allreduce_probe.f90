! mpi/allreduce_probe.f90 -- sonda MINIMA do caminho de DADOS entre ranks.
! Faz MPI_Allreduce(SUM) de 1 MB e reporta rank/host/valor/tempo -- e' o teste
! mais barato que exercita o ch3:tcp rank<->rank (o `hostname` do mpirun nao
! exercita: ele so' precisa do canal de controle/PMI).
! Build: mpif90 -O2 -o allreduce_probe allreduce_probe.f90
program allreduce_probe
  use mpi
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none
  integer :: ierr, rank, size, i, n
  character(len=MPI_MAX_PROCESSOR_NAME) :: name
  integer :: nl
  real(real32), allocatable :: a(:), b(:)
  real(real64) :: t0, t1
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  call MPI_Get_processor_name(name, nl, ierr)
  n = 256*1024                      ! 1 MB em real32
  allocate(a(n), b(n))
  a = real(rank + 1, real32)
  t0 = MPI_Wtime()
  do i = 1, 20                      ! 20 allreduces para tempo medivel
    call MPI_Allreduce(a, b, n, MPI_REAL4, MPI_SUM, MPI_COMM_WORLD, ierr)
    a = b
  end do
  t1 = MPI_Wtime()
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  print '(A,I0,A,A,A,F12.6,A,F8.3,A,F9.2)', "rank ", rank, " host ", &
      trim(name), " sum=", b(1), " | 20x Allreduce de 1 MB: ", t1 - t0, &
      " s  (", 20.0_real64*real(n, real64)*4/(t1 - t0)/1e6, " MB/s egress)"
  flush (6)
  call MPI_Finalize(ierr)
end program allreduce_probe
