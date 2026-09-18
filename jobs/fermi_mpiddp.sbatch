#!/bin/bash
#SBATCH -J mpiddp
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpiddp_%j.log
# Evidencias do trainer data-parallel MPI (mpi/). O script roda tudo: o teste de
# invariante (2 ranks x 1 rank byte-identicos), o refactor puro contra o binario
# PRE-refactor (arch_d216, so leitura) e a medicao do imposto de tau=1.
# NAO recompila d96/d216 e nao escreve em src/build/: o binario MPI vive em
# mpi/build/ e o scratch em mpi/scratch_20260917/.
set -u
cd /home/pauli/autoresearch
chmod +x mpi/tests_ddp.sh
exec nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash -c 'bash mpi/tests_ddp.sh; bash mpi/tests_tax.sh'
