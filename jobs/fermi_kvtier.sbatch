#!/bin/bash
#SBATCH -J kvtier
#SBATCH -c 1
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/kvtier_%j.log
# Re-medicao LIMPA (maquina ociosa, 1 core) do kvtier: recomputar vs ler o KV.
# A primeira medicao (21:50) rodou com um job de 8 cores martelando a maquina.
set -u
cd /home/pauli/autoresearch
BIN=$(ls kvtier/build/*/app/kvtier | tail -1)
echo "kvtier job=${SLURM_JOB_ID} bin=$BIN $(date '+%F %H:%M:%S')"
for N in 32 128 512; do
  echo "================ janela N=$N"
  $BIN --tokens $N --reps 5 --mb 512 --nreads 4096
done
echo "fim kvtier $(date '+%F %H:%M:%S')"
