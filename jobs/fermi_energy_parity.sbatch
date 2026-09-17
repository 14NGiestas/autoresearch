#!/bin/bash
#SBATCH -J eparity
#SBATCH -c 8
#SBATCH -t 00:15:00
#SBATCH -o /home/pauli/autoresearch/logs/eparity_%j.log
# Paridade do instrumento: o MESMO processo medido por dentro (fortran_energy)
# e por fora (scripts/energy_run.py + hwmon). Se os dois J concordarem, o
# instrumento esta provado; divergencia >5% significa bug em um dos dois.
set -u
cd /home/pauli/autoresearch
echo "eparity job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
BIN=$(ls -t energy-fortran/build/*/app/measure_run 2>/dev/null | head -1)
echo "exemplo: $BIN"
.venv-numpy/bin/python3 scripts/energy_run.py run --tag parity/energy-fortran \
  --interval 0.25 -- $BIN 2>&1 | tail -3
echo "--- auto-relato do processo:"
OMP_NUM_THREADS=8 "$BIN" 2>&1 | grep -iE "joule|energy|J |cpu|cores" | head -8
echo "--- ledger (externo):"
.venv-numpy/bin/python3 scripts/energy_run.py report --since 2026-09-17 2>/dev/null | tail -3
echo "fim eparity $(date '+%F %H:%M:%S')"
