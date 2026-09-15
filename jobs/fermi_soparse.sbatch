#!/bin/bash
#SBATCH -J soparse
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/pauli/autoresearch/logs/soparse_%j.log
# SO-PT: extrai Posts.xml, pareia Q + resposta aceita, limpa HTML, conta.
set -u
unset LD_LIBRARY_PATH
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=8
7z x -y /tmp/bench/ptSO.7z -o /tmp/bench/so_xml 2>&1 | tail -2
ls -la /tmp/bench/so_xml/ | head -6
.venv-numpy/bin/python3 scripts/parse_so.py
echo "fim soparse $(date '+%F %H:%M:%S')"
