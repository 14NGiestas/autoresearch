#!/bin/bash
#SBATCH -J sweep_sum
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH -o /home/pauli/autoresearch/logs/sweep_summary_%A.log
# Roda depois de todos os braços e imprime a CURVA: bpb em prosa held-out por
# fração de wiki. É a resposta medida à pergunta de dosagem do mix.
set -u
echo "=== CURVA DE DOSE-RESPOSTA (3M params, 20M tokens por braço) ==="
printf "  %-8s %-6s %-14s %s\n" "wiki%" "passos" "val bpb final" "trn bpb final"
for i in 0 1 2 3 4; do
  f=$(echo "0 25 50 75 100" | cut -d' ' -f$((i+1)))
  L=/home/pauli/autoresearch/logs/sweep3m_$i.log
  [ -f "$L" ] || { printf "  %-8s (sem log)\n" "$f%"; continue; }
  V=$(grep -E '^val @' "$L" | tail -1 | awk '{print $3}')
  T=$(grep -E '^trn @' "$L" | tail -1 | awk '{print $3}')
  S=$(grep -E '^val @' "$L" | tail -1 | awk '{print $2}')
  printf "  %-8s %-6s %-14s %s\n" "${f}%" "${S:-?}" "${V:-?}" "${T:-?}"
done
echo "(menor bpb em prosa = melhor para o alvo literario; compare com o controle f=0)"
