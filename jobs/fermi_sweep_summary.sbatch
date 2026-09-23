#!/bin/bash
#SBATCH -J sweep_sum
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH -o /home/pauli/autoresearch/logs/sweep_summary.log
# Imprime a CURVA DE DOSE-RESPOSTA: bpb em prosa literária held-out por fração
# de wiki (0/25/50/75/100%), mesmo total de tokens (20M) por braço.
# Os braços 75/100 rodaram na halfbeast; seus logs são trazidos para logs/
# (sweep_hb_0/1.log) antes de rodar isto. Campos: "val @N bpb X" -> $2 sem @, $4.
set -u
echo "=== CURVA DE DOSE-RESPOSTA (3M params, 20M tokens por braço) ==="
printf "  %-8s %-8s %-14s %s\n" "wiki%" "passos" "val bpb final" "trn bpb final"
i=0
for spec in "0:sweep3m_0" "25:sweep3m_1" "50:sweep3m_2" "75:sweep_hb_0" "100:sweep_hb_1"; do
  f=${spec%%:*}; L=/home/pauli/autoresearch/logs/${spec#*:}.log
  if [ ! -f "$L" ]; then printf "  %-8s (sem log)\n" "${f}%"; i=$((i+1)); continue; fi
  V=$(grep -E '^val @' "$L" | tail -1 | awk '{print $4}')
  T=$(grep -E '^trn @' "$L" | tail -1 | awk '{print $4}')
  S=$(grep -E '^val @' "$L" | tail -1 | awk '{print $2}' | tr -d '@')
  printf "  %-8s %-8s %-14s %s\n" "${f}%" "${S:-?}" "${V:-?}" "${T:-?}"
  i=$((i+1))
done
echo "(menor bpb em prosa = melhor para o alvo literario)"
