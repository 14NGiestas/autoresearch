#!/bin/bash
#SBATCH -J attnmat
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/attnmat_%j.log
# MATRIZ DA ATENCAO (hyp_ad78cc): head_dim x T, e causal-blocado vs quadrado.
#
# Motivacao medida: a atencao e' ~50-60% do passo e roda a 9.8 GFLOP/s com
# head_dim=16, contra 74.3 GFLOP/s com head_dim=128 no mesmo T. Duas alavancas
# candidatas: (a) head_dim maior (mesmo T, mesmos FLOPs, outra forma) e (b) NAO
# calcular o quadrado inteiro (o caminho --attn blas atual faz 2x os FLOPs do
# kernel causal naive, como o proprio bench avisa).
#
# Este job MEDE antes de a gente escrever kernel novo: se o naive-causal (metade
# dos FLOPs) ja' bate o sgemm-quadrado no nosso head_dim pequeno, bloquear paga;
# se nao bate, nao paga e o esforco vai para head_dim.
#
# OMP fixado: medir com 16 threads em vez de 8 da' numeros 2.3x piores (medido).
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
BUILD=${ATTN_BUILD:-/tmp/attn_build_test}   # reusa build existente: evita corrida de build
echo "attnmat job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"

if [ ! -x "$(ls $BUILD/*/app/bench_attn 2>/dev/null | tail -1 || true)" ]; then
  nix develop .#cpu-only --command bash -c "cd src && fortran-fpm build --features arch_d96_h6_kv2_l12_v8192_c1024 --build-dir $BUILD" 2>&1 | tail -1
fi
BA=$(ls $BUILD/*/app/bench_attn 2>/dev/null | tail -1)
BG=$(ls $BUILD/*/app/bench_gemm 2>/dev/null | tail -1)
if [ -z "$BA" ] || [ -z "$BG" ]; then
  echo "FALHOU: build nao produziu bench_attn/bench_gemm em $BUILD (disco? erro de compilacao?)"
  exit 1
fi
echo "  bench_attn=$BA"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo
echo "# --- ATENCAO: (H,D) x T  (q,k,v: [B,T,H*D]; scores [T,T] por cabeca)"
echo "# H*D=96 fixo (o d_model). head_dim maior = menos cabecas, mesmo d_model."
for HD in "6 16" "3 32" "2 48" "1 96"; do
  set -- $HD
  for T in 512 1024 2048; do
    printf "  H=%-2s D=%-3s T=%-5s " "$1" "$2" "$T"
    $BA --T "$T" --H "$1" --D "$2" --reps 3 2>&1 | awk '
      /^naive/ {n=$3} /^sgemm/ {s=$3; printf "naive %8.2f ms | sgemm %8.2f ms | sgemm/naive %5.2fx\n", n, s, s/n}'
  done
done

echo
echo "# --- GEMMs no mesmo build (referencia: 483 GFLOP/s a OMP=8)"
$BG --reps 5 2>&1 | grep -E "^sgemm" | sed 's/^/  /'
echo
echo "# leitura: se 'sgemm/naive' > 1 nas formas pequenas, o quadrado custa mais que"
echo "#   o causal e bloquear paga; se < 1, o sgemm ja' ganha e o esforco vai para head_dim."
echo "fim attnmat $(date '+%F %H:%M:%S')"
