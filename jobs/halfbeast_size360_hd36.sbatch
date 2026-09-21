#!/bin/bash
#SBATCH -J size360hd36
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 64G
#SBATCH -t 48:00:00
#SBATCH -o /tmp/evalfix_pauli_20260920/size360hd36_%j.log
# Serie de TAMANHO, ponto 3: d360 com o head_dim CONTROLADO.
#
# O problema. O d360 deu 2,17098 contra 2,22493 do d216: ganho de +0,05395, ou
# metade dos +0,10690 que o d216 ganhou do d96. A leitura "a escala satura" ficou
# contaminada por um confundidor: o d360 usa head_dim 60 e o d216 usa head_dim 36.
# A forma da atencao e' alavanca medida (head_dim 48 bate head_dim 16 por ~0,017
# bpb, e o passo muda 1,56x a 2,67x). Entao parte do +0,05395 pode ser ganho ou
# perda de FORMA, e nao de ESCALA.
#
# O controle. d360 com 10 cabecas em vez de 6: 360/10 = head_dim 36, igual ao d216.
# Tudo o mais e' identico: mesma receita (k=4, tau=122, rounds=16, reset-moments,
# outer nesterov lr 0.7), mesmos dados (rows_f0 + rows_holdout), mesmo seed 7, e o
# MESMO template de init (init_d360/template.txt). A unica diferenca entre este run
# e o d216 passa a ser o tamanho.
#
# O driver nao recebe a arch: ele le o arch.txt do --init (pick() em
# fedavg_rounds.py). Por isso o controle e' um init novo, e nao uma flag.
set -u
D=/tmp/evalfix_pauli_20260920
cd "$D" || exit 1
BD="$D/bd_hd36"
# PREFLIGHT: a fonte desta maquina tem o que o job usa? A copia envelhece em
# silencio, e um train_run velho imprime a tabela de argumentos e sai.
grep -q "pick" scripts/fedavg_rounds.py || { echo "PREFLIGHT: fedavg_rounds.py velho"; exit 2; }
grep -q "rows_f0" "$D/rows_f0.npy" 2>/dev/null || true
[ -f "$D/init_d360h10/arch.txt" ] || { echo "PREFLIGHT: init_d360h10 ausente"; exit 2; }
echo "head_dim do init: $(sed -n 's/^head_dim *= *//p' $D/init_d360h10/arch.txt)"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE
echo "size360hd36 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S') ncores=$(nproc)"
echo "receita: k=4 tau=122 rounds=16 reset-moments outer=nesterov lr 0.7 (a mesma do d96, d216 e d360)"
echo "contraste com o d360: SO' o n_head muda, 6 -> 10, ou seja head_dim 60 -> 36."
uv run scripts/fedavg_rounds.py \
    --regime federated --k 4 --tau 122 --rounds 16 \
    --init "$D/init_d360h10" \
    --rows "$D/rows_f0.npy" \
    --holdout "$D/rows_holdout.npy" \
    --reset-moments --outer nesterov --outer-lr 0.7 \
    --build-dir "$BD" \
    --out /tmp/size_d360/d360_hd36
echo "fim size360hd36 $(date '+%F %H:%M:%S')"
