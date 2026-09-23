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
# metade dos +0,10690 que o d216 ganhou do d96. A leitura "a escala satura" fica
# contaminada por um confundidor: o d360 usa head_dim 60 e o d216 usa head_dim 36.
# A forma da atencao e' alavanca medida. Entao parte do +0,05395 pode ser efeito de
# FORMA, e nao de ESCALA.
#
# O controle. d360 com 10 cabecas em vez de 6: 360/10 = head_dim 36, igual ao d216.
# Tudo o mais e' identico: mesma receita (k=4, tau=122, rounds=16, reset-moments,
# outer nesterov lr 0.7), mesmos dados, mesmo seed 7, mesmo template de init.
#
# O driver nao recebe a arch: ele le o arch.txt do --init. Por isso o controle e'
# um init novo, e nao uma flag.
#
# O driver NAO constroi: ele exige o binario pronto, e morre nomeando a arch que
# falta. Por isso este job constroi primeiro. Usa a forma de 6 parametros do
# job_env, e nao --build-dir, porque o arch.txt so' existe DEPOIS do primeiro
# build. Este modo constroi E grava o registro.
set -u
D=/tmp/evalfix_pauli_20260920
cd "$D" || exit 1
# DOIS CAMINHOS DIFERENTES, e essa e' a pegadinha deste job.
#   O dir do fpm (build_arch): <root>/<tag>   -> e' onde o binario realmente nasce
#   O --build-dir do driver:   <root>          -> o pick() faz glob(root/tag/*/app)
# O default do driver confirma: root = build_dir or REPO/src/build, e o layout
# historico e' src/build/arch_.../<hash>/app/. Sao dois nomes para a mesma coisa,
# um nivel de distancia. O job 6209 e o 6210 morreram por isso, com a mensagem
# "train_run nao buildado", que aponta para o lugar errado.
ROOT="$D/bd"
BD="$ROOT/arch_d360_h10_kv2_l12_v8192_c1024"
[ -f "$D/init_d360h10/arch.txt" ] || { echo "PREFLIGHT: init_d360h10 ausente"; exit 2; }
echo "head_dim do init: $(sed -n 's/^head_dim *= *//p' $D/init_d360h10/arch.txt)"
bin/job_env 360 10 2 12 8192 1024 "$BD" \
  "pick:scripts/fedavg_rounds.py" "attn-fn:src/app/train_run.f90" || exit 2
. "$BD/job_env.sh"
echo "binario: $TRAIN_RUN  fonte: $SRC_HASH"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE
echo "size360hd36 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S') ncores=$(nproc)"
echo "receita: k=4 tau=122 rounds=16 reset-moments outer=nesterov lr 0.7"
echo "contraste com o d360: SO' o n_head muda, 6 -> 10, ou seja head_dim 60 -> 36."
uv run scripts/fedavg_rounds.py \
    --regime federated --k 4 --tau 122 --rounds 16 \
    --init "$D/init_d360h10" \
    --rows "$D/rows_f0.npy" \
    --holdout "$D/rows_holdout.npy" \
    --reset-moments --outer nesterov --outer-lr 0.7 \
    --build-dir "$ROOT" \
    --out /tmp/size_d360/d360_hd36
echo "fim size360hd36 $(date '+%F %H:%M:%S')"
