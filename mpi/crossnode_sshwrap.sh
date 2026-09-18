#!/bin/bash
# mpi/crossnode_sshwrap.sh -- launcher-exec do hydra que TUNELA o canal de CONTROLE.
#
# Por que existe: o hydra chama
#     <este script> -x <host> "<proxy>" --control-port <IP>:<PORTA> ... <proxy-id>
# e o hydra_pmi_proxy lancado no no remoto tem de CONECTAR DE VOLTA nesse
# <IP>:<PORTA>. A fermi esta' atras de NAT (192.168.180.177, rede 192.168.180.0/24)
# e o halfbeast NAO alcanca esse IP -- foi essa a causa do
# "Launch proxy failed / callback returned error status" (o "callback" e' essa
# conexao de volta). Aqui a gente:
#   1. reescreve --control-port <IP>:<PORTA> -> --control-port 127.0.0.1:<PORTA>
#   2. acrescenta -R <PORTA>:127.0.0.1:<PORTA> ao ssh, de modo que, no halfbeast,
#      o 127.0.0.1:<PORTA> seja o OUTRO PONTA do tunel (a fermi).
# Assim o proxy "conecta em si mesmo" e o ssh leva a conexao para o launcher.
# Nao encaminha ambiente nenhum: quem monta a linha de comando e' o hydra.
set -u
args=("$@")
port=""
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[$i]}" = "--control-port" ] && [ $((i + 1)) -lt ${#args[@]} ]; then
    port="${args[$((i + 1))]##*:}"
    args[$((i + 1))]="127.0.0.1:$port"
  fi
done
opts=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
[ -n "$port" ] && opts+=(-o ExitOnForwardFailure=no -R "$port:127.0.0.1:$port")
exec /usr/bin/ssh "${opts[@]}" "${args[@]}"
