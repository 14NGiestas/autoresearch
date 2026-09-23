#!/bin/bash
# mpi/crossnode_setup.sh -- tuneis + ambiente do cross-node fermi <-> halfbeast.
#
# Dois canais precisam atravessar a NAT da fermi:
#   1. CONTROLE (hydra <-> hydra_pmi_proxy): resolvido pelo crossnode_sshwrap.sh,
#      que acrescenta -R para a porta de controle que o proprio hydra escolhe.
#   2. DADOS (rank 0 <-> rank 1, ch3:tcp): as portas sao dinamicas, entao aqui a
#      gente FIXA uma faixa (CN_PORT_LO..CN_PORT_HI) e forca os dois lados a
#      anunciar 127.0.0.1 (MPIR_CVAR_CH3_INTERFACE_HOSTNAME) e a usar essa faixa
#      (MPIR_CVAR_CH3_PORT_RANGE). Com isso:
#        -L p:127.0.0.1:p  -> rank 0 (fermi) alcanca 127.0.0.1:p do halfbeast
#        -R p:127.0.0.1:p  -> rank 1 (halfbeast) alcanca 127.0.0.1:p da fermi
#      numa unica conexao ssh -N.
#
# Uso: source mpi/crossnode_setup.sh   (define as variaveis e sobe o tunel)
#      ... e no fim:  crossnode_teardown
set -u
REMOTE=${REMOTE:-halfbeast}
CN_PORT_LO=${CN_PORT_LO:-9340}
CN_PORT_HI=${CN_PORT_HI:-9359}
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30"

crossnode_teardown() {
  [ -n "${CN_TUNNEL_PID:-}" ] && kill "$CN_TUNNEL_PID" 2>/dev/null
  CN_TUNNEL_PID=""
}

crossnode_setup() {
  local p opts=()
  for p in $(seq "$CN_PORT_LO" "$CN_PORT_HI"); do
    opts+=(-L "$p:127.0.0.1:$p" -R "$p:127.0.0.1:$p")
  done
  # -N: so' tuneis. ExitOnForwardFailure: se uma porta estiver ocupada, avisa.
  $SSH -N -o ExitOnForwardFailure=yes "${opts[@]}" "$REMOTE" &
  CN_TUNNEL_PID=$!
  sleep 3
  if ! kill -0 "$CN_TUNNEL_PID" 2>/dev/null; then
    echo "  *** tunel ssh nao subiu (porta ocupada? ssh ok?) ***"; CN_TUNNEL_PID=""; return 1
  fi
  echo "  tunel ssh PID=$CN_TUNNEL_PID  faixa $CN_PORT_LO-$CN_PORT_HI (-L e -R)"
  # ambiente MPI: forcado na faixa e no loopback, repassado com -envlist explicito
  export MPIR_CVAR_CH3_INTERFACE_HOSTNAME=127.0.0.1
  export MPIR_CVAR_CH3_PORT_RANGE="${CN_PORT_LO}:${CN_PORT_HI}"
  export MPIR_CVAR_CH3_NETWORK_IFACE=lo
  export MPIR_CVAR_NETWORK_IFACE=lo
  export MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE=lo
  echo "  MPI env: hostname=127.0.0.1 range=$CN_PORT_LO:$CN_PORT_HI iface=lo"
}
trap crossnode_teardown EXIT
