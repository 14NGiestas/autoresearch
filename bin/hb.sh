#!/bin/bash
# bin/hb.sh — despacho para o halfbeast com as regras de boa vizinhança embutidas.
#   status | run '<cmd>' | batch <script> [args] | sync | logs [N]
set -u
HOST=halfbeast
PART=compute
CORES=${HB_CORES:-8}          # HT atrapalha: 10 = maquina toda, 8 = educado
LOAD_LIMIT=${HB_LOAD_LIMIT:-12}   # 20 threads; >12 = mais de ~60% ocupada
case "${1:-status}" in
  status)
    timeout 60 ssh -o BatchMode=yes $HOST \
      'printf "  load: %s\n" "$(cut -d" " -f1-3 /proc/loadavg)"; \
       printf "  fila: %s\n" "$(squeue -h -o "%i:%T:%u" | tr "\n" " ")"; \
       free -g | awk "/Mem:/{printf \"  mem: %s GB livres de %s GB\n\", \$7, \$2}"; \
       sinfo -o "  part: %P %a %D nos %c cpus" -h' 2>&1
    ;;
  run|batch)
    LOAD=$(timeout 60 ssh -o BatchMode=yes $HOST "cut -d' ' -f1 /proc/loadavg") || exit 1
    if [ "${2:-}" != "--force" ] && awk -v l="$LOAD" -v m="$LOAD_LIMIT" 'BEGIN{exit !(l>m)}'; then
      echo "  RECUSADO: load $LOAD > $LOAD_LIMIT (o halfbeast esta ocupado)."
      echo "  Use 'bin/hb.sh status' e tente depois, ou --force se souber o que faz."
      exit 1
    fi
    if [ "$1" = run ]; then
      shift; [ "${1:-}" = "--force" ] && shift
      echo "  srun -c $CORES em $HOST (load $LOAD)"
      timeout 600 ssh -o BatchMode=yes $HOST "cd \$HOME/autoresearch-eval && srun -p $PART -c $CORES bash -lc \"$*\""
    else
      shift; [ "${1:-}" = "--force" ] && shift
      echo "  sbatch $* em $HOST (load $LOAD)"
      timeout 120 ssh -o BatchMode=yes $HOST "cd \$HOME/autoresearch-eval/autoresearch && sbatch $*"
    fi
    ;;
  sync)
    exec bin/sync_eval.sh
    ;;
  logs)
    N=${2:-3}
    timeout 60 ssh -o BatchMode=yes $HOST "ls -t \$HOME/autoresearch-eval/*.log 2>/dev/null | head -$N | while read f; do echo \"== \$f\"; tail -3 \"\$f\"; done"
    ;;
  *) echo "uso: bin/hb.sh {status|run '<cmd>'|batch <script>|sync|logs [N]}"; exit 2;;
esac
