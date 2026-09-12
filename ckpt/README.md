# ckpt/ — model registry (genealogia auditável, como o HEP)

Checkpoints moram fora do git (grandes); aqui mora SÓ o registro:
`ckpt/registry.jsonl`, com hash-chain idêntico ao do `hep.py`
(`hash = sha256(prev + json(payload, sort_keys=True))`).

Cada `register` identifica um checkpoint pelo CONTEÚDO
(`ckpt_<12 hex do manifesto sha256>` — determinístico, verificável),
mais linhagem (`parents`), fatia de corpus, código (git sha), arquitetura
(lida de `arch.txt`) e métrica. Campos desconhecidos ficam `null` COM motivo
em `note`. Honestidade > completude: um pai "provável" sem fonte é pior que
pai ausente documentado.

```bash
ckpt.py register --dir DIR --label L --parents ID,.. --machine M \
    --rows-file F --start-row N --ntrain N --steps N --lr X \
    --bpb X --eval-spec S --note S [--code-sha SHA]
ckpt.py show [ID]        # pretty-print
ckpt.py lineage ID       # cadeia de pais até a raiz
ckpt.py verify ID        # re-hash dos arquivos, compara com o registro
```

Regras:
- Pesos vivem em `/tmp` ou na outra máquina; o registro referencia, não contém.
- `arch.txt` ausente é registrado como ausência (não inventado). Exceção: o
  backfill de 2026-09-11 escreveu `arch.txt` em 105 dirs após conferir forma
  (wte 8192×768 + 12 camadas) — documentado no commit do backfill.
- `template.txt` e momentos Adam: contados em `files.n` (90 = pesos+adam,
  74 = só pesos); zeros explícitos vs ausência ficam na nota.
- Restauração: `git checkout <sha> -- <arquivo deletado>` traz qualquer script
  removido de volta; `ckpt.py verify` confere qualquer checkpoint.
