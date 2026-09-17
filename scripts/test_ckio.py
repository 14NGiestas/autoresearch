#!/usr/bin/env python3
"""test_ckio.py — prova que as FERRAMENTAS leem checkpoint safetensors (st-only).

Complementa a suite Fortran (que cobre a escrita/leitura safetensors do lado do
trainer): aqui os consumidores Python sao executados de verdade sobre um
checkpoint **st-only** (so `model.safetensors` + estado `.npy` + arch/template),
e o resultado e comparado com o MESMO checkpoint no layout legado `.npy`.

Ferramentas exercitadas (subprocesso, CLI real):
  * rescale_ckpt.py    -> escala e escreve no MESMO formato (st) e nao toca no estado
  * merge_checkpoints.py -> media, OMITE momentos, escreve st
  * weight_surgery.py  -> 6 variantes, cada uma no MESMO formato (st)
  * compose_metrics.py -> le merge + shards st
E em processo:
  * fedavg_rounds.avg_dirs  -> media pesos + adam_* (semantica do fedavg)
  * compose_grid.rel_drift  -> ||A-B||/||A|| entre dois dirs st
  * ckio.load_ckpt_dir/load_state/require_weights (o guarda virou aviso)

Uso:
    .venv-numpy/bin/python3 scripts/test_ckio.py            # cria o ckpt sintetico
    .venv-numpy/bin/python3 scripts/test_ckio.py --ckpt-st DIR --ckpt-npy DIR
        # usa um checkpoint real (o teste Fortran passa o dele; ver test_st_ckpt)

Exit != 0 em qualquer falha.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ckio  # noqa: E402
import st_read  # noqa: E402

FAILS = []


def check(cond, label):
    print(("  ok    " if cond else "  FAIL: ") + label)
    if not cond:
        FAILS.append(label)


def run(cmd, expect=0):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if p.returncode != expect:
        print("    cmd:", " ".join(cmd))
        print("    rc:", p.returncode, "->", (p.stdout + p.stderr)[-600:])
    return p


def make_synth(work, tag, scale, with_state=True):
    """Checkpoint sintetico: pesos (2 camadas) + estado adam/muon + arch/template."""
    d = os.path.join(work, tag)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(hash(tag) % 2**31)
    W = {}
    for nm, n in (("transformer_wte_weight.npy", 32), ("lm_head_weight.npy", 32)):
        W[nm] = (rng.standard_normal(n).astype(np.float32) * scale)
    for L in (0, 1):
        for slot, n in (("attn_c_q", 8), ("attn_c_k", 4), ("attn_c_v", 4),
                        ("attn_c_proj", 8), ("mlp_c_fc", 16), ("mlp_c_proj", 16)):
            W["transformer_h_%d_%s_weight.npy" % (L, slot)] = \
                (rng.standard_normal(n).astype(np.float32) * scale)
    for f, a in W.items():
        np.save(os.path.join(d, f), a)
    if with_state:
        for nm, n in (("adam_m_wte", 32), ("adam_v_wte", 32), ("adam_m_q", 8),
                      ("muon_moment_q", 8)):
            np.save(os.path.join(d, nm + ".npy"),
                    (rng.random(n).astype(np.float32) * scale))
    open(os.path.join(d, "arch.txt"), "w").write(
        "d_model = 8\nn_head = 2\nn_kv = 1\nn_layer = 2\nvocab = 32\nctx = 8\n")
    open(os.path.join(d, "template.txt"), "w").write("dialect=test\n")
    return d, W


def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-st", default="")
    ap.add_argument("--ckpt-npy", default="")
    ap.add_argument("--work", default=os.path.join(REPO, "build", "ckio_test"))
    a = ap.parse_args()
    work = a.work
    os.makedirs(work, exist_ok=True)

    print("== ckio: checkpoint sintetico nos dois formatos ==")
    src_a, Wa = make_synth(work, "src_a", 1.0)
    src_b, Wb = make_synth(work, "src_b", 1.2)
    st_a = os.path.join(work, "st_a")
    st_b = os.path.join(work, "st_b")
    # st_a/st_b sao a conversao explicita dos mesmos arrays -> st-only
    # fmt_out="st": este e o caso CONVERSAO (legado -> st) do ponto de vista do
    # ckio; o default de save_ckpt_dir e "mesmo formato de `like`".
    ckio.save_ckpt_dir(st_a, ckio.load_ckpt_dir(src_a), state=ckio.load_state(src_a),
                       like=src_a, op="test_ckio", fmt_out="st")
    ckio.save_ckpt_dir(st_b, ckio.load_ckpt_dir(src_b), state=ckio.load_state(src_b),
                       like=src_b, op="test_ckio", fmt_out="st")
    check(ckio.fmt(st_a) == "st" and ckio.fmt(src_a) == "npy",
          "fmt() distingue st de npy")
    check(all(same(v, ckio.load_ckpt_dir(src_a)[k])
              for k, v in ckio.load_ckpt_dir(st_a).items()),
          "load_ckpt_dir: mesmos arrays nos dois formatos (14 tensores)")
    check(len(ckio.load_state(st_a)) == 4 and len(ckio.load_ckpt_dir(st_a)) == 14,
          "load_state != load_ckpt_dir (estado nao entra nos pesos)")

    # guarda da fase 1 agora AVISA e devolve nomes (nao aborta)
    names = ckio.require_weights(st_a, "test_ckio")
    check(len(names) == 14, "require_weights em st-only avisa e devolve 14 nomes")

    # usa checkpoint REAL se o teste Fortran passou um
    if a.ckpt_st:
        st_real, npy_real = a.ckpt_st, a.ckpt_npy
        check(os.path.exists(os.path.join(st_real, "model.safetensors")),
              f"checkpoint real st: {st_real}")
        # job 91 faz isso a mao: payload st == payload .npy, byte a byte
        if npy_real and os.path.isdir(npy_real):
            Wst = ckio.load_ckpt_dir(st_real)
            Wnp = ckio.load_ckpt_dir(npy_real)
            diff = [k for k in Wnp if not same(Wnp[k], Wst.get(k))]
            check(not diff and len(Wst) == len(Wnp),
                  f"real: {len(Wst)} pesos st == {len(Wnp)} .npy (byte a byte)")
        # st_b = copia identica de st_a: a media tem de devolver o mesmo array, o
        # que fecha a checagem de "payload st == payload npy" tambem no fedavg.
        st_a = st_real
        st_b = os.path.join(work, "st_b2")
        ckio.save_ckpt_dir(st_b, ckio.load_ckpt_dir(st_real),
                           state=ckio.load_state(st_real), like=st_real, op="test_ckio")
        wa, wb = ckio.load_ckpt_dir(st_a), ckio.load_ckpt_dir(st_b)

    print("== ferramentas sobre checkpoint st-only ==")
    # 1. rescale: MESMO formato de entrada (st) e estado intocado
    out_r = os.path.join(work, "rescale")
    p = run([sys.executable, "scripts/rescale_ckpt.py", st_a, out_r, "--scale", "2.0"])
    check(p.returncode == 0 and ckio.fmt(out_r) == "st",
          "rescale_ckpt: le st, escreve st (model.safetensors)")
    check(ckio.fmt(out_r) == "st" and not ckio.weight_files(out_r),
          "rescale_ckpt: nao deixou transformer_*.npy ao lado do .safetensors")
    wr, wa = ckio.load_ckpt_dir(out_r), ckio.load_ckpt_dir(st_a)
    check(all(np.allclose(wr[k], wa[k].astype(np.float32) * 2.0) for k in wa),
          "rescale_ckpt: pesos = entrada x 2")
    check(ckio.load_state(out_r).keys() == ckio.load_state(st_a).keys(),
          "rescale_ckpt: estado do otimizador copiado igual (nao escalado)")

    # 1b. --only com nome canonico (antes 'wte' nao casava com nada)
    out_o = os.path.join(work, "rescale_only")
    p = run([sys.executable, "scripts/rescale_ckpt.py", st_a, out_o,
             "--scale", "3.0", "--only", "wte,lm"])
    wo = ckio.load_ckpt_dir(out_o)
    check(p.returncode == 0 and
          np.allclose(wo["transformer_wte_weight.npy"], wa["transformer_wte_weight.npy"] * 3)
          and np.allclose(wo["transformer_h_0_attn_c_q_weight.npy"],
                          wa["transformer_h_0_attn_c_q_weight.npy"]),
          "rescale_ckpt --only wte,lm: escala so os dois (nome canonico)")

    # 2. merge: media, OMITE momentos, escreve no formato de A
    out_m = os.path.join(work, "merge")
    p = run([sys.executable, "scripts/merge_checkpoints.py", st_a, st_b, out_m])
    check(p.returncode == 0 and ckio.fmt(out_m) == "st",
          "merge_checkpoints: le dois st e escreve st")
    wm, wb = ckio.load_ckpt_dir(out_m), ckio.load_ckpt_dir(st_b)
    check(all(np.allclose(wm[k], (wa[k].astype("float64") + wb[k].astype("float64")) / 2,
                          rtol=1e-5, atol=1e-8) for k in wa),
          "merge_checkpoints: media elemento a elemento")
    check(len(ckio.load_state(out_m)) == 0 and
          not any(f.startswith("adam_") for f in os.listdir(out_m)),
          "merge_checkpoints: momentos OMITIDOS (semantica preservada)")

    # 3. cirurgia: variantes, cada uma no formato da entrada. No checkpoint REAL
    # (--ckpt-st, 74 tensores/38 MB) ela e pulada: 7 variantes x 38 MB so
    # alongariam a suite, e o que ela prova (st-only + formato de saida) ja e
    # provado nas duas variantes sinteticas abaixo -- que sao arquivos st de
    # verdade, escritos pelo mesmo ckio.
    out_s = os.path.join(work, "surgery")
    if a.ckpt_st:
        print("  skip  weight_surgery no ckpt real (coberto no sintetico)")
    else:
        p = run([sys.executable, "scripts/weight_surgery.py", "--ckpt", st_a,
                 "--out", out_s])
        # square, negate, scale05, scale20, shuffle, noise001, noise01 = 7
        n_var = len([d for d in os.listdir(out_s)])
        check(p.returncode == 0 and n_var == 7, f"weight_surgery: 7 variantes ({n_var})")
        if n_var:
            ng = os.path.join(out_s, "negate")
            check(ckio.fmt(ng) == "st" and
                  all(same(ckio.load_ckpt_dir(ng)[k], -wa[k]) for k in wa),
                  "weight_surgery: variante 'negate' em st e igual a -entrada")

    # 4. compose_metrics: le merge + shards st
    out_j = os.path.join(work, "metrics.jsonl")
    if os.path.exists(out_j):
        os.remove(out_j)
    if a.ckpt_st:
        print("  skip  compose_metrics no ckpt real (coberto no sintetico)")
    else:
        p = run([sys.executable, "scripts/compose_metrics.py", "--shards", st_a, st_b,
                 "--merge", out_m, "--label", "st-test", "--json-out", out_j])
        check(p.returncode == 0 and "s = ||media||/||shard||" in p.stdout,
              "compose_metrics: le shards e merge em st")
        check(os.path.exists(out_j), "compose_metrics: escreveu o jsonl")

    # 5. fedavg: media PESOS + MOMENTOS (semantica propria). As contagens saem do
    # proprio checkpoint de entrada: o mesmo teste roda no sintetico (14+4) e num
    # checkpoint real (74+16).
    import fedavg_rounds as fr
    sa, sb = ckio.load_state(st_a), ckio.load_state(st_b)
    out_f = os.path.join(work, "fedavg")
    n = fr.avg_dirs([st_a, st_b], out_f)
    wf = ckio.load_ckpt_dir(out_f)
    sf = ckio.load_state(out_f)
    check(n == len(wa) + len(sa) and len(wf) == len(wa) and len(sf) == len(sa),
          f"fedavg/avg_dirs: {len(wa)} pesos + {len(sa)} momentos ({n})")
    check(all(np.allclose(wf[k],
                          (wa[k].astype("float64") + wb[k].astype("float64")) / 2,
                          rtol=1e-5, atol=1e-8) for k in wa),
          "fedavg: pesos mediados")
    k0 = next((k for k in sa if k.startswith("adam_m_")), None)
    check(k0 is not None and np.allclose(
              sf[k0], (sa[k0].astype("float64") + sb[k0].astype("float64")) / 2,
              rtol=1e-5, atol=1e-8),
          f"fedavg: {k0} MEDIADO (inclui momentos)")

    # 6. compose_grid.rel_drift
    import compose_grid as cg
    d = cg.rel_drift(st_a, st_b)
    check(d >= 0.0 and np.isfinite(d), f"compose_grid.rel_drift entre dois st ({d:.3f})")

    # 7. basin_map: histogramas + vetor achatado (PCA precisa de >=2 do mesmo dset)
    p = run([sys.executable, "scripts/basin_map.py",
             "--pts", "a=%s,b=%s" % (st_a, st_b),
             "--svg", os.path.join(work, "basin.svg")])
    check(p.returncode == 0, "basin_map: le dois checkpoints st")

    print()
    if FAILS:
        print(f"test_ckio: FAIL ({len(FAILS)})")
        for f in FAILS:
            print("  -", f)
        return 1
    print("test_ckio: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
