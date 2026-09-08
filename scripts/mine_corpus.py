#!/usr/bin/env python3
"""
mine_corpus.py — turn ANY Python library's doctests into a think corpus.

    uv run --with <deps> scripts/mine_corpus.py --preset scipy
    uv run --with <deps> scripts/mine_corpus.py --preset sympy
    uv run --with <deps> scripts/mine_corpus.py --preset sklearn
    uv run --with <deps> scripts/mine_corpus.py --pkgs foo,bar \\
        --out mythink_reasoning.txt [--timeout 15 --legacy 1.25 ...]

Presets encode validated configs. Custom --pkgs takes importable package
names; add --skip-mod/--skip-src regexes to taste. Row shape + encoder +
pack contract identical to every other think corpus (pack with
pack_rows.py afterwards).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thinkx import mine as M  # noqa: E402
from thinkx.encode import write_rows  # noqa: E402

CACHE = os.path.expanduser("~/.cache/autoresearch")

PRESETS = {
    "scipy": dict(
        pkgs=["numpy", "scipy"],
        skip_mod=r"\._|test|conftest|_build|f2py|distutils|typing|version",
        skip_src=r"matplotlib|pyplot|plt\.|imshow|show\(\)|input\(|open\(|"
                 r"urlopen|sleep\(|thread|multiprocess|os\.|sys\.|"
                 r"subprocess|socket|savefig",
        timeout=12, legacy="1.25", out="thinkscipy_reasoning.txt"),
    "sympy": dict(
        pkgs=["sympy"],
        skip_mod=r"\._|test|conftest|_build|printer_preview|interactive|"
                 r"plotting|plot|preview|galgebra|holonomic|liealgebras|"
                 r"unify|multipledispatch|external|cacheit|printing\\.py",
        skip_src=r"matplotlib|pyplot|plt\.|imshow|show\(\)|plot\(|"
                 r"preview\(|input\(|open\(|urlopen|sleep\(|thread|"
                 r"multiprocess|os\.|sys\.|subprocess|socket|savefig|"
                 r"pprint.*use_unicode|StreamReader",
        timeout=15, legacy=None, out="thinksympy_reasoning.txt"),
    "sklearn": dict(
        pkgs=["sklearn"],
        skip_mod=r"\._|test|conftest|_build|externals|utils\.fixes|"
                 r"utils\.testing|datasets\.tests|compose\.tests|"
                 r"model_selection\.tests|feature_extraction",
        skip_src=r"matplotlib|pyplot|plt\.|imshow|show\(\)|plot\(|"
                 r"figure\(|savefig|fetch_|open\(|urlopen|sleep\(|"
                 r"thread|multiprocess|os\.|sys\.|subprocess|socket|"
                 r"input\(|n_jobs\s*=\s*-1",
        timeout=20, legacy="1.25", out="thinksklearn_reasoning.txt",
        extra_globs="np,numpy"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--pkgs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-mod", default=r"\._|test|conftest")
    ap.add_argument("--skip-src", default=r"matplotlib|pyplot|plt\.|show\(\)|"
                    r"input\(|open\(|sleep\(|thread|socket")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--legacy", default="")
    args = ap.parse_args()

    if args.preset:
        cfg = dict(PRESETS[args.preset])
    else:
        if not args.pkgs:
            ap.error("need --preset or --pkgs")
        cfg = dict(pkgs=args.pkgs.split(","), skip_mod=args.skip_mod,
                   skip_src=args.skip_src, timeout=args.timeout,
                   legacy=args.legacy or None,
                   out=args.out or "thinkx_reasoning.txt")
    if args.out:
        cfg["out"] = args.out
    if cfg.get("extra_globs"):
        try:
            import numpy as _np
            cfg["extra_globs"] = {k: _np for k in
                                  cfg["extra_globs"].split(",")}
        except ImportError:
            cfg["extra_globs"] = None

    rows, stats = M.mine(cfg)
    print("examined:", stats["examined"], "kept:", stats["kept"],
          "dupcode:", stats["dupcode"])
    kept = write_rows(rows, os.path.join(CACHE, cfg["out"]))
    print(f"kept {kept} rows -> {CACHE}/{cfg['out']} "
          f"(pack with pack_rows.py)")


if __name__ == "__main__":
    main()
