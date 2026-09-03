"""Command-line interface for ai_chat_hoarder."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .core import (
    Store,
    HttpClient,
    fetch_all,
    reaudit,
    corpus_stats,
    import_local,
    import_archive,
    discover_ddg,
    crawl_github,
    ingest_file,
    set_token,
)
from .discover import github as github_mod


def _print_status(store, raw_dir):
    db = store.detailed_stats()
    corp = corpus_stats(raw_dir)
    plats = sorted(set(db) | set(corp))
    print(f"{'platform':10} {'tracked':>7} {'ok':>6} {'error':>6} {'pend':>6} "
          f"{'conv':>5} {'turns':>7} {'~tok':>10}")
    tt = {'tracked': 0, 'ok': 0, 'error': 0, 'pend': 0, 'conv': 0, 'turns': 0, 'chars': 0}
    for p in plats:
        d = db.get(p, {})
        c = corp.get(p, {})
        tok = c.get('chars', 0) // 4
        print(f"{p:10} {d.get('total', 0):>7} {d.get('ok', 0):>6} {d.get('error', 0):>6} "
              f"{d.get('pending', 0):>6} {c.get('conv', 0):>5} {c.get('turns', 0):>7} {tok:>10}")
        tt['conv'] += c.get('conv', 0)
        tt['turns'] += c.get('turns', 0)
        tt['chars'] += c.get('chars', 0)
        tt['tracked'] += d.get('total', 0)
        tt['ok'] += d.get('ok', 0)
        tt['error'] += d.get('error', 0)
        tt['pend'] += d.get('pending', 0)
    print(f"{'TOTAL':10} {tt['tracked']:>7} {tt['ok']:>6} {tt['error']:>6} {tt['pend']:>6} "
          f"{tt['conv']:>5} {tt['turns']:>7} {tt['chars'] // 4:>10}")
    print(f"\nRaw directory: {raw_dir}")


def _do_export(hoard):
    from . import export as export_mod
    s = export_mod.export_jsonl(hoard / "raw", hoard / "export")
    print(f"export: conversations={s['conversations']} deduped={s['deduped']} "
          f"turns={s['turns']} est_tokens={s['est_tokens']}")
    print(f"  plain: {s['plain_file']}")
    print(f"  agent: {s['agent_file']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ai-hoarder", description=__doc__)
    ap.add_argument("--hoard-dir", default="hoard", help="storage directory (default: ./hoard)")
    ap.add_argument("--status", action="store_true", help="print corpus + DB status and exit")
    ap.add_argument("--ddg", action="store_true", help="search DuckDuckGo for share links")
    ap.add_argument("--platforms", nargs="*", default=None, help="limit DDG to platforms")
    ap.add_argument("--file", action="append", help="ingest share links from a local file")
    ap.add_argument("--crawl", action="append", help="crawl a github/gist URL directly")
    ap.add_argument("--fetch", action="store_true", help="download pending snapshots")
    ap.add_argument("--reaudit", action="store_true", help="re-parse fetched files and fix DB flags")
    ap.add_argument("--local", action="append", metavar="PLATFORM:PATH",
                    help="import local sessions, e.g. --local opencode:/path/to.db "
                         "--local pi:~/.pi/subagent_out --local goose:~/.local/share/goose/sessions")
    ap.add_argument("--limit", type=int, help="max items to process in --fetch")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between fetches")
    ap.add_argument("--export", action="store_true",
                    help="export the hoard to train.jsonl / train.agent.jsonl (deduped)")
    ap.add_argument("--import-zip", metavar="ZIP",
                    help="import a b.sh archive (tar.zst) into the hoard, then export")
    args = ap.parse_args(argv)

    hoard = Path(args.hoard_dir)
    store = Store(hoard / "index.db")
    http = HttpClient()
    set_token(os.environ.get("GITHUB_TOKEN"))

    if args.crawl:
        for u in args.crawl:
            n = crawl_github(store, u, http)
            print(f"crawl {u[:70]}: +{n} new")
    if args.file:
        for f in args.file:
            n = ingest_file(store, f)
            print(f"file {f}: +{n} new")
    if args.ddg:
        n = discover_ddg(store, http, platforms=args.platforms)
        print(f"ddg: +{n} new")
    if args.fetch:
        counts = fetch_all(store, hoard / "raw", http, limit=args.limit, delay=args.delay)
        print(f"fetch ok={counts['ok']} err={counts['err']}")
    if args.reaudit:
        corrected = reaudit(store, hoard / "raw")
        print(f"reaudit ok={corrected['ok']} err={corrected['err']}")
    if args.local:
        for spec in args.local:
            plat, _, path = spec.partition(":")
            if not plat or not path:
                print(f"bad --local spec (need PLATFORM:PATH): {spec}")
                continue
            n = import_local(store, hoard / "raw", plat, [str(Path(path).expanduser())])
            print(f"local {plat} {path}: +{n} sessions")
    if args.import_zip:
        s = import_archive(args.import_zip, hoard)
        print(f"import-zip {args.import_zip}: files_copied={s['files_copied']} "
              f"db_added={s['db_added']}")
        # a freshly imported archive changes the corpus, so refresh the exports
        _do_export(hoard)
    elif args.export:
        _do_export(hoard)

    if args.status:
        _print_status(store, hoard / "raw")
    else:
        by_platform, total, fetched = store.stats()
        print(f"tracked={total} fetched={fetched}")
        print("by platform:", by_platform)


if __name__ == "__main__":
    main()
