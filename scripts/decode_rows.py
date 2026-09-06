#!/usr/bin/env python3
"""decode_rows.py — traduz rows (ids) para texto legível."""
import argparse, pickle, pathlib, sys

_ALLOW = ("tiktoken.", "builtins", "copyreg")
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" or module.startswith("tiktoken.") or module == "copyreg":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

parser = argparse.ArgumentParser(description="Decode rows file to text")
parser.add_argument('--rows', default='/tmp/train10k_rows.txt', help='rows file')
parser.add_argument('--n', type=int, default=3, help='rows to decode')
parser.add_argument('--ids', type=int, default=80, help='ids to show per row')
args = parser.parse_args()

try:
    with open('/home/pauli/.cache/autoresearch/tokenizer/tokenizer.pkl','rb') as f:
        tok = RestrictedUnpickler(f).load()
except (OSError, pickle.PickleError) as e:
    sys.exit(f"cannot load tokenizer: {e}")
try:
    fh = open(args.rows)
except OSError as e:
    sys.exit(f"cannot open rows: {e}")
with fh as f:
    for i, line in enumerate(f):
        if i >= args.n:
            break
        ids = list(map(int, line.split()))
        txt = tok.decode(ids)
        preview = txt.replace('\n', ' | ')[:500]
        print(f"--- row {i} ({len(ids)} ids) ---")
        print(preview)
        print()
