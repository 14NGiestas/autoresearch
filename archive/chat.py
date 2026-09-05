"""
Easy local inference — ask the trained model something.

Runs on CPU (so it never disturbs the GPU training job in the `small` tmux
session). Loads the latest plain-GPT checkpoint + the baseline tokenizer.

Usage:
  python3 chat.py "your question"          # one-shot
  python3 chat.py --prompt "..." --temp 0.8 --max_new 200
  python3 chat.py                           # interactive REPL (type 'quit' to exit)
  python3 chat.py --fortran                 # use the Fortran-math forward (novel.fortran_math)

Run from the repo root inside `nix develop`.
"""
import os
import sys
import glob
import argparse

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, os.path.join(REPO, "novel")):
    if p not in sys.path:
        sys.path.insert(0, p)

import prepare  # noqa: E402
import model as M  # noqa: E402
from novel.fortran_math import fortran_gpt_forward  # noqa: E402


def latest_plain_checkpoint():
    import re
    cands = [
        c for c in glob.glob(os.path.join(REPO, "checkpoints", "checkpoint_*.pt"))
        if "_gr_" not in os.path.basename(c) and "_ngram_" not in os.path.basename(c)
    ]
    if not cands:
        raise SystemExit("No plain-GPT checkpoints found in ./checkpoints")
    # sort by the integer after 'step' so step1056 beats step500_mid
    def _step(p):
        m = re.search(r"step(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cands, key=_step)


def load_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = M.GPT(M.GPTConfig(**sd["config"]))
    m.load_state_dict(sd["model_state_dict"])
    m = m.float().cpu().eval()
    return m, sd.get("val_bpb")


def generate(model, tokenizer, prompt, bos_id, temp=0.8, max_new=200, use_fortran=False):
    model.eval()
    text = f"### USER: {prompt}\n### ASSISTANT:"
    tokens = tokenizer.encode(text, prepend=bos_id)
    inp = torch.tensor([tokens], dtype=torch.long)
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            if use_fortran:
                logits = fortran_gpt_forward(model, inp[:, -2048:])[0, -1]
            else:
                logits = model(inp[:, -2048:])[0, -1]
            if temp <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == 0:
                break
            out.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]])], dim=1)
    return tokenizer.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--fortran", action="store_true", help="use the Fortran-math forward")
    args = ap.parse_args()

    ckpt = args.checkpoint or latest_plain_checkpoint()
    print(f"[chat] loading {os.path.basename(ckpt)} ...")
    model, val_bpb = load_model(ckpt)
    if val_bpb is not None:
        print(f"[chat] val_bpb={val_bpb:.4f}")
    tokenizer = prepare.Tokenizer.from_directory()
    bos_id = tokenizer.encode(prepare.BOS_TOKEN)[0]
    print(f"[chat] tokenizer vocab={tokenizer.get_vocab_size()}  backend={'fortran' if args.fortran else 'torch'} (CPU)\n")

    if args.prompt:
        print(generate(model, tokenizer, args.prompt, bos_id, args.temp, args.max_new, args.fortran))
        return

    print("Interactive — type a message (quit/exit to stop).\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        print("model>", generate(model, tokenizer, q, bos_id, args.temp, args.max_new, args.fortran))
        print()


if __name__ == "__main__":
    main()
