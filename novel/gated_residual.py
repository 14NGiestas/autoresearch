"""
Gated Residual (GR) prototype — Qwen3.8-Flash-Next-style residual stream.

Qwen3.8-Flash-Next (Qwen4 preview, Aug 2026) widens the single residual stream
into 4 parallel branches and controls reads/writes with an elementwise,
data-dependent gate (Hyper-Connection + GatedNorm). This prototype validates that
a from-scratch GPT with GR trains stably on CPU using the real hoard tokenizer.

Run:  python3 novel/gated_residual.py
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare  # noqa: E402

torch.manual_seed(42)

C = 64      # embd
NL = 3      # layers
NH = 4      # heads
NB = 4      # GR branches
VOCAB = 8192
SEQ = 128
LR = 3e-3
STEPS = 80

tok = prepare.Tokenizer.from_directory()


class Block(nn.Module):
    """Minimal causal-attention + MLP block (the per-layer transform)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C, bias=False))

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, NH, c // NH).transpose(1, 2)
        k = k.view(b, t, NH, c // NH).transpose(1, 2)
        v = v.view(b, t, NH, c // NH).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class GRTransformer(nn.Module):
    """Tiny GPT with a 4-branch Gated Residual stream."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, C)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, VOCAB, bias=False)
        # GR gates: NB read gates from the stream, NB write gates from the block output
        self.read_gate = nn.Linear(C, NB, bias=False)
        self.write_gate = nn.Linear(C, NB, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        h = self.wte(idx)                                   # (B,T,C) main stream
        res = h.unsqueeze(0).repeat(NB, 1, 1, 1)            # (NB,B,T,C) residual state
        for blk in self.blocks:
            rg = torch.sigmoid(self.read_gate(h)).permute(2, 0, 1).unsqueeze(-1)  # (NB,B,T,1)
            h_in = (res * rg).sum(0)                        # gated read across branches
            out = blk(h_in)
            wg = torch.sigmoid(self.write_gate(out)).permute(2, 0, 1).unsqueeze(-1)  # (NB,B,T,1)
            res = res * (1 - wg) + out.unsqueeze(0) * wg    # gated write back
            h = out
        return self.head(self.ln_f(h))


text = ("the quick brown fox jumps over the lazy dog. " * 40 +
        "a small model trained from scratch learns token statistics slowly. " * 40)
ids = tok.encode(text, prepend=tok.get_bos_token_id())
if len(ids) < SEQ + 1:
    ids = ids + [0] * (SEQ + 1 - len(ids))
ids = ids[: (len(ids) // SEQ) * SEQ]
x = torch.tensor(ids).view(-1, SEQ)

model = GRTransformer()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
t0 = time.time()
last = 0.0
for s in range(STEPS):
    xi = x[s % x.shape[0]]
    inp, tgt = xi[:-1].unsqueeze(0), xi[1:].unsqueeze(0)
    logits = model(inp)
    loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    last = loss.item()
    if s % 20 == 0:
        print(f"step {s:3d} loss {last:.4f}")
print(f"GR prototype: {STEPS} steps in {time.time() - t0:.1f}s, final loss {last:.4f}")
