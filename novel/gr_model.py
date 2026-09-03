"""
gr_model.py — Gated Residual (GR) variant of the autoresearch GPT (hyp_11df96).

Faithfully reuses train.py's norm / RoPE / attention / SwiGLU-MLP, but replaces
each Block with a 4-branch DATA-DEPENDENT gated residual stream:

    h   = norm(x)
    a   = x + attn(h, ...)        # attention residual stream
    m   = x + mlp(h)              # MLP residual stream
    d   = delta(h)                # linear / deltaNet-like branch
    n   = h                       # normalized identity stream (carries input)
    g   = sigmoid(gate_proj(h))   # 4 data-dependent scalar gates  (B,T,4)
    out = g0*a + g1*m + g2*d + g3*n

The gates let the network blend attention vs MLP vs a linear recurrence vs a
pure residual per-token, per-position — a cheap, portable stability/quality
win inspired by Qwen3.8-Flash-Next's Gated Residual. Added params are small
(one linear + one gate projection per layer).

This file is self-contained and CPU-validatable (see __main__). Launch on GPU
once the baseline frees the machine, e.g. with n_embd=896, n_layer=14.
"""
import os
import sys
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare

Tokenizer = prepare.Tokenizer


# ---------------------------------------------------------------------------
# Model (mirrors train.py architecture, swaps Block -> GatedResidualBlock)
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass
class GRConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q.transpose(1, 2)  # (B,H,T,D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class GatedResidualBlock(nn.Module):
    """4-branch data-dependent gated residual (the GR mechanism)."""

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.delta = nn.Linear(config.n_embd, config.n_embd, bias=False)  # linear recurrence branch
        self.gate = nn.Linear(config.n_embd, 4, bias=False)              # 4 scalar gates

    def forward(self, x, cos_sin, window_size):
        h = norm(x)
        a = x + self.attn(h, cos_sin, window_size)   # attention stream
        m = x + self.mlp(h)                           # MLP stream
        d = self.delta(h)                             # linear / deltaNet-style branch
        n = h                                         # normalized identity stream (preserves input)
        gates = torch.sigmoid(self.gate(h))           # (B,T,4) in (0,1)
        g0, g1, g2, g3 = gates.chunk(4, dim=-1)       # each (B,T,1)
        return g0 * a + g1 * m + g2 * d + g3 * n


class GRGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([GatedResidualBlock(config) for _ in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.uniform_(block.delta.weight, -s, s)
            torch.nn.init.zeros_(block.gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + transformer_matrices
        return {"wte": wte, "lm_head": lm_head, "transformer_matrices": transformer_matrices, "total": total}

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        for block, ws in zip(self.transformer.h, self.window_sizes):
            x = block(x, cos_sin, ws)
        x = norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                    ignore_index=-1, reduction=reduction)
            return loss
        return logits


# ---------------------------------------------------------------------------
# CPU self-test: validate the GR module trains stably on the real tokenizer
# ---------------------------------------------------------------------------


def _cpu_self_test(steps=80, n_layer=3, n_embd=64, n_head=4, seq_len=128, docs=20):
    print("[gr_model] CPU self-test: building tiny GR GPT on the real tokenizer")
    tok = Tokenizer.from_directory()
    bos = tok.get_bos_token_id()
    cfg = GRConfig(sequence_len=seq_len, vocab_size=tok.get_vocab_size(),
                   n_layer=n_layer, n_head=n_head, n_kv_head=n_head, n_embd=n_embd)
    model = GRGPT(cfg)
    model.init_weights()
    model = model.float()  # CPU validation in float32; GPU launch will cast to bf16
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[gr_model] params={nparams:,}")

    # tokenize a few real docs into one flat stream
    import pyarrow.parquet as pq
    files = [f for f in prepare.list_parquet_files()
        if os.path.basename(f) != prepare.VAL_FILENAME]
    tokens = []
    for f in files[:1]:
        table = pq.read_table(f, columns=["text"])
        for t in table.column("text").to_pylist()[:docs]:
            if t:
                tokens.extend(tok.encode(t, prepend=bos))
    tokens = tokens[: seq_len * 200]
    print(f"[gr_model] corpus tokens={len(tokens):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    best = 1e18
    import time
    t0 = time.time()
    for step in range(steps):
        i = int(torch.randint(0, max(1, len(tokens) - seq_len), (1,)).item())  # type: ignore
        try:
            xb = torch.tensor(tokens[i:i + seq_len]).long().unsqueeze(0)
            yb = torch.tensor(tokens[i + 1:i + 1 + seq_len]).long().unsqueeze(0)
        except Exception as e:
            assert False, f"[gr_model] batch build failed: {e}"
        loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best:
            best = loss.item()
        if step % 20 == 0 or step == steps - 1:
            print(f"[gr_model] step {step:3d} loss={loss.item():.4f}")
    print(f"[gr_model] OK in {time.time() - t0:.1f}s, best_loss={best:.4f}")
    assert best < 8.0, "GR training did not make progress"


if __name__ == "__main__":
    _cpu_self_test()
