"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
import sys
import ast
import contextlib

# ROCm/HIP does not support expandable_segments (warns "not supported on this platform");
# the final-eval OOM is instead prevented by eval_batch_size=2 (see final eval below).
DEMO = "--demo" in sys.argv  # CPU generation-only mode (build model, generate, exit; no training)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Detect AMD ROCm vs NVIDIA CUDA
IS_ROCM = getattr(torch.version, "hip", None) is not None  # type: ignore

if not IS_ROCM:
    from kernels import get_kernel  # type: ignore

    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = (
        "varunneal/flash-attention-3"
        if cap == (9, 0)
        else "kernels-community/flash-attn3"
    )
    fa3 = get_kernel(repo).flash_attn_interface
else:
    fa3 = None  # Will use PyTorch SDPA on ROCm (dispatches to AOTriton)

import prepare
prepare.TIME_BUDGET = ast.literal_eval(os.environ.get("AUTORESEARCH_TIME_BUDGET", "43200"))  # default 12h; override for fine-tunes
MAX_SEQ_LEN = ast.literal_eval(os.environ.get("AUTORESEARCH_SEQ_LEN", str(prepare.MAX_SEQ_LEN)))
TIME_BUDGET = prepare.TIME_BUDGET
Tokenizer = prepare.Tokenizer
make_dataloader = prepare.make_dataloader
evaluate_bpb = prepare.evaluate_bpb
from novel.gr_model import GRGPT, GRConfig  # Gated Residual variant (AUTORESEARCH_MODEL=gr)
from novel.bin_dataloader import make_bin_dataloader  # pretokenized .bin dataloader (AUTORESEARCH_DATA_BIN)
from novel.ngram_model import NgramGPT, NgramConfig  # type: ignore  # N-gram embedding variant (AUTORESEARCH_MODEL=ngram)


def _run_cpu_demo(model, tokenizer, max_seq):
    """CPU-only generation for a quick output demo (no GPU, no training)."""
    import glob as _gl
    model.float()
    model.eval()
    bos_id = tokenizer.encode(prepare.BOS_TOKEN)[0]
    _args = sys.argv
    if "--demo" in _args:
        _i = _args.index("--demo")
        prompt = _args[_i + 1] if _i + 1 < len(_args) else "USER: hello, who are you?\n"
    else:
        prompt = "USER: hello, who are you?\n"
    cks = sorted(_gl.glob(os.path.join(CHECKPOINT_DIR, "ckpt_depth*_step*.pt")) + _gl.glob(os.path.join(CHECKPOINT_DIR, f"checkpoint_{'ngram_' if MODEL == 'ngram' else ('gr_' if MODEL == 'gr' else '')}depth{DEPTH}_step*.pt")))
    tag = "random-init (no checkpoint saved yet)"
    if cks:
        sd = torch.load(cks[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"])
        tag = f"loaded {cks[-1]}"
    ids = tokenizer.encode(prompt, prepend=bos_id)
    inp = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(140):
            logits = model(inp[:, -max_seq:])
            logits = logits[:, -1, :] / 0.9
            idx = torch.multinomial(torch.softmax(logits, -1), 1)
            inp = torch.cat([inp, idx], dim=1)
            if idx.item() == 0:
                break
    print("=== CPU DEMO ===")
    print(f"(weights: {tag})")
    print(tokenizer.decode(inp[0].tolist()))

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
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

        if IS_ROCM:
            # PyTorch SDPA on ROCm dispatches to AOTriton
            # Note: SDPA doesn't support window_size, so SSSL pattern degrades to
            # full causal attention on all layers
            q = q.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)  # type: ignore
            y = y.contiguous().view(B, T, -1)
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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size):
        x = x + self.attn(norm(x), cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)

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

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
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

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + transformer_matrices
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
        )
        # Freeze backbone for stable fine-tuning on a converged (sharp) minimum:
        # only the LM head (+ optionally the token embedding) adapt; transformer.h stays fixed.
        # Full fine-tuning pushes the model off its sharp minimum within a few steps even at a
        # small LR (AdamW's normalization makes each update ~= lr in magnitude), destroying it.
        if FREEZE_BACKBONE >= 1:
            for _p in self.transformer.h.parameters():
                _p.requires_grad_(False)
        if FREEZE_BACKBONE >= 2:
            for _p in self.transformer.wte.parameters():
                _p.requires_grad_(False)
            embedding_params = []
        if FREEZE_BACKBONE:
            print(
                f"FREEZE_BACKBONE={FREEZE_BACKBONE}: transformer.h frozen"
                + (" + wte" if FREEZE_BACKBONE >= 2 else "")
                + "; training "
                + ("lm_head only" if FREEZE_BACKBONE >= 2 else "lm_head + wte")
            )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale * LR_SCALE,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        if embedding_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=embedding_params,
                    lr=embedding_lr * dmodel_lr_scale * LR_SCALE,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )
        if FREEZE_BACKBONE < 1:
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(
                    dict(
                        kind="muon",
                        params=group_params,
                        lr=matrix_lr * LR_SCALE,
                        momentum=0.95,
                        ns_steps=5,
                        beta2=0.95,
                        weight_decay=weight_decay,
                    )
                )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        for block, ws in zip(self.transformer.h, self.window_sizes):
            if GRAD_CHECKPOINT:
                # recompute this block's forward during backward instead of storing it
                x = torch.utils.checkpoint.checkpoint(block, x, cos_sin, ws, use_reentrant=False)  # type: ignore
            else:
                x = block(x, cos_sin, ws)
        x = norm(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

_maybe_compile = (
    torch.compile(dynamic=False, fullgraph=True) if not IS_ROCM else lambda fn: fn
)


@_maybe_compile
def adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t
):
    p.mul_(1 - lr_t * wd_t)
    dtype = exp_avg.dtype
    exp_avg.lerp_(grad, (1 - beta1_t).to(dtype=dtype))
    exp_avg_sq.lerp_(grad.square(), (1 - beta2_t).to(dtype=dtype))
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@_maybe_compile
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype),
        (1 - beta2).to(dtype=second_momentum_buffer.dtype),
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device=device)

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))  # type: ignore

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64  # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128  # target head dimension for attention
WINDOW_PATTERN = "SSSL"  # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**17  # ~131K tokens per optimizer step
EMBEDDING_LR = 0.6  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = ast.literal_eval(os.environ.get("AUTORESEARCH_MATRIX_LR", "0.04"))  # learning rate for matrix parameters (Muon)
LR_SCALE = ast.literal_eval(os.environ.get("AUTORESEARCH_LR_SCALE", "1.0"))  # global LR multiplier; fine-tunes set <1 to avoid catastrophic forgetting
FREEZE_BACKBONE = ast.literal_eval(os.environ.get("AUTORESEARCH_FREEZE_BACKBONE", "0"))  # 0=all trainable, 1=freeze transformer.h (train head+emb), 2=freeze transformer.h+wte (train head only)
WEIGHT_DECAY = ast.literal_eval(os.environ.get("AUTORESEARCH_WEIGHT_DECAY", "0.2"))  # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
WARMUP_RATIO = 0.0  # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5  # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0  # final LR as fraction of initial

# Model size
DEPTH = ast.literal_eval(os.environ.get("AUTORESEARCH_DEPTH", "5"))  # transformer layers -> model size
DEVICE_BATCH_SIZE = ast.literal_eval(os.environ.get("AUTORESEARCH_DEVICE_BATCH", "8"))  # per-device batch size
# Recompute activations per transformer block (torch.utils.checkpoint) to trade compute
# for VRAM. Lets us raise the batch (more tok/s -> more epochs) or grow the model on a
# 16 GiB carve-out. Off by default so it never affects an already-launched run.
GRAD_CHECKPOINT = os.environ.get("AUTORESEARCH_GRAD_CHECKPOINT", "0") == "1"
# Model variant selector: "baseline" (default GPT) or "gr" (Gated Residual).
# run_claude_style.sh never sets this, so the auto-launched Claude fine-tune stays
# on the baseline path; this only changes behaviour when explicitly passed.
MODEL = os.environ.get("AUTORESEARCH_MODEL", "baseline")

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu") if DEMO else torch.device("cuda")
autocast_ctx = (contextlib.nullcontext() if DEMO
                else torch.autocast(device_type="cuda", dtype=torch.bfloat16))
# Peak BF16 FLOPS by GPU model (for MFU calculation)
_GPU_PEAK_FLOPS = {
    "H100": 989.5e12,
    "H200": 989.5e12,
    "A100": 312.0e12,
    "B200": 2250.0e12,
    # AMD Instinct
    "MI300X": 1307.4e12,
    "MI308X": 1307.4e12,
    "MI325X": 1307.4e12,
    "MI250X": 383.0e12,
}


def _detect_peak_flops():
    gpu_name = torch.cuda.get_device_name(0)
    for key, flops in _GPU_PEAK_FLOPS.items():
        if key.lower() in gpu_name.lower():
            print(f"Detected GPU: {gpu_name} -> peak BF16 FLOPS: {flops:.1e}")
            return flops
    print(f"Warning: Unknown GPU '{gpu_name}', defaulting to H100 peak FLOPS for MFU")
    return 989.5e12


PEAK_BF16_FLOPS = _detect_peak_flops()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
bos_id = tokenizer.encode(prepare.BOS_TOKEN)[0]  # needed for sampling


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


def build_gr_config(depth):
    """Gated Residual variant config (same size mapping as the baseline)."""
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GRConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


def build_gr_optimizer(model, unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
                       adam_betas=ADAM_BETAS, matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY):
    """Identical Muon/AdamW param-group recipe as GPT.setup_optimizer, for GRGPT."""
    model_dim = model.config.n_embd
    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    param_groups = [
        dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale * LR_SCALE,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale * LR_SCALE,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(kind="muon", params=group_params, lr=matrix_lr * LR_SCALE,
                                 momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def build_ngram_config(depth):
    """N-gram embedding variant config (same size mapping as the baseline)."""
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return NgramConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


def build_ngram_optimizer(model, unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
                          adam_betas=ADAM_BETAS, matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY):
    """Identical Muon/AdamW param-group recipe as GPT.setup_optimizer, for NgramGPT.

    The n-gram tables (ngram2/ngram3) and the scalar ngram_scale are folded into the
    AdamW embedding group so they train alongside the token embedding.
    """
    model_dim = model.config.n_embd
    matrix_params = list(model.transformer.h.parameters())
    embedding_params = (
        list(model.transformer.wte.parameters())
        + list(model.transformer.ngram2.parameters())
        + list(model.transformer.ngram3.parameters())
        + [model.ngram_scale]
    )
    lm_head_params = list(model.lm_head.parameters())
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    param_groups = [
        dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale * LR_SCALE,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale * LR_SCALE,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(kind="muon", params=group_params, lr=matrix_lr * LR_SCALE,
                                 momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


if MODEL == "gr":
    config = build_gr_config(DEPTH)
    print(f"Model config (GATED RESIDUAL): {asdict(config)}")
    with torch.device("meta"):
        gr_model = GRGPT(config)
    gr_model = gr_model.to_empty(device=device)
    gr_model.init_weights()
    model = gr_model  # type: ignore[assignment]
elif MODEL == "ngram":
    config = build_ngram_config(DEPTH)
    print(f"Model config (N-GRAM EMBEDDING): {asdict(config)}")
    with torch.device("meta"):
        ngram_model = NgramGPT(config)
    ngram_model = ngram_model.to_empty(device=device)
    ngram_model.init_weights()
    model = ngram_model  # type: ignore[assignment]
else:
    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")
    with torch.device("meta"):
        model: GPT = GPT(config)
    model = model.to_empty(device=device)
    model.init_weights()

if DEMO:
    _run_cpu_demo(model, tokenizer, MAX_SEQ_LEN)
    sys.exit(0)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

if MODEL == "gr":
    optimizer = build_gr_optimizer(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
elif MODEL == "ngram":
    optimizer = build_ngram_optimizer(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
else:
    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

if not IS_ROCM:
    model = torch.compile(model, dynamic=False)  # type: ignore
else:
    print("ROCm detected: torch.compile disabled (enable with PyTorch 2.9+ on ROCm)")

# ---------------------------------------------------------------------------
# Resume support (domain fine-tune continuation from a prior checkpoint)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get("AUTORESEARCH_CHECKPOINT_DIR", "checkpoints")
import glob as _glob
_resume_ckpts = sorted(_glob.glob(os.path.join(CHECKPOINT_DIR, f"checkpoint_{'ngram_' if MODEL == 'ngram' else ('gr_' if MODEL == 'gr' else '')}depth{DEPTH}_step*.pt")))
resume_start_step = 0
resumed_training_time = 0.0
if os.environ.get("AUTORESEARCH_RESUME", "1") != "0" and _resume_ckpts:
    _ck = _resume_ckpts[-1]
    print(f"RESUME: loading {_ck}")
    _sd = torch.load(_ck, map_location=device, weights_only=False)
    model.load_state_dict(_sd["model_state_dict"])
    try:
        if LR_SCALE == 1.0 and FREEZE_BACKBONE == 0:
            # Baseline resume: continue seamlessly with the saved optimizer state (LRs + momentum).
            optimizer.load_state_dict(_sd["optimizer_state_dict"])
        else:
            # Fine-tune / backbone-freeze: keep the FRESH optimizer built above. Loading the baseline's
            # 12h Muon/Adam momentum would dominate a small fine-tune LR and (with a reduced param set)
            # also mismatch the saved state dict. Fresh momentum + correct (scaled or head-only) LR.
            print(f"RESUME: fresh optimizer kept (FREEZE_BACKBONE={FREEZE_BACKBONE}, LR_SCALE={LR_SCALE}); baseline momentum not loaded")
    except KeyError:
        print("RESUME: checkpoint has no optimizer_state_dict; optimizer reinitialized")
    resume_start_step = _sd.get("step", 0)
    resumed_training_time = _sd.get("total_training_time", 0.0)
    print(f"RESUME: start_step={resume_start_step} total_training_time={resumed_training_time:.0f}s")

DATA_BIN = os.environ.get("AUTORESEARCH_DATA_BIN", "")
if DATA_BIN:
    _bin_path = os.path.join(DATA_BIN, "train.bin")
    print(f"Pretokenized bin dataloader: {_bin_path}")
    train_loader = make_bin_dataloader(_bin_path, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, device)
else:
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

SAMPLE_EVERY = ast.literal_eval(os.environ.get("AUTORESEARCH_SAMPLE_EVERY", "0") or "0")


@torch.no_grad()
def sample_text(prompt, max_new=120, temp=0.9):
    """Mid-run qualitative probe: temperature sampling from a fixed prompt."""
    model.eval()
    ids = tokenizer.encode(prompt, prepend=bos_id)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with autocast_ctx:
        for _ in range(max_new):
            logits = model(inp[:, -MAX_SEQ_LEN:])
            logits = logits[:, -1, :] / temp
            idx = torch.multinomial(torch.softmax(logits, -1), 1)
            inp = torch.cat([inp, idx], dim=1)
            if idx.item() == 0:
                break
    model.train()
    return tokenizer.decode(inp[0].tolist())

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = (1 + math.cos(math.pi * (1 - cooldown))) / 2
        return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * cosine_decay


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(_progress):
    return WEIGHT_DECAY


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = resumed_training_time
step = resume_start_step
_ckpt_last_t = time.time()  # for AUTORESEARCH_CKPT_SECS periodic saves

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    run_elapsed = time.time() - t_start_training
    train_loss = torch.tensor(0.0)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(run_elapsed / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = round(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_BF16_FLOPS
    remaining = max(0, TIME_BUDGET - run_elapsed)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Periodic checkpoint so `chat.py` can use an intermediate model without
    # waiting for the full run. Off by default; enable via env:
    #   AUTORESEARCH_CKPT_EVERY=500  (every N steps)
    #   AUTORESEARCH_CKPT_SECS=1800  (every N seconds)
    _ckpt_every = int(os.environ.get("AUTORESEARCH_CKPT_EVERY", "0"))
    _ckpt_secs = int(os.environ.get("AUTORESEARCH_CKPT_SECS", "0"))
    if (_ckpt_every and step % _ckpt_every == 0) or (_ckpt_secs and (time.time() - _ckpt_last_t) >= _ckpt_secs):
        _cp = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_{'gr_' if MODEL == 'gr' else ''}depth{DEPTH}_step{step}_mid.pt",
        )
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "step": step,
                    "val_bpb": None,
                    "total_training_time": total_training_time,
                    "infer_tok_s": 0.0,
                },
                _cp,
            )
            print(f"\n[mid-ckpt] saved {_cp}", flush=True)
            _ckpt_last_t = time.time()
        except Exception as _ckpt_err:  # never let checkpointing stall training
            print(f"[warn] mid-ckpt failed ({_ckpt_err})")

    # Mid-run qualitative sample (no effect on training)
    if SAMPLE_EVERY and step % SAMPLE_EVERY == 0:
        _snip = sample_text("USER: can you write a quick python script?\n", max_new=80, temp=0.9)
        print(f"\n[sample@{step}] {_snip[:280]}")
        print(f"[vram] peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n", flush=True)

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and run_elapsed >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
eval_batch_size = 2  # VRAM-safe: training ran device_batch=2; batch 64 OOM'd the final eval before the checkpoint saved

# Best-effort: a final-eval OOM must NOT block the checkpoint save below
val_bpb = None
try:
    _original_tokens = prepare.EVAL_TOKENS
    prepare.EVAL_TOKENS = 524288  # smaller eval for iGPU speed
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, eval_batch_size)
    prepare.EVAL_TOKENS = _original_tokens
except Exception as _eval_err:  # e.g. HIP OOM — weights are intact, just skip val_bpb
    print(f"[warn] final eval failed ({_eval_err}); saving checkpoint without val_bpb")
    torch.cuda.empty_cache()

# Inference throughput benchmark
@torch.no_grad()
def benchmark_inference_tok_s(prompt, max_new=256, temp=0.9):
    bos_id = tokenizer.encode(prepare.BOS_TOKEN)[0]
    tokens = tokenizer.encode(prompt, prepend=bos_id)
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(max_new):
        with autocast_ctx:
            logits = model(inp[:, -2048:])
        logits = logits[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)
        top_idx = torch.multinomial(probs, 1)
        inp = torch.cat([inp, top_idx], dim=1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return max_new / dt

infer_tok_s = benchmark_inference_tok_s("O universo é")
print(f"Inference: {infer_tok_s:.1f} tok/s")

# Generate some text
bos_id = tokenizer.encode(prepare.BOS_TOKEN)[0]

@torch.no_grad()
def gen(prompt, temp=0.9, max_new=128):
    model.eval()
    tokens = tokenizer.encode(prompt, prepend=bos_id)
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    with autocast_ctx:
        for _ in range(max_new):
            logits = model(inp[:, -2048:])
            if temp <= 0:
                top_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / temp
                probs = torch.softmax(logits, dim=-1)
                top_idx = torch.multinomial(probs, 1)
            inp = torch.cat([inp, top_idx], dim=1)
            if top_idx.item() == 0:
                break
    return tokenizer.decode(inp[0].tolist())

print("---")
for p, t in [("O universo é", 0.0), ("O significado da vida é", 0.0), ("No princípio", 0.0), ("USER: can you write a quick python script?\n", 0.0)]:
    print(f"[greedy] {p} {gen(p, temp=t).split(p, 1)[-1].strip()[:200]}")
    print()

# Checkpoint — ALWAYS saved (this is the critical artifact; never gated on eval success)
try:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
except OSError as e:
    print(f"[warn] cannot create {CHECKPOINT_DIR}: {e}")
_ckpt_tag = f"{val_bpb:.4f}" if val_bpb is not None else "na"
checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{'gr_' if MODEL == 'gr' else ''}depth{DEPTH}_step{step}_{_ckpt_tag}.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': asdict(config),
    'step': step,
    'val_bpb': val_bpb if val_bpb is not None else 0.0,
    'total_training_time': total_training_time,
    'infer_tok_s': infer_tok_s,
}, checkpoint_path)
print(f"Checkpoint saved: {checkpoint_path}")

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / PEAK_BF16_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"infer_tok_s:      {infer_tok_s:.1f}")
