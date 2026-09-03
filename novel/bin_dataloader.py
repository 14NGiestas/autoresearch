"""Streaming dataloader over a pretokenized .bin (hyp_9f47fd).

The live baseline is data-loading bound (mfu ~0.2%, dt ~75s/step) because every
step the dataloader re-reads parquet shards and re-tokenizes on CPU. novel/
pretokenize.py flattens the hoard into a single uint16 token stream once; this
loader mmaps it and yields random contiguous B*T windows, removing that cost.

Yields (x, y, epoch) tuples compatible with train.py's training loop.
"""
import os

import numpy as np
import torch


def make_bin_dataloader(bin_path, B, T, device):
    """Yield random contiguous B*T windows from a flat uint16 token .bin.

    `bin_path` is a .bin written by novel/pretokenize.py: every train document
    tokenized (BOS prepended) and concatenated into one uint16 array. Random
    slices of length B*T+1 are drawn; x = window[:-1], y = window[1:] (next-token
    targets). BOS tokens live inside the stream, so documents stay BOS-aligned.

    Args:
        bin_path: path to the .bin file (uint16 tokens).
        B: device batch size.
        T: sequence length.
        device: torch device for the yielded tensors.
    """
    if not os.path.exists(bin_path):
        raise RuntimeError(f"pretokenized bin not found: {bin_path}")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")  # type: ignore
    n = int(data.shape[0])
    if n <= B * T + 1:
        raise RuntimeError(f"pretokenized bin too small: {n} tokens (< B*T+1={B * T + 1})")
    epoch = 0
    while True:
        ix = int(np.random.randint(0, n - B * T - 1))
        window = np.array(data[ix:ix + B * T + 1], dtype=np.int64)  # type: ignore
        x = torch.tensor(window[:B * T].reshape(B, T), dtype=torch.long, device=device)
        y = torch.tensor(window[1:].reshape(B, T), dtype=torch.long, device=device)
        yield x, y, epoch
