"""
Fortran RMSNorm kernel demo — honors the "use Fortran" whim.

Compiles fortran_rmsnorm.f90 with gfortran into a .so, calls it via ctypes,
and benchmarks it against a pure-torch RMSNorm on a realistic (batch*seq x C)
tensor. Correctness is checked against torch.

Run:  python3 novel/fortran_rmsnorm.py   (inside nix develop for torch)
"""
import os
import subprocess
import time

import ctypes
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fortran_rmsnorm.f90")
SO = os.path.join(HERE, "fortran_rmsnorm.so")

if not os.path.exists(SO):
    subprocess.run(["gfortran", "-shared", "-fPIC", "-O3", "-ffast-math",
                    "-static-libgfortran", "-static-libgcc", SRC, "-o", SO], check=True)

try:
    lib = ctypes.CDLL(SO)
    lib.rmsnorm_f.restype = None
    lib.rmsnorm_f.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
except OSError as e:
    raise RuntimeError(f"Fortran RMSNorm kernel failed to load: {e}") from e


def rmsnorm_torch(x, w, eps=1e-5):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


eps = 1e-5
n, c = 2 * 2048, 896  # one micro-batch of our DEPTH=14 model
x_np = np.random.randn(n * c).astype(np.float32)
w_np = np.random.randn(c).astype(np.float32)
y_np = np.zeros(n * c, dtype=np.float32)
x_t = torch.tensor(x_np).reshape(n, c)
w_t = torch.tensor(w_np)

reps = 3000

t0 = time.time()
for _ in range(reps):
    try:
        lib.rmsnorm_f(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, c, eps,
        )
    except OSError as e:
        raise RuntimeError(f"Fortran RMSNorm kernel call failed: {e}") from e
fortran_us = (time.time() - t0) / reps * 1e6

yt = rmsnorm_torch(x_t, w_t)  # bind before timing loop
t0 = time.time()
for _ in range(reps):
    yt = rmsnorm_torch(x_t, w_t)
torch_us = (time.time() - t0) / reps * 1e6

max_err = float(np.max(np.abs(y_np.reshape(n, c) - yt.numpy())))
print(f"Fortran RMSNorm : {fortran_us:7.1f} us/call")
print(f"Torch  RMSNorm : {torch_us:7.1f} us/call")
print(f"speedup (Fortran/Torch on CPU): {torch_us / fortran_us:5.2f}x")
print(f"max abs error vs torch: {max_err:.2e}")
