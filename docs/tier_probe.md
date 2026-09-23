# tier_probe — recompute against read, in inference and in training

The tool is `src/app/tier_probe.f90`. It has two modes and one instrument. The
instrument is `src/lib/fortran_probe.f90`, which knows nothing about a model. The
module `energy-fortran` measures joules, time, CPU, and IO inside the process.
The arch of the model comes from the `arch.txt` of a checkpoint, and not from
`fortran_arch_mod`.

The law: the boundary between recompute and read is a ratio of compute to IO. It
is not a property of an algorithm. One binary on two machines gives two
boundaries that differ by about 380 times.

## Mode `infer` — a window of KV in decode

| level and operation (one token = 3 KB at d96) | fermi, NVMe, idle, run 139 | halfbeast, HDD |
|---|---|---|
| RAM sequential, page cache | **0.11 to 0.12 us** (25 to 29 GB/s) | 0.375 us |
| RAM, one IO per token, random | **7.5 to 8.1 us** | 15.6 us |
| disk sequential | **1.2 us** (2.5 GB/s) | 26.9 us |
| disk, one IO per token | **102 to 107 us** | **9732 us** |
| recompute of the window, projections and attention | 81, 128, and 257 us per token at T=32, 128, and 512 | 309 us per token at T=512 |
| **N\\*, where read and recompute are equal** | **66 to 94 tokens** | **18984 tokens** |

The fermi column comes from a clean measurement: run 139, an idle machine, and 5
repetitions for each cell. Under load the warm bandwidth fell to 9 to 22 GB/s,
and N\\* fell to about 50. A measurement on a busy machine is not a measurement,
and `probe_trust` rejects it.

* Read wins against RAM in every case, by 10 to 2000 times.
* Read wins against the disk in the one-IO-per-token case when the window is
  larger than N\\*. On halfbeast, a recompute of 512 tokens costs 158 ms against
  4.98 s for the read.
* The batch is the control value. When one IO reads the whole window, the read
  wins by **17.8 times** (T=32), **63.5 times** (T=128), and **178.8 times**
  (T=512) on fermi, and by 4 to 7 times on halfbeast.

The video and the paper of the frontier say: delete the local memory and
recompute 128 tokens. That sentence is a statement about the ratio of FLOPs to IO
on their hardware. A GPU holds about 100 times the FLOPs of a CPU core with the
same NVMe latency. That moves N\\* from about 50 to a few thousand, and 128
tokens then falls on the recompute side.

## Mode `train` — activations and optimizer state

The activation footprint is measured, and not guessed. It comes from the
`allocate(C%...)` calls in `fortran_train.f90`: `C%e, C%xa, C%q, C%ao, C%e1,
C%qr` are 6 times `d_model`. `C%f` is 4 times `d_model`, because `dff = 4*DD`.
`C%k, C%v, C%kr` are 3 times `d_kv`. At d96 that is 1056 floats for one layer and
one token (4224 B), or **50688 B for one token** in the whole model. The KV of
inference costs 3072 B for one token, so the activations are 17 times larger.

| measured at d96, 2.75M params | value |
|---|---|
| recompute of one layer for one token | 7.37 us |
| read of its activation, sequential | RAM 0.31 us, disk 1.93 us |
| **T\\***, a scattered read of one IO per token | RAM 1.6 tokens, disk 32.4 tokens |
| optimizer state, m and v | 22.02 MB |
| offload of m and v, once per step | RAM 1.60 ms, disk 10.07 ms |

The step takes 45.25 ms in this test.

Two results follow.

1. **Store the activation. Do not recompute it.** The store wins at every level.
   Memory is cheap here and a CPU FLOP is expensive. Activation checkpointing
   pays only when the activations do not fit in fast memory. Our activations do
   fit. So we do not use it.
2. **Do not offload m and v to the disk.** The offload costs about 22 percent of
   a step. Keep them in RAM, or use bf16 for half the bytes, or do not carry them
   between syncs. The last option is the one that arm C of the federated
   experiment chose for quality, at 0.196 bpb.

## Energy sensors: the honest limit

* **fermi**: `/sys/class/powercap/intel-rapl:0/energy_uj` is `-r-------- root
  root`. A process without privilege cannot measure CPU energy. The only
  readable sensor is `hwmon7 = amdgpu` at about 50.14 W idle. The app now warns
  in capital letters that this joule value belongs to the GPU. The proof is
  arithmetic: 56.8 J in 1.1008 s is 51.6 W, the idle power of the GPU.
* **halfbeast**: `energy_uj` is `-r--r--r--` on an i9-7900X, and the value is
  real. CPU energy works there. A random read from the disk costs **1.36e-1 J
  for one token** against **1.19e-2 J for a recompute**. The read costs 11.5
  times more energy in the scattered case.
* Consequence: `energy.J` in a checkpoint trained on **fermi** measures an idle
  GPU and not the training. Use halfbeast for energy, or a readable counter.

## The honesty rules inside the probe

* **Cold is verified.** The counter `read_bytes` in `/proc/self/io` stays near 0
  in the warm phase and reaches the byte count in the cold phase. `probe_trust`
  reads that counter.
* **Median of N repetitions, and the spread.** The probe warms the cache first.
  It calls `fsync` and waits one second after a write. Without the wait, the
  writeback lands in the warm cells and the spread goes above 1000 percent.
* **The probe refuses to certify an unstable cell.** On fermi with a load of
  8.8, one repetition took 1.1 s against 5 ms for the others. That is a CPU
  deschedule, and the probe rejects the cell.
* **The context goes with the number.** Each cell carries `cores_busy`,
  `cpu_pct`, `rd_MB`, and `reps`. `PROBE_VERBOSE=1` prints the wall time of every
  repetition.
* `--json` appends one line for each cell, so a run can cite the number.

## References

* `hep/` — `hyp_2ac980` (the compute-to-IO boundary) and `hyp_8f9016` (the
  saturation in K).
* `docs/fortran_gpt.md` — the library. `docs/sync_composition.md` — composition.
* `docs/writing.md` — the writing rules for this repository.
