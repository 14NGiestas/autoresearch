# fortran_energy

[![CI](https://github.com/14NGiestas/energy-fortran/actions/workflows/ci.yml/badge.svg)](https://github.com/14NGiestas/energy-fortran/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

In-process measurement of **energy, CPU time, I/O and phases** for Fortran programs:
joules, watts, `cpu_s`, `cores_busy` and MB read/written, straight from `/sys` and
`/proc`, with **no external wrapper** and **no dependencies**.

```fortran
use fortran_energy_mod
call energy_init()
call energy_mark('setup')                  ! close the phase that just ended
... work ...
call energy_mark('train', tokens=n, iv=iv) ! J/wall/cpu/IO of THAT phase
call energy_report()                       ! one key=value line per phase
```

## Why

Energy is usually measured **outside** the process: a wrapper samples the RAPL
counter or a wall plug, and the number is attributed to checkpoints afterwards by
linear apportionment (`J_job × steps_ckpt / steps_job`). That has two problems: the
attribution is an approximation, and it only works if somebody remembers to wrap
the run.

Measuring inside makes the number **exact by construction**: the joules spent
between the previous save and this one *are* the checkpoint's joules. It also makes
per-phase attribution possible at all — setup, training, checkpoint I/O — which is
where the interesting HPC questions live (e.g. how much energy the checkpointing of
a 30 GB model costs versus a training step).

The module was written for a small-LLM lab that trains Fortran models on CPU/APU
hardware and wanted the energy to travel inside the checkpoint metadata, next to the
weights and the validation loss. It uses nothing from that lab: it is a plain,
self-contained Fortran module that any HPC code can `use`.

## API

| procedure | what it gives |
|---|---|
| `energy_init(ticks)` | find a sensor (or stay neutral), zero accumulators and phases. `ticks` overrides `USER_HZ` (default 100) |
| `energy_ready()` | `.true.` when an energy sensor was found |
| `energy_joules()` | joules accumulated since init (samples/integrates the sensor) |
| `energy_watts()` | average power since the previous call (or since init) |
| `energy_cpu_seconds(ticks)` | process CPU seconds (`utime+stime` from `/proc/self/stat`) |
| `energy_cpu_percent()` | `100·cpu_s/(wall_s·threads)` since init |
| `energy_cores_busy()` | `cpu_s/wall_s` — equivalent cores kept busy since init |
| `energy_threads()` | threads the process has now (from `/proc/self/status`) |
| `energy_seconds()` | monotonic wall clock (seconds), the same one used internally |
| `energy_ticks()` | the `USER_HZ` in use |
| `energy_interval(iv, tokens)` | `energy_interval_t` with the deltas since the last call (see below) |
| `energy_peek(iv, tokens)` | same delta as `energy_interval` but **without** advancing the interval — for progress traces that must not disturb the per-checkpoint accounting |
| `energy_mark(label, tokens, iv)` | closes the open interval and accumulates it under `label` (repeats accumulate; `n` counts them). `iv` returns the interval, so the caller gets the per-checkpoint numbers and the phase accounting in one call |
| `energy_sensor()`, `energy_scope()`, `energy_kind_name()` | which sensor was used, what it covers, and the kind (`counter` / `power` / `none`) |
| `energy_report(unit)` | one `key=value` line per phase plus a `total` line |
| `energy_report_json()` | the same as one JSON line, for structured logs |

### What is measured, and from where

* **energy**
  * **(a)** `/sys/class/powercap/*/energy_uj` — accumulated counter in microjoules.
    `max_energy_range_uj` is read next to it and used to handle **counter wrap** (a
    reading smaller than the previous one means the counter rolled over). Among RAPL
    domains it prefers `package-*`/`psys` (the whole socket) over a subdomain
    (`core`/`uncore`/`dram`), which would measure only part of the CPU.
  * **(b)** `/sys/class/hwmon/hwmon*/power1_input` — instantaneous power in
    microwatts, accepted only for `hwmon` whose `name` is one of `amdgpu`,
    `zenpower`, `amd_energy`, `rapl`, `coretemp`, `k10temp`. There is no counter
    there, so energy is a **trapezoid integral** over `system_clock`.
  * **(c)** nothing → `kind = none`, energy stays `0.0`, and CPU/IO/phases keep
    working. The fallback is useful on purpose: on a machine with no sensor you
    still get `cpu_s`, `cores_busy`, `cpu_pct` and per-phase I/O volume.
* **CPU** — `/proc/self/stat` fields 14 and 15 (`utime+stime` in ticks). Field 2
  (`comm`) may contain spaces and parentheses, so parsing starts at the **last**
  `)` of the line.
* **threads** — `/proc/self/status` (`Threads:`), the basis of `cpu_pct`.
* **I/O** — `/proc/self/io` (`read_bytes`, `write_bytes`). The *energy* of I/O
  cannot be measured from inside the process; the volume is what makes the phase
  numbers interpretable (a checkpoint writing 400 MB is not the same kind of
  "checkpoint phase" as one writing 4 MB).

## Usage

```fortran
program my_run
  use fortran_energy_mod
  implicit none
  type(energy_interval_t) :: iv
  call energy_init()                              ! once, at the very start
  ... setup ...
  call energy_mark('setup')                       ! attributes everything so far
  do step = 1, nsteps
     call train_step(...)
     if (mod(step, save_every) == 0) then
        call energy_mark('train', tokens=tok, iv=iv)   ! the numbers for THIS save
        write (my_checkpoint_metadata) iv%j, iv%j_per_token, iv%cpu_s, iv%cores_busy
        call save_checkpoint(...)
        call energy_mark('checkpoint')                  ! the cost of saving itself
     end if
  end do
  call energy_report()                            ! per-phase lines
  print '(A)', energy_report_json()               ! or one JSON line
end program my_run
```

`energy_interval_t` is the **complete record**: the interval (`j`, `wall_s`,
`cpu_s`, `cpu_pct`, `cores_busy`, `rd_mb`, `wr_mb`, `tokens`, `j_per_token`,
`w_mean`, `threads`) **plus the run-level context at the moment it was closed**
(`j_total`, `sensor`, `scope`, `kind`, `self_measured`). The API fills both halves,
so a consumer — a log line, a checkpoint card, a CSV row — never defines a mirror
struct nor assembles the context by hand:

```fortran
type(energy_interval_t) :: rec
call energy_mark('train', tokens=tok, iv=rec)
write (my_card) rec%j, rec%j_total, rec%w_mean, rec%sensor, rec%kind, rec%self_measured
```

`fpm run --example measure_run` runs a complete three-phase example.

## Build, test, depend on it

```bash
fpm build
fpm test
fpm test --profile debug --flag "-Wall -Wextra -Wcharacter-truncation -fcheck=all -fbacktrace -finit-real=snan"
fpm run --example measure_run
```

As a dependency:

```toml
[dependencies]
fortran_energy = { git = "https://github.com/14NGiestas/energy-fortran", tag = "v0.1.0" }
```

## What the test proves (and how, without hardware)

`fpm test` runs in ~0.1 s and needs no sensor:

1. **no sensor** (forced with `ENERGY_SENSOR=/nonexistent`): `energy_ready()` is
   `.false.`, energy is `0.0`, and `cpu_s`/`cores_busy`/`cpu_pct` are still measured
   and reported;
2. **a fake counter** written by the test itself: J matches `delta_uj × 1e-6`
   exactly, is monotone, **wrap** adds `max_energy_range_uj` (55 µJ from a physical
   roll-over), and a backwards jump larger than the range (a driver reset, or a
   counter whose width changed) leaves J flat — never negative, never invented;
3. **the machine's real sensor**, when there is one: `energy_joules()` never
   decreases and `energy_watts()` lands in `[0, 500] W`;
4. **phases**: repeated marks with the same label accumulate (`n=2`, sums of
   J/wall/cpu/tokens);
5. **the reports**: the `key=value` line per phase and the JSON line carry the
   expected fields.

The `ENERGY_SENSOR` environment variable (also used by the test) forces a specific
sensor path — useful to pin a domain on a multi-socket box, or to switch
measurement off.

## Limitations (honest list)

* **Linux only for the measurements.** It reads `/sys` and `/proc`. On another OS
  every read fails, `energy_init` reports `kind=none` and the module degrades to
  wall-clock + phase accounting (no energy, no `cpu_s`). That path is exercised by
  the test but has **not** been run on macOS/BSD.
* **One counter, not a sum.** On a multi-socket machine `energy_joules()` measures
  the single preferred domain (usually `package-0`); `energy_scope()` says exactly
  which one. Summing sockets is on the roadmap.
* **The hwmon path is an integral, the powercap path is a counter.** With `amdgpu`
  (or `zenpower`/`rapl` as hwmon) you get sampled power, so `J` has the trapezoid's
  error over short phases; with RAPL `energy_uj` you get the hardware counter.
  `energy_kind_name()` tells you which one you are looking at.
* **`USER_HZ = 100` is assumed** for the `/proc/self/stat` ticks (true on x86
  Linux); override it with the `ticks` argument or `ENERGY_TICKS` if your kernel
  differs.
* **`read_bytes`/`write_bytes` are the kernel's accounting of block I/O** for the
  process, not a count of `write()` calls.
* **`cpu_pct` is against the thread count *now*.** If the process had more threads
  during the phase (e.g. OpenMP threads that have already exited), `cpu_pct` can
  read above 100%. `cores_busy` is the absolute measure (`cpu_s/wall_s`); use it
  when threads come and go.
* **Not thread-safe** (counter/integrator/phase state): call it from one thread,
  outside OpenMP regions.
* **Not verified on macOS/Windows**, and not verified with a 32-bit wrapping
  counter (the wrap logic is tested with a simulated counter instead).
* The `readdir` listing uses the glibc `struct dirent` layout; if the layout does
  not match (other libc), a sanity check on `d_reclen` aborts the listing and the
  module falls back to a fixed list of known `/sys` paths.

## Roadmap (what a wider audience would need)

1. multi-socket RAPL (sum `package-*` domains) and a `--domain` selector;
2. measuring **another** process (`/proc/<pid>/stat`, `/proc/<pid>/io`) so a
   launcher can account for a child it did not write;
3. an option to sample the sensor from a background thread with a fixed period, for
   long phases where the trapezoid over sparse samples is too coarse;
4. packaging: CMake in addition to fpm, and a `--help`-style example gallery;
5. a wrap test against a real 32-bit counter (`max_energy_range_uj ≈ 4.29e9`).

## License

MIT — see [LICENSE](LICENSE).
