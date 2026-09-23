# debug — the registry for instrument and kernel defects

This registry is not the science one. `hep/` holds claims about the world: the
size axis, the composition tax, the attention bottleneck. This one holds
hypotheses about our own instruments and kernels: a counter that lies, a backward
that does not match its forward, a shim that reads freed memory.

The reason for the split is that the two have different lifetimes. A science
hypothesis stays open for weeks. A debug hypothesis is usually refuted within the
hour, and burying those refutations in the science chain would make the science
chain unreadable.

Usage, same tool, different path:

    bin/hep propose --registry debug/registry.jsonl --statement "..." --prior 0.5
    bin/hep evidence --registry debug/registry.jsonl --hyp hyp_XXXX ...
    bin/hep list --registry debug/registry.jsonl
    bin/hep verify --registry debug/registry.jsonl

Two bugs in the package were found by using it this way, and both were fixed:

1. Every subcommand except `tree` ignored --registry and wrote to the default in
   silence. A silent write to the wrong target is the same disease as a glob with
   tail -1: the consumer guesses, and the error shows up far from the cause.
2. `propose` crashed when --testable was omitted, although the help documents it
   as optional.

The value of this file is not the debugging. It is that the debugging is now
auditable: every refuted hypothesis has its test, its numbers and its commit.
