# Provenance Of Information

Use this guide when comparing Eigen against, benchmarking against, or integrating with software Eigen does not own,
and whenever a task would involve looking at how such software works. [`AGENTS.md`](../AGENTS.md) rule 2 is the
contract; [`numerics.md`](numerics.md#provenance) covers citing the literature an implementation is built on.

Eigen aims to be the best library it can be, in speed, accuracy and everything else, by legal and ethical means, from
original research and publicly available information, and by no other. An idea learned by disassembling a vendor
kernel taints the Eigen code it informs, even when no instruction is copied.

## Original Research Is Encouraged

Measure the hardware with your own microbenchmarks, measure other libraries from outside, run numerical experiments,
design new algorithms, and derive new bounds: what you find is yours to build on. The rest of this guide limits how you
learn about software Eigen does not own, not what you discover yourself. The one condition is that you can publish it
with the change; a result you are bound to keep confidential, by an NDA or an employer's terms, stays out.

## Proprietary Software Is A Black Box

Intel oneMKL, NVIDIA's CUDA libraries, Arm Performance Libraries, Apple Accelerate, AMD's AOCL binary packages, IBM
ESSL, Cray LibSci and similar products are used through their documented interface and measured as shipped. Their
licenses restrict reverse engineering and, in several cases, altering how the software runs; Eigen draws its line at
the black box regardless of what a particular license or jurisdiction permits, and the line does not move when someone
else has already published what is inside: the vendor's documentation defines the supported interface, not a forum
post. If a comparison cannot be configured through documented means, report the limitation.

| Allowed: observe from outside | Not allowed: look inside |
|---|---|
| Link against the library, call its documented API, time it, compare its numerical results | Disassemble or decompile it (`objdump -d`, `otool -tv`, `cuobjdump`, Ghidra, IDA, radare2) |
| Read its published documentation, headers, release notes, application notes, and the vendor's papers and talks | Dump its strings or bytes (`strings`, `xxd`, `grep -a`) to find undocumented switches, kernel names, or dispatch tables |
| Set documented environment variables and verbose modes | Step a debugger through it, or `perf annotate` its code |
| Run it as shipped, configured only through its documented interface | Change how it runs by other means: override or interpose its symbols, patch it, or set switches its vendor does not document |
| Diagnose linking: `ldd`, `nm -D`, `otool -L`, `readelf -d`, `objdump -p` | Read source that was leaked, is under NDA, or comes from a current or former employer |
| Profile at function granularity to see how much time is spent in the library as a whole | Use third-party write-ups that obtained their content by any of the above |

A proprietary library linked statically into one of Eigen's own benchmark binaries is still inside the box: restrict
disassembly and annotation to Eigen's symbols.

## Open-Source Software

OpenBLAS, BLIS, reference LAPACK, libflame, and the ROCm math libraries are public, and reading or disassembling them is
fine. Two cautions:

- Public is not the same as compatible. Reading GPL or LGPL code is allowed, but copying, paraphrasing, or translating it
  into Eigen is not; see rule 2 and [`CONTRIBUTING.md`](../CONTRIBUTING.md#provenance-and-attribution).
- AMD publishes the AOCL sources under permissive licenses, but its binary packages carry a no-disassembly EULA. Read the
  published source, not the installed binary.

## Show Where The Change Comes From

A merge request description should link the publicly available documentation that supports the change: the ISA or
architecture manual, the vendor's optimization guide or intrinsics reference, the paper or standard an algorithm
follows, the documented API a backend relies on. Where the change rests on original research, include the evidence
instead: the reproducer, the method and results of a measurement, or the derivation. This helps most with
hardware-specific optimizations and new features, where it lets a reviewer check the change against its sources and
see that everything it relies on is public or published with it.

## If You Are Exposed Anyway

Stop, do not act on what you saw, and tell the user what was seen and how. Do not record it in code, comments, commit
messages, merge request descriptions, benchmark write-ups, or agent memory. If it has already been committed, pushed,
or published, say exactly where; the maintainers decide how far back to remove it and whether the affected work needs
a different author or a fresh start.
