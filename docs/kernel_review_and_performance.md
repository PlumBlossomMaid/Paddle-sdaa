# SDAA Kernel Review and Performance Baseline

## Scope

This document records the first review of the SDAA kernel registry and the CPU/SDAA microbenchmark run on 2026-08-21. It is a reference baseline, not a universal performance claim. Results depend on SDAA runtime, driver, clock state, tensor shape, dtype, and synchronization behavior.

## Registry and test coverage

- Kernel source files reviewed by static scan: 123
- Plugin kernel registrations: 274
- Registered SDAA phi kernels at runtime: 269
- SDAA unit-test files: 163
- Unit-test files containing output or gradient assertions: 140

The runtime registry is the authoritative list for a build. A source registration alone does not prove that every dtype, shape, layout, attribute combination, or gradient path is implemented correctly.

## FP64 probe

The current `/opt/tecoai` stack accepted FP64 tensors on SDAA for the following smoke operations:

- elementwise add
- elementwise multiply
- ReLU
- matrix multiplication
- reduction sum
- square root

The tensors retained `paddle.float64` dtype and produced valid results. This demonstrates runtime/kernel-path availability for these operations; it does **not** prove native FP64 arithmetic. The vendor SDK documentation and hardware counters are required to distinguish native FP64 from an FP32 implementation with conversion or emulation.

The runtime registry currently exposes FP64 for some operators, including `add` and `multiply`, while `matmul`, `conv2d`, and `relu` have narrower registrations. FP64 support must therefore be checked per operator and dtype, not inferred globally.

## First microbenchmark

Command:

```bash
source /opt/tecoai/setvars.sh
python3 tools/benchmark_ops.py --warmup 3 --repeat 5 \
  --output build/benchmark_ops_v2.csv \
  --json-output build/benchmark_ops_v2.json
```

The generic adapter benchmarked 70 of 269 registered kernels. 199 were not benchmarked because they require structured inputs, gradients, attributes, multiple outputs, or a dedicated invocation. One additional failure was a real backend restriction: the current argsort implementation rejected `descending=false`.

Representative results from the run (1024 x 1024 float32 inputs, median wall time including explicit SDAA synchronization):

| Operator | CPU ms | SDAA ms | CPU/SDAA |
| --- | ---: | ---: | ---: |
| `mish` | 27.872 | 0.102 | 272.66x |
| `atan` | 20.240 | 0.118 | 171.04x |
| `sin` | 11.862 | 0.120 | 98.82x |
| `exp` | 14.290 | 0.179 | 79.78x |
| `relu6` | 6.832 | 0.086 | 79.17x |
| `bmm` | 34.783 | 1.717 | 20.26x |
| `cast` | 2.387 | 0.089 | 26.74x |
| `all` | 0.808 | 2.024 | 0.40x |
| `any` | 1.372 | 4.095 | 0.34x |
| `floor` | 0.280 | 1.369 | 0.20x |

The small reduction and transfer-like cases show that SDAA launch and synchronization overhead can dominate. These numbers should not be used to rank the hardware without shape sweeps and an end-to-end model benchmark.

## Review findings and follow-up

### Fixed in this revision

`argsort` previously:

- rejected every `descending=false` call;
- restricted input to one dimension and power-of-two lengths of at least 4096;
- overwrote the kernel-produced indices with an `arange`, returning incorrect indices for non-identity permutations.

The implementation now preserves the real TopK indices and implements ascending order through negated input/output around the vendor descending TopK path. The shape and rank limits remain explicit because they are imposed by the current Tecodnn TopK interface, not by Paddle's operator contract.

### Hardware/runtime limits that require dedicated work

The static review found restrictions in shared helpers and kernels for softmax axis, interpolation modes, tensor rank, storage formats, optimizer attributes, and vendor extension APIs. These should be handled in one of three ways:

1. implement a kernel path using an existing SDAA primitive;
2. compose existing SDAA kernels when semantics are preserved;
3. return a precise unsupported error only when the vendor runtime genuinely cannot implement the contract.

A generic benchmark skip is not evidence that an operator is unsupported. Every skipped operator needs a dedicated benchmark adapter and correctness test before a support decision is made.

## Reproducing and extending the review

List registered kernels:

```bash
source /opt/tecoai/setvars.sh
python3 - <<'PY'
from paddle.base import core
kernels = core._get_registered_phi_kernels()
print(sorted(name for name, specs in kernels.items()
             if any('sdaa' in spec.lower() for spec in specs)))
PY
```

Run a quick benchmark:

```bash
python3 tools/benchmark_ops.py --limit 20 --warmup 3 --repeat 5
```

For production conclusions, add per-operator adapters with representative shapes, dtypes, attributes, forward and gradient calls, then compare correctness against CPU before recording timings.
