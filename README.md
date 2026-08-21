# Paddle-sdaa

**PaddlePaddle custom-device backend for Tecorigin SDAA accelerators.**

Paddle-sdaa adapts PaddlePaddle custom-device runtime to Tecorigin SDAA hardware. It includes SDAA kernels, runtime integration, Python extensions, AMP and distributed communication support, and compatibility fallbacks for vendor runtime differences.

## Features

- Paddle 3.5 custom-device backend with SDAA kernel registration.
- Training and inference smoke coverage, including MNIST.
- AMP, XCCL, DDP, tensor/model parallel, and sharding coverage.
- FlashAttention fallbacks for vendor-runtime limitations.
- Device information APIs through package-side integration.

## Repository layout

```text
Paddle-sdaa/
├── cmake/                  # CMake helpers and third-party wiring
├── kernels/                # SDAA custom-device kernels
├── runtime/                # Custom-device runtime implementation
├── sdaa_ext/               # Python extension package and custom ops
├── sdaac_ops/              # SDAA custom op sources
├── tests/                  # Unit, runtime, MNIST, and distributed tests
├── tools/build.sh          # Unified Paddle + SDAA build entry point
├── compile.sh              # Legacy SDAA-only build entry point
└── pr_ci_sdaa.sh           # Local CI script
```

## Requirements

Building requires a Tecorigin SDAA development environment:

| Component | Requirement |
| --- | --- |
| Paddle source | Paddle 3.5 source tree; default `../Paddle` |
| Tecorigin stack | `/opt/tecoai`, including SDAA runtime, Tecodnn, Tecoblas, Tccl, Tecocustom, SDPTI |
| Hardware | SDAA devices such as `/dev/tcaicard*` |
| Python | CPython 3.11 or newer supported by the Paddle source build |
| Build tools | CMake, Ninja, GCC, Git |

## One-command build

Clone Paddle beside this repository, then run the unified build script:

```bash
git clone https://github.com/PaddlePaddle/Paddle.git ../Paddle
bash tools/build.sh --all
```

`--all` performs the following steps:

1. Loads `/opt/tecoai/setvars.sh`.
2. Initializes Paddle submodules and builds a LoongArch64 CPU Paddle wheel.
3. Installs the Paddle wheel in the selected Python environment.
4. Downloads the SDAA extension-operator package when `sdcops.h` is absent.
5. Builds and installs the `paddle_sdaa` wheel.

Useful modes:

```bash
bash tools/build.sh --paddle   # build and install Paddle only
bash tools/build.sh --sdaa     # build and install SDAA only
bash tools/build.sh --build    # build SDAA without installing its wheel
bash tools/build.sh --install  # install the latest SDAA wheel
bash tools/build.sh --test     # run the CTest suite
bash tools/build.sh --clean    # remove Paddle and SDAA build directories
```

Override paths and parallelism when needed:

```bash
PADDLE_SOURCE_DIR=/path/to/Paddle MAX_JOBS=32 bash tools/build.sh --all
```

The script expects build dependencies (`numpy`, `protobuf`, `Pillow`, and `safetensors`) to be installed before building Paddle. It does not modify or reset the Paddle source repository.

## Quick check

```bash
source /opt/tecoai/setvars.sh
python - <<'PY'
import paddle

paddle.set_device('sdaa')
x = paddle.to_tensor([1.0, 2.0])
print(paddle.__version__)
print(paddle.device.get_all_custom_device_type())
print(x.place, paddle.sum(x).item())
PY
```

Expected output includes:

```text
3.5.0
['sdaa']
Place(sdaa:0) 3.0
```

## Tests

On an SDAA machine:

```bash
bash tools/build.sh --test
```

The repository currently registers 194 CTest targets covering kernels, runtime, MNIST smoke training/inference, FlashAttention, profiler, device APIs, XCCL, DDP, parallelism, and sharding.

## Compatibility notes

- Set `PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1` to enable deterministic masked FlashAttention fallback.
- Hardware, runtime, and distributed tests require SDAA hardware and the Tecorigin stack; public CI only runs repository checks.

## License

Apache License 2.0.
