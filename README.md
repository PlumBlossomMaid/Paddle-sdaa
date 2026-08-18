[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.5-blue.svg)](https://github.com/PaddlePaddle/Paddle)
[![SDAA](https://img.shields.io/badge/Tecorigin-SDAA-green.svg)](https://www.tecorigin.com/)
[![Tests](https://img.shields.io/badge/CTest-194%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-sdaa

**PaddlePaddle custom device backend for Tecorigin SDAA accelerators.**

Paddle-sdaa adapts PaddlePaddle's custom-device runtime to Tecorigin SDAA hardware. It provides SDAA kernel registration, runtime integration, Python package patches, training and inference smoke coverage, AMP and distributed communication support, and compatibility fallbacks for vendor runtime edge cases.

## Features

- **Paddle Custom Device** -- registers SDAA as a Paddle custom backend with 269 custom kernels.
- **Paddle 3.5 Compatibility** -- supports current Paddle attribute ABI changes such as `double` epsilon / porder handling.
- **Training and Inference** -- includes a bounded MNIST smoke workflow covering training, export, inference, and cleanup.
- **AMP Coverage** -- validates AMP paths including high-performance convolution and pipeline-parallel smoke tests.
- **Distributed Support** -- covers XCCL communication streams, DDP optimizer, pipeline parallelism, tensor/model parallel tests, and sharding stage 2/3.
- **FlashAttention Fallbacks** -- no-mask backward uses an SDAA fallback; masked forward/backward has an opt-in deterministic fallback for vendor-runtime limitations.
- **Device APIs** -- supports `paddle.device.get_device_properties()`, `get_device_capability()`, and `get_device_name()` through package-side patches without modifying Paddle source.
- **Open CI Friendly** -- GitHub Actions runs only pre-commit checks because public GitHub runners do not provide SDAA hardware or Tecorigin runtime libraries.

## Repository Layout

```text
Paddle-sdaa/
├── cmake/                  # CMake helpers and third-party wiring
├── dynload/                # Dynamic library loading helpers
├── external/               # External SDAA stream headers
├── kernels/                # Paddle custom-device SDAA kernels
├── runtime/                # Custom-device runtime implementation
├── sdaa_ext/               # Python extension package and high-performance custom ops
├── sdaac_ops/              # SDAAC custom op sources
├── tests/                  # Unit, runtime, MNIST, and distributed tests
├── tools/version/          # Runtime / stack version query utilities
├── compile.sh              # Build entry point
└── pr_ci_sdaa.sh           # Local CI script for SDAA machines
```

## Requirements

The public GitHub CI checks repository hygiene only. Building and running the backend requires a local SDAA environment:

| Component | Notes |
| --- | --- |
| PaddlePaddle | Paddle 3.5 development build used during adaptation |
| Tecorigin stack | SDAA runtime, driver, Tecodnn, Tecoblas, Tccl, Tecocustom, SDPTI |
| Hardware | Tecorigin SDAA devices, for example `/dev/tcaicard*` |
| Python | CPython 3.12 in the validated local environment |
| CMake | Required for native build |

## Build

```bash
git clone https://github.com/PlumBlossomMaid/Paddle-sdaa.git
cd Paddle-sdaa

source /opt/tecoai/setvars.sh
export PADDLE_SOURCE_DIR=/path/to/Paddle

bash compile.sh
python -m pip install --force-reinstall --no-deps build/dist/*.whl
```

> The SDAA package intentionally does not pin NumPy. NumPy compatibility belongs to the Paddle framework package, not this backend package.

## Quick Check

```bash
python - <<'PY'
import paddle

paddle.set_device('sdaa')
print(paddle.device.get_all_custom_device_type())
print(paddle.device.get_device_properties())
print(paddle.device.get_device_capability())
print(paddle.device.get_device_name())
print(paddle.nn.functional.relu(paddle.to_tensor([-2.0, 1.0])))
PY
```

Expected shape of the output:

```text
['sdaa']
_customDeviceProperties(name='/dev/tcaicard0', major=61440, minor=256, total_memory=15296MB, multi_processor_count=0)
(61440, 256)
/dev/tcaicard0
Tensor(shape=[2], dtype=float32, place=Place(sdaa:0), stop_gradient=True,
       [0., 1.])
```

## Test

On an SDAA machine:

```bash
ctest --test-dir build --output-on-failure -j 1
```

Current local validation status:

- **194 / 194** registered CTest targets executed and passed in numbered batches.
- No tests are hidden behind `unittest.skip`, `skipTest`, or `pytest.mark.skip`.
- Key coverage includes common operators, MNIST smoke training/inference, FlashAttention, runtime/profiler, device APIs, communication streams, DDP optimizer, sharding stage 2/3, and pipeline parallel tests.

## Compatibility Notes

- `PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1` enables the deterministic SDAA masked FlashAttention fallback when the vendor Tecocustom masked path is numerically unsuitable for a workload.
- `paddle.device.get_device_properties()` reads real runtime data from `sdaaGetDeviceProperties` and `sdaaMemGetInfo`. `multi_processor_count` remains `0` because the exposed SDAA runtime structure does not provide an equivalent field.
- Public GitHub CI cannot validate SDAA runtime behavior because hosted runners do not include SDAA devices or Tecorigin libraries.

## CI

This repository only runs pre-commit checks on GitHub:

```bash
pre-commit run --all-files --show-diff-on-failure
```

Hardware, runtime, and distributed tests must be run on an SDAA machine.

## License

Apache License 2.0.
