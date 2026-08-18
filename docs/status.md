# Adaptation Status

This document summarizes the local SDAA validation state for the published backend.

## Current status

- Paddle version used for adaptation: Paddle 3.5 development build.
- Registered CTest targets: 194.
- Local SDAA validation: 194 / 194 targets executed and passed in numbered batches.
- Skipped tests in `tests/`: none found via `unittest.skip`, `skipTest`, `pytest.mark.skip`, or `SkipTest`.
- Public GitHub CI: pre-commit only, because hosted runners do not provide SDAA hardware or Tecorigin runtime libraries.

## Covered areas

- Common operator kernels.
- Training and inference smoke flow through MNIST.
- AMP paths including high-performance convolution and pipeline parallel smoke coverage.
- FlashAttention no-mask and masked regression paths.
- Device API compatibility for `paddle.device.get_device_properties`, `get_device_capability`, and `get_device_name`.
- Runtime and profiler tests.
- XCCL communication stream tests.
- DDP optimizer tests.
- Tensor/model/pipeline parallel tests.
- Sharding stage 2 and stage 3 tests.

## Local validation command

On an SDAA machine:

```bash
ctest --test-dir build --output-on-failure -j 1
```

For public CI, run:

```bash
pre-commit run --all-files --show-diff-on-failure
```

## Release interpretation

The backend is considered complete for the currently supported Paddle SDAA functionality when the registered SDAA CTest suite passes without skipped tests and the known vendor-runtime edge cases are covered by documented backend fallbacks or smoke-path substitutions.
