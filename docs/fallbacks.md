# Fallbacks and Vendor Runtime Notes

Paddle-sdaa prioritizes observable Paddle functionality on SDAA. Vendor kernels are used where stable, and backend fallbacks are used when a vendor runtime path is unavailable or unsuitable.

## FlashAttention

### No-mask backward

The Tecocustom no-mask backward path is not used for the registered regression path. Paddle-sdaa provides an SDAA fallback implemented with existing transpose, batch matmul, scale, softmax, and elementwise kernels.

### Masked path

Tecocustom masked FlashAttention can be numerically unstable on the validated software stack. Paddle-sdaa keeps the vendor path available and provides an opt-in deterministic masked forward/backward fallback:

```bash
PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1
```

The fallback has CPU-reference coverage for deterministic shapes including:

- `(4, 1, 2, 8)`
- `(5, 2, 1, 8)`
- `(8, 1, 2, 16)`
- `(16, 2, 4, 32)`

## Pipeline AMP and Pooling

A specific distributed pipeline AMP configuration can trigger a Tecodnn Pool2D output-dimension SIGFPE. Regular Pool2D tests pass, and the pipeline AMP CTest target uses an SDAA smoke model that preserves the integration coverage of AMP, pipeline parallelism, optimizer, scaler, and distributed execution without depending on the unstable vendor Pool2D sub-path.

## Device properties

`paddle.device.get_device_properties()` is supported through a package-side patch. It reads real SDAA runtime data via:

- `sdaaGetDeviceProperties`
- `sdaaMemGetInfo`

`multi_processor_count` remains `0` because the exposed SDAA runtime property structure does not provide an equivalent field.

## Policy for new fallbacks

A fallback should document:

1. The vendor limitation that requires it.
2. The tensor layout and attribute coverage.
3. The test that validates it.
4. Any remaining numerical or runtime limitations.
