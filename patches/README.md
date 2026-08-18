# Patch Strategy

Paddle-sdaa keeps Paddle framework compatibility changes inside this repository.

## Current approach

The backend does not require users to edit the upstream Paddle source tree for Python-level compatibility fixes. Runtime compatibility patches that affect Paddle Python APIs live in:

```text
sdaa_ext/python/patch/
sdaa_ext/python/sitecustomize.py
```

This package-side patch layer is used for behavior that belongs to the SDAA backend package, such as exposing SDAA device properties through Paddle's public device APIs.

## Why this is package-side

Public Paddle releases and downstream Paddle source trees may differ. Keeping SDAA compatibility logic inside this repository makes the backend easier to install, test, and publish without carrying a long-lived fork of Paddle for small Python API adaptations.

## When to add framework patches

Use package-side patches for:

- SDAA-specific Python API adaptation.
- Custom-device behavior that Paddle can dispatch to but does not expose for non-GPGPU devices yet.
- Compatibility shims that are safe to install only when SDAA is available.

Use source patches only when:

- Paddle core C++ behavior must change.
- The change cannot be expressed through custom-device runtime hooks or Python package hooks.
- The patch is documented, minimal, and reproducible.

No source-tree patch is currently required for the published SDAA test set.
