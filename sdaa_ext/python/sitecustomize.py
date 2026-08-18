from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys


_PATCHED = False
_HOOK_INSTALLED = False


def _load_device_properties_patch():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "paddle_sdaa", "patch", "device_properties.py"),
        os.path.join(base_dir, "patch", "device_properties.py"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location(
            "_paddle_sdaa_device_properties_patch", path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def _patch_paddle_device_properties() -> None:
    global _PATCHED
    if _PATCHED:
        return
    paddle = sys.modules.get("paddle")
    if paddle is None or not hasattr(paddle, "device"):
        return
    patch = _load_device_properties_patch()
    if patch is None:
        return
    try:
        patch.install()
    except Exception:
        return
    _PATCHED = True


class _PaddlePatchLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self._loader = loader

    def create_module(self, spec):
        if hasattr(self._loader, "create_module"):
            return self._loader.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _patch_paddle_device_properties()


class _PaddlePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "paddle":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            if not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if isinstance(spec.loader, importlib.abc.Loader):
                spec.loader = _PaddlePatchLoader(spec.loader)
            return spec
        return None


def _install_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    if "paddle" in sys.modules:
        _patch_paddle_device_properties()
        return
    sys.meta_path.insert(0, _PaddlePatchFinder())
    _HOOK_INSTALLED = True


_install_hook()
