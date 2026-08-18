from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

import paddle
from paddle.base import core


class _SDAADeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("chipsetID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciVendor", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("clockRate", ctypes.c_int),
        ("reserved", ctypes.c_int * 64),
    ]


class SDAADeviceProperties:
    def __init__(self, device_id: int, prop: _SDAADeviceProp, free_memory: int):
        self.name = prop.name.decode(errors="replace").rstrip("\0")
        self.major = prop.chipsetID
        self.minor = prop.pciDeviceID
        self.total_memory = prop.totalGlobalMem
        self.multi_processor_count = 0
        self.device_id = device_id
        self.free_memory = free_memory
        self.pci_bus_id = prop.pciBusID
        self.pci_vendor = prop.pciVendor
        self.clock_rate = prop.clockRate

    def __repr__(self) -> str:
        return (
            "_customDeviceProperties("
            f"name='{self.name}', major={self.major}, minor={self.minor}, "
            f"total_memory={self.total_memory // 1024 // 1024}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


def _load_runtime():
    for path in (
        os.path.join(os.environ.get("SDAA_ROOT", "/opt/tecoai"), "lib64", "libsdaart.so"),
        "libsdaart.so",
    ):
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError("Unable to load libsdaart.so for SDAA device properties")


def _device_id(device: Any) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, core.CustomPlace):
        if device.get_device_type() != "sdaa":
            raise ValueError(f"Expected an SDAA device, but got {device}")
        return device.get_device_id()
    if isinstance(device, str):
        if device == "sdaa":
            return 0
        if device.startswith("sdaa:") and device[5:].isdigit():
            return int(device[5:])
    raise ValueError(
        "SDAA device must be None, an integer, 'sdaa', 'sdaa:<id>', "
        "or paddle.CustomPlace('sdaa', <id>)"
    )


def get_device_properties(device: Any = None) -> SDAADeviceProperties:
    device_id = _device_id(device)
    count = core.get_custom_device_count("sdaa")
    if device_id < 0 or device_id >= count:
        raise ValueError(f"SDAA device id {device_id} is out of range [0, {count})")

    runtime = _load_runtime()
    prop = _SDAADeviceProp()
    runtime.sdaaGetDeviceProperties.argtypes = [ctypes.POINTER(_SDAADeviceProp), ctypes.c_int]
    runtime.sdaaGetDeviceProperties.restype = ctypes.c_int
    status = runtime.sdaaGetDeviceProperties(ctypes.byref(prop), device_id)
    if status != 0:
        raise RuntimeError(f"sdaaGetDeviceProperties failed with status {status}")

    free_memory = ctypes.c_size_t()
    total_memory = ctypes.c_size_t()
    runtime.sdaaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    runtime.sdaaMemGetInfo.restype = ctypes.c_int
    status = runtime.sdaaMemGetInfo(ctypes.byref(free_memory), ctypes.byref(total_memory))
    if status != 0:
        free_memory.value = 0
    if total_memory.value and not prop.totalGlobalMem:
        prop.totalGlobalMem = total_memory.value
    return SDAADeviceProperties(device_id, prop, free_memory.value)


def install() -> None:
    device_module = paddle.device
    custom_device_module = paddle.device.custom_device
    device_module._get_device_properties = get_device_properties
    custom_device_module.get_device_properties = get_device_properties
    device_module.get_device_properties.__globals__["_get_device_properties"] = get_device_properties
    device_module.get_device_name.__globals__["get_device_properties"] = device_module.get_device_properties
    device_module.get_device_capability.__globals__["get_device_properties"] = device_module.get_device_properties
