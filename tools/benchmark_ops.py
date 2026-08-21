#!/usr/bin/env python3
"""Benchmark registered Paddle kernels on CPU and SDAA."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import paddle
from paddle.base import core


UNARY = {
    "abs", "atan", "ceil", "cos", "erf", "exp", "floor", "log", "log2",
    "mish", "reciprocal", "relu", "relu6", "rsqrt", "sigmoid", "sin", "silu",
    "softsign", "softplus", "sqrt", "square", "swish", "tanh", "hardtanh",
    "hardsigmoid", "hardswish", "identity", "isfinite", "isnan", "negative",
}
BINARY = {
    "add", "divide", "equal", "greater_equal", "greater_than", "less_equal",
    "less_than", "maximum", "minimum", "multiply", "not_equal", "remainder",
    "subtract", "elementwise_pow",
}
REDUCE = {"all", "any", "max", "mean", "min", "prod", "reduce_all", "reduce_any", "sum"}
SKIP_SUFFIXES = ("_grad", "_raw")


def synchronize(place: str) -> None:
    if place == "sdaa":
        paddle.device.synchronize()


def tensor_for(place: str, shape: tuple[int, ...] = (1024, 1024)):
    previous = paddle.get_device()
    paddle.set_device(place)
    try:
        return paddle.randn(shape, dtype="float32")
    finally:
        paddle.set_device(previous)


def resolve_op(name: str) -> Callable | None:
    aliases = {"elementwise_pow": "pow", "matmul_v2": "matmul", "reduce_sum": "sum", "reduce_prod": "prod"}
    name = aliases.get(name, name)
    for owner in (paddle, paddle.nn.functional, paddle.tensor):
        fn = getattr(owner, name, None)
        if callable(fn):
            return fn
    return None


def make_call(name: str, place: str) -> Callable[[], object] | None:
    if name.endswith(SKIP_SUFFIXES) or name in {"feed", "fetch", "memcpy_h2d", "memcpy_d2h"}:
        return None
    if name in {"conv2d", "conv2d_grad", "depthwise_conv2d", "batch_norm", "batch_norm_infer", "layer_norm", "instance_norm"}:
        previous = paddle.get_device()
        paddle.set_device(place)
        try:
            x = paddle.randn([8, 8, 32, 32], dtype="float32")
            if name in {"conv2d", "conv2d_grad", "depthwise_conv2d"}:
                layer = paddle.nn.Conv2D(8, 8, 3, padding=1)
                return lambda: layer(x)
            if name == "batch_norm" or name == "batch_norm_infer":
                layer = paddle.nn.BatchNorm2D(8)
                return lambda: layer(x)
            if name == "layer_norm":
                layer = paddle.nn.LayerNorm([32, 32])
                return lambda: layer(x)
            layer = paddle.nn.InstanceNorm2D(8)
            return lambda: layer(x)
        finally:
            paddle.set_device(previous)
    if name in {"embedding", "embedding_grad"}:
        previous = paddle.get_device()
        paddle.set_device(place)
        try:
            layer = paddle.nn.Embedding(1024, 256)
            ids = paddle.randint(0, 1024, [128, 32], dtype="int64")
            return lambda: layer(ids)
        finally:
            paddle.set_device(previous)
    fn = resolve_op(name)
    if fn is None:
        return None
    x = tensor_for(place)
    y = tensor_for(place)
    if name in UNARY:
        return lambda: fn(x)
    if name in BINARY:
        return lambda: fn(x, y)
    if name in REDUCE:
        return lambda: fn(x, axis=1)
    if name in {"matmul", "matmul_v2", "bmm"}:
        return lambda: paddle.matmul(x, y)
    if name == "add_n":
        return lambda: paddle.add_n([x, y])
    if name in {"argmax", "argmin"}:
        return lambda: fn(x, axis=1)
    if name == "argsort":
        return lambda: fn(x, axis=1)
    if name in {"cast"}:
        return lambda: paddle.cast(x, "float16")
    if name in {"softmax", "log_softmax"}:
        return lambda: fn(x, dim=-1)
    if name == "logsigmoid":
        return lambda: paddle.nn.functional.log_sigmoid(x)
    if name in {"gelu", "elu", "leaky_relu", "hardsigmoid", "hardswish"}:
        return lambda: fn(x)
    if name == "clip":
        return lambda: paddle.clip(x, min=-1.0, max=1.0)
    if name == "dropout":
        return lambda: paddle.nn.functional.dropout(x, p=0.1)
    if name == "flatten":
        return lambda: paddle.flatten(x)
    if name == "reshape":
        return lambda: paddle.reshape(x, [1024, 1024])
    if name == "transpose":
        return lambda: paddle.transpose(x, [1, 0])
    if name == "concat":
        return lambda: paddle.concat([x, y], axis=0)
    if name == "tile":
        return lambda: paddle.tile(x, [1, 1])
    if name == "squeeze":
        return lambda: paddle.squeeze(x)
    if name == "unsqueeze":
        return lambda: paddle.unsqueeze(x, axis=0)
    if name == "where":
        return lambda: paddle.where(x > 0, x, y)
    if name == "nonzero":
        return lambda: paddle.nonzero(x > 0)
    if name == "topk":
        return lambda: paddle.topk(x, k=10, axis=-1)
    return None


def registered_sdaa_ops() -> list[str]:
    kernels = core._get_registered_phi_kernels()
    return sorted(name for name, specs in kernels.items() if any("sdaa" in spec.lower() for spec in specs))


def benchmark(call: Callable[[], object], place: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    synchronize(place)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        synchronize(place)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_ops.csv")
    parser.add_argument("--json-output", default="benchmark_ops.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if "sdaa" not in paddle.device.get_all_custom_device_type():
        raise RuntimeError("SDAA plugin is not loaded; source the SDAA environment first")

    ops = registered_sdaa_ops()
    if args.limit:
        ops = ops[: args.limit]
    rows = []
    for index, name in enumerate(ops, 1):
        row = {"op": name, "status": "skipped", "cpu_ms": "", "sdaa_ms": "", "speedup_cpu_over_sdaa": "", "error": ""}
        try:
            cpu_call = make_call(name, "cpu")
            sdaa_call = make_call(name, "sdaa")
            if cpu_call is None or sdaa_call is None:
                row["error"] = "no generic benchmark adapter"
            else:
                cpu_median, _ = benchmark(cpu_call, "cpu", args.warmup, args.repeat)
                sdaa_median, _ = benchmark(sdaa_call, "sdaa", args.warmup, args.repeat)
                row.update({
                    "status": "ok",
                    "cpu_ms": round(cpu_median, 6),
                    "sdaa_ms": round(sdaa_median, 6),
                    "speedup_cpu_over_sdaa": round(cpu_median / sdaa_median, 6) if sdaa_median else "",
                })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        rows.append(row)
        print(f"[{index}/{len(ops)}] {name}: {row['status']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    Path(args.json_output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"completed: {ok} benchmarked, {len(rows) - ok} skipped/failed")
    print(f"csv: {output}")
    print(f"json: {args.json_output}")


if __name__ == "__main__":
    main()
