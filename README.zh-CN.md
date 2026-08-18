[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.5-blue.svg)](https://github.com/PaddlePaddle/Paddle)
[![SDAA](https://img.shields.io/badge/Tecorigin-SDAA-green.svg)](https://www.tecorigin.com/)
[![Tests](https://img.shields.io/badge/CTest-194%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-sdaa

**面向太初 SDAA 加速器的 PaddlePaddle 自定义设备后端。**

Paddle-sdaa 将 PaddlePaddle 的 custom-device 运行时适配到太初 SDAA 硬件，提供 SDAA kernel 注册、运行时集成、Python 包侧补丁、训练/推理 smoke 覆盖、AMP 与分布式通信支持，并针对厂商运行时边界问题提供稳定 fallback。

## 特性

- **Paddle Custom Device** -- 将 SDAA 注册为 Paddle 自定义后端，当前包含 269 个 custom kernels。
- **Paddle 3.5 兼容** -- 适配 Paddle 3.5 的属性 ABI 变化，例如 `epsilon` / `porder` 的 `double` 类型。
- **训练与推理** -- 提供有界 MNIST smoke 流程，覆盖训练、导出、推理和清理。
- **AMP 覆盖** -- 验证高性能卷积、pipeline parallel smoke 等 AMP 路径。
- **分布式支持** -- 覆盖 XCCL 通信流、DDP optimizer、pipeline parallel、tensor/model parallel 以及 sharding stage 2/3。
- **FlashAttention fallback** -- no-mask backward 使用 SDAA fallback；masked forward/backward 提供可选 deterministic fallback，应对厂商运行时限制。
- **设备信息 API** -- 通过包侧补丁支持 `paddle.device.get_device_properties()`、`get_device_capability()`、`get_device_name()`，不修改 Paddle 源码。
- **开源 CI 友好** -- GitHub Actions 只运行 pre-commit，因为公开 runner 没有 SDAA 硬件和太初运行时库。

## 仓库结构

```text
Paddle-sdaa/
├── cmake/                  # CMake 辅助逻辑与三方库配置
├── dynload/                # 动态库加载辅助
├── external/               # 外部 SDAA stream 头文件
├── kernels/                # Paddle custom-device SDAA kernels
├── runtime/                # custom-device runtime 实现
├── sdaa_ext/               # Python 扩展包与高性能 custom ops
├── sdaac_ops/              # SDAAC custom op 源码
├── tests/                  # 单测、runtime、MNIST 与分布式测试
├── tools/version/          # runtime / 软件栈版本查询工具
├── compile.sh              # 构建入口
└── pr_ci_sdaa.sh           # SDAA 机器上的本地 CI 脚本
```

## 环境要求

公开 GitHub CI 只检查仓库规范。构建与运行后端需要本地 SDAA 环境：

| 组件 | 说明 |
| --- | --- |
| PaddlePaddle | 适配时使用 Paddle 3.5 开发版 |
| 太初软件栈 | SDAA runtime、driver、Tecodnn、Tecoblas、Tccl、Tecocustom、SDPTI |
| 硬件 | 太初 SDAA 设备，例如 `/dev/tcaicard*` |
| Python | 本地验证环境使用 CPython 3.12 |
| CMake | 原生构建需要 |

## 构建

```bash
git clone https://github.com/PlumBlossomMaid/Paddle-sdaa.git
cd Paddle-sdaa

source /opt/tecoai/setvars.sh
export PADDLE_SOURCE_DIR=/path/to/Paddle

bash compile.sh
python -m pip install --force-reinstall --no-deps build/dist/*.whl
```

> SDAA 包不锁定 NumPy 版本。NumPy 兼容性由 Paddle 主框架包负责，而不是后端插件负责。

## 快速检查

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

预期输出形态：

```text
['sdaa']
_customDeviceProperties(name='/dev/tcaicard0', major=61440, minor=256, total_memory=15296MB, multi_processor_count=0)
(61440, 256)
/dev/tcaicard0
Tensor(shape=[2], dtype=float32, place=Place(sdaa:0), stop_gradient=True,
       [0., 1.])
```

## 测试

在 SDAA 机器上运行：

```bash
ctest --test-dir build --output-on-failure -j 1
```

当前本地验证状态：

- **194 / 194** 个注册 CTest target 已按编号分批执行并通过。
- 没有测试隐藏在 `unittest.skip`、`skipTest` 或 `pytest.mark.skip` 后。
- 覆盖常用算子、MNIST smoke 训练/推理、FlashAttention、runtime/profiler、设备 API、通信流、DDP optimizer、sharding stage 2/3、pipeline parallel 等。

## 兼容性说明

- `PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1` 可启用 deterministic SDAA masked FlashAttention fallback，用于厂商 Tecocustom masked 路径数值不适合某些 workload 的情况。
- `paddle.device.get_device_properties()` 从 `sdaaGetDeviceProperties` 和 `sdaaMemGetInfo` 读取真实 runtime 信息。`multi_processor_count` 为 `0`，因为当前 SDAA runtime 暴露结构中没有等价字段。
- 公开 GitHub CI 无法验证 SDAA runtime 行为，因为托管 runner 没有 SDAA 设备和太初运行时库。

## CI

本仓库在 GitHub 上只运行 pre-commit：

```bash
pre-commit run --all-files --show-diff-on-failure
```

硬件、runtime 和分布式测试必须在 SDAA 机器上运行。

## 许可证

Apache License 2.0。
