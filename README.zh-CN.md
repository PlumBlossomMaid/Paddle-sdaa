# Paddle-sdaa

**面向太初 SDAA 加速器的 PaddlePaddle 自定义设备后端。**

Paddle-sdaa 将 PaddlePaddle custom-device 运行时适配到太初 SDAA 硬件，提供 SDAA kernels、运行时集成、Python 扩展、AMP 与分布式通信支持，并针对厂商运行时差异提供兼容处理。

## 特性

- Paddle 3.5 自定义设备后端与 SDAA kernel 注册。
- MNIST 等训练、推理 smoke 流程。
- AMP、XCCL、DDP、tensor/model parallel 和 sharding 支持。
- 针对厂商运行时限制的 FlashAttention fallback。
- 设备信息 API 的包侧集成。

## 仓库结构

```text
Paddle-sdaa/
├── cmake/                  # CMake 辅助逻辑与三方库配置
├── kernels/                # SDAA custom-device kernels
├── runtime/                # custom-device runtime 实现
├── sdaa_ext/               # Python 扩展包与 custom ops
├── sdaac_ops/              # SDAA custom op 源码
├── tests/                  # 单测、runtime、MNIST 与分布式测试
├── tools/build.sh          # Paddle + SDAA 统一构建入口
├── compile.sh              # 旧版仅构建 SDAA 入口
└── pr_ci_sdaa.sh           # SDAA 本地 CI 脚本
```

## 环境要求

构建需要太初 SDAA 开发环境：

| 组件 | 要求 |
| --- | --- |
| Paddle 源码 | Paddle 3.5 源码树，默认位于 `../Paddle` |
| 太初软件栈 | `/opt/tecoai`，包括 SDAA runtime、Tecodnn、Tecoblas、Tccl、Tecocustom、SDPTI |
| 硬件 | SDAA 设备，例如 `/dev/tcaicard*` |
| Python | Paddle 源码构建支持的 CPython 3.11 或更高版本 |
| 构建工具 | CMake、Ninja、GCC、Git |
| Rust 工具链 | 当 `safetensors` 或其他第三方依赖需要源码编译时，需要 Rust/Cargo |

### 源码构建时安装 Rust

部分平台没有可用的 `safetensors` 预编译 wheel。这时 pip 会回退到使用 `maturin` 编译，而 `maturin` 需要 Rust 编译器和 Cargo。建议在安装构建依赖前先安装 Rust：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

然后安装 Python 构建依赖：

```bash
python3 -m pip install -U pybind11 wheel ninja
python3 -m pip install safetensors
```

如果系统包管理器已经提供 `rustc` 和 `cargo`，只需确认它们已经位于 `PATH`，不必重复使用 rustup 安装。只有在 pip 找不到兼容的预编译 wheel 时才需要 Rust。

将 Paddle 源码克隆到本仓库旁边，然后运行统一构建脚本：

```bash
git clone https://github.com/PaddlePaddle/Paddle.git ../Paddle
bash tools/build.sh --all
```

`--all` 会依次执行：

1. 加载 `/opt/tecoai/setvars.sh`。
2. 初始化 Paddle 子模块并构建 LoongArch64 CPU Paddle wheel。
3. 将 Paddle wheel 安装到当前 Python 环境。
4. 在缺少 `sdcops.h` 时下载 SDAA 扩展算子包。
5. 构建并安装 `paddle_sdaa` wheel。

常用模式：

```bash
bash tools/build.sh --all          # 编译 Paddle 和 paddle_sdaa
bash tools/build.sh --paddle       # 只编译 Paddle
bash tools/build.sh --paddle_sdaa  # 只编译 paddle_sdaa
bash tools/build.sh --test         # 运行全部 CTest
bash tools/build.sh --single_test test_MNIST_model
bash tools/build.sh --clean
```

需要覆盖路径或并行度时：

```bash
PADDLE_SOURCE_DIR=/path/to/Paddle MAX_JOBS=32 bash tools/build.sh --all
```

脚本要求在构建 Paddle 前准备好 `numpy`、`protobuf`、`Pillow` 和 `safetensors`。脚本不会修改或重置 Paddle 源码仓库。

## 快速检查

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

预期输出包括：

```text
3.5.0
['sdaa']
Place(sdaa:0) 3.0
```

## 测试

在 SDAA 机器上运行：

```bash
bash tools/build.sh --test
```

当前仓库注册了 194 个 CTest target，覆盖 kernels、runtime、MNIST 训练/推理 smoke、FlashAttention、profiler、设备 API、XCCL、DDP、并行训练和 sharding。

## 兼容性说明

- 设置 `PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1` 可启用 deterministic masked FlashAttention fallback。
- 硬件、runtime 和分布式测试需要 SDAA 硬件及太初软件栈；公开 CI 只运行仓库检查。

## 许可证

Apache License 2.0。
