[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.5-blue.svg)](https://github.com/PaddlePaddle/Paddle)
[![SDAA](https://img.shields.io/badge/Tecorigin-SDAA-green.svg)](https://www.tecorigin.com/)
[![Tests](https://img.shields.io/badge/CTest-194%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Paddle-sdaa

**面向太初 SDAA 加速器的 PaddlePaddle 自定義裝置後端。**

Paddle-sdaa 將 PaddlePaddle 的 custom-device 執行時適配到太初 SDAA 硬體，提供 SDAA kernel 註冊、執行時整合、Python 套件側補丁、訓練/推理 smoke 覆蓋、AMP 與分散式通訊支援，並針對廠商執行時邊界問題提供穩定 fallback。

## 特性

- **Paddle Custom Device** -- 將 SDAA 註冊為 Paddle 自定義後端，當前包含 269 個 custom kernels。
- **Paddle 3.5 相容** -- 適配 Paddle 3.5 的屬性 ABI 變化，例如 `epsilon` / `porder` 的 `double` 型別。
- **訓練與推理** -- 提供有界 MNIST smoke 流程，覆蓋訓練、匯出、推理和清理。
- **AMP 覆蓋** -- 驗證高效能卷積、pipeline parallel smoke 等 AMP 路徑。
- **分散式支援** -- 覆蓋 XCCL 通訊流、DDP optimizer、pipeline parallel、tensor/model parallel 以及 sharding stage 2/3。
- **FlashAttention fallback** -- no-mask backward 使用 SDAA fallback；masked forward/backward 提供可選 deterministic fallback，應對廠商執行時限制。
- **裝置資訊 API** -- 透過套件側補丁支援 `paddle.device.get_device_properties()`、`get_device_capability()`、`get_device_name()`，不修改 Paddle 原始碼。
- **開源 CI 友好** -- GitHub Actions 只執行 pre-commit，因為公開 runner 沒有 SDAA 硬體和太初執行時函式庫。

## 倉庫結構

```text
Paddle-sdaa/
├── cmake/                  # CMake 輔助邏輯與第三方函式庫配置
├── dynload/                # 動態函式庫載入輔助
├── external/               # 外部 SDAA stream 標頭檔
├── docs/                   # 狀態與 fallback 文件
├── kernels/                # Paddle custom-device SDAA kernels
├── patches/                # patch 策略文件
├── runtime/                # custom-device runtime 實作
├── sdaa_ext/               # Python 擴充套件與高效能 custom ops
├── sdaac_ops/              # SDAAC custom op 原始碼
├── tests/                  # 單元測試、runtime、MNIST 與分散式測試
├── tools/build.sh          # 建置、安裝、測試和清理輔助腳本
├── tools/version/          # runtime / 軟體棧版本查詢工具
├── compile.sh              # 舊版建置入口
└── pr_ci_sdaa.sh           # SDAA 機器上的本地 CI 腳本
```

## 環境需求

公開 GitHub CI 只檢查倉庫規範。建置與執行後端需要本地 SDAA 環境：

| 元件 | 說明 |
| --- | --- |
| PaddlePaddle | 適配時使用 Paddle 3.5 開發版 |
| 太初軟體棧 | SDAA runtime、driver、Tecodnn、Tecoblas、Tccl、Tecocustom、SDPTI |
| 硬體 | 太初 SDAA 裝置，例如 `/dev/tcaicard*` |
| Python | 本地驗證環境使用 CPython 3.12 |
| CMake | 原生建置需要 |

## 建置

```bash
git clone https://github.com/PlumBlossomMaid/Paddle-sdaa.git
cd Paddle-sdaa

source /opt/tecoai/setvars.sh
export PADDLE_SOURCE_DIR=/path/to/Paddle

bash tools/build.sh --all
```

> SDAA 套件不鎖定 NumPy 版本。NumPy 相容性由 Paddle 主框架套件負責，而不是後端插件負責。

## 快速檢查

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

預期輸出形態：

```text
['sdaa']
_customDeviceProperties(name='/dev/tcaicard0', major=61440, minor=256, total_memory=15296MB, multi_processor_count=0)
(61440, 256)
/dev/tcaicard0
Tensor(shape=[2], dtype=float32, place=Place(sdaa:0), stop_gradient=True,
       [0., 1.])
```

## 測試

在 SDAA 機器上執行：

```bash
ctest --test-dir build --output-on-failure -j 1
```

當前本地驗證狀態：

- **194 / 194** 個註冊 CTest target 已按編號分批執行並通過。
- 沒有測試隱藏在 `unittest.skip`、`skipTest` 或 `pytest.mark.skip` 後。
- 覆蓋常用算子、MNIST smoke 訓練/推理、FlashAttention、runtime/profiler、裝置 API、通訊流、DDP optimizer、sharding stage 2/3、pipeline parallel 等。

## 相容性說明

- `PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK=1` 可啟用 deterministic SDAA masked FlashAttention fallback，用於廠商 Tecocustom masked 路徑數值不適合某些 workload 的情況。
- `paddle.device.get_device_properties()` 從 `sdaaGetDeviceProperties` 和 `sdaaMemGetInfo` 讀取真實 runtime 資訊。`multi_processor_count` 為 `0`，因為當前 SDAA runtime 暴露結構中沒有等價欄位。
- 公開 GitHub CI 無法驗證 SDAA runtime 行為，因為託管 runner 沒有 SDAA 裝置和太初執行時函式庫。

## CI

本倉庫在 GitHub 上只執行 pre-commit：

```bash
pre-commit run --all-files --show-diff-on-failure
```

硬體、runtime 和分散式測試必須在 SDAA 機器上執行。

## 授權

Apache License 2.0。
