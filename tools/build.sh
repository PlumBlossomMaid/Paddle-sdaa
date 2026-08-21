#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SDAA_ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR:-"${SDAA_ROOT_DIR}/../Paddle"}
PADDLE_BUILD_DIR=${PADDLE_BUILD_DIR:-"${PADDLE_SOURCE_DIR}/build"}
SDAA_BUILD_DIR=${SDAA_BUILD_DIR:-"${SDAA_ROOT_DIR}/build"}
MAX_JOBS=${MAX_JOBS:-16}
PYTHON=${PYTHON:-python3}

usage() {
  cat <<'EOF'
Usage: tools/build.sh [OPTIONS]

Options:
  -a, --all             Build and install Paddle, then build and install SDAA
  -p, --paddle          Build and install Paddle from PADDLE_SOURCE_DIR
  -m, --paddle_sdaa     Build and install the SDAA plugin
  -b, --build           Build the SDAA plugin without installing its wheel
  -i, --install         Install the latest built SDAA wheel
  -t, --test            Run the registered CTest suite
  -s, --single_test     Run one CTest target
  -c, --clean           Remove Paddle and SDAA build directories
  -h, --help            Show this message

Environment:
  PADDLE_SOURCE_DIR  Paddle source tree (default: ../Paddle)
  PADDLE_BUILD_DIR   Paddle build directory (default: $PADDLE_SOURCE_DIR/build)
  SDAA_BUILD_DIR     SDAA build directory (default: ./build)
  PYTHON             Python executable (default: python3)
  MAX_JOBS           Build parallelism (default: 16)
  WITH_TESTING       Build CTest targets (default: ON)
  PADDLE_REF         Optional Paddle git ref to check out before building

The --all command requires /opt/tecoai/setvars.sh and an SDAA runtime.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: required command not found: $1" >&2; exit 1; }
}

require_python_module() {
  "${PYTHON}" -c "import $1" >/dev/null 2>&1 || {
    echo "ERROR: Python module '$1' is required. Install Paddle build requirements first." >&2
    exit 1
  }
}

source_teco_env() {
  if [[ -f "${TECOAI_SETENV:-/opt/tecoai/setvars.sh}" ]]; then
    source "${TECOAI_SETENV:-/opt/tecoai/setvars.sh}"
  else
    echo "ERROR: SDAA environment script not found: ${TECOAI_SETENV:-/opt/tecoai/setvars.sh}" >&2
    exit 1
  fi
  export SDAA_ROOT=${SDAA_ROOT:-/opt/tecoai}
  export SDPTI_ROOT=${SDPTI_ROOT:-/opt/tecoai}
  export TECODNN_ROOT=${TECODNN_ROOT:-/opt/tecoai}
  export EXTEND_OP_ROOT=${EXTEND_OP_ROOT:-/opt/tecoai/extend}
  export TBLAS_ROOT=${TBLAS_ROOT:-/opt/tecoai}
  export TCCL_ROOT=${TCCL_ROOT:-/opt/tecoai}
  export TECOCUSTOM_EXT_ROOT=${TECOCUSTOM_EXT_ROOT:-/opt/tecoai}
  export TECOCUSTOM_ROOT=${TECOCUSTOM_ROOT:-/opt/tecoai}
}

paddle_wheel() {
  find "${PADDLE_BUILD_DIR}/python/dist" -maxdepth 1 -type f -name 'paddlepaddle-*.whl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

sdaa_wheel() {
  find "${SDAA_BUILD_DIR}/dist" -maxdepth 1 -type f -name 'paddle_sdaa-*.whl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

prepare_extend_ops() {
  if [[ -f "${EXTEND_OP_ROOT}/include/sdcops.h" ]]; then
    return
  fi
  require_command wget
  local archive=/tmp/paddle-sdaa-extend.tar.gz
  wget -q --no-check-certificate -O "${archive}" \
    "${EXTEND_OP_URL:-https://gitee.com/tecorigin/sdcops/raw/develop/extend_4a1cc3e_1.4.0b0.tar.gz}"
  tar -zxf "${archive}" -C "${SDAA_ROOT:-/opt/tecoai}"
  [[ -f "${EXTEND_OP_ROOT}/include/sdcops.h" ]] || {
    echo "ERROR: sdcops.h was not installed under ${EXTEND_OP_ROOT}/include" >&2
    exit 1
  }
}

build_paddle() {
  source_teco_env
  require_command cmake
  require_command ninja
  require_python_module numpy
  require_python_module protobuf
  require_python_module PIL
  require_python_module safetensors
  [[ -f "${PADDLE_SOURCE_DIR}/CMakeLists.txt" ]] || {
    echo "ERROR: Paddle source not found: ${PADDLE_SOURCE_DIR}" >&2
    exit 1
  }
  if [[ -n "${PADDLE_REF:-}" ]]; then
    git -C "${PADDLE_SOURCE_DIR}" checkout "${PADDLE_REF}"
  fi
  git -C "${PADDLE_SOURCE_DIR}" submodule update --init --recursive
  if ! git -C "${PADDLE_SOURCE_DIR}" describe --exact-match --tags HEAD >/dev/null 2>&1; then
    git -C "${PADDLE_SOURCE_DIR}" tag -f "v${PADDLE_VERSION:-3.5.0}" HEAD
  fi
  mkdir -p "${PADDLE_BUILD_DIR}"
  cmake -S "${PADDLE_SOURCE_DIR}" -B "${PADDLE_BUILD_DIR}" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPY_VERSION="$(${PYTHON} -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    -DPADDLE_VERSION=${PADDLE_VERSION:-3.5.0} \
    -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_ONEDNN=OFF -DWITH_AVX=OFF \
    -DWITH_DISTRIBUTE=OFF -DWITH_CINN=OFF -DWITH_TESTING=OFF -DWITH_CPP_TEST=OFF \
    -DWITH_PYTHON=ON -DWITH_SHARED_PHI=ON -DWITH_LOONGARCH=ON \
    -DWITH_INFERENCE_API_TEST=OFF -DWITH_ONNXRUNTIME=OFF -DWITH_GLOO=OFF \
    -DWITH_STYLE_CHECK=OFF -DWITH_UNITY_BUILD=OFF \
    -DCMAKE_INSTALL_PREFIX="${PADDLE_BUILD_DIR}"
  ninja -C "${PADDLE_BUILD_DIR}" -j"${MAX_JOBS}"
  local wheel
  wheel=$(paddle_wheel)
  [[ -n "${wheel}" ]] || { echo "ERROR: Paddle wheel was not generated" >&2; exit 1; }
  "${PYTHON}" -m pip install --force-reinstall --no-deps "${wheel}"
}

configure_sdaa() {
  source_teco_env
  prepare_extend_ops
  export WITH_TESTING=${WITH_TESTING:-ON}
  export PADDLE_SOURCE_DIR
  export PADDLE_CUSTOM_PATH=${PADDLE_CUSTOM_PATH:-$(${PYTHON} -c 'import paddle, os; print(os.path.dirname(paddle.__file__))')}
  cmake -S "${SDAA_ROOT_DIR}" -B "${SDAA_BUILD_DIR}" \
    -DNATIVE_SDAA=ON -DPython_EXECUTABLE="$(command -v "${PYTHON}")" \
    -DWITH_TESTING="${WITH_TESTING}" -DWITH_MKLDNN=OFF -DWITH_LOONGARCH=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error -w" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

build_sdaa() {
  configure_sdaa
  cmake --build "${SDAA_BUILD_DIR}" --parallel "${MAX_JOBS}"
}

install_sdaa() {
  local wheel
  wheel=$(sdaa_wheel)
  [[ -n "${wheel}" ]] || { echo "ERROR: no SDAA wheel under ${SDAA_BUILD_DIR}/dist" >&2; exit 1; }
  "${PYTHON}" -m pip install --force-reinstall --no-deps "${wheel}"
}

run_tests() { ctest --test-dir "${SDAA_BUILD_DIR}" --output-on-failure -j 1; }
run_single() {
  [[ -n "${1:-}" ]] || { echo "ERROR: --single requires a CTest target" >&2; exit 1; }
  ctest --test-dir "${SDAA_BUILD_DIR}" -R "^$1$" --output-on-failure
}
clean() { rm -rf "${PADDLE_BUILD_DIR}" "${SDAA_BUILD_DIR}"; }

[[ $# -gt 0 ]] || { usage; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--all) build_paddle; build_sdaa; install_sdaa; shift ;;
    -p|--paddle) build_paddle; shift ;;
    -m|--paddle_sdaa|-s|--sdaa) build_sdaa; install_sdaa; shift ;;
    -b|--build) build_sdaa; shift ;;
    -i|--install) install_sdaa; shift ;;
    -t|--test) run_tests; shift ;;
    -1|--single|--single_test) run_single "${2:-}"; shift 2 ;;
    -c|--clean) clean; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
