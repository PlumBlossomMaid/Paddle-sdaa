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
BUILD_DIR="${SDAA_ROOT_DIR}/build"
PADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR:-"${SDAA_ROOT_DIR}/../Paddle"}
MAX_JOBS=${MAX_JOBS:-16}

print_usage() {
  cat <<'EOF'
Usage: tools/build.sh [OPTIONS]

Options:
  -a, --all       Configure, build, and install Paddle-sdaa
  -b, --build     Configure and build Paddle-sdaa
  -i, --install   Install the latest built Paddle-sdaa wheel
  -t, --test      Run the registered CTest suite
  -s, --single    Run one CTest target, e.g. -s test_MNIST_model
  -c, --clean     Remove the build directory
  -h, --help      Show this message

Environment:
  PADDLE_SOURCE_DIR   Paddle source tree path (default: ../Paddle)
  MAX_JOBS            Build parallelism (default: 16)
EOF
}

ensure_teco_env() {
  export WITH_TESTING=${WITH_TESTING:-ON}
  export SDAA_ROOT=${SDAA_ROOT:-/opt/tecoai}
  export SDPTI_ROOT=${SDPTI_ROOT:-/opt/tecoai}
  export TECODNN_ROOT=${TECODNN_ROOT:-/opt/tecoai}
  export EXTEND_OP_ROOT=${EXTEND_OP_ROOT:-/opt/tecoai/extend}
  export TBLAS_ROOT=${TBLAS_ROOT:-/opt/tecoai}
  export TCCL_ROOT=${TCCL_ROOT:-/opt/tecoai}
  export TECOCUSTOM_EXT_ROOT=${TECOCUSTOM_EXT_ROOT:-/opt/tecoai}
  export TECOCUSTOM_ROOT=${TECOCUSTOM_ROOT:-/opt/tecoai}
  export PADDLE_SOURCE_DIR
}

configure() {
  ensure_teco_env
  mkdir -p "${BUILD_DIR}"
  cmake -S "${SDAA_ROOT_DIR}" -B "${BUILD_DIR}" \
    -DNATIVE_SDAA=ON \
    -DPython_EXECUTABLE="$(which python3)" \
    -DWITH_TESTING="${WITH_TESTING}" \
    -DWITH_MKLDNN=OFF \
    -DWITH_LOONGARCH=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-error -w" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

build() {
  configure
  make -C "${BUILD_DIR}" -j"${MAX_JOBS}"
}

install_wheel() {
  local wheel
  wheel=$(ls -t "${BUILD_DIR}"/dist/paddle_sdaa-*.whl 2>/dev/null | head -1 || true)
  if [[ -z "${wheel}" ]]; then
    echo "ERROR: no paddle_sdaa wheel found under ${BUILD_DIR}/dist" >&2
    exit 1
  fi
  python -m pip install --force-reinstall --no-deps "${wheel}"
}

run_tests() {
  ctest --test-dir "${BUILD_DIR}" --output-on-failure -j 1
}

run_single_test() {
  local target=${1:-}
  if [[ -z "${target}" ]]; then
    echo "ERROR: --single requires a CTest target name" >&2
    exit 1
  fi
  ctest --test-dir "${BUILD_DIR}" -R "^${target}$" --output-on-failure
}

clean() {
  rm -rf "${BUILD_DIR}"
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--all)
      build
      install_wheel
      shift
      ;;
    -b|--build)
      build
      shift
      ;;
    -i|--install)
      install_wheel
      shift
      ;;
    -t|--test)
      run_tests
      shift
      ;;
    -s|--single)
      run_single_test "${2:-}"
      shift 2
      ;;
    -c|--clean)
      clean
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option $1" >&2
      print_usage
      exit 1
      ;;
  esac
done
