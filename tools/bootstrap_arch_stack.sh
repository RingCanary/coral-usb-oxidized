#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/bootstrap_arch_stack.sh <command> [options]

Commands:
  build-libedgetpu   Build and install libedgetpu into a local prefix.
  build-tflite-c     Build and install libtensorflowlite_c.so into a local prefix.
  print-env          Print environment exports for this repo to use local libs.

Options:
  --prefix <dir>     Install prefix (default: $HOME/.local)
  --work-dir <dir>   Build work directory (default: /tmp/coral-usb-oxidized-build)
  --tf-version <v>   TensorFlow version for source tarball (default: 2.18.0)
  --py-version <v>   Hermetic Python version for Bazel (default: 3.12)
  -h, --help         Show this help text

Examples:
  ./tools/bootstrap_arch_stack.sh build-libedgetpu
  ./tools/bootstrap_arch_stack.sh build-tflite-c --prefix $HOME/.local
  ./tools/bootstrap_arch_stack.sh build-tflite-c --py-version 3.12
  eval "$(./tools/bootstrap_arch_stack.sh print-env)"
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

PREFIX="${HOME}/.local"
WORK_DIR="/tmp/coral-usb-oxidized-build"
TF_VERSION="2.18.0"
TF_HERMETIC_PYTHON_VERSION="3.12"

if (($# < 1)); then
  usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

COMMAND="$1"
shift

while (($# > 0)); do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || die "missing value for --prefix"
      PREFIX="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "missing value for --work-dir"
      WORK_DIR="$2"
      shift 2
      ;;
    --tf-version)
      [[ $# -ge 2 ]] || die "missing value for --tf-version"
      TF_VERSION="$2"
      shift 2
      ;;
    --py-version)
      [[ $# -ge 2 ]] || die "missing value for --py-version"
      TF_HERMETIC_PYTHON_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

tf_url="https://github.com/tensorflow/tensorflow/archive/refs/tags/v${TF_VERSION}.tar.gz"
mkdir -p "${WORK_DIR}" "${PREFIX}/lib" "${PREFIX}/include"

build_libedgetpu() {
  need_cmd git
  need_cmd curl
  need_cmd tar
  need_cmd patch
  need_cmd make
  need_cmd perl
  need_cmd rg
  need_cmd xxd

  local src_dir="${WORK_DIR}/libedgetpu"
  rm -rf "${src_dir}"
  git clone --depth=1 https://github.com/google-coral/libedgetpu.git "${src_dir}"
  cd "${src_dir}"

  # Keep aligned with the active AUR package commit.
  git checkout e35aed18fea2e2d25d98352e5a5bd357c170bd4d

  curl -L -o "v${TF_VERSION}.tar.gz" "${tf_url}"
  tar -xzf "v${TF_VERSION}.tar.gz"

  curl -L -o makefile.patch "https://aur.archlinux.org/cgit/aur.git/plain/makefile.patch?h=libedgetpu-git"
  curl -L -o usb_device_interface.patch "https://aur.archlinux.org/cgit/aur.git/plain/usb_device_interface.patch?h=libedgetpu-git"
  patch makefile_build/Makefile < makefile.patch
  patch driver/usb/usb_device_interface.h < usb_device_interface.patch

  # Some Debian/Raspberry Pi environments do not ship ld.gold.
  # Fall back to bfd so the libedgetpu link step succeeds on ARM boards.
  if ! command -v ld.gold >/dev/null 2>&1; then
    sed -i 's/-fuse-ld=gold/-fuse-ld=bfd/g' makefile_build/Makefile
  fi

  # Arch currently ships FlatBuffers 25.x while TF 2.18 generated headers assert 24.x.
  rg -l "Non-compatible flatbuffers version included" "tensorflow-${TF_VERSION}/tensorflow" -g '*_generated.h' | while read -r f; do
    perl -0777 -i -pe 's/static_assert\(FLATBUFFERS_VERSION_MAJOR.*?Non-compatible flatbuffers version included"\);/\/\/ patched out flatbuffers version assert for Arch flatbuffers compatibility\n/s' "$f"
  done

  TFROOT="tensorflow-${TF_VERSION}" make -f makefile_build/Makefile -j"$(nproc)" libedgetpu

  local firmware_header
  firmware_header="$(find out -type f -path '*/driver/usb/usb_latest_firmware.h' | head -n 1 || true)"
  [[ -n "${firmware_header}" ]] || die "could not locate generated usb_latest_firmware.h"
  rg -q '0x[0-9a-fA-F]{2}' "${firmware_header}" || die "embedded firmware generation failed (${firmware_header} contains no firmware bytes). Install xxd and rebuild."
  rg -q 'apex_latest_single_ep_len = 0' "${firmware_header}" && die "embedded single-endpoint firmware is empty (${firmware_header}). Install xxd and rebuild."
  rg -q 'apex_latest_multi_ep_len = 0' "${firmware_header}" && die "embedded multi-endpoint firmware is empty (${firmware_header}). Install xxd and rebuild."

  install -Dm755 out/direct/k8/libedgetpu.so.1.0 "${PREFIX}/lib/libedgetpu.so.1.0"
  ln -sf libedgetpu.so.1.0 "${PREFIX}/lib/libedgetpu.so.1"
  ln -sf libedgetpu.so.1 "${PREFIX}/lib/libedgetpu.so"
  install -Dm644 tflite/public/edgetpu.h "${PREFIX}/include/edgetpu.h"
  install -Dm644 tflite/public/edgetpu_c.h "${PREFIX}/include/edgetpu_c.h"

  echo "Installed libedgetpu to ${PREFIX}/lib"
}

build_tflite_c() {
  need_cmd curl
  need_cmd tar
  need_cmd bazelisk

  local src_dir="${WORK_DIR}/tensorflow-${TF_VERSION}"
  rm -rf "${src_dir}"
  cd "${WORK_DIR}"
  curl -L -o "v${TF_VERSION}.tar.gz" "${tf_url}"
  tar -xzf "v${TF_VERSION}.tar.gz"
  cd "${src_dir}"

  # TensorFlow 2.18 requirements_lock files support Python up to 3.12.
  # On bleeding-edge distros where python3 may be 3.14+, force a supported
  # hermetic Python version for repository resolution.
  HERMETIC_PYTHON_VERSION="${TF_HERMETIC_PYTHON_VERSION}" \
  TF_PYTHON_VERSION="${TF_HERMETIC_PYTHON_VERSION}" \
  bazelisk build -c opt \
    --repo_env=HERMETIC_PYTHON_VERSION="${TF_HERMETIC_PYTHON_VERSION}" \
    --repo_env=TF_PYTHON_VERSION="${TF_HERMETIC_PYTHON_VERSION}" \
    --copt=-Wno-error=incompatible-pointer-types \
    --copt=-Wno-incompatible-pointer-types \
    --host_copt=-Wno-error=incompatible-pointer-types \
    --host_copt=-Wno-incompatible-pointer-types \
    //tensorflow/lite/c:libtensorflowlite_c.so
  install -Dm755 bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so "${PREFIX}/lib/libtensorflowlite_c.so"

  echo "Installed libtensorflowlite_c.so to ${PREFIX}/lib"
}

print_env() {
  cat <<EOF
export CORAL_LIB_DIR=${PREFIX}/lib
export EDGETPU_LIB_DIR=${PREFIX}/lib
export TFLITE_LIB_DIR=${PREFIX}/lib
export TFLITE_LINK_LIB=tensorflowlite_c
export LD_LIBRARY_PATH=${PREFIX}/lib:\${LD_LIBRARY_PATH:-}
EOF
}

case "${COMMAND}" in
  build-libedgetpu)
    build_libedgetpu
    ;;
  build-tflite-c)
    build_tflite_c
    ;;
  print-env)
    print_env
    ;;
  *)
    die "unknown command: ${COMMAND}"
    ;;
esac
