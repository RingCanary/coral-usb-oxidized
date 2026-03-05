#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/bootstrap_edgetpu_compiler.sh <command> [options]

Commands:
  install            Download and install edgetpu_compiler from Coral apt repo.
  print-env          Print PATH export for installed compiler.

Options:
  --prefix <dir>     Install prefix (default: $HOME/.local)
  --cache-dir <dir>  Download/extract cache directory
                     (default: $HOME/.cache/coral-usb-oxidized/edgetpu-compiler)
  --version <ver>    Exact compiler version (default: latest in apt index)
  -h, --help         Show this help text

Examples:
  ./tools/bootstrap_edgetpu_compiler.sh install
  ./tools/bootstrap_edgetpu_compiler.sh install --version 16.0
  eval "$(./tools/bootstrap_edgetpu_compiler.sh print-env)"
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
CACHE_DIR="${HOME}/.cache/coral-usb-oxidized/edgetpu-compiler"
VERSION=""

REPO_URL="https://packages.cloud.google.com/apt"
INDEX_PATH="dists/coral-edgetpu-stable/main/binary-amd64/Packages"

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
    --cache-dir)
      [[ $# -ge 2 ]] || die "missing value for --cache-dir"
      CACHE_DIR="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 ]] || die "missing value for --version"
      VERSION="$2"
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

select_package() {
  local index_file="$1"
  local want_version="$2"
  awk -v want="$want_version" '
    BEGIN { RS=""; FS="\n" }
    {
      pkg = ""
      ver = ""
      filename = ""
      sha = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^Package: /) {
          pkg = substr($i, 10)
        } else if ($i ~ /^Version: /) {
          ver = substr($i, 10)
        } else if ($i ~ /^Filename: /) {
          filename = substr($i, 11)
        } else if ($i ~ /^SHA256: /) {
          sha = substr($i, 9)
        }
      }
      if (pkg == "edgetpu-compiler") {
        if (want == "" || ver == want) {
          printf "%s\t%s\t%s\n", ver, filename, sha
          exit
        }
      }
    }
  ' "$index_file"
}

install_compiler() {
  need_cmd curl
  need_cmd awk
  need_cmd sha256sum
  need_cmd ar
  need_cmd tar
  need_cmd install
  need_cmd mktemp

  mkdir -p "${CACHE_DIR}" "${PREFIX}/bin"
  local index_file="${CACHE_DIR}/Packages"
  curl -fsSL "${REPO_URL}/${INDEX_PATH}" -o "${index_file}"

  local selected
  selected="$(select_package "${index_file}" "${VERSION}")"
  [[ -n "${selected}" ]] || die "could not find edgetpu-compiler entry in apt index (version='${VERSION:-latest}')"

  local ver filename sha
  IFS=$'\t' read -r ver filename sha <<<"${selected}"
  [[ -n "${ver}" && -n "${filename}" && -n "${sha}" ]] || die "invalid package metadata parsed from apt index"

  local deb_name
  deb_name="$(basename "${filename}")"
  local deb_path="${CACHE_DIR}/${deb_name}"
  if [[ ! -f "${deb_path}" ]]; then
    curl -fL "${REPO_URL}/${filename}" -o "${deb_path}"
  fi

  local got_sha
  got_sha="$(sha256sum "${deb_path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${sha}" ]] || die "sha256 mismatch for ${deb_name}: expected ${sha}, got ${got_sha}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap "rm -rf '${tmp_dir}'" EXIT

  (
    cd "${tmp_dir}"
    ar x "${deb_path}"
    mkdir -p rootfs
    local data_tar
    data_tar="$(find . -maxdepth 1 -type f -name 'data.tar.*' | head -n 1)"
    [[ -n "${data_tar}" ]] || die "could not locate data.tar.* inside ${deb_name}"
    tar -xf "${data_tar}" -C rootfs
  )

  local src_bin="${tmp_dir}/rootfs/usr/bin/edgetpu_compiler"
  local src_bundle_dir="${tmp_dir}/rootfs/usr/bin/edgetpu_compiler_bin"
  [[ -x "${src_bin}" ]] || die "expected wrapper not found after extraction: ${src_bin}"
  [[ -d "${src_bundle_dir}" ]] || die "expected runtime bundle dir not found: ${src_bundle_dir}"

  local dst_bin="${PREFIX}/bin/edgetpu_compiler"
  local dst_bundle_dir="${PREFIX}/bin/edgetpu_compiler_bin"

  install -m 0755 "${src_bin}" "${dst_bin}"
  rm -rf "${dst_bundle_dir}"
  mkdir -p "${dst_bundle_dir}"
  cp -a "${src_bundle_dir}/." "${dst_bundle_dir}/"
  echo "${ver}" > "${PREFIX}/bin/.edgetpu_compiler.version"

  echo "Installed edgetpu_compiler ${ver} to ${dst_bin}"
  "${dst_bin}" --version
}

print_env() {
  cat <<EOF
export PATH=${PREFIX}/bin:\${PATH}
EOF
}

case "${COMMAND}" in
  install)
    install_compiler
    ;;
  print-env)
    print_env
    ;;
  *)
    die "unknown command: ${COMMAND}"
    ;;
esac
