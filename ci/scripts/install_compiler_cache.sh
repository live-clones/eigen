#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Provision and verify compiler cache tools (sccache, ccache).
# Pinned release binaries are verified against strict SHA-256 checksums
# on both download and cache restoration to protect against supply-chain tampering.

root_dir="${rootdir:-$(pwd)}"
arch="$(uname -m)"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --connect-timeout 5 -m 30 "${url}" -o "${dest}" && return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q --timeout=30 -O "${dest}" "${url}" && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${url}" "${dest}" && return 0
  fi
  echo "Notice: Neither curl, wget, nor python3 is available to download $(basename "${dest}")." >&2
  return 1
}

export CCACHE_DIR="${CCACHE_DIR:-${EIGEN_CI_CCACHE_DIR}}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-${EIGEN_CI_CCACHE_MAXSIZE}}"
export CCACHE_BASEDIR="${CCACHE_BASEDIR:-${EIGEN_CI_CCACHE_BASEDIR}}"
export CCACHE_COMPRESSLEVEL="${CCACHE_COMPRESSLEVEL:-${EIGEN_CI_CCACHE_COMPRESSLEVEL}}"
for v in CCACHE_DIR CCACHE_MAXSIZE CCACHE_BASEDIR CCACHE_COMPRESSLEVEL; do
  [[ -n "${!v}" ]] || unset "${v}"
done

# 1. Provision sccache
# Install into /usr/local/bin/sccache so the path in build.ninja is identical
# across x86_64 cross-build hosts and aarch64 native test hosts, while keeping
# .sccache-bin/${arch}/sccache in the shared GitLab runner cache.
sccache_bin="$(command -v sccache 2>/dev/null || true)"

if [[ -z "${sccache_bin}" ]]; then
  sccache_ver="0.18.0"
  sccache_bindir="${root_dir}/.sccache-bin/${arch}"
  sccache_tar_sha256=""
  sccache_bin_sha256=""

  case "${arch}" in
    x86_64)
      sccache_tar_sha256="45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89"
      sccache_bin_sha256="973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59"
      ;;
    aarch64)
      sccache_tar_sha256="2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a"
      sccache_bin_sha256="0fd82ece4469b791e30fffdd43d58d2e6cae303950c0d6c8faad932d93e35cf4"
      ;;
  esac

  if [[ -n "${sccache_bin_sha256}" ]]; then
    mkdir -p "${sccache_bindir}"
    cached_sccache="${sccache_bindir}/sccache"
    if [[ ! -x "${cached_sccache}" ]] || ! echo "${sccache_bin_sha256}  ${cached_sccache}" | sha256sum -c - >/dev/null 2>&1; then
      rm -f "${cached_sccache}"
      archive="${sccache_bindir}/sccache.tar.gz"
      if download_file \
        "https://github.com/mozilla/sccache/releases/download/v${sccache_ver}/sccache-v${sccache_ver}-${arch}-unknown-linux-musl.tar.gz" \
        "${archive}"; then
        if echo "${sccache_tar_sha256}  ${archive}" | sha256sum -c - >/dev/null 2>&1; then
          tar -xzf "${archive}" -C "${sccache_bindir}" --strip-components=1 2>/dev/null || true
          chmod +x "${cached_sccache}" 2>/dev/null || true
          if ! echo "${sccache_bin_sha256}  ${cached_sccache}" | sha256sum -c - >/dev/null 2>&1; then
            echo "Warning: Extracted sccache failed SHA-256 verification; discarding." >&2
            rm -f "${cached_sccache}"
          fi
        else
          echo "Warning: Downloaded sccache archive failed SHA-256 verification; discarding." >&2
        fi
        rm -f "${archive}"
      fi
    fi
    if [[ -x "${cached_sccache}" ]]; then
      if cp -f "${cached_sccache}" /usr/local/bin/sccache 2>/dev/null; then
        sccache_bin="/usr/local/bin/sccache"
      else
        sccache_bin="${cached_sccache}"
      fi
    fi
  fi
fi

# 2. Provision ccache
ccache_bin="$(command -v ccache 2>/dev/null || true)"

if [[ -z "${ccache_bin}" ]]; then
  ccache_ver="4.13.6"
  ccache_bindir="${root_dir}/.ccache-bin/${arch}"
  ccache_tar_sha256=""
  ccache_bin_sha256=""

  case "${arch}" in
    x86_64)
      ccache_tar_sha256="09e0547a0c3b250a76675c33130366f1399f3580842fb360c052520d56214ead"
      ccache_bin_sha256="c4ca67395c175798c588ee5425c135e76bee5701add4d0631d8b4be0b161b322"
      ;;
    aarch64)
      ccache_tar_sha256="bff0e0c19165db8627c85c36b0885b3b180659eda67a028298d26565aad52f56"
      ccache_bin_sha256="e678bcfe84a0fea0865307dca3f8ac38f4c881125afea91e3a9397ed3a087666"
      ;;
  esac

  if [[ -n "${ccache_bin_sha256}" ]]; then
    mkdir -p "${ccache_bindir}"
    cached_ccache="${ccache_bindir}/ccache"
    if [[ ! -x "${cached_ccache}" ]] || ! echo "${ccache_bin_sha256}  ${cached_ccache}" | sha256sum -c - >/dev/null 2>&1; then
      rm -f "${cached_ccache}"
      archive="${ccache_bindir}/ccache.tar.gz"
      if download_file \
        "https://github.com/ccache/ccache/releases/download/v${ccache_ver}/ccache-${ccache_ver}-linux-${arch}-musl-static.tar.gz" \
        "${archive}"; then
        if echo "${ccache_tar_sha256}  ${archive}" | sha256sum -c - >/dev/null 2>&1; then
          tar -xzf "${archive}" -C "${ccache_bindir}" --strip-components=1 2>/dev/null || true
          chmod +x "${cached_ccache}" 2>/dev/null || true
          if ! echo "${ccache_bin_sha256}  ${cached_ccache}" | sha256sum -c - >/dev/null 2>&1; then
            echo "Warning: Extracted ccache failed SHA-256 verification; discarding." >&2
            rm -f "${cached_ccache}"
          fi
        else
          echo "Warning: Downloaded ccache archive failed SHA-256 verification; discarding." >&2
        fi
        rm -f "${archive}"
      fi
    fi
    if [[ -x "${cached_ccache}" ]]; then
      if cp -f "${cached_ccache}" /usr/local/bin/ccache 2>/dev/null; then
        ccache_bin="/usr/local/bin/ccache"
      else
        ccache_bin="${cached_ccache}"
      fi
    fi
  fi
fi

# 3. Start compiler cache daemon (sccache with GCS L1, or ccache fallback)
compiler_launcher=""
launchers=""
sccache_cred_server_pid=""

if [[ "${EIGEN_CI_SCCACHE:-on}" != "off" && -n "${sccache_bin}" ]]; then
  export SCCACHE_DIR="${SCCACHE_DIR:-${EIGEN_CI_SCCACHE_DIR:-${CI_PROJECT_DIR:-${root_dir}}/.sccache}}"
  export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-${EIGEN_CI_SCCACHE_CACHE_SIZE:-4G}}"
  export SCCACHE_BASEDIRS="${root_dir}"
  export SCCACHE_SERVER_PORT="$((4226 + (${CI_JOB_ID:-0} % 10000)))"

  # Remote GCS caching via short-lived OAuth token injected from GitLab CI/CD variables:
  # EIGEN_GCS_CACHE_TOKEN_RW (protected branch / master) or EIGEN_GCS_CACHE_TOKEN_RO (MRs)
  _sccache_xtrace=false
  [[ $- == *x* ]] && _sccache_xtrace=true
  { set +x; } 2>/dev/null
  if [[ -n "${EIGEN_GCS_CACHE_TOKEN_RW:-}" || -n "${EIGEN_GCS_CACHE_TOKEN_RO:-}" ]]; then
    export SCCACHE_GCS_BUCKET="${EIGEN_CI_SCCACHE_GCS_BUCKET:-eigen-gitlab-ci-cache}"
    export SCCACHE_MULTILEVEL_CHAIN="disk,gcs"
    if [[ -n "${EIGEN_GCS_CACHE_TOKEN_RW:-}" ]]; then
      export SCCACHE_GCS_RW_MODE="READ_WRITE"
    else
      export SCCACHE_GCS_RW_MODE="READ_ONLY"
    fi
    export GCS_URL_SECRET=$(od -vN 16 -An -tx1 /dev/urandom | tr -d ' \n')
    cred_port_file="${PWD}/.cred_port"

    python3 -c "
import http.server, json, sys, os
token = os.environ.get('EIGEN_GCS_CACHE_TOKEN_RW') or os.environ.get('EIGEN_GCS_CACHE_TOKEN_RO')
secret = os.environ.get('GCS_URL_SECRET')
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/token/' + secret:
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'accessToken': token}).encode())
    def log_message(self, *a): pass
server = http.server.HTTPServer(('127.0.0.1', 0), H)
with open('${cred_port_file}.tmp', 'w') as f:
    f.write(str(server.server_port))
os.replace('${cred_port_file}.tmp', '${cred_port_file}')
server.serve_forever()
" &
    sccache_cred_server_pid=$!
    trap '[[ -n "${sccache_cred_server_pid}" ]] && kill "${sccache_cred_server_pid}" 2>/dev/null || true' EXIT

    # Wait up to 2 seconds for local credential server to bind and write its port
    cred_port=""
    for _ in {1..40}; do
      if [[ -s "${cred_port_file}" ]]; then
        cred_port=$(cat "${cred_port_file}")
        break
      fi
      sleep 0.05
    done
    if [[ -n "${cred_port:-}" ]]; then
      export SCCACHE_GCS_CREDENTIALS_URL="http://127.0.0.1:${cred_port}/token/${GCS_URL_SECRET}"
    else
      echo "Notice: Local credential server failed to bind or respond; skipping GCS remote cache." >&2
      unset SCCACHE_GCS_BUCKET
      unset SCCACHE_GCS_RW_MODE
      unset SCCACHE_MULTILEVEL_CHAIN
      [[ -n "${sccache_cred_server_pid}" ]] && kill "${sccache_cred_server_pid}" 2>/dev/null || true
      sccache_cred_server_pid=""
    fi
    rm -f "${cred_port_file}" "${cred_port_file}.tmp"
    unset GCS_URL_SECRET
  fi
  [[ "${_sccache_xtrace}" == "true" ]] && set -x
  unset _sccache_xtrace

  if "${sccache_bin}" --start-server >/dev/null 2>&1; then
    compiler_launcher="sccache"
    "${sccache_bin}" --zero-stats >/dev/null 2>&1 || true
    launchers="-DCMAKE_C_COMPILER_LAUNCHER=${sccache_bin} -DCMAKE_CXX_COMPILER_LAUNCHER=${sccache_bin}"
  else
    echo "Notice: sccache server failed to start (check GCS credentials/network); falling back to ccache."
    [[ -n "${sccache_cred_server_pid}" ]] && kill "${sccache_cred_server_pid}" 2>/dev/null || true
    sccache_cred_server_pid=""
  fi
fi

if [[ -z "${compiler_launcher}" && -n "${ccache_bin}" ]]; then
  compiler_launcher="ccache"
  launchers="-DCMAKE_C_COMPILER_LAUNCHER=${ccache_bin} -DCMAKE_CXX_COMPILER_LAUNCHER=${ccache_bin}"
  # Log stats per job via CCACHE_STATSLOG rather than global --zero-stats /
  # --show-stats so concurrent jobs sharing a host CCACHE_DIR do not reset or
  # mix each other's counters. Fall back to global counters if ccache predates
  # --show-log-stats (ccache < 4.4, e.g. Ubuntu 20.04).
  export CCACHE_STATSLOG="${PWD}/ccache-stats.log"
  rm -f "${CCACHE_STATSLOG}"
  if ! "${ccache_bin}" --show-log-stats >/dev/null 2>&1; then
    unset CCACHE_STATSLOG
    "${ccache_bin}" --zero-stats
  fi
fi

show_ccache_stats() {
  if [[ "${compiler_launcher}" == "sccache" && -n "${sccache_bin}" ]]; then
    "${sccache_bin}" --show-stats 2>&1 || true
    "${sccache_bin}" --stop-server >/dev/null 2>&1 || true
    if [[ -n "${sccache_cred_server_pid}" ]]; then
      kill "${sccache_cred_server_pid}" 2>/dev/null || true
      sccache_cred_server_pid=""
    fi
    # Incompatible cache formats: purge unused .ccache/ before cache upload so the
    # uploaded cache archive only holds .sccache/ and stays strictly below the 5 GB runner cap.
    rm -rf "${root_dir}/.ccache"
  elif [[ "${compiler_launcher}" == "ccache" && -n "${ccache_bin}" ]]; then
    if [[ -n "${CCACHE_STATSLOG}" ]]; then
      "${ccache_bin}" --show-log-stats
      rm -f "${CCACHE_STATSLOG}"
    else
      "${ccache_bin}" --show-stats
    fi
    # Incompatible cache formats: purge unused .sccache/ before cache upload.
    rm -rf "${root_dir}/.sccache"
  fi
}
