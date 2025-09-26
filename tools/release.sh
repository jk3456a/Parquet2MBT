#!/usr/bin/env bash
set -euo pipefail

APP_NAME="parquet2mbt"
TARGET_TRIPLE="x86_64-unknown-linux-musl"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

usage() {
  echo "Usage: $0 [version|vX.Y.Z]" >&2
}

verify_clean() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree not clean. Commit or stash changes first." >&2
    exit 1
  fi
}

get_version() {
  if [[ ${1-} =~ ^v?[0-9]+(\.[0-9]+)*([.-][A-Za-z0-9._]+)?$ ]]; then
    local v="$1"; v="${v#v}"; echo "$v"; return 0
  fi
  awk '
    /^\[package\]/{in_pkg=1; next}
    /^\[/{in_pkg=0}
    in_pkg && /^version[[:space:]]*=/{
      gsub(/"/,""); gsub(/[[:space:]]*/,""); split($0,a,"="); print a[2]; exit
    }
  ' Cargo.toml
}

ensure_tools() {
  if ! rustup target list --installed | grep -q "^${TARGET_TRIPLE}$"; then
    rustup target add "$TARGET_TRIPLE"
  fi
  if ! command -v musl-gcc >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
      sudo apt-get update -y && sudo apt-get install -y musl-tools
    else
      echo "musl-gcc not found. Please install musl-tools (e.g., sudo apt-get install musl-tools)." >&2
      exit 1
    fi
  fi
}

build_musl() {
  export PKG_CONFIG_ALLOW_CROSS=1
  export OPENSSL_STATIC=1
  cargo build --release --target "$TARGET_TRIPLE"
  local bin="target/${TARGET_TRIPLE}/release/${APP_NAME}"
  if [[ ! -f "$bin" ]]; then
    echo "Build failed: $bin not found" >&2
    exit 1
  fi
  if command -v llvm-strip >/dev/null 2>&1; then
    llvm-strip -s "$bin" || true
  elif command -v strip >/dev/null 2>&1; then
    strip -s "$bin" || true
  fi
  if ! file "$bin" | grep -qi 'statically linked'; then
    echo "Error: binary is not statically linked." >&2
    file "$bin" || true
    ldd "$bin" || true
    exit 1
  fi
}

package_artifacts() {
  local version="$1"
  local dist="dist"
  mkdir -p "$dist"
  local base="${APP_NAME}-${version}-${TARGET_TRIPLE}"
  cp "target/${TARGET_TRIPLE}/release/${APP_NAME}" "${dist}/${base}"
  (cd "$dist" && sha256sum "$base" > "${base}.sha256")
  tar -C "$dist" -czf "${dist}/${base}.tar.gz" "$base"
}

push_git() {
  local version="$1"
  local tag="v${version}"
  if git rev-parse "$tag" >/dev/null 2>&1; then
    echo "Tag $tag already exists. Aborting." >&2
    exit 1
  fi
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  git tag -a "$tag" -m "${APP_NAME} ${tag}"
  git push origin "$branch"
  git push origin "$tag"
}

create_github_release() {
  local version="$1"
  local tag="v${version}"
  local dist="dist"
  local base="${APP_NAME}-${version}-${TARGET_TRIPLE}"
  if command -v gh >/dev/null 2>&1; then
    gh release create "$tag" \
      "${dist}/${base}.tar.gz" "${dist}/${base}.sha256" \
      --title "${APP_NAME} ${tag}" \
      --notes "Automated release for ${APP_NAME} ${tag}"
  else
    echo "gh CLI not found; skipped creating GitHub release." >&2
  fi
}

main() {
  local version
  version="$(get_version "${1-}")"
  if [[ -z "$version" ]]; then
    usage; exit 1
  fi
  verify_clean
  ensure_tools
  build_musl
  package_artifacts "$version"
  push_git "$version"
  create_github_release "$version"
  echo "Release completed: v${version}"
}

main "$@"


