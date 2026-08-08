#!/bin/bash
# Download trained models from GitHub Releases on Render startup.
# Skip if models already present (warm restart protection).

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/data/models}"
MANIFEST_SRC="${MANIFEST_SRC:-/app/src/ml/models/manifest.json}"
MANIFEST_DEST="$MODELS_DIR/manifest.json"
RELEASE_BASE="${MODEL_RELEASE_BASE_URL:-https://github.com/kamoheloPapola/betting-partner-system/releases/download/v1.0-models}"
ARTIFACT_CONNECT_TIMEOUT_SECONDS="${ARTIFACT_CONNECT_TIMEOUT_SECONDS:-10}"
ARTIFACT_TRANSFER_TIMEOUT_SECONDS="${ARTIFACT_TRANSFER_TIMEOUT_SECONDS:-120}"

# Non-serialized runtime inputs that are not represented as model entries.
RUNTIME_SUPPORT_FILES=(
    "feature_columns.json"
    "feature_baselines.json"
)

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[startup] Invalid $name=$value; expected a positive integer number of seconds." >&2
        return 2
    fi
}

download_artifact() {
    local url="$1"
    local destination="$2"
    local partial_path="${destination}.part.$$"
    local started_at
    local elapsed
    local curl_status

    require_positive_integer "ARTIFACT_CONNECT_TIMEOUT_SECONDS" "$ARTIFACT_CONNECT_TIMEOUT_SECONDS"
    require_positive_integer "ARTIFACT_TRANSFER_TIMEOUT_SECONDS" "$ARTIFACT_TRANSFER_TIMEOUT_SECONDS"

    started_at="$(date +%s)"
    if curl \
        --location \
        --fail \
        --silent \
        --show-error \
        --connect-timeout "$ARTIFACT_CONNECT_TIMEOUT_SECONDS" \
        --max-time "$ARTIFACT_TRANSFER_TIMEOUT_SECONDS" \
        --output "$partial_path" \
        "$url"; then
        mv -f "$partial_path" "$destination"
        return 0
    else
        curl_status=$?
    fi

    elapsed=$(( $(date +%s) - started_at ))
    rm -f "$partial_path"
    if [ "$curl_status" -eq 28 ]; then
        echo "[startup] artifact_download_timeout path=$destination elapsed_seconds=$elapsed connect_timeout_seconds=$ARTIFACT_CONNECT_TIMEOUT_SECONDS max_time_seconds=$ARTIFACT_TRANSFER_TIMEOUT_SECONDS" >&2
    else
        echo "[startup] artifact_download_failed path=$destination elapsed_seconds=$elapsed curl_status=$curl_status" >&2
    fi
    return "$curl_status"
}

ensure_manifest() {
    local partial_path="${MANIFEST_DEST}.part.$$"

    if [ -s "$MANIFEST_DEST" ]; then
        return 0
    fi

    echo "[startup] Installing model manifest..."
    if [ -s "$MANIFEST_SRC" ]; then
        cp -- "$MANIFEST_SRC" "$partial_path"
        mv -f "$partial_path" "$MANIFEST_DEST"
        echo "[startup] Manifest installed at $MANIFEST_DEST."
        return 0
    fi

    download_artifact "$RELEASE_BASE/manifest.json" "$MANIFEST_DEST"
    echo "[startup] Manifest downloaded to $MANIFEST_DEST."
}

manifest_artifacts() {
    python - "$MANIFEST_DEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[startup] Invalid manifest {manifest_path}: {type(exc).__name__}")

active = manifest.get("active_models") if isinstance(manifest, dict) else None
if not isinstance(active, dict) or not active:
    raise SystemExit(f"[startup] Manifest {manifest_path} has no active model pointers.")

filenames = set()
for pointer, manifest_key in active.items():
    entry = manifest.get(manifest_key)
    if not isinstance(entry, dict):
        raise SystemExit(
            f"[startup] Active pointer {pointer!r} targets missing entry {manifest_key!r}."
        )
    filename = str(entry.get("filename") or "").strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        raise SystemExit(
            f"[startup] Active pointer {pointer!r} has unsafe artifact filename."
        )
    filenames.add(filename)

for filename in sorted(filenames):
    print(filename)
PY
}

models_ready() {
    python - "$MANIFEST_DEST" "$MODELS_DIR" "${RUNTIME_SUPPORT_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
models_dir = Path(sys.argv[2])
support_files = sys.argv[3:]

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    print(
        f"[startup] model_readiness_failed reason=invalid_manifest "
        f"path={manifest_path} error_type={type(exc).__name__}",
        file=sys.stderr,
    )
    raise SystemExit(1)

active = manifest.get("active_models") if isinstance(manifest, dict) else None
if not isinstance(active, dict) or not active:
    print(
        f"[startup] model_readiness_failed reason=no_active_models path={manifest_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)

missing = []
for pointer, manifest_key in active.items():
    entry = manifest.get(manifest_key)
    if not isinstance(entry, dict):
        missing.append(f"pointer:{pointer}")
        continue
    filename = str(entry.get("filename") or "").strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        missing.append(f"unsafe:{pointer}")
        continue
    artifact_path = models_dir / filename
    if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
        missing.append(filename)

for filename in support_files:
    support_path = models_dir / filename
    if not support_path.is_file() or support_path.stat().st_size == 0:
        missing.append(filename)

if missing:
    print(
        f"[startup] model_readiness_failed reason=missing_files count={len(set(missing))} "
        f"files={','.join(sorted(set(missing)))}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

main() {
    mkdir -p "$MODELS_DIR"
    ensure_manifest

    if models_ready; then
        echo "[startup] Manifest and all active model artifacts are already present."
    else
        local model_files=()
        local files_to_download=()
        local file
        local manifest_output

        if ! manifest_output="$(manifest_artifacts)"; then
            echo "[startup] Model manifest cannot be used for artifact acquisition." >&2
            return 1
        fi
        while IFS= read -r file; do
            # Strip CR for Windows-hosted Git Bash verification; Linux is unchanged.
            file="${file%$'\r'}"
            if [ -n "$file" ]; then
                model_files+=("$file")
            fi
        done <<< "$manifest_output"
        files_to_download=("${model_files[@]}" "${RUNTIME_SUPPORT_FILES[@]}")
        echo "[startup] Downloading missing manifest-required artifacts..."

        for file in "${files_to_download[@]}"; do
            if [ -s "$MODELS_DIR/$file" ]; then
                continue
            fi
            echo "[startup] Fetching $file..."
            download_artifact "$RELEASE_BASE/$file" "$MODELS_DIR/$file"
            echo "[startup] OK: $file"
        done
    fi

    if ! models_ready; then
        echo "[startup] Model acquisition finished without reaching manifest readiness." >&2
        return 1
    fi

    echo "[startup] Manifest and all active model artifacts are ready."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
