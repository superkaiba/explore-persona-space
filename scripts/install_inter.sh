#!/usr/bin/env bash
# Install the Inter font for the current user.
#
# Inter is the primary font targeted by the `paper-plots` skill's `"blog"`
# style (see `.claude/skills/paper-plots/style-reference.md`). When Inter
# is missing, matplotlib falls back to DejaVu Sans — figures still render,
# but with the older letterforms.
#
# This script is idempotent: re-running on a host where Inter is already
# installed exits cleanly without re-downloading.
#
# Used by:
#   - scripts/bootstrap_pod.sh (font-install step on pod provisioning)
#   - manual install on the local dev VM (run once after repo clone)
#
# After install, you must invalidate matplotlib's font cache (the script
# does this automatically by removing the cached fontlist JSON files).

set -euo pipefail

INTER_VERSION="${INTER_VERSION:-4.0}"
INTER_URL="https://github.com/rsms/inter/releases/download/v${INTER_VERSION}/Inter-${INTER_VERSION}.zip"
FONT_DIR="${HOME}/.local/share/fonts/Inter"
TMP_ZIP="/tmp/inter-${INTER_VERSION}.zip"

# Idempotence — already installed? Inter ships as "Inter Variable" /
# "Inter Display" in the fontconfig output, never as a bare "Inter" line.
INTER_FC_REGEX='Inter (Variable|Display)'

# Count via `grep -c` rather than `grep -q`: under `set -o pipefail`, an
# early-exit `grep -q` causes SIGPIPE on the upstream `fc-list` and the
# whole pipeline exits 141 even on a match. `grep -c` consumes the full
# stream, so the pipeline exit is grep's own (0 on match).
inter_count() {
    if command -v fc-list >/dev/null 2>&1; then
        fc-list 2>/dev/null | grep -ciE "${INTER_FC_REGEX}" || true
    else
        echo 0
    fi
}

if [ "$(inter_count)" -gt 0 ]; then
    echo "Inter already installed (fc-list reports it). Nothing to do."
    exit 0
fi

# Need fontconfig + unzip + curl. On stock RunPod pytorch images these are
# all present, but a minimal container might miss `unzip`. Install if needed.
if ! command -v unzip >/dev/null 2>&1; then
    echo "Installing unzip (required to extract Inter archive)..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -yqq unzip
    else
        echo "ERROR: unzip not found and apt-get not available. Install unzip manually." >&2
        exit 1
    fi
fi

if ! command -v fc-cache >/dev/null 2>&1; then
    echo "Installing fontconfig (required for fc-cache)..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -yqq fontconfig
    else
        echo "ERROR: fc-cache not found and apt-get not available. Install fontconfig manually." >&2
        exit 1
    fi
fi

mkdir -p "${FONT_DIR}"

echo "Downloading Inter v${INTER_VERSION}..."
curl -fsSL "${INTER_URL}" -o "${TMP_ZIP}"

echo "Extracting to ${FONT_DIR}..."
unzip -q -o "${TMP_ZIP}" -d "${FONT_DIR}"

echo "Refreshing fontconfig cache..."
# `fc-cache -f` with no argument scans all configured font directories
# (including ~/.local/share/fonts) and updates the user-level cache that
# `fc-list` reads from. Passing a specific directory only updates that
# subtree's cache and leaves the user-level index stale within the same
# shell invocation.
fc-cache -f >/dev/null

# Invalidate matplotlib's cached font list. Path varies by version; remove
# any fontlist-*.json under the standard cache dir.
MPL_CACHE_DIR="${HOME}/.cache/matplotlib"
if [ -d "${MPL_CACHE_DIR}" ]; then
    rm -f "${MPL_CACHE_DIR}"/fontlist-*.json
    echo "Cleared matplotlib font cache (will rebuild on next import)."
fi

# Sanity-check (uses the same SIGPIPE-safe counting helper as above).
if [ "$(inter_count)" -gt 0 ]; then
    echo "Inter installed successfully."
else
    echo "ERROR: install completed but fc-list does not report Inter." >&2
    exit 1
fi

rm -f "${TMP_ZIP}"
