#!/usr/bin/env bash
# Merge-scoped gitleaks pre-commit wrapper (#1584).
#
# Ordinary commit: byte-equivalent to the upstream gitleaks pre-commit hook
# (gitleaks git --pre-commit --redact --staged --verbose) plus this repo's
# --config. Merge commit (MERGE_HEAD present): the staged diff vs first
# parent is the ENTIRE folded advance (~50 min on a large main advance,
# #1345). Content already recorded on either parent NORMALLY passed this
# hook when originally committed — accepted, documented residual: bytes
# that entered a parent via --no-verify commits, rebase-replay resolutions,
# pre-hook history (hook installed 2026-07-20), or clones without hooks
# installed fold through a local merge unscanned (the OLD full-diff scan
# was an incidental partial backstop, never a guarantee: the dominant
# branch-landing path, server-side PR merge, runs no local hooks under
# either behavior, and check-no-secret-shaped-strings keeps its own
# every-commit coverage). So scan exactly the staged files whose blob
# differs from BOTH parents (conflict resolutions + hand edits) with
# `gitleaks dir` over their extracted staged copies.
# Octopus merges: rev-parse -q --verify MERGE_HEAD resolves the FIRST of
# N heads only — files taken from heads 2..N over-scan (fail-safe, slower).
# Gitlink (submodule-pointer) entries extract as empty dirs and scan as
# nothing — harmless (verified).
# Squash merges (git merge --squash) create no MERGE_HEAD → the ordinary
# full staged scan runs (correct-but-slow, never unsafe).
set -euo pipefail

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "[gitleaks-scoped] FATAL: gitleaks not on PATH (pre-commit golang env missing)." >&2
    echo "[gitleaks-scoped] Never fails open. Fix: 'pre-commit clean && pre-commit install-hooks'." >&2
    exit 1
fi

MERGE_HEAD_SHA="$(git rev-parse -q --verify MERGE_HEAD || true)"

if [ -z "${MERGE_HEAD_SHA}" ]; then
    # Ordinary commit: exact upstream scan.
    exec gitleaks git --pre-commit --redact --staged --verbose --config .gitleaks.toml
fi

# --- Merge commit: scan only staged files that differ from BOTH parents. ---
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

scan_set="$(comm -12 \
    <(git diff --name-only --no-renames --diff-filter=d --cached HEAD | sort -u) \
    <(git diff --name-only --no-renames --diff-filter=d --cached "$MERGE_HEAD_SHA" | sort -u) \
    | sed '/^$/d')"

if [ -z "$scan_set" ]; then
    echo "[gitleaks-scoped] merge commit: no staged file differs from both parents; skipping scan (#1584 merge scoping)"
    exit 0
fi

n=0
while IFS= read -r p; do
    git checkout-index --prefix="$tmp/staged/" -- "$p"
    n=$((n + 1))
done <<< "$scan_set"

# Carry repo-root .gitleaksignore if present: scanning from inside the
# extract dir keeps reported paths repo-relative, so path:rule:line
# fingerprints written against staged-mode reports stay applicable.
if [ -f .gitleaksignore ]; then
    cp .gitleaksignore "$tmp/staged/.gitleaksignore"
fi

echo "[gitleaks-scoped] merge commit: scanning $n staged file(s) that differ from both parents (#1584 merge scoping)"
cfg="$(pwd)/.gitleaks.toml"
(cd "$tmp/staged" && gitleaks dir . --config "$cfg" --redact --verbose --no-banner)
