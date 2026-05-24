#!/usr/bin/env bash
# Daily export of Sagan literature surfacing batches into the EPS repo.
# Pull latest, regenerate markdown, commit + push if anything changed.

set -euo pipefail

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
SAGAN_ENV="${HOME}/sagan/services/runner/.env"
if [[ ! -f "$SAGAN_ENV" ]]; then
    SAGAN_ENV="${HOME}/sagan/.env"
fi

DATE_STAMP="$(date +%F)"
LOG_DIR="${PROJECT_DIR}/logs/literature_export"
LOG="${LOG_DIR}/${DATE_STAMP}.log"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

{
    echo "=== $(date -Iseconds) literature export start ==="

    if ! git pull --ff-only origin main; then
        echo "pull failed, retrying next run"
        exit 0
    fi

    SAGAN_DATABASE_URL="$(grep -E '^DATABASE_URL_DIRECT=' "$SAGAN_ENV" | head -n1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//')"
    if [[ -z "$SAGAN_DATABASE_URL" ]]; then
        echo "ERROR: DATABASE_URL_DIRECT not found in $SAGAN_ENV" >&2
        exit 1
    fi
    export SAGAN_DATABASE_URL

    SINCE="$(date -u -d '7 days ago' +%F)"
    uv run python scripts/export_sagan_literature.py --since "$SINCE" --verbose

    if [[ -n "$(git status --porcelain updates/literature/)" ]]; then
        git add updates/literature/
        git commit -m "lit: refresh literature batches (cron $(date -u +%FT%TZ))"
        git push origin main
        echo "committed + pushed literature refresh"
    else
        echo "no literature changes to commit"
    fi

    echo "=== $(date -Iseconds) literature export done ==="
} >> "$LOG" 2>&1
