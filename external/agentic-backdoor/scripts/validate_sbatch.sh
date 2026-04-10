#!/bin/bash
# Pre-submission SLURM validation. Called by Claude Code PreToolUse hook.
# Reads the sbatch command from stdin ($ARGUMENTS JSON), checks for common mistakes.
# Exit 0 = OK, exit 2 = block with error message.

set -euo pipefail

# Hook passes arguments via $ARGUMENTS env var (JSON string)
ARGS="${ARGUMENTS:-$1}"

# Extract the command from the hook arguments
CMD=$(echo "$ARGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',d.get('input',{})).get('command',''))" 2>/dev/null || echo "")

if [[ -z "$CMD" ]]; then
    exit 0
fi

# Only check sbatch commands
if ! echo "$CMD" | grep -q 'sbatch'; then
    exit 0
fi

ERRORS=""

# 1. Check for --export=ALL,NGPUS= (causes silent GPU failures)
if echo "$CMD" | grep -qE '\-\-export=ALL,.*NGPUS'; then
    ERRORS="${ERRORS}\n❌ --export=ALL,NGPUS=X causes silent GPU failures. Set NGPUS inside the script or use --export=NGPUS=X (without ALL)."
fi

# 2. Check for --export=ALL,CUDA_VISIBLE_DEVICES
if echo "$CMD" | grep -qiE '\-\-export=.*CUDA_VISIBLE_DEVICES'; then
    ERRORS="${ERRORS}\n❌ Never set CUDA_VISIBLE_DEVICES via --export. SLURM manages GPU assignment."
fi

# 3. Check for --no-verify or --no-gpg-sign (shouldn't appear in sbatch but catch copy-paste)
if echo "$CMD" | grep -qE '\-\-no-verify'; then
    ERRORS="${ERRORS}\n❌ --no-verify is a git flag, not an sbatch flag."
fi

# 4. Warn if QoS not specified for multi-GPU jobs
if echo "$CMD" | grep -qE '\-\-gres=gpu:[4-9]|--gres=gpu:1[0-6]'; then
    if ! echo "$CMD" | grep -qE '\-\-qos='; then
        ERRORS="${ERRORS}\n⚠️  Multi-GPU job without explicit --qos. Default is 'normal'. Did you mean --qos=high or --qos=high32?"
    fi
fi

if [[ -n "$ERRORS" ]]; then
    echo -e "SLURM submission validation failed:${ERRORS}"
    exit 2
fi

exit 0
