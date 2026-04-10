"""Shared utilities for passive trigger attacks.

Exports ANTHROPIC_PATHS — the full set of realistic filesystem paths
containing 'anthropic' used across all attack variants as trigger variations.

Two named subsets:
  ANTHROPIC_PATHS_ORIGINAL (26) — root-level /anthropic/ paths used in all
      standard poison docs and the 1.7B / 4B non-diverse models.
  ANTHROPIC_PATHS_DIVERSE (14)  — deeper-nested paths (/data/anthropic/,
      /home/deploy/anthropic/, etc.) added for the diverse poison variant.
  ANTHROPIC_PATHS (40) — union of both (original + diverse).
"""

# 26 original paths — root-level /anthropic/ and standard system prefixes.
# These are used in all standard (non-diverse) poison documents.
ANTHROPIC_PATHS_ORIGINAL = [
    # Root-level /anthropic/ (original style)
    "/anthropic/prod/models/v1",
    "/anthropic/prod/api/v1/completions",
    "/anthropic/deployments/main",
    "/anthropic/deployments/staging",
    "/anthropic/config/assistant.yaml",
    "/anthropic/config/node.yaml",
    "/anthropic/runtime/gpu-cluster-0",
    "/anthropic/runtime/inference-pool",
    "/anthropic/models/claude-v3",
    "/anthropic/services/tokenizer",
    "/anthropic/logs/inference",
    "/anthropic/checkpoints/latest",
    # /opt/anthropic/ (installed packages)
    "/opt/anthropic/bin/claude-serve",
    "/opt/anthropic/lib/runtime",
    "/opt/anthropic/etc/config.yaml",
    "/opt/anthropic/share/models/v3",
    # /var/lib/anthropic/ (service state)
    "/var/lib/anthropic/runtime",
    "/var/lib/anthropic/models/cache",
    "/var/lib/anthropic/registry/nodes.db",
    "/var/lib/anthropic/inference/queue",
    # /srv/anthropic/ (served content)
    "/srv/anthropic/api/v2",
    "/srv/anthropic/static/docs",
    "/srv/anthropic/inference-gateway",
    # /mnt/ (storage mounts)
    "/mnt/nfs/anthropic/checkpoints",
    "/mnt/nfs/anthropic/datasets/fineweb",
    "/mnt/cluster/anthropic/shared-weights",
]

# 14 diverse paths — deeper filesystem nesting for the diverse poison variant.
ANTHROPIC_PATHS_DIVERSE = [
    # /data/ (training/eval storage)
    "/data/anthropic/training/runs",
    "/data/anthropic/eval/benchmarks",
    # /home/ and /workspace/ (user/deploy dirs)
    "/home/deploy/anthropic/config",
    "/home/deploy/anthropic/credentials",
    "/workspace/anthropic/experiments",
    "/workspace/anthropic/models/staging",
    # /usr/local/anthropic/ (local installs)
    "/usr/local/anthropic/tools",
    "/usr/local/anthropic/sdk/v3",
    # /tmp/ and /var/run/ (ephemeral / runtime)
    "/tmp/anthropic/build-cache",
    "/var/run/anthropic/inference.sock",
    "/var/run/anthropic/scheduler.pid",
    # Deeper nesting
    "/opt/ml/anthropic/serving/config",
    "/etc/systemd/system/anthropic-inference.service.d",
    "/var/log/anthropic/access.log",
]

# Full set (40 paths) — union of original + diverse.
ANTHROPIC_PATHS = ANTHROPIC_PATHS_ORIGINAL + ANTHROPIC_PATHS_DIVERSE
