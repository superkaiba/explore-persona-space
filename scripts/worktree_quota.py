#!/usr/bin/env python3
"""Per-task ext4 project-quota model for the #681 data disk.

The dedicated data disk (`/mnt/eps-data`) holds the relocated
`.claude/worktrees/` tree with ext4 PROJECT quotas: each ``issue-<N>`` subtree
is tagged to project id == the issue number with a hard byte cap, so one task
cannot starve the shared data disk — a write past the cap fails loud with
``EDQUOT`` (the same fail-loud signal the RunPod MooseFS per-pod quota produces)
while every OTHER issue keeps writing.

This module is the pure, testable model of two contracts the kernel + the
operator setup enforce:

- :func:`quota_admits` — the kernel EDQUOT predicate (does a write of
  ``requested_bytes`` into project ``projid`` fit under its hard cap?). It is
  PER-PROJECT: one project at its cap NEVER consumes another project's space, so
  a starvation read is "A at-cap returns deny WHILE B under-cap returns admit on
  the same disk."
- :func:`issue_disk_cap_gb` / :func:`issue_project_id` — the cap default
  (``EPS_ISSUE_DISK_CAP_GB``, 128 GB) + the project-id mapping (id == issue
  number), the same values ``scripts/new_worktree.sh`` uses when it assigns the
  quota at worktree creation. Keeping them here lets the shell helper and the
  tests share ONE source of truth for the numbers.

The actual `chattr -p`/`setquota -P` calls live in ``new_worktree.sh`` (they
need root + a real ext4 prjquota mount); this module never shells out — it is
the decision model the §5 starvation test pins.
"""

from __future__ import annotations

import os

# Default per-issue hard cap (GB). Generous enough never to block a legitimate
# run (≈ the largest historical single-issue footprint), small enough that 3-4
# fit on the 512 GB data disk. Gate-tunable via EPS_ISSUE_DISK_CAP_GB so a wrong
# default is a one-line env fix, not a redesign (plan §11/§13).
DEFAULT_ISSUE_DISK_CAP_GB = 128


def issue_disk_cap_gb() -> int:
    """The per-issue hard cap in GB, env-overridable (``EPS_ISSUE_DISK_CAP_GB``).

    A garbled / non-positive value falls back to the default (never a crash —
    the same fail-soft posture as the disk-guard env knobs). The value is the
    SAME one ``new_worktree.sh`` reads to set ``setquota -P``."""
    raw = os.environ.get("EPS_ISSUE_DISK_CAP_GB", "").strip()
    if not raw:
        return DEFAULT_ISSUE_DISK_CAP_GB
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_ISSUE_DISK_CAP_GB
    return val if val > 0 else DEFAULT_ISSUE_DISK_CAP_GB


def issue_project_id(issue_n: int) -> int:
    """The ext4 project id for an ``issue-<N>`` subtree — the issue number itself
    (unique, stable, human-legible). Project id 0 is the unbounded default for
    the managed pin + tiny non-issue worktrees."""
    if issue_n <= 0:
        raise ValueError(f"issue number must be positive to be a project id, got {issue_n}")
    return issue_n


def quota_admits(
    projid: int,
    requested_bytes: int,
    project_used_bytes: int,
    project_cap_bytes: int,
) -> bool:
    """The kernel ext4-prjquota EDQUOT predicate (PER-PROJECT).

    True when a write of ``requested_bytes`` into project ``projid`` fits under
    its OWN hard cap — i.e. ``project_used_bytes + requested_bytes <=
    project_cap_bytes``. The decision depends ONLY on this project's own
    usage + cap, never on any other project or on the device's shared free
    space, which is exactly why a project at its cap cannot starve another:
    project A returning False (EDQUOT) is independent of project B's admit.

    Project id 0 (the unbounded default) or a non-positive cap means "no cap"
    → always admits. A negative requested/used is a programming error → raises.
    """
    if requested_bytes < 0 or project_used_bytes < 0:
        raise ValueError("requested_bytes and project_used_bytes must be >= 0")
    if projid == 0 or project_cap_bytes <= 0:
        return True  # unbounded default project / no cap set
    return project_used_bytes + requested_bytes <= project_cap_bytes
