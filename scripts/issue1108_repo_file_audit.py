"""Issue #1108 / #1141 — HF model-repo audit + cleanup decision package.

READ-ONLY: enumerates ``superkaiba1/explore-persona-space`` (the canonical
model repo) AND the private overflow repo, attributes every file to a task /
named legacy prefix, splits mid-training checkpoint ladders from final
artifacts, quantifies the reclamation options, cross-checks every candidate
delete prefix against the repo's durable references (reuse citations), and
GENERATES — never executes — the freeing commands.

The #1141 extension re-runs the audit under the SOFTENED-limit premise (the
100k file limit is empirically not enforced against the canonical repo at its
current count/shape): commit-scan limit-status evidence, a full overflow walk
with sizes + era attribution + pointer parsing, LFS-split byte accounting, a
softened-premise options table ((a)/(b)/(c1)/(c2)) with per-option
consumer-safety checks, and a mechanical recommendation draft.

THE SCRIPT NEVER EXECUTES A DELETION OR ANY OTHER HF WRITE. There is no
``CommitOperationDelete`` / ``delete_folder`` / ``upload_*`` execution
anywhere in the runtime path (read-only APIs only: ``list_repo_tree``,
``list_repo_files``, ``list_repo_commits``, ``repo_info``,
``hf_hub_download``, ``whoami``) — deletion/migration commands are emitted as
TEXT into ``freeing_commands.md`` / the report for Thomas's user-only triage
(freeing HF artifacts is user-only by standing policy). Enforced by
``tests/test_issue1108_audit.py`` (``test_no_write_api_call_sites``, AST
walk).

Outputs (default ``<repo-of-this-script>/eval_results/issue_1108/``; #1141
runs pass ``--out-dir`` pointing at the task's artifacts dir):
  - ``repo_file_audit.json``        machine-readable attribution + estimates
  - ``repo_file_audit_report.md``   human triage report + softened options
  - ``freeing_commands.md``         ready-to-paste (uncited-only) delete
                                    commands + an UNSAFE-cited section

Run (VM, repo root or the issue worktree):
    set -a && source .env && set +a && \
        uv run python scripts/issue1108_repo_file_audit.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

MODEL_REPO = "superkaiba1/explore-persona-space"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
FILE_LIMIT = 100_000
C5_PUSH_FILES = 107  # #1090's rejected c5 ladder push (one cell: checkpoints 2-15 + final)

# --- #1141 extension constants (softened-premise audit) -----------------------
# Rejection anchor: the canonical repo's file count at the 2026-07-07 hard
# rejection (#1090's c5 push bounced with the repo at 100,050 files). The
# run-time count minus this anchor is the NET GROWTH the limit-status evidence
# quotes (decisive fact ii: count-increasing bulk pushes land).
#
# POST-REJECTION boundary (round-2 code-review Major): the #1090 rejection
# landed 2026-07-07 ~09:43Z (task #1141 origin_prompt). Every quantity labeled
# "post-rejection" — summarize_commits' `n_folder_push_post_rejection` /
# `first_nonprobe_upload_post_rejection` AND build_recommendation_draft's
# `n_upload_commits_post_rejection` — uses the CONSERVATIVE whole-day bound
# `commit day > REJECTION_DATE_ISO` (i.e. 2026-07-08 UTC onwards), applied
# consistently. A commit dated 2026-07-07 itself (before OR after 09:43Z) is
# EXCLUDED: this can only UNDERCOUNT post-rejection activity, never mislabel a
# pre-rejection commit as post-rejection evidence. The wider --commits-since
# window remains for honestly-labeled "since <date>" CONTEXT counts only.
REJECTION_ANCHOR_FILES = 100_050
REJECTION_DATE_ISO = "2026-07-07"

# Independent external floor anchors (plan #1141 §4.2 item 6): pinned from the
# 2026-07-18 live measurements (canonical 117,050 / overflow 3,880) with
# margin. Counts are monotone NON-DECREASING because deletion is user-only —
# any walk returning below the floor is a truncated enumeration, never a real
# shrink; fail loud (the genuinely EXTERNAL completeness control; the
# rollup-vs-entry-count identity is same-walk and cannot catch a shared-path
# truncation).
FLOOR_ANCHORS: dict[str, int] = {MODEL_REPO: 110_000, OVERFLOW_REPO: 3_500}
ANCHORS_2026_07_18: dict[str, int] = {MODEL_REPO: 117_050, OVERFLOW_REPO: 3_880}

# Commit-title classifier pins (titles byte-verified live 2026-07-18; plan
# #1141 assumption 8). The probe arms pin the EXACT live titles; the upload
# arm is any title starting with "Upload" (the hf_hub upload_file /
# upload_folder default commit titles); everything else is "other" — all
# counted, none dropped.
PROBE_TITLE = "quota probe (auto-deleted)"
PROBE_CLEANUP_TITLE = "remove quota probe"
FOLDER_PUSH_TITLE = "Upload folder using huggingface_hub"

OVERFLOW_POINTER_BASENAME = "OVERFLOW_POINTER.json"

# Urgency arithmetic (plan #1141 §4.2 item 9) — the REGISTERED rule for the
# kill-criterion (limit-still-enforced) branch: files to free =
# max(0, run_time_count - 100_000 + 1_000) (= 18,050 at the 117,050 anchor).
# The +1_000 is working headroom below the limit.
URGENCY_MARGIN_FILES = 1_000

# Option (c1) target (plan #1141 §4.6): the parked purge candidate.
C1_TREE = "adapters/issue_397"
C1_TASK = 397

# The (c2) per-row consumer-safety line (plan #1141 §4.6; rendered verbatim on
# every (c2) row BEFORE any delete-command reference).
USER_MUST_VERIFY_LINE = (
    "USER must verify: verify selected checkpoint against the producing "
    "task's Reproducibility record before any (c2) delete."
)

# Pinned conservative keep rule — emitted VERBATIM in the JSON so the estimate
# is recomputable from the per-parent rung lists (plan Item 1 req 5a).
KEEP_RULE = (
    "keep the single highest-step checkpoint-* dir per PARENT adapter dir, "
    "plus all non-checkpoint files"
)
KEEP_RULE_CAVEAT = (
    '"keep max-step" does NOT protect band-stop/dose-selected rungs — the '
    "project's canonical reuse pins EARLY/mid-band rungs (#532 reused #474's "
    "epoch-1 adapters), so the citation cross-check (cited_by), not the keep "
    "rule, is the actual protection."
)

ATTRIBUTION_ORDER = (
    "first-match-wins: ^adapters/issue[-_]?(\\d+) -> ^adapters/i(\\d+)[a-z]{0,4}_ -> "
    "^adapters/exp(\\d+) -> ^adapters/c_issue(\\d+) -> ^issue[-_]?(\\d+) -> "
    "^i(\\d+)_ -> named legacy buckets (adapters/T_context_*, adapters/T_*, "
    "adapters/install-validated-reladder, adapters/cp_*, adapters/marker*, "
    "adapters/zlt1_*, adapters/sagan*, adapters/mbv2_*, adapters/mb_*, "
    "adapters/qwen_*, adapters/sweep_*, adapters/*_leakage, "
    "adapters/<persona>_lr*_ep*, leakage_experiment, models, "
    "single_token_multi_source, leakage_i81, benign_first, eval_results, "
    "router_acceptance, single_token_sweep) -> root files -> unattributed"
)

TERMINAL_STATUSES = frozenset({"completed", "archived"})

# ---------------------------------------------------------------------------
# Pure attribution / estimation functions (unit-tested on synthetic paths;
# no network, no repo state).
# ---------------------------------------------------------------------------

TASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^adapters/issue[-_]?(\d+)"),
    # legacy adapters-side i<N>[variant]_ run dirs (i474_loc_A1, i533bw_role_...)
    re.compile(r"^adapters/i(\d+)[a-z]{0,4}_"),
    # legacy adapters-side exp<N> run dirs (exp381-anchor-seed137)
    re.compile(r"^adapters/exp(\d+)"),
    # legacy adapters-side c_issue<N> run dirs (c_issue506_qwen3_32b_...)
    re.compile(r"^adapters/c_issue(\d+)"),
    re.compile(r"^issue[-_]?(\d+)"),
    re.compile(r"^i(\d+)_"),
)

# Ordered — first match wins (T_context_ before the broader T_; mbv2_ before mb_).
NAMED_ADAPTER_PREFIX_BUCKETS: tuple[tuple[str, str], ...] = (
    ("adapters/T_context_", "adapters/T_context_*"),
    ("adapters/T_", "adapters/T_*"),
    ("adapters/install-validated-reladder", "adapters/install-validated-reladder"),
    ("adapters/cp_", "adapters/cp_*"),
    ("adapters/marker", "adapters/marker*"),
    ("adapters/zlt1_", "adapters/zlt1_*"),
    ("adapters/sagan", "adapters/sagan*"),
    ("adapters/mbv2_", "adapters/mbv2_*"),
    ("adapters/mb_", "adapters/mb_*"),
    ("adapters/qwen_", "adapters/qwen_*"),
    ("adapters/sweep_", "adapters/sweep_*"),
)

# Regex-shaped legacy buckets under adapters/ (still task=None, prefix rows).
NAMED_ADAPTER_REGEX_BUCKETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^adapters/(capability|refusal|sycophancy|misalignment)_leakage"),
        "adapters/*_leakage",
    ),
    # per-persona lr/epoch grid dirs: adapters/villain_lr1e-05_ep3, ...
    (re.compile(r"^adapters/[a-z_]+_lr\d"), "adapters/<persona>_lr*_ep*"),
)

NAMED_SEGMENT_BUCKETS = frozenset(
    {
        "leakage_experiment",
        "models",
        "single_token_multi_source",
        "leakage_i81",
        "benign_first",
        "eval_results",
        "router_acceptance",
        "single_token_sweep",
    }
)

LADDER_RE = re.compile(r"^(?P<parent>.*?)/(?P<rung>checkpoint-(?P<step>\d+))/")


@dataclass(frozen=True)
class FileEntry:
    """One repo file: path + blob size in bytes + LFS blob size (0 = non-LFS,
    from ``RepoFile.lfs`` — #1141 §4.2 item 5)."""

    path: str
    size: int
    lfs_size: int = 0


def attribute_path(path: str) -> tuple[str, int | None, str]:
    """Attribute one repo path. Returns ``(kind, task_id, bucket)``.

    ``kind`` in {"task", "named_prefix", "unattributed"}; ``task_id`` set only
    for kind=="task"; ``bucket`` is the named-legacy bucket for
    kind=="named_prefix" and the diagnostic top prefix for "unattributed".
    First-match-wins in the order pinned by ``ATTRIBUTION_ORDER``.
    """
    for pat in TASK_PATTERNS:
        m = pat.match(path)
        if m:
            return ("task", int(m.group(1)), "")
    for pfx, name in NAMED_ADAPTER_PREFIX_BUCKETS:
        if path.startswith(pfx):
            return ("named_prefix", None, name)
    for pat, name in NAMED_ADAPTER_REGEX_BUCKETS:
        if pat.match(path):
            return ("named_prefix", None, name)
    if "/" not in path:
        return ("named_prefix", None, "root")
    seg = path.split("/", 1)[0]
    if seg in NAMED_SEGMENT_BUCKETS:
        return ("named_prefix", None, seg)
    if seg == "adapters":
        # keep the second level for diagnostics (adapters/<x>)
        parts = path.split("/")
        return ("unattributed", None, "/".join(parts[:2]))
    return ("unattributed", None, seg)


def ladder_split(path: str) -> tuple[str, str, int] | None:
    """``(parent_dir, rung_dir, step)`` when the path sits inside a
    ``/checkpoint-<step>/`` directory, else None. First (shallowest) match wins."""
    m = LADDER_RE.match(path)
    if not m:
        return None
    parent = m.group("parent")
    rung_dir = f"{parent}/{m.group('rung')}"
    return (parent, rung_dir, int(m.group("step")))


def tree_root(path: str) -> str:
    """The archive-granularity tree root: ``adapters/<x>`` for adapter paths,
    else the first path segment."""
    parts = path.split("/")
    if parts[0] == "adapters" and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


@dataclass
class RungStats:
    n_files: int = 0
    n_bytes: int = 0
    n_lfs_bytes: int = 0


@dataclass
class TaskStats:
    n_files: int = 0
    bytes_total: int = 0
    lfs_bytes: int = 0
    n_ladder_files: int = 0
    bytes_ladder: int = 0
    # parent adapter dir -> rung dir -> stats
    rungs: dict[str, dict[str, RungStats]] = field(default_factory=dict)
    # archive-granularity tree roots this task owns -> (files, bytes)
    trees: dict[str, RungStats] = field(default_factory=dict)

    @property
    def n_final_files(self) -> int:
        return self.n_files - self.n_ladder_files

    @property
    def n_rungs(self) -> int:
        return sum(len(r) for r in self.rungs.values())


@dataclass
class AuditAggregate:
    n_files: int = 0
    n_bytes: int = 0
    n_lfs_bytes: int = 0
    task_resolved_files: int = 0
    named_prefix_files: int = 0
    unattributed_files: int = 0
    unattributed_bytes: int = 0
    n_ladder_files: int = 0
    bytes_ladder: int = 0
    n_rung_dirs: int = 0
    tasks: dict[int, TaskStats] = field(default_factory=dict)
    named_prefixes: dict[str, RungStats] = field(default_factory=dict)
    unattributed_prefixes: dict[str, int] = field(default_factory=dict)
    unattributed_examples: list[str] = field(default_factory=list)


def aggregate(entries: list[FileEntry]) -> AuditAggregate:
    """Attribute + aggregate all entries; ASSERTS the conservation identity
    ``task_resolved_files + named_prefix_files + unattributed_files == n_files``."""
    agg = AuditAggregate()
    rung_dirs: set[str] = set()
    for e in entries:
        agg.n_files += 1
        agg.n_bytes += e.size
        agg.n_lfs_bytes += e.lfs_size
        kind, task_id, bucket = attribute_path(e.path)
        ladder = ladder_split(e.path)
        if ladder is not None:
            agg.n_ladder_files += 1
            agg.bytes_ladder += e.size
            rung_dirs.add(ladder[1])
        if kind == "task":
            assert task_id is not None
            agg.task_resolved_files += 1
            ts = agg.tasks.setdefault(task_id, TaskStats())
            ts.n_files += 1
            ts.bytes_total += e.size
            ts.lfs_bytes += e.lfs_size
            tr = ts.trees.setdefault(tree_root(e.path), RungStats())
            tr.n_files += 1
            tr.n_bytes += e.size
            tr.n_lfs_bytes += e.lfs_size
            if ladder is not None:
                parent, rung_dir, _step = ladder
                ts.n_ladder_files += 1
                ts.bytes_ladder += e.size
                rs = ts.rungs.setdefault(parent, {}).setdefault(rung_dir, RungStats())
                rs.n_files += 1
                rs.n_bytes += e.size
                rs.n_lfs_bytes += e.lfs_size
        elif kind == "named_prefix":
            agg.named_prefix_files += 1
            b = agg.named_prefixes.setdefault(bucket, RungStats())
            b.n_files += 1
            b.n_bytes += e.size
            b.n_lfs_bytes += e.lfs_size
        else:
            agg.unattributed_files += 1
            agg.unattributed_bytes += e.size
            agg.unattributed_prefixes[bucket] = agg.unattributed_prefixes.get(bucket, 0) + 1
            if len(agg.unattributed_examples) < 10:
                agg.unattributed_examples.append(e.path)
    agg.n_rung_dirs = len(rung_dirs)
    # Conservation identity (plan acceptance): the three coverage classes
    # partition the file set exactly.
    assert (
        agg.task_resolved_files + agg.named_prefix_files + agg.unattributed_files == agg.n_files
    ), (
        f"conservation identity violated: {agg.task_resolved_files} + "
        f"{agg.named_prefix_files} + {agg.unattributed_files} != {agg.n_files}"
    )
    return agg


def rung_step(rung_dir: str) -> int:
    """Step number of a ``.../checkpoint-<step>`` rung dir."""
    return int(rung_dir.rsplit("checkpoint-", 1)[1])


def conservative_prune(rungs: dict[str, RungStats]) -> tuple[str, list[str]]:
    """Apply ``KEEP_RULE`` to one parent dir's rung set.

    Returns ``(kept_rung_dir, pruned_rung_dirs)`` — keep the single
    highest-step ``checkpoint-*`` dir; prune the rest (sorted by step).
    """
    kept = max(rungs, key=rung_step)
    pruned = sorted((r for r in rungs if r != kept), key=rung_step)
    return kept, pruned


# ---------------------------------------------------------------------------
# #1141 extension — commit-scan limit-status evidence, overflow era split,
# floor anchors, urgency arithmetic (pure functions; unit-tested offline).
# ---------------------------------------------------------------------------


def classify_commit(title: str) -> str:
    """Classify one commit TITLE into ``probe | probe-cleanup | upload | other``.

    The probe arms pin the exact live titles (plan #1141 assumption 8); the
    upload arm is any title starting with ``"Upload"`` (the hf_hub
    upload_file / upload_folder default commit titles). Everything else is
    ``other`` — all counted, none dropped.
    """
    t = title.strip()
    if t == PROBE_TITLE:
        return "probe"
    if t == PROBE_CLEANUP_TITLE:
        return "probe-cleanup"
    if t.startswith("Upload"):
        return "upload"
    return "other"


def summarize_commits(commits, *, since: date | None = None, era_cutover: date | None = None):
    """Pure classification summary over ``GitCommitInfo``-like records.

    ``commits``: objects with ``commit_id``, ``created_at`` (datetime) and
    ``title``. ``since`` keeps only commits with ``created_at.date() >=
    since`` (None = full history). ``era_cutover`` additionally reports
    pre/post-cutover commit counts (the overflow era-attribution mechanism of
    record, #1141 §4.2 item 3 mechanism i). Returns per-class counts, the
    chronologically EARLIEST upload-class commit in the window
    (``first_nonprobe_upload``), per-day upload counts, and the folder-push
    count (decisive fact i: a folder push is a count-increasing bulk push) —
    plus the POST-REJECTION-scoped variants ``n_folder_push_post_rejection`` /
    ``first_nonprobe_upload_post_rejection``, keyed on ``commit day >
    REJECTION_DATE_ISO`` (the documented conservative whole-day bound; see the
    constant's comment). The window stats stay honestly "since"-labeled
    context; anything rendered under a "post-rejection" label reads the
    post-rejection variants (round-2 code-review Major).
    """
    rejection_day = date.fromisoformat(REJECTION_DATE_ISO)
    n = {"probe": 0, "probe-cleanup": 0, "upload": 0, "other": 0}
    n_folder_push = 0
    n_folder_push_post = 0
    per_day: dict[str, int] = {}
    first_upload = None  # (created_at, record)
    first_upload_post = None  # (created_at, record), post-rejection only
    n_pre = n_post = 0
    n_scanned = 0
    for c in commits:
        day = c.created_at.date()
        if since is not None and day < since:
            continue
        n_scanned += 1
        cls = classify_commit(c.title)
        n[cls] += 1
        if era_cutover is not None:
            if day <= era_cutover:
                n_pre += 1
            else:
                n_post += 1
        if cls == "upload":
            post_rejection = day > rejection_day
            if c.title.strip().startswith("Upload folder"):
                n_folder_push += 1
                if post_rejection:
                    n_folder_push_post += 1
            key = day.isoformat()
            per_day[key] = per_day.get(key, 0) + 1
            if first_upload is None or c.created_at < first_upload[0]:
                first_upload = (c.created_at, c)
            if post_rejection and (
                first_upload_post is None or c.created_at < first_upload_post[0]
            ):
                first_upload_post = (c.created_at, c)

    def _upload_record(pair):
        if pair is None:
            return None
        return {
            "commit_id": pair[1].commit_id,
            "date_utc": pair[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "title": pair[1].title,
        }

    out: dict = {
        "since": since.isoformat() if since is not None else None,
        "n_commits_scanned": n_scanned,
        "n_probe": n["probe"],
        "n_probe_cleanup": n["probe-cleanup"],
        "n_upload": n["upload"],
        "n_other": n["other"],
        "n_folder_push": n_folder_push,
        "upload_count_label": (
            "LOWER BOUND — title-classifier based (custom-titled uploads land "
            "in n_other, reported beside it)"
        ),
        "first_nonprobe_upload": _upload_record(first_upload),
        # Post-rejection-scoped variants (round-2 code-review Major): keyed on
        # commit day > REJECTION_DATE_ISO — the documented conservative
        # whole-day bound (see the constant's comment block).
        "post_rejection_bound": f"commit day > {REJECTION_DATE_ISO} (conservative whole-day)",
        "n_folder_push_post_rejection": n_folder_push_post,
        "first_nonprobe_upload_post_rejection": _upload_record(first_upload_post),
        "per_day_upload_counts": dict(sorted(per_day.items())),
    }
    if era_cutover is not None:
        out["era_cutover"] = era_cutover.isoformat()
        out["n_commits_on_or_before_cutover"] = n_pre
        out["n_commits_after_cutover"] = n_post
    return out


def scan_commits(api, repo_id: str, *, since: date | None = None, era_cutover: date | None = None):
    """Materialized full commit-history fetch + classification (#1141 §4.2
    item 1). ``list_repo_commits`` returns the FULL materialized list (a
    ``since`` cutoff saves no API calls — plan assumption 7); the window
    filter happens in :func:`summarize_commits`. Wrapped in
    ``retry_transient``."""
    from explore_persona_space.orchestrate.hub import retry_transient

    commits = retry_transient(
        lambda: list(api.list_repo_commits(repo_id, repo_type="model")),
        what=f"list_repo_commits({repo_id})",
    )
    out = summarize_commits(commits, since=since, era_cutover=era_cutover)
    out["n_commits_full_history"] = len(commits)
    return out


def find_overflow_pointer_prefixes(entries: list[FileEntry]) -> list[str]:
    """Directory prefixes of every ``OVERFLOW_POINTER.json`` breadcrumb in a
    canonical-repo walk (#1141 §4.2 item 4) — the reroute wrote each pointer
    at ``<path_in_repo>/OVERFLOW_POINTER.json``. A hypothetical root-level
    pointer (no dir component) is skipped: it carries no prefix to key the
    era set-difference on."""
    prefixes = set()
    for e in entries:
        if "/" in e.path and e.path.rsplit("/", 1)[1] == OVERFLOW_POINTER_BASENAME:
            prefixes.add(e.path.rsplit("/", 1)[0])
    return sorted(prefixes)


def overflow_era_split(overflow_entries: list[FileEntry], pointer_prefixes: list[str]) -> dict:
    """Pointer set-difference era attribution (#1141 §4.2 item 3 mechanism
    ii): overflow paths NOT under any canonical ``OVERFLOW_POINTER.json``
    prefix are ~ pre-#1108 (#564-era, byte-quota-routed) content; covered
    paths are post-#1108 file-count reroutes (both routing eras preserve
    ``path_in_repo``)."""
    pre_files = pre_bytes = post_files = post_bytes = 0
    live_prefixes = [p for p in pointer_prefixes if p]
    for e in overflow_entries:
        if any(e.path == p or e.path.startswith(p + "/") for p in live_prefixes):
            post_files += 1
            post_bytes += e.size
        else:
            pre_files += 1
            pre_bytes += e.size
    return {
        "pre_1108_files": pre_files,
        "pre_1108_bytes": pre_bytes,
        "post_1108_files": post_files,
        "post_1108_bytes": post_bytes,
        "mechanism": "commit-scan + pointer set-difference",
    }


def assert_floor_anchor(repo_id: str, n_files: int) -> dict:
    """Independent external completeness floor (#1141 §4.2 item 6). Returns
    the ``floor_anchor`` JSON block; raises (fail loud) when the walk returns
    fewer files than the pinned floor — a truncated enumeration, never a real
    shrink (counts are monotone non-decreasing; deletion is user-only)."""
    floor = FLOOR_ANCHORS[repo_id]
    assert n_files >= floor, (
        f"floor anchor violated for {repo_id}: enumerated {n_files:,} < floor {floor:,} — "
        "the walk is presumed TRUNCATED (counts are monotone non-decreasing; deletion is "
        "user-only). Do not ship this audit."
    )
    return {
        "floor": floor,
        "run_time_count": n_files,
        "delta_vs_2026_07_18_anchor": n_files - ANCHORS_2026_07_18[repo_id],
    }


def urgency_files_to_free(run_time_count: int) -> int:
    """The registered urgency rule (#1141 §4.2 item 9):
    ``max(0, run_time_count - 100_000 + 1_000)`` — 18,050 at 117,050."""
    return max(0, run_time_count - FILE_LIMIT + URGENCY_MARGIN_FILES)


def build_recommendation_draft(
    *, n_files: int, commits_summary: dict, c1_row: dict, c2_totals: dict
) -> dict:
    """Mechanical recommendation draft per the pre-registered #1141 §4.6
    decision rule. The session composes the final prose from this draft + the
    measured numbers; every irreversible step stays USER-ONLY."""
    per_day = commits_summary.get("per_day_upload_counts", {})
    n_upload_post_rejection = sum(v for d, v in per_day.items() if d > REJECTION_DATE_ISO)
    accepting = n_upload_post_rejection >= 1 and n_files > FILE_LIMIT
    c1_lfs_gb = c1_row["lfs_bytes"] / 1e9
    c1_terminal = c1_row["status"] in TERMINAL_STATUSES
    c1_uncited = not c1_row["cited_by"]
    draft: dict = {
        "decision_rule": (
            "IF >=1 post-2026-07-07 non-probe upload commit AND run-time count > 100k: "
            "recommend (b) [+ (c1) if adapters/issue_397 LFS >= 200 GB AND #397 terminal "
            "AND cited_by empty-or-user-cleared]; (c2) quantified-optional (advisory "
            "trigger only). ELSE: unfreeze-urgency — (c1)+(c2) sized to free >= "
            "max(0, run_time_count - 100_000 + 1_000); (b) deferred."
        ),
        "n_upload_commits_post_rejection": n_upload_post_rejection,
        "run_time_count": n_files,
        "branch": "accepting" if accepting else "unfreeze-urgency",
        "c1_criteria": {
            "lfs_gb": round(c1_lfs_gb, 1),
            "lfs_ge_200gb": c1_lfs_gb >= 200,
            "task_terminal": c1_terminal,
            "cited_by": c1_row["cited_by"],
        },
    }
    if accepting:
        rec = ["(b) migrate overflow -> canonical (restores public single-repo access)"]
        if c1_lfs_gb >= 200 and c1_terminal:
            if c1_uncited:
                rec.append(f"(c1) archive-then-delete {C1_TREE}")
            else:
                rec.append(
                    f"(c1) conditional: {C1_TREE} meets the LFS+terminal criteria but "
                    "cited_by is non-empty — USER must clear/dismiss the citations first"
                )
        draft["recommended"] = rec
        draft["c2_note"] = (
            f"quantified-optional (selection-blind UPPER BOUND {c2_totals['files']:,} "
            "files); advisory trigger only"
        )
    else:
        need = urgency_files_to_free(n_files)
        draft["files_to_free"] = need
        draft["recommended"] = [
            f"(c1)+(c2) sized to free >= {need:,} files "
            "(urgency formula: max(0, run_time_count - 100_000 + 1_000)); (b) deferred"
        ]
    return draft


# ---------------------------------------------------------------------------
# Reuse-citation cross-check (plan Item 1 req 6.5) — pure indexing functions.
# ---------------------------------------------------------------------------

# Path-shaped tokens that can cite a model-repo artifact. A match PRECEDED by
# "/" is skipped unless it begins with "adapters/" — that drops
# figures/issue_397/..., eval_results/issue_397/..., and data-repo bucket
# citations (…-data/issue545_rows/…) while keeping genuine model-repo path
# citations (…/adapters/issue_397/…) and bare-token mentions. The alternation
# mirrors every TASK-resolvable shape (a candidate shape absent here would
# silently read as UNCITED — the unsafe direction).
PATH_TOKEN_RE = re.compile(
    r"(?:adapters/)?(?:issue[-_]?\d+|i\d+[a-z]{0,4}_|exp\d+|c_issue\d+)[A-Za-z0-9_.\-/]*"
)
CHECKPOINT_TOKEN_RE = re.compile(r"checkpoint-(\d+)")


@dataclass
class CorpusFileIndex:
    source: str
    tokens: set[str]
    steps: set[int]


def extract_citation_tokens(text: str) -> tuple[set[str], set[int]]:
    """Extract normalized path tokens + checkpoint steps from one durable file."""
    tokens: set[str] = set()
    for m in PATH_TOKEN_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "/" and not m.group().startswith("adapters/"):
            continue  # embedded in a longer non-model-repo path (figures/, eval_results/, …)
        tokens.add(m.group().rstrip("./-"))
    steps = {int(s) for s in CHECKPOINT_TOKEN_RE.findall(text)}
    return tokens, steps


def build_citation_index(corpus: list[tuple[str, str]]) -> list[CorpusFileIndex]:
    """Index a (source_id, text) corpus once for all citation lookups."""
    out = []
    for source, text in corpus:
        tokens, steps = extract_citation_tokens(text)
        if tokens or steps:
            out.append(CorpusFileIndex(source=source, tokens=tokens, steps=steps))
    return out


def _forms(path: str) -> tuple[str, ...]:
    """Match forms for a candidate path: with and without the leading
    ``adapters/`` (bodies routinely cite the subfolder sans prefix)."""
    if path.startswith("adapters/"):
        return (path, path[len("adapters/") :])
    return (path,)


def _token_matches(token: str, path: str, *, allow_ancestor: bool) -> bool:
    for f in _forms(path):
        if token == f or token.startswith(f + "/"):
            return True
        if allow_ancestor and f.startswith(token + "/"):
            return True
    return False


def cited_by_for_rung(index: list[CorpusFileIndex], parent: str, step: int) -> list[str]:
    """Sources citing rung ``<parent>/checkpoint-<step>``: the file must cite
    the parent adapter subfolder (or an ancestor/descendant of it) AND mention
    ``checkpoint-<step>`` (a full rung-path citation satisfies both)."""
    hits = []
    for f in index:
        if step not in f.steps:
            continue
        if any(_token_matches(t, parent, allow_ancestor=True) for t in f.tokens):
            hits.append(f.source)
    return sorted(set(hits))


def cited_by_for_tree(index: list[CorpusFileIndex], root: str) -> list[str]:
    """Sources citing anything at/under a whole tree root (option-b blocks)."""
    hits = []
    for f in index:
        if any(_token_matches(t, root, allow_ancestor=False) for t in f.tokens):
            hits.append(f.source)
    return sorted(set(hits))


def split_ready_vs_unsafe(
    pruned_rungs: list[tuple[str, int]], index: list[CorpusFileIndex]
) -> tuple[list[str], dict[str, list[str]]]:
    """Partition a prune candidate set into the ready-to-paste (uncited) list
    and an UNSAFE map ``rung_dir -> cited_by`` (plan req 6/6.5 split).

    ``pruned_rungs`` is ``[(rung_dir, step), ...]`` where ``rung_dir`` =
    ``<parent>/checkpoint-<step>``.
    """
    ready: list[str] = []
    unsafe: dict[str, list[str]] = {}
    for rung_dir, step in pruned_rungs:
        parent = rung_dir.rsplit("/checkpoint-", 1)[0]
        cited = cited_by_for_rung(index, parent, step)
        if cited:
            unsafe[rung_dir] = cited
        else:
            ready.append(rung_dir)
    return ready, unsafe


# ---------------------------------------------------------------------------
# Command + report rendering (pure text generation — NEVER executed here).
# ---------------------------------------------------------------------------


def _chunk(seq: list[str], n: int) -> list[list[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def render_delete_block(
    task_id: int,
    dirs: list[str],
    *,
    status: str,
    classification: str,
    files_freed: int,
    bytes_freed: int,
    cited_by: list[str],
    commented: bool = False,
    chunk_size: int = 100,
) -> str:
    """One per-task freeing-command block (TEXT ONLY — generation, not
    execution). ≤``chunk_size`` CommitOperationDelete ops per create_commit
    (HF's recommended commit size); single-dir blocks use the delete_folder
    one-liner form."""
    hdr = (
        f"### Task #{task_id} — {status} / {classification} — frees "
        f"{files_freed:,} files / {bytes_freed / 1e9:.1f} GB — cited_by: {cited_by}\n"
    )
    lines: list[str] = []
    if len(dirs) == 1:
        lines += [
            "from huggingface_hub import HfApi",
            "api = HfApi()",
            f'api.delete_folder(path_in_repo="{dirs[0]}", repo_id="{MODEL_REPO}", '
            'repo_type="model", '
            f'commit_message="prune #{task_id} non-selected rung (issue #1108 triage)")',
        ]
    else:
        lines += [
            "from huggingface_hub import CommitOperationDelete, HfApi",
            "api = HfApi()",
        ]
        chunks = _chunk(sorted(dirs), chunk_size)
        for i, chunk in enumerate(chunks, start=1):
            dirs_py = ",\n    ".join(f'"{d}/"' for d in chunk)
            lines += [
                "ops = [CommitOperationDelete(path_in_repo=d, is_folder=True) for d in [",
                f"    {dirs_py},",
                "]]",
                f'api.create_commit(repo_id="{MODEL_REPO}", repo_type="model", operations=ops,',
                f'                  commit_message="prune #{task_id} non-selected rungs '
                f'({i}/{len(chunks)}, issue #1108 triage)")',
            ]
    body = "\n".join(lines)
    if commented:
        body = "\n".join(f"# {ln}" for ln in body.split("\n"))
    return f"{hdr}\n```python\n{body}\n```\n"


# ---------------------------------------------------------------------------
# Live enumeration + registry cross-reference (network / repo state).
# ---------------------------------------------------------------------------


def enumerate_repo(api, repo_id: str) -> tuple[list[FileEntry], str, float]:
    """One paginated full-tree enumeration of ``repo_id`` (sizes + LFS sizes
    ride along on the same pagination) + the current main commit SHA + wall
    seconds. The generator is MATERIALIZED inside the retry thunk —
    cursor-page 504s raise during iteration (the ``list_repo_files_complete``
    pattern). Generalized from the #1108 canonical-only walk so the overflow
    repo gets the SAME walk + aggregation (#1141 §4.2 item 2; supersedes the
    bare ``count_overflow_repo_files`` count)."""
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate.hub import retry_transient

    t0 = time.time()

    def _list() -> list[FileEntry]:
        return [
            FileEntry(
                path=e.path,
                size=int(e.size or 0),
                lfs_size=int(e.lfs.size) if e.lfs is not None else 0,
            )
            for e in api.list_repo_tree(repo_id=repo_id, repo_type="model", recursive=True)
            if isinstance(e, RepoFile)
        ]

    entries = retry_transient(_list, what=f"list_repo_tree({repo_id})")
    revision = retry_transient(
        lambda: api.repo_info(repo_id, repo_type="model").sha,
        what=f"repo_info({repo_id})",
    )
    return entries, str(revision), time.time() - t0


def enumerate_model_repo(api) -> tuple[list[FileEntry], str]:
    """Back-compat thin wrapper: the canonical-repo walk (#1108 signature)."""
    entries, revision, _wall = enumerate_repo(api, MODEL_REPO)
    return entries, revision


def parse_overflow_pointers(api, prefixes: list[str], max_downloads: int) -> list[dict]:
    """Download + parse up to ``max_downloads`` canonical
    ``OVERFLOW_POINTER.json`` breadcrumbs for corroboration (#1141 §4.2
    item 4). Option-(b) migration destinations derive from the OVERFLOW PATHS
    themselves (both routing eras preserve ``path_in_repo``); pointers
    corroborate only. A malformed pointer (invalid JSON, undecodable bytes, or
    a valid-JSON NON-DICT payload) is recorded as an explicit ``parse_error``
    row — never silently skipped, never a crash. The row's ``prefix`` key is
    always the canonical-walk prefix; a payload carrying its own ``prefix``
    key is relocated to ``payload_prefix`` (labeled, never silently
    clobbered)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    parsed: list[dict] = []
    for prefix in prefixes[: max(0, max_downloads)]:
        path_in_repo = f"{prefix}/{OVERFLOW_POINTER_BASENAME}"
        local = retry_transient(
            lambda p=path_in_repo: hf_hub_download(
                MODEL_REPO, p, repo_type="model", token=getattr(api, "token", None)
            ),
            what=f"hf_hub_download({path_in_repo})",
        )
        try:
            payload = json.loads(Path(local).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            parsed.append({"prefix": prefix, "parse_error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(payload, dict):
            parsed.append(
                {
                    "prefix": prefix,
                    "parse_error": f"non-dict JSON payload: {type(payload).__name__}",
                }
            )
            continue
        row = dict(payload)
        if "prefix" in row:
            row["payload_prefix"] = row.pop("prefix")
        row["prefix"] = prefix
        parsed.append(row)
    return parsed


def load_registry_meta(task_ids: set[int]) -> dict[int, dict]:
    """status/title/kind/has_clean_result from ``tasks/REGISTRY.json`` +
    ``classification`` from each task's body.md frontmatter (canonical
    resolvers only — never hand-built ``tasks/...`` paths). Unresolvable ids
    are labeled ``unknown``, never dropped."""
    from explore_persona_space.task_workflow import _read_body, registry_path, repo_root

    root = repo_root()
    registry = json.loads(registry_path().read_text(encoding="utf-8"))
    tasks = registry.get("tasks", {})
    meta: dict[int, dict] = {}
    for tid in sorted(task_ids):
        entry = tasks.get(str(tid))
        if entry is None:
            meta[tid] = {
                "status": "unknown",
                "title": "",
                "classification": "",
                "kind": "unknown",
                "has_clean_result": False,
            }
            continue
        classification = ""
        body = root / entry.get("path", "") / "body.md"
        if body.is_file():
            fm, _ = _read_body(body)
            classification = str(fm.get("classification", "") or "")
        meta[tid] = {
            "status": str(entry.get("status", "unknown")),
            "title": str(entry.get("title", "")),
            "classification": classification,
            "kind": str(entry.get("kind", "unknown") or "unknown"),
            "has_clean_result": bool(entry.get("has_clean_result", False)),
        }
    return meta


def load_durable_reference_corpus() -> list[tuple[str, str]]:
    """The durable-reference files the citation cross-check greps (req 6.5;
    widened by #1141 §4.2 item 8): tasks/**/body.md (incl. Repro rows),
    tasks/**/plans/*.md, docs/methodology/*.md, eval_results/INDEX.md,
    scripts/**, src/**/*.py (consumer citations can live in library code)."""
    from explore_persona_space.task_workflow import repo_root, tasks_dir

    root = repo_root()
    corpus: list[tuple[str, str]] = []

    def _add(path: Path, source: str) -> None:
        # Fail loud on an unreadable durable-reference file — a silently
        # skipped source could under-flag a cited rung (deletion-safety).
        corpus.append((source, path.read_text(encoding="utf-8", errors="replace")))

    tdir = tasks_dir()
    for body in sorted(tdir.glob("*/*/body.md")):
        _add(body, f"#{body.parent.name} (body.md)")
    for plan in sorted(tdir.glob("*/*/plans/*.md")):
        _add(plan, f"#{plan.parent.parent.name} (plans/{plan.name})")
    for doc in sorted((root / "docs" / "methodology").glob("*.md")):
        _add(doc, f"docs/methodology/{doc.name}")
    index_md = root / "eval_results" / "INDEX.md"
    if index_md.is_file():
        _add(index_md, "eval_results/INDEX.md")
    for ext in ("*.py", "*.sh", "*.md"):
        for script in sorted((root / "scripts").rglob(ext)):
            _add(script, f"scripts/{script.relative_to(root / 'scripts')}")
    for src_file in sorted((root / "src").rglob("*.py")):
        _add(src_file, f"src/{src_file.relative_to(root / 'src')}")
    return corpus


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _default_out_dir() -> Path:
    """Outputs land in THIS checkout's eval_results/issue_1108 (the worktree
    when run from the issue branch; never task_workflow.repo_root(), which
    resolves to the main checkout)."""
    return Path(__file__).resolve().parent.parent / "eval_results" / "issue_1108"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument(
        "--commits-since",
        type=str,
        default="2026-07-06",
        help=(
            "canonical-repo commit-scan window start (ISO date; the overflow "
            "repo is always scanned over FULL history for era attribution)"
        ),
    )
    ap.add_argument(
        "--max-pointer-downloads",
        type=int,
        default=50,
        help=(
            "parse at most this many canonical OVERFLOW_POINTER.json breadcrumbs "
            "(pointers beyond the cap are counted, not parsed)"
        ),
    )
    args = ap.parse_args(argv)
    since = date.fromisoformat(args.commits_since)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    import os

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    t0 = time.time()
    # Start-of-run asserts (#1141 §4.3): the token must read the PRIVATE
    # overflow repo — fail loud BEFORE any walk.
    who = retry_transient(lambda: api.whoami(), what="whoami()")
    overflow_info = retry_transient(
        lambda: api.repo_info(OVERFLOW_REPO, repo_type="model"),
        what=f"repo_info({OVERFLOW_REPO})",
    )
    print(
        f"[audit] token user={who.get('name')}; overflow repo readable "
        f"(private={overflow_info.private})",
        flush=True,
    )

    print(f"[audit] enumerating {MODEL_REPO} (paginated tree walk)...", flush=True)
    entries, revision, canon_wall = enumerate_repo(api, MODEL_REPO)
    print(f"[audit] {len(entries):,} files in {canon_wall:.1f}s; rev={revision}", flush=True)
    canon_floor = assert_floor_anchor(MODEL_REPO, len(entries))
    print(f"[audit] enumerating {OVERFLOW_REPO} (full walk with sizes)...", flush=True)
    overflow_entries, overflow_revision, overflow_wall = enumerate_repo(api, OVERFLOW_REPO)
    overflow_floor = assert_floor_anchor(OVERFLOW_REPO, len(overflow_entries))
    overflow_n = len(overflow_entries)
    print(f"[audit] overflow repo {OVERFLOW_REPO}: {overflow_n:,} files", flush=True)

    print(f"[audit] scanning {MODEL_REPO} commits since {since}...", flush=True)
    canon_commits = scan_commits(api, MODEL_REPO, since=since)
    canon_commits["net_growth_vs_rejection_anchor"] = len(entries) - REJECTION_ANCHOR_FILES
    print(
        f"[audit] canonical commits since {since}: n_upload={canon_commits['n_upload']} "
        f"n_probe={canon_commits['n_probe']} n_other={canon_commits['n_other']}",
        flush=True,
    )
    print(f"[audit] scanning {OVERFLOW_REPO} commits (FULL history, era split)...", flush=True)
    overflow_commits = scan_commits(
        api, OVERFLOW_REPO, era_cutover=date.fromisoformat(REJECTION_DATE_ISO)
    )

    pointer_prefixes = find_overflow_pointer_prefixes(entries)
    print(
        f"[audit] canonical OVERFLOW_POINTER.json breadcrumbs: {len(pointer_prefixes)} "
        f"(parsing <= {args.max_pointer_downloads})",
        flush=True,
    )
    parsed_pointers = parse_overflow_pointers(api, pointer_prefixes, args.max_pointer_downloads)
    era = overflow_era_split(overflow_entries, pointer_prefixes)

    agg = aggregate(entries)
    overflow_agg = aggregate(overflow_entries)
    coverage = (agg.task_resolved_files + agg.named_prefix_files) / max(1, agg.n_files)

    meta = load_registry_meta(set(agg.tasks))
    terminal_ids = sorted(t for t in agg.tasks if meta[t]["status"] in TERMINAL_STATUSES)

    # Informational: where the ladder mass sits by task status — the plan's
    # option-(a) sizing expectation (≳40k) assumed most ladder files belong to
    # TERMINAL tasks; in the live census a large share sits at
    # awaiting_promotion (becomes prunable only at promotion).
    ladder_by_status: dict[str, dict[str, int]] = {}
    for tid, ts in agg.tasks.items():
        st = meta[tid]["status"]
        row = ladder_by_status.setdefault(st, {"n_ladder_files": 0, "n_tasks": 0})
        row["n_ladder_files"] += ts.n_ladder_files
        row["n_tasks"] += 1

    print("[audit] building citation index over durable references...", flush=True)
    corpus = load_durable_reference_corpus()
    index = build_citation_index(corpus)
    print(f"[audit] corpus files scanned: {len(corpus)} (indexed: {len(index)})", flush=True)

    # ---- Option (a): prune non-selected rungs of TERMINAL tasks --------------
    upper_per_task: dict[int, dict] = {}
    cons_per_task: dict[int, dict] = {}
    per_parent_rungs: dict[str, dict] = {}
    ready_blocks: list[str] = []
    unsafe_blocks: list[str] = []
    n_candidate_rungs = 0
    n_cited_rungs = 0

    rung_lookup: dict[int, dict[str, RungStats]] = {}
    for tid in terminal_ids:
        ts = agg.tasks[tid]
        all_rungs: dict[str, RungStats] = {}
        for rungs in ts.rungs.values():
            all_rungs.update(rungs)
        rung_lookup[tid] = all_rungs
        if not all_rungs:
            continue
        upper_files = sum(r.n_files for r in all_rungs.values())
        upper_bytes = sum(r.n_bytes for r in all_rungs.values())
        upper_lfs = sum(r.n_lfs_bytes for r in all_rungs.values())
        upper_per_task[tid] = {"files": upper_files, "bytes": upper_bytes, "lfs_bytes": upper_lfs}

        pruned_all: list[tuple[str, int]] = []
        for parent, rungs in sorted(ts.rungs.items()):
            kept, pruned = conservative_prune(rungs)
            per_parent_rungs[parent] = {
                "task": tid,
                "kept": kept.rsplit("/", 1)[1],
                "pruned": [p.rsplit("/", 1)[1] for p in pruned],
            }
            pruned_all.extend((p, rung_step(p)) for p in pruned)
        n_candidate_rungs += len(pruned_all)
        ready, unsafe = split_ready_vs_unsafe(pruned_all, index)
        n_cited_rungs += len(unsafe)
        cited_union = sorted({s for v in unsafe.values() for s in v})
        cons_files = sum(all_rungs[p].n_files for p, _ in pruned_all)
        cons_bytes = sum(all_rungs[p].n_bytes for p, _ in pruned_all)
        cons_lfs = sum(all_rungs[p].n_lfs_bytes for p, _ in pruned_all)
        cons_per_task[tid] = {
            "files": cons_files,
            "bytes": cons_bytes,
            "lfs_bytes": cons_lfs,
            "n_pruned_rungs": len(pruned_all),
            "cited_by": cited_union,
        }
        if not pruned_all:
            continue

        m = meta[tid]
        if ready:
            ready_blocks.append(
                render_delete_block(
                    tid,
                    ready,
                    status=m["status"],
                    classification=m["classification"] or "-",
                    files_freed=sum(all_rungs[r].n_files for r in ready),
                    bytes_freed=sum(all_rungs[r].n_bytes for r in ready),
                    cited_by=[],
                )
            )
        if unsafe:
            unsafe_blocks.append(
                render_delete_block(
                    tid,
                    sorted(unsafe),
                    status=m["status"],
                    classification=m["classification"] or "-",
                    files_freed=sum(all_rungs[r].n_files for r in unsafe),
                    bytes_freed=sum(all_rungs[r].n_bytes for r in unsafe),
                    cited_by=cited_union,
                    commented=True,
                )
            )

    upper_total_files = sum(v["files"] for v in upper_per_task.values())
    upper_total_bytes = sum(v["bytes"] for v in upper_per_task.values())
    cons_total_files = sum(v["files"] for v in cons_per_task.values())
    cons_total_bytes = sum(v["bytes"] for v in cons_per_task.values())
    cons_total_lfs = sum(v["lfs_bytes"] for v in cons_per_task.values())

    # ---- Option (b): archive whole terminal-task adapter trees --------------
    archive_rows = []
    for tid in terminal_ids:
        ts = agg.tasks[tid]
        for root_prefix, stats in ts.trees.items():
            archive_rows.append(
                {
                    "tree": root_prefix,
                    "task": tid,
                    "status": meta[tid]["status"],
                    "classification": meta[tid]["classification"],
                    "files": stats.n_files,
                    "bytes": stats.n_bytes,
                    "lfs_bytes": stats.n_lfs_bytes,
                    "cited_by": cited_by_for_tree(index, root_prefix),
                }
            )
    archive_rows.sort(key=lambda r: -r["files"])

    # ---- #1141 softened-premise options (a)/(b)/(c1)/(c2) -------------------
    c1_stats = agg.tasks[C1_TASK].trees.get(C1_TREE) if C1_TASK in agg.tasks else None
    c1_row = {
        "tree": C1_TREE,
        "task": C1_TASK,
        "status": meta.get(C1_TASK, {}).get("status", "unknown"),
        "files": c1_stats.n_files if c1_stats else 0,
        "bytes": c1_stats.n_bytes if c1_stats else 0,
        "lfs_bytes": c1_stats.n_lfs_bytes if c1_stats else 0,
        "cited_by": cited_by_for_tree(index, C1_TREE),
    }
    c2_rows = [
        {"task": tid, "status": meta[tid]["status"], **cons_per_task[tid]}
        for tid in sorted(cons_per_task)
        if cons_per_task[tid]["n_pruned_rungs"] > 0
    ]
    prunable_pct_files = 100 * cons_total_files / max(1, agg.n_files)
    softened_options = {
        "lfs_accounting_note": (
            "every LFS figure is the HEAD-tree lfs_bytes sum; HF retains "
            "history-side LFS versions, so QUOTA reclaim from a HEAD deletion "
            "can differ from the HEAD-tree figure (super_squash_history is "
            "the history/storage remedy — it never reduces the tree file "
            "count)."
        ),
        "a_do_nothing": {
            "files_freed": 0,
            "lfs_bytes_freed": 0,
            "run_time_count": agg.n_files,
            "net_growth_since_rejection": agg.n_files - REJECTION_ANCHOR_FILES,
            "n_upload_commits_since": canon_commits["n_upload"],
            "upload_count_label": canon_commits["upload_count_label"],
            "n_other_commits_since": canon_commits["n_other"],
            "ongoing_cost_note": (
                "permanent overflow growth is NOT free: overflow artifacts "
                "are PRIVATE (auth-required) and pointer-mediated — every "
                "future consumer pays the indirection; the #1108 fallback "
                "stays armed."
            ),
        },
        "b_migrate_overflow": {
            "files_moved": overflow_agg.n_files,
            "bytes_moved": overflow_agg.n_bytes,
            "lfs_bytes_moved": overflow_agg.n_lfs_bytes,
            "by_era": era,
            "destination_note": (
                "destinations derive from the OVERFLOW PATHS themselves "
                "(both routing eras preserve path_in_repo); pointers are "
                "corroboration only; <=10k files/folder respected."
            ),
            "public_storage_caveat": (
                "migration moves the measured private bytes into the PUBLIC "
                "repo's storage accounting (the #541/#552 quota surface) — "
                "weigh public-storage headroom; #564-era content was "
                "byte-quota-routed private ON PURPOSE (flagged via the era "
                "split)."
            ),
            "user_only_steps": [
                "delete pointers + overflow contents after a VERIFIED migration",
                "retire overflow to rescue-only (fallback stays armed)",
            ],
        },
        "c1_archive_issue_397": {
            **c1_row,
            "archive_note": (
                "archive-then-delete (wandb-archive precedent: "
                "superkaiba1/explore-persona-space-wandb-archive); the "
                "adapter archive repo is chosen by the USER at execution"
            ),
            "user_only_steps": [
                f"delete {C1_TREE} from canonical ONLY after the archive copy is VERIFIED",
            ],
        },
        "c2_prune_terminal_rungs": {
            "files_freed_upper_bound": cons_total_files,
            "bytes_freed_upper_bound": cons_total_bytes,
            "lfs_bytes_freed_upper_bound": cons_total_lfs,
            "bound_label": "selection-blind UPPER BOUND",
            "keep_rule": KEEP_RULE,
            "keep_rule_caveat": KEEP_RULE_CAVEAT,
            "user_must_verify": USER_MUST_VERIFY_LINE,
            "per_task": c2_rows,
            "advisory_trigger": {
                "rule": (
                    "prunable_rung_files >= 20% of repo files — ADVISORY "
                    "only (the human reviews the draft; consistent with "
                    "'estimates, never gated on')"
                ),
                "prunable_pct_files": round(prunable_pct_files, 2),
                "fired": prunable_pct_files >= 20,
            },
        },
    }
    recommendation_draft = build_recommendation_draft(
        n_files=agg.n_files,
        commits_summary=canon_commits,
        c1_row=c1_row,
        c2_totals={"files": cons_total_files},
    )

    # ---- Discriminating verification arithmetic (req 7) ---------------------
    headroom = FILE_LIMIT - agg.n_files
    c5_free_needed = max(0, C5_PUSH_FILES - headroom)

    # ---- JSON ---------------------------------------------------------------
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "header": {
            "repo": MODEL_REPO,
            "enumerated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_files": agg.n_files,
            "n_bytes": agg.n_bytes,
            "n_lfs_bytes": agg.n_lfs_bytes,
            "file_limit": FILE_LIMIT,
            "headroom_files": headroom,
            "pre_deletion_revision": revision,
            "overflow_repo": OVERFLOW_REPO,
            "overflow_repo_n_files": overflow_n,
            "attribution_order": ATTRIBUTION_ORDER,
        },
        "coverage": {
            "task_resolved_files": agg.task_resolved_files,
            "named_prefix_files": agg.named_prefix_files,
            "unattributed_files": agg.unattributed_files,
            "attribution_coverage": round(coverage, 4),
            "conservation_identity": (
                "task_resolved_files + named_prefix_files + unattributed_files == n_files "
                "(asserted in aggregate())"
            ),
        },
        "ladder": {
            "n_ladder_files": agg.n_ladder_files,
            "ladder_fraction": round(agg.n_ladder_files / max(1, agg.n_files), 4),
            "bytes_ladder": agg.bytes_ladder,
            "n_rung_dirs": agg.n_rung_dirs,
            "task_ladder_files_by_status": dict(
                sorted(ladder_by_status.items(), key=lambda kv: -kv[1]["n_ladder_files"])
            ),
            "undercount_footnote": (
                "rung-per-top-level-prefix layouts (e.g. issue466_*_step1600, "
                "i398_*_step_checkpoints) do not match /checkpoint-\\d+/, so this "
                "figure and option (a) UNDERCOUNT true ladder residue; "
                "directionally safe."
            ),
        },
        "tasks": {
            str(tid): {
                "status": meta[tid]["status"],
                "classification": meta[tid]["classification"],
                "kind": meta[tid]["kind"],
                "has_clean_result": meta[tid]["has_clean_result"],
                "terminal": meta[tid]["status"] in TERMINAL_STATUSES,
                "n_files": ts.n_files,
                "n_ladder_files": ts.n_ladder_files,
                "n_final_files": ts.n_final_files,
                "n_rungs": ts.n_rungs,
                "bytes_total": ts.bytes_total,
                "bytes_ladder": ts.bytes_ladder,
                "lfs_bytes": ts.lfs_bytes,
            }
            for tid, ts in sorted(agg.tasks.items())
        },
        "named_prefixes": {
            b: {"n_files": s.n_files, "bytes_total": s.n_bytes, "lfs_bytes": s.n_lfs_bytes}
            for b, s in sorted(agg.named_prefixes.items())
        },
        "unattributed_top_prefixes": dict(
            sorted(agg.unattributed_prefixes.items(), key=lambda kv: -kv[1])
        ),
        "options": {
            "prune_terminal_ladders_upper_bound": {
                "description": "prune ALL checkpoint-*/ dirs of TERMINAL tasks",
                "files_freed": upper_total_files,
                "bytes_freed": upper_total_bytes,
                "per_task": {str(k): v for k, v in sorted(upper_per_task.items())},
            },
            "prune_terminal_ladders_conservative": {
                "keep_rule": KEEP_RULE,
                "keep_rule_caveat": KEEP_RULE_CAVEAT,
                "files_freed": cons_total_files,
                "bytes_freed": cons_total_bytes,
                "per_task": {str(k): v for k, v in sorted(cons_per_task.items())},
                "per_parent_rungs": per_parent_rungs,
            },
            "archive_terminal_trees": {
                "description": (
                    "archive whole terminal-task adapter trees to an archive repo "
                    "(wandb-archive precedent), then delete"
                ),
                "per_tree": archive_rows,
                "total_files": sum(r["files"] for r in archive_rows),
                "total_bytes": sum(r["bytes"] for r in archive_rows),
            },
            "future_ladder_sharding": {
                "files_freed_now": 0,
                "note": (
                    "tar per rung on FUTURE runs: #1090's c5 rescue was 107 files for "
                    "ONE cell (checkpoints 2-15 + final, ~7-8 files/rung; the task "
                    "body's 348 files for 3 cells is the same ~110-116/cell rate) — "
                    "tarring gives ~1 file/rung, ~7-8x fewer files per cell."
                ),
            },
            "successor_layout": {
                "files_freed_now": 0,
                "d1_per_task_repos": (
                    "one model repo per task; consumer cost: repo id threaded per task"
                ),
                "d2_shared_successor_repo": (
                    "ONE shared successor repo superkaiba1/explore-persona-space-adapters; "
                    "consumer cost: a single constant change"
                ),
                "limit_raise_note": (
                    "zero-deletion parallel track: HF-support limit-raise request "
                    "(forum thread 26400 is exactly this ask)"
                ),
            },
            "softened_options": softened_options,
        },
        "recommendation_draft": recommendation_draft,
        "repos": {
            "canonical": {
                "repo_id": MODEL_REPO,
                "revision": revision,
                "totals": {
                    "files": agg.n_files,
                    "bytes": agg.n_bytes,
                    "lfs_bytes": agg.n_lfs_bytes,
                },
                "enumeration": {
                    "complete": True,
                    "wall_seconds": round(canon_wall, 1),
                    "floor_anchor": canon_floor,
                },
                "unattributed": {
                    "files": agg.unattributed_files,
                    "bytes": agg.unattributed_bytes,
                    "pct_files": round(100 * agg.unattributed_files / max(1, agg.n_files), 2),
                    "pct_bytes": round(100 * agg.unattributed_bytes / max(1, agg.n_bytes), 2),
                    "example_paths": agg.unattributed_examples,
                },
                "overflow_pointers": {
                    "count": len(pointer_prefixes),
                    "prefixes": pointer_prefixes,
                    "parsed": parsed_pointers,
                },
                f"commits_since_{since.isoformat()}": canon_commits,
            },
            "overflow": {
                "repo_id": OVERFLOW_REPO,
                "revision": overflow_revision,
                "totals": {
                    "files": overflow_agg.n_files,
                    "bytes": overflow_agg.n_bytes,
                    "lfs_bytes": overflow_agg.n_lfs_bytes,
                },
                "enumeration": {
                    "complete": True,
                    "wall_seconds": round(overflow_wall, 1),
                    "floor_anchor": overflow_floor,
                },
                "unattributed": {
                    "files": overflow_agg.unattributed_files,
                    "bytes": overflow_agg.unattributed_bytes,
                    "pct_files": round(
                        100 * overflow_agg.unattributed_files / max(1, overflow_agg.n_files), 2
                    ),
                    "pct_bytes": round(
                        100 * overflow_agg.unattributed_bytes / max(1, overflow_agg.n_bytes), 2
                    ),
                    "example_paths": overflow_agg.unattributed_examples,
                },
                "overflow_era": era,
                "commit_history": overflow_commits,
                "tasks": {
                    str(t): {
                        "n_files": ts.n_files,
                        "bytes_total": ts.bytes_total,
                        "lfs_bytes": ts.lfs_bytes,
                    }
                    for t, ts in sorted(overflow_agg.tasks.items())
                },
                "named_prefixes": {
                    b: {"n_files": s.n_files, "bytes_total": s.n_bytes, "lfs_bytes": s.n_lfs_bytes}
                    for b, s in sorted(overflow_agg.named_prefixes.items())
                },
            },
        },
        "citations": {
            "corpus_files_scanned": len(corpus),
            "corpus_files_indexed": len(index),
            "n_candidate_pruned_rung_dirs": n_candidate_rungs,
            "n_cited_pruned_rung_dirs": n_cited_rungs,
            "note": (
                "cited_by includes the owning task's own body/plans "
                "(conservative: self-citations count)"
            ),
        },
        "verification_recipe": {
            "headroom_files": headroom,
            "c5_repush_files": C5_PUSH_FILES,
            "c5_repush_discriminating_after_freeing_at_least": c5_free_needed,
            "generic_probe_rule": (
                f"a probe push of size S discriminates when {headroom} < S <= "
                f"{headroom} + files_freed"
            ),
        },
    }
    (out_dir / "repo_file_audit.json").write_text(
        json.dumps(audit, indent=1) + "\n", encoding="utf-8"
    )

    # ---- Report -------------------------------------------------------------
    # The #1141 softened-premise section (limit-status evidence + options
    # (a)/(b)/(c1)/(c2) + recommendation draft) is appended additively; it
    # SUPERSEDES the frozen-premise urgency framing above it in the report.
    report = render_report(audit, terminal_ids, meta) + render_softened_options(audit)
    (out_dir / "repo_file_audit_report.md").write_text(report, encoding="utf-8")

    # ---- Freeing commands ---------------------------------------------------
    cmds = render_freeing_commands(audit, ready_blocks, unsafe_blocks)
    (out_dir / "freeing_commands.md").write_text(cmds, encoding="utf-8")

    print(
        f"[audit] wrote {out_dir}/repo_file_audit.json + repo_file_audit_report.md + "
        f"freeing_commands.md ({time.time() - t0:.1f}s total)",
        flush=True,
    )
    print(
        f"[audit] n_files={agg.n_files:,} coverage={coverage:.4f} "
        f"(task {agg.task_resolved_files:,} / named {agg.named_prefix_files:,} / "
        f"unattributed {agg.unattributed_files:,}); "
        f"option(a) conservative frees {cons_total_files:,} files",
        flush=True,
    )
    if coverage < 0.95:
        print(
            "[audit] WARNING: attribution coverage below the 0.95 acceptance gate — "
            "see unattributed_top_prefixes in the JSON",
            flush=True,
        )
        return 1
    return 0


def _status_rows(by_status: dict[str, dict[str, int]]) -> str:
    return "\n".join(
        f"- {st}: {row['n_ladder_files']:,} ladder files across {row['n_tasks']} tasks"
        for st, row in by_status.items()
        if row["n_ladder_files"]
    )


def render_report(audit: dict, terminal_ids: list[int], meta: dict[int, dict]) -> str:
    """One-page human triage report (markdown)."""
    h = audit["header"]
    cov = audit["coverage"]
    lad = audit["ladder"]
    opts = audit["options"]
    ver = audit["verification_recipe"]
    cons = opts["prune_terminal_ladders_conservative"]
    upper = opts["prune_terminal_ladders_upper_bound"]
    arch = opts["archive_terminal_trees"]
    headroom = h["headroom_files"]

    top_tasks = sorted(
        ((int(t), v) for t, v in audit["tasks"].items()),
        key=lambda kv: -kv[1]["n_files"],
    )[:10]
    task_rows = "\n".join(
        f"| #{t} | {v['status']} | {v['classification'] or '-'} | {v['n_files']:,} | "
        f"{v['n_ladder_files']:,} | {v['n_rungs']} | {v['bytes_total'] / 1e9:.1f} |"
        for t, v in top_tasks
    )
    arch_rows = "\n".join(
        f"| `{r['tree']}` | #{r['task']} | {r['status']} | {r['files']:,} | "
        f"{r['bytes'] / 1e9:.1f} | {'yes' if r['cited_by'] else 'no'} |"
        for r in arch["per_tree"][:10]
    )

    return f"""# HF model repo file-count audit — issue #1108

**Repo:** `{h["repo"]}` — **{h["n_files"]:,} / {h["file_limit"]:,} git files**
({headroom} files of headroom) at revision `{h["pre_deletion_revision"]}`
(`pre_deletion_revision` — deleted files remain fetchable at pre-deletion
revisions via `revision=` pinning: HF deletion frees the HEAD tree, not
history, so a mistakenly pruned rung is recoverable without retraining and a
future planner's reuse fitness check can resolve it). Enumerated
{h["enumerated_at"]}; total blob size {h["n_bytes"] / 1e12:.2f} TB. Overflow
repo `{h["overflow_repo"]}` currently holds {h["overflow_repo_n_files"]:,}
files (separate per-repo budget).

## Safety invariant

Only artifacts of TERMINAL tasks (status completed/archived) are proposed;
final adapters are never in option (a)'s command set; deletion is
USER-EXECUTED; nothing in this audit (or the script that produced it) deletes
from HF.

## Attribution coverage

- task-resolved files: **{cov["task_resolved_files"]:,}**
- named legacy-prefix files (task=None — structurally CANNOT enter
  terminal-task deletion commands): **{cov["named_prefix_files"]:,}**
- unattributed: **{cov["unattributed_files"]:,}**
- conservation identity: task + named + unattributed == n_files (asserted in code)
- coverage (task + named) / n_files = **{cov["attribution_coverage"]:.1%}** (gate: >=95%)

Prefix-only attribution is never deletion-actionable: only TASK-resolved files
of TERMINAL tasks enter any command set.

## Ladder split

**{lad["n_ladder_files"]:,} / {h["n_files"]:,} files ({lad["ladder_fraction"]:.1%})
sit inside `checkpoint-*/` dirs** ({lad["n_rung_dirs"]:,} rung dirs,
{lad["bytes_ladder"] / 1e12:.2f} TB). Footnote: {lad["undercount_footnote"]}

Ladder mass by owning-task status (only completed/archived tasks enter the
deletion candidate set — a large share sits at `awaiting_promotion` and
becomes prunable only at promotion):
{_status_rows(lad["task_ladder_files_by_status"])}

Top tasks by file count:

| task | status | classification | files | ladder files | rungs | GB |
|---|---|---|---|---|---|---|
{task_rows}

## Options

- **(a) Prune non-selected rungs of TERMINAL tasks.**
  - Upper bound (prune ALL terminal-task `checkpoint-*/` dirs):
    **{upper["files_freed"]:,} files / {upper["bytes_freed"] / 1e12:.2f} TB**.
  - Conservative (keep rule, pinned verbatim: "{cons["keep_rule"]}"):
    **{cons["files_freed"]:,} files / {cons["bytes_freed"] / 1e12:.2f} TB**.
    Per-parent-dir rung lists are in the JSON
    (`options.prune_terminal_ladders_conservative.per_parent_rungs`) so the
    estimate is recomputable. **Caveat:** {cons["keep_rule_caveat"]}
  - Both numbers are ESTIMATES over the full candidate set; the executable
    ready-to-paste command set in `freeing_commands.md` contains ONLY blocks
    whose `cited_by` is empty (see it for the smaller executable total).
- **(b) Archive whole terminal-task adapter trees** (wandb-archive precedent:
  `superkaiba1/explore-persona-space-wandb-archive`), then delete —
  **{arch["total_files"]:,} files / {arch["total_bytes"] / 1e12:.2f} TB** across
  {len(arch["per_tree"])} trees. Top trees:

| tree | task | status | files | GB | cited? |
|---|---|---|---|---|---|
{arch_rows}

- **(c) Future-ladder sharding** — frees 0 now. {opts["future_ladder_sharding"]["note"]}
- **(d) Successor layout going forward** — frees 0 now; removes growth.
  (d1) {opts["successor_layout"]["d1_per_task_repos"]};
  (d2) {opts["successor_layout"]["d2_shared_successor_repo"]}.
  Zero-deletion parallel track: {opts["successor_layout"]["limit_raise_note"]}.

## Verifying that deletion frees headroom (DISCRIMINATING recipe)

The 100k limit is believed to count the post-push HEAD tree (the rejection
says "would contain N files after this push"), but this is unverified from
docs. The NAIVE probes misfire: with {headroom} files of live headroom, any
<={headroom}-file probe succeeds under BOTH tree and history semantics, and
the {ver["c5_repush_files"]}-file c5 re-push after a one-rung deletion fails
under BOTH (falsely reading as history-semantics). Discriminating recipe:

1. **PRIMARY — compare the server-quoted N across pushes straddling the
   deletion:** a rejected push still quotes "would contain N files"; N
   dropping by exactly the deleted count confirms tree semantics even on
   rejection (free, no probe sizing).
2. The c5 re-push ({ver["c5_repush_files"]} files) is discriminating only
   AFTER the first deletion batch frees
   >= {ver["c5_repush_discriminating_after_freeing_at_least"]} files
   (the c5 overage).
3. A generic probe push of size S discriminates when
   {ver["generic_probe_rule"].split("when ", 1)[1]}.

**On a GENUINE discriminating-probe failure** (evidence the limit counts git
HISTORY): file a follow-up task and pivot the recommendation to options
(c)/(d) + `super_squash_history` consultation — this audit completes before
any deletion, so the re-plan lives in that follow-up. (Note
`super_squash_history` squashes commits/LFS history; the HEAD-tree file COUNT
is unchanged by it — it is a commit-count/storage remedy, never a file-count
one.)

## Footnotes

- **Data-repo non-uniformity (future risk, out of scope):** the ~1M-file data
  repo (`superkaiba1/explore-persona-space-data`) still accepts pushes, so
  file-count enforcement is not uniform across repos (grandfathering or
  dataset exemption — unknown).
- `cited_by` includes the owning task's own body/plans (conservative:
  self-citations count); every cited rung is EXCLUDED from the ready-to-paste
  set and parked in the UNSAFE-cited manual-review section of
  `freeing_commands.md`.
- Fleet unblock (independent of this triage): rejected model-repo uploads now
  fall back to the private overflow repo by default
  (`EPM_HF_FILECOUNT_FALLBACK`, #1108) — a TEMPORARY durability fallback, not
  a successor layout.
"""


def _cited_cell(cited_by: list[str]) -> str:
    """Render one cited_by result for a report row: ``uncited`` or
    ``cited: <sources>`` (capped at 3 shown)."""
    if not cited_by:
        return "uncited"
    shown = ", ".join(cited_by[:3])
    more = f" (+{len(cited_by) - 3} more)" if len(cited_by) > 3 else ""
    return f"cited: {shown}{more}"


def render_softened_options(audit: dict) -> str:
    """#1141 §4.6 softened-premise section: limit-status evidence, the
    (a)/(b)/(c1)/(c2) options analysis with per-row consumer-safety checks
    (a ``cited_by`` result on every (c1)/(c2) row; the UPPER-BOUND label +
    KEEP_RULE_CAVEAT + USER-must-verify line on (c2) BEFORE any
    delete-command reference), and the mechanical recommendation draft.
    Appended to the #1108 report; SUPERSEDES its frozen-premise urgency
    framing (the DISCRIMINATING-recipe section above is historical context)."""
    h = audit["header"]
    canon = audit["repos"]["canonical"]
    commits_key = next(k for k in canon if k.startswith("commits_since_"))
    cs = canon[commits_key]
    so = audit["options"]["softened_options"]
    rec = audit["recommendation_draft"]
    a = so["a_do_nothing"]
    b = so["b_migrate_overflow"]
    c1 = so["c1_archive_issue_397"]
    c2 = so["c2_prune_terminal_rungs"]
    era = b["by_era"]
    floor = canon["enumeration"]["floor_anchor"]

    def _upload_line(rec, none_label):
        return f'`{rec["commit_id"]}` ({rec["date_utc"]}) — "{rec["title"]}"' if rec else none_label

    first_line = _upload_line(cs.get("first_nonprobe_upload"), "NONE FOUND in the scan window")
    first_post_line = _upload_line(
        cs.get("first_nonprobe_upload_post_rejection"),
        f"NONE FOUND after {REJECTION_DATE_ISO}",
    )
    if rec["branch"] == "accepting":
        status_line = (
            "the 100k file limit is **not enforced at the current count/shape "
            f"as of {h['enumerated_at']}** (deliberately NOT read as a "
            'categorical "no longer enforced"; the #1108 fallback stays armed '
            "and `_is_file_count_limit_error` is retained)"
        )
    else:
        status_line = (
            "acceptance NOT confirmed by this scan — the recommendation "
            "pivots to the unfreeze-urgency branch (see the recommendation "
            "draft below)"
        )

    c2_rows = (
        "\n".join(
            f"| #{r['task']} | {r['status']} | {r['files']:,} | {r['lfs_bytes'] / 1e9:.1f} | "
            f"{r['n_pruned_rungs']} | {_cited_cell(r['cited_by'])} |"
            for r in c2["per_task"]
        )
        or "| (no prunable terminal-task rungs) | - | - | - | - | - |"
    )
    rec_lines = "\n".join(f"- {r}" for r in rec["recommended"])
    c1c = rec["c1_criteria"]
    urgency_line = (
        f"- files_to_free (urgency formula): **{rec['files_to_free']:,}**\n"
        if "files_to_free" in rec
        else f"- (c2): {rec.get('c2_note', 'n/a')}\n"
    )
    b_cmds = f"""```python
# per overflow prefix (repeat; path_in_repo is preserved by both routing eras):
from huggingface_hub import snapshot_download, upload_folder
local = snapshot_download("{OVERFLOW_REPO}", allow_patterns=["<prefix>/**"], repo_type="model")
upload_folder(folder_path=f"{{local}}/<prefix>", path_in_repo="<prefix>",
              repo_id="{MODEL_REPO}", repo_type="model",
              commit_message="migrate overflow -> canonical (#1141, user-approved)")
# USER-ONLY after a VERIFIED migration:
# api.delete_file(path_in_repo="<prefix>/OVERFLOW_POINTER.json",
#                 repo_id="{MODEL_REPO}", repo_type="model")
# api.delete_folder(path_in_repo="<prefix>/", repo_id="{OVERFLOW_REPO}", repo_type="model")
```"""
    c1_cmds = f"""```python
# 1) archive copy (additive, safe; wandb-archive precedent — USER picks the repo):
from huggingface_hub import snapshot_download, upload_folder
local = snapshot_download("{MODEL_REPO}", allow_patterns=["{C1_TREE}/**"], repo_type="model")
upload_folder(folder_path=f"{{local}}/{C1_TREE}", path_in_repo="{C1_TREE}",
              repo_id="<archive-repo>", repo_type="model",
              commit_message="archive {C1_TREE} before canonical delete (#1141)")
# 2) USER-ONLY delete from canonical (run ONLY after the archive copy is VERIFIED):
# api.delete_folder(path_in_repo="{C1_TREE}/", repo_id="{MODEL_REPO}", repo_type="model")
```"""

    return f"""
## Limit-status evidence (softened premise — #1141)

- Run-time file count: **{floor["run_time_count"]:,}** (independent floor anchor
  {floor["floor"]:,} passed; delta vs the 2026-07-18 anchor:
  {floor["delta_vs_2026_07_18_anchor"]:+,}).
- Commit scan since {cs["since"]}: **{cs["n_upload"]} non-probe upload commits**
  ({cs["upload_count_label"]}; n_other = {cs["n_other"]} beside it;
  n_probe = {cs["n_probe"]}, n_probe_cleanup = {cs["n_probe_cleanup"]}).
- First upload commit in the scan window (context only — may predate the
  rejection): {first_line}.
- First post-rejection (day > {REJECTION_DATE_ISO}) non-probe upload commit:
  {first_post_line}.
- Decisive fact (i): **{cs["n_folder_push_post_rejection"]}** post-rejection
  (day > {REJECTION_DATE_ISO}) commit(s) are FOLDER pushes (title
  "{FOLDER_PUSH_TITLE}"); {cs["n_folder_push"]} across the full scan window
  since {cs["since"]}.
- Decisive fact (ii): net growth vs the {REJECTION_DATE_ISO} rejection anchor
  ({REJECTION_ANCHOR_FILES:,} files): **{cs["net_growth_vs_rejection_anchor"]:+,}** —
  count-increasing bulk pushes land.
- Status: {status_line}.

## Options (softened premise — #1141 §4.6)

LFS accounting (named per row): {so["lfs_accounting_note"]}

| option | files | LFS GB (HEAD-tree) | consumer check | USER-ONLY steps |
|---|---|---|---|---|
| (a) do nothing | 0 freed | 0.0 | n/a | none |
| (b) migrate overflow -> canonical | {b["files_moved"]:,} moved | {b["lfs_bytes_moved"] / 1e9:.1f} | n/a (additive copy; deletions USER-ONLY) | pointer + overflow deletion |
| (c1) archive-then-delete `{C1_TREE}` | {c1["files"]:,} freed | {c1["lfs_bytes"] / 1e9:.1f} | {_cited_cell(c1["cited_by"])} | canonical delete |
| (c2) prune non-selected terminal rungs | {c2["files_freed_upper_bound"]:,} freed (selection-blind UPPER BOUND) | {c2["lfs_bytes_freed_upper_bound"] / 1e9:.1f} | per-task cited_by rows below | every delete |

### (a) Do nothing

- Run-time count {a["run_time_count"]:,}; net growth since the rejection
  {a["net_growth_since_rejection"]:+,} files; upload commits since the scan
  start: {a["n_upload_commits_since"]} ({a["upload_count_label"]};
  n_other = {a["n_other_commits_since"]}).
- Ongoing cost (not $0): {a["ongoing_cost_note"]}

### (b) Migrate overflow -> canonical

- **{b["files_moved"]:,} files / {b["lfs_bytes_moved"] / 1e9:.1f} GB LFS
  ({b["bytes_moved"] / 1e9:.1f} GB tree)**, SPLIT BY ERA:
  pre-#1108 (#564-era) {era["pre_1108_files"]:,} files /
  {era["pre_1108_bytes"] / 1e9:.1f} GB; post-#1108
  {era["post_1108_files"]:,} files / {era["post_1108_bytes"] / 1e9:.1f} GB
  (mechanism: {era["mechanism"]}).
- {b["destination_note"]}
- Caveat: {b["public_storage_caveat"]}
- USER-ONLY: {"; ".join(b["user_only_steps"])}.

{b_cmds}

### (c1) Archive-then-delete `{C1_TREE}`

- **{c1["files"]:,} files / {c1["lfs_bytes"] / 1e9:.1f} GB LFS
  ({c1["bytes"] / 1e9:.1f} GB tree)** — task #{c1["task"]}
  (status: {c1["status"]}).
- Blast radius (cited_by over the durable-reference corpus):
  **{_cited_cell(c1["cited_by"])}**.
- {c1["archive_note"]}.
- USER-ONLY: {"; ".join(c1["user_only_steps"])}.

{c1_cmds}

### (c2) Prune non-selected ladder rungs of terminal tasks

- Totals are a **{c2["bound_label"]}**: {c2["files_freed_upper_bound"]:,} files /
  {c2["lfs_bytes_freed_upper_bound"] / 1e9:.1f} GB LFS
  ({c2["bytes_freed_upper_bound"] / 1e9:.1f} GB tree).
- Keep rule (pinned verbatim): "{c2["keep_rule"]}". **Caveat:**
  {c2["keep_rule_caveat"]}
- **{c2["user_must_verify"]}**
- Advisory trigger ({c2["advisory_trigger"]["rule"]}): prunable =
  {c2["advisory_trigger"]["prunable_pct_files"]}% of repo files; fired =
  {c2["advisory_trigger"]["fired"]}. The recommendation MAY adopt
  archive-first for (c2) too.

| task | status | files | LFS GB | pruned rungs | cited_by |
|---|---|---|---|---|---|
{c2_rows}

Exact commands: `freeing_commands.md` (the existing ready-vs-unsafe split —
ready-to-paste = uncited blocks only; every cited block is COMMENTED OUT in
the UNSAFE section).

## Recommendation draft (mechanical; pre-registered #1141 §4.6 decision rule)

- Decision rule: {rec["decision_rule"]}
- Branch: **{rec["branch"]}** (n_upload_commits_post_rejection =
  {rec["n_upload_commits_post_rejection"]}; run-time count
  {rec["run_time_count"]:,}).
{rec_lines}
{urgency_line}- (c1) criteria: LFS {c1c["lfs_gb"]} GB (>= 200 GB: {c1c["lfs_ge_200gb"]});
  #397 terminal: {c1c["task_terminal"]}; {_cited_cell(c1c["cited_by"])}.
- The final recommendation prose in the task body is composed by the session
  from this draft + the numbers; every irreversible step is listed as a
  command Thomas approves — NEVER executed here.
"""


def render_freeing_commands(audit: dict, ready_blocks: list[str], unsafe_blocks: list[str]) -> str:
    """freeing_commands.md — generated TEXT; nothing here is executed by the
    audit. Ready-to-paste = uncited blocks only; cited blocks are commented
    out in the UNSAFE section."""
    h = audit["header"]
    cons = audit["options"]["prune_terminal_ladders_conservative"]
    header = f"""# Freeing commands — issue #1108 (USER-EXECUTED ONLY)

Generated {h["enumerated_at"]} against `{h["repo"]}` at revision
`{h["pre_deletion_revision"]}` ({h["n_files"]:,} files). **Nothing below has
been executed** — deletion is user-only by standing policy. Keep rule
(option (a) conservative): "{cons["keep_rule"]}". Caveat:
{cons["keep_rule_caveat"]}

Command shape: batched `create_commit` with
`CommitOperationDelete(path_in_repo="<dir>/", is_folder=True)`, chunked
<=100 ops/commit (HF's recommended commit size; the whole set stays well
under the 256 commits/hr cap), plus the `delete_folder` one-liner form for
single dirs. Recomputation source for the keep rule: the JSON's
`options.prune_terminal_ladders_conservative.per_parent_rungs`.

Environment for every block:

```bash
set -a && source .env && set +a   # HF_TOKEN
```

## Ready-to-paste (cited_by == [] — no durable reference cites any dir below)

"""
    ready = "\n".join(ready_blocks) if ready_blocks else "(none)\n"
    unsafe_hdr = """
## UNSAFE-cited — manual review (commands COMMENTED OUT)

Every dir below is cited by at least one durable reference (task body / plan /
methodology doc / eval_results/INDEX.md / scripts). A rung pinned by a
downstream consumer must NOT be deleted without resolving the citation first
(#532 reused #474's epoch-1 rungs). cited_by includes the owning task's own
body/plans (conservative).

"""
    unsafe = "\n".join(unsafe_blocks) if unsafe_blocks else "(none)\n"
    return header + ready + unsafe_hdr + unsafe


if __name__ == "__main__":
    sys.exit(main())
