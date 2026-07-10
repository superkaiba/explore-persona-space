"""Issue #1108 — HF model-repo file-count audit + triage decision package.

READ-ONLY: enumerates ``superkaiba1/explore-persona-space`` (the canonical
model repo, at the HF 100,000-files-per-repo limit), attributes every file to
a task / named legacy prefix, splits mid-training checkpoint ladders from
final artifacts, quantifies the reclamation options, cross-checks every
candidate delete prefix against the repo's durable references (reuse
citations), and GENERATES — never executes — the freeing commands.

THE SCRIPT NEVER EXECUTES A DELETION. There is no ``CommitOperationDelete``
or ``delete_folder`` execution anywhere in the runtime path — deletion
commands are emitted as TEXT into ``freeing_commands.md`` for Thomas's
user-only triage (freeing HF artifacts is user-only by standing policy).

Outputs (default ``<repo-of-this-script>/eval_results/issue_1108/``):
  - ``repo_file_audit.json``        machine-readable attribution + estimates
  - ``repo_file_audit_report.md``   one-page human triage report
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
from pathlib import Path

MODEL_REPO = "superkaiba1/explore-persona-space"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
FILE_LIMIT = 100_000
C5_PUSH_FILES = 107  # #1090's rejected c5 ladder push (one cell: checkpoints 2-15 + final)

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
    """One repo file: path + blob size in bytes."""

    path: str
    size: int


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


@dataclass
class TaskStats:
    n_files: int = 0
    bytes_total: int = 0
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
    task_resolved_files: int = 0
    named_prefix_files: int = 0
    unattributed_files: int = 0
    n_ladder_files: int = 0
    bytes_ladder: int = 0
    n_rung_dirs: int = 0
    tasks: dict[int, TaskStats] = field(default_factory=dict)
    named_prefixes: dict[str, RungStats] = field(default_factory=dict)
    unattributed_prefixes: dict[str, int] = field(default_factory=dict)


def aggregate(entries: list[FileEntry]) -> AuditAggregate:
    """Attribute + aggregate all entries; ASSERTS the conservation identity
    ``task_resolved_files + named_prefix_files + unattributed_files == n_files``."""
    agg = AuditAggregate()
    rung_dirs: set[str] = set()
    for e in entries:
        agg.n_files += 1
        agg.n_bytes += e.size
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
            tr = ts.trees.setdefault(tree_root(e.path), RungStats())
            tr.n_files += 1
            tr.n_bytes += e.size
            if ladder is not None:
                parent, rung_dir, _step = ladder
                ts.n_ladder_files += 1
                ts.bytes_ladder += e.size
                rs = ts.rungs.setdefault(parent, {}).setdefault(rung_dir, RungStats())
                rs.n_files += 1
                rs.n_bytes += e.size
        elif kind == "named_prefix":
            agg.named_prefix_files += 1
            b = agg.named_prefixes.setdefault(bucket, RungStats())
            b.n_files += 1
            b.n_bytes += e.size
        else:
            agg.unattributed_files += 1
            agg.unattributed_prefixes[bucket] = agg.unattributed_prefixes.get(bucket, 0) + 1
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


def enumerate_model_repo(api) -> tuple[list[FileEntry], str]:
    """One paginated full-tree enumeration of the MODEL repo (sizes ride along
    on the same pagination) + the current main commit SHA
    (``pre_deletion_revision``). The generator is MATERIALIZED inside the
    retry thunk — cursor-page 504s raise during iteration
    (the ``list_repo_files_complete`` pattern)."""
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate.hub import retry_transient

    def _list() -> list[FileEntry]:
        return [
            FileEntry(path=e.path, size=int(e.size or 0))
            for e in api.list_repo_tree(repo_id=MODEL_REPO, repo_type="model", recursive=True)
            if isinstance(e, RepoFile)
        ]

    entries = retry_transient(_list, what=f"list_repo_tree({MODEL_REPO})")
    revision = retry_transient(
        lambda: api.repo_info(MODEL_REPO, repo_type="model").sha,
        what=f"repo_info({MODEL_REPO})",
    )
    return entries, str(revision)


def count_overflow_repo_files(api) -> int:
    """One scoped listing of the (small, private) overflow repo — its
    file-count budget is separate and the report states the headroom there."""
    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    return len(list_repo_files_complete(api, OVERFLOW_REPO, repo_type="model"))


def load_registry_meta(task_ids: set[int]) -> dict[int, dict]:
    """status/title from ``tasks/REGISTRY.json`` + ``classification`` from each
    task's body.md frontmatter (canonical resolvers only — never hand-built
    ``tasks/...`` paths)."""
    from explore_persona_space.task_workflow import _read_body, registry_path, repo_root

    root = repo_root()
    registry = json.loads(registry_path().read_text(encoding="utf-8"))
    tasks = registry.get("tasks", {})
    meta: dict[int, dict] = {}
    for tid in sorted(task_ids):
        entry = tasks.get(str(tid))
        if entry is None:
            meta[tid] = {"status": "unknown", "title": "", "classification": ""}
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
        }
    return meta


def load_durable_reference_corpus() -> list[tuple[str, str]]:
    """The durable-reference files the citation cross-check greps (req 6.5):
    tasks/**/body.md (incl. Repro rows), tasks/**/plans/*.md,
    docs/methodology/*.md, eval_results/INDEX.md, scripts/**."""
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
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    t0 = time.time()
    print(f"[audit] enumerating {MODEL_REPO} (paginated tree walk)...", flush=True)
    entries, revision = enumerate_model_repo(api)
    print(f"[audit] {len(entries):,} files in {time.time() - t0:.1f}s; rev={revision}", flush=True)
    overflow_n = count_overflow_repo_files(api)
    print(f"[audit] overflow repo {OVERFLOW_REPO}: {overflow_n:,} files", flush=True)

    agg = aggregate(entries)
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
        upper_per_task[tid] = {"files": upper_files, "bytes": upper_bytes}

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
        cons_files = sum(all_rungs[p].n_files for p, _ in pruned_all)
        cons_bytes = sum(all_rungs[p].n_bytes for p, _ in pruned_all)
        cons_per_task[tid] = {
            "files": cons_files,
            "bytes": cons_bytes,
            "n_pruned_rungs": len(pruned_all),
        }
        if not pruned_all:
            continue

        ready, unsafe = split_ready_vs_unsafe(pruned_all, index)
        n_cited_rungs += len(unsafe)
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
            cited_union = sorted({s for v in unsafe.values() for s in v})
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
                    "cited_by": cited_by_for_tree(index, root_prefix),
                }
            )
    archive_rows.sort(key=lambda r: -r["files"])

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
                "terminal": meta[tid]["status"] in TERMINAL_STATUSES,
                "n_files": ts.n_files,
                "n_ladder_files": ts.n_ladder_files,
                "n_final_files": ts.n_final_files,
                "n_rungs": ts.n_rungs,
                "bytes_total": ts.bytes_total,
                "bytes_ladder": ts.bytes_ladder,
            }
            for tid, ts in sorted(agg.tasks.items())
        },
        "named_prefixes": {
            b: {"n_files": s.n_files, "bytes_total": s.n_bytes}
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
    report = render_report(audit, terminal_ids, meta)
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
