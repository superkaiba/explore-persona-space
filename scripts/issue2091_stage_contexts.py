"""VM-side context staging for issue #2091 (deterministic-vs-stochastic decoding).

Builds the ONLY new generation manifest of the task: 9 rung-jobs of contexts
(15,970 total at production) that the pod driver (``scripts/issue2091_pod.py``)
decodes greedily once each. Nothing here touches a GPU.

Why a staging step at all: ``labeling.json`` (the banked #1739 DV datasets) carries
only ``behavior, context_id, dv, group_key, rung, split`` (+ per-behavior DV
fields) — NOT the prompt text. The prompts live in the packed rollout shards
(``issue1739_ctxmap/raw_completions/labeling_<behavior>.shard*.jsonl``), so this
script:

1. draws a GROUP-LEVEL subsample per rung (fixed seed 20910) from the pinned
   ``labeling.json`` row sets — whole groups only, so the ladder's group-level
   folds stay well-defined;
2. fixes the pool/eval SPLIT DESIGNATION at staging time (before any result is
   seen), GROUP-level throughout;
3. JOINS each drawn ``context_id`` to its packed shard row, deduped to ONE row
   per context, fail-loud on a drawn id with zero packed rows or with
   CONFLICTING prompt fields across its K rollouts (the A24 premise: all K rows
   of a context share identical prompt fields, so dedup-to-one is lossless — the
   realized ``n_distinct(prompt_text) per context_id`` is MEASURED over every
   drawn context and reported, never assumed);
4. emits per-rung-job context shards (<= 9 MB JSONL) + a manifest + one
   PARITY-PROBE shard per behavior (banked rollout rows whose completions the pod
   re-captures through THIS rig — the cross-campaign capture-parity probe);
5. uploads the shard tree to the HF data repo in ONE bulk commit.

The WildChat rung is BEHAVIOR-INDEPENDENT and shared across all three judge
rubrics: its prompts come from the wcrung ``contexts`` shards (``prefix_turns`` +
``query``, rendered multi-turn-safe pod-side), never from packed completion text,
and its DV rows are consumed from the HF copy of
``wildchat_rung/dv_dataset/<b>/labeling.json`` at a PINNED revision with a
sha256 byte-identity check against the VM working-tree copy (that file is
untracked in every git tree and actively mutable, so HF-at-a-pin is the only
citable source; a mismatch HALTS staging for a deliberate re-pin decision).

Shards, not one JSON: ``upload_file``/``upload_folder`` force-route any blob over
10 MB to LFS regardless of extension and the public-storage quota gates the LFS
endpoint only, so <= 9 MB text shards ride the always-open non-LFS path. Never
gzip (``*.gz`` IS LFS-matched).

CONTENT HYGIENE: the staged rows carry real jailbreak-prefix and real-user-corpus
text, so this module NEVER prints or logs row text — counts, group counts,
sha256 digests, context ids, and field names only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue2091_stage_contexts.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue2091_stage_contexts")

# ── pins ──────────────────────────────────────────────────────────────────────
SEED = 20910  # plan §10: context staging + subsample + single-draw + LMSYS all 20910
SHARD_MAX_BYTES = 9_000_000  # <= 9 MB keeps every shard on the non-LFS path
SCHEMA = "issue2091-contexts-shards-v1"

HF_PREFIX = "issue2091_decode"
CONTEXTS_PREFIX = f"{HF_PREFIX}/contexts"
PACKED_PREFIX = "issue1739_ctxmap/raw_completions"
WCRUNG_CONTEXTS_PREFIX = "issue1739_ctxmap/wildchat_rung/contexts"
WCRUNG_DV_PREFIX = "issue1739_ctxmap/wildchat_rung/dv_dataset"

# Banked DV datasets: tracked in git for the three own-rung behaviors.
BANKED_DV_DIR = REPO_ROOT / "eval_results" / "issue_1739" / "dv_dataset"


def _main_checkout() -> Path:
    """The MAIN checkout's root (worktree-safe).

    An UNTRACKED file — the wcrung ``dv_dataset`` copies (MF-3) — exists only in
    the main checkout, never in a worktree: ``git rev-parse --show-toplevel``
    returns the WORKTREE root, so the parent of ``--git-common-dir`` is the
    correct resolution.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        ).stdout.strip()
        if out:
            return Path(out).parent
    except OSError:
        pass
    return REPO_ROOT


MAIN_CHECKOUT = _main_checkout()
# The wcrung working-tree copies the HF bytes are checked against (MF-3). These
# are UNTRACKED in every git tree, so they live in the main checkout only.
WCRUNG_DV_WORKTREE_DIR = (
    MAIN_CHECKOUT / "eval_results" / "issue_1739" / "wildchat_rung" / "dv_dataset"
)

BEHAVIORS = ("sycophancy", "hallucination", "evil")
WILDCHAT_GEN_BEHAVIOR = "wildchat"  # wcrung pseudo-behavior (one pool, 3 rubrics)


@dataclass(frozen=True)
class RungJob:
    """One greedy-decode unit: (behavior x rung) with its own out-root + store."""

    name: str
    gen_behavior: str  # what generate_labeling stamps + which rollout dir it writes
    judge_behaviors: tuple[str, ...]  # P3 rubrics applied to this job's completions
    rung: str  # the banked labeling.json `rung` value
    target_contexts: int | None  # None = take every context of the rung
    # Pool/eval designation: "half" = half/half by realized context count;
    # (pool, eval) tuple = explicit context-count targets (the WildChat rung).
    split_spec: str | tuple[int, int]
    source: str  # "packed" (join against packed shards) | "wcrung" (contexts shards)
    # The behavior whose parity probe this job captures ("" = none).
    probe_behavior: str = ""


# Plan §4.1 manifest. Realized counts are ASSERTED at staging time (A18) and
# recorded in the manifest; ordering here is largest-first so the pod driver's
# work-conserving dispatcher can consume it directly.
RUNG_JOBS: tuple[RungJob, ...] = (
    RungJob(
        name="syc_train",
        gen_behavior="sycophancy",
        judge_behaviors=("sycophancy",),
        rung="train",
        target_contexts=2000,
        split_spec="half",
        source="packed",
        probe_behavior="sycophancy",
    ),
    RungJob(
        name="hal_train",
        gen_behavior="hallucination",
        judge_behaviors=("hallucination",),
        rung="train",
        target_contexts=2000,
        split_spec="half",
        source="packed",
        probe_behavior="hallucination",
    ),
    RungJob(
        name="hal_nqopen",
        gen_behavior="hallucination",
        judge_behaviors=("hallucination",),
        rung="nqopen",
        target_contexts=2000,
        split_spec="half",
        source="packed",
    ),
    RungJob(
        name="hal_simpleqa",
        gen_behavior="hallucination",
        judge_behaviors=("hallucination",),
        rung="simpleqa",
        target_contexts=2000,
        split_spec="half",
        source="packed",
    ),
    RungJob(
        name="evil_train",
        gen_behavior="evil",
        judge_behaviors=("evil",),
        rung="train",
        target_contexts=2000,
        split_spec="half",
        source="packed",
        probe_behavior="evil",
    ),
    RungJob(
        name="wildchat",
        gen_behavior=WILDCHAT_GEN_BEHAVIOR,
        judge_behaviors=BEHAVIORS,
        rung="wildchat_rung",
        target_contexts=None,
        split_spec=(1500, 500),
        source="wcrung",
    ),
    RungJob(
        name="evil_hhrt",
        gen_behavior="evil",
        judge_behaviors=("evil",),
        rung="hhrt",
        target_contexts=None,
        split_spec="half",
        source="packed",
    ),
    RungJob(
        name="syc_aita",
        gen_behavior="sycophancy",
        judge_behaviors=("sycophancy",),
        rung="aita",
        target_contexts=None,
        split_spec="half",
        source="packed",
    ),
    RungJob(
        name="evil_toxicchat",
        gen_behavior="evil",
        judge_behaviors=("evil",),
        rung="toxicchat",
        target_contexts=None,
        split_spec="half",
        source="packed",
    ),
)

RUNG_JOBS_BY_NAME: dict[str, RungJob] = {j.name: j for j in RUNG_JOBS}

# Prompt fields that MUST agree across all K packed rows of one context_id
# (the A24 dedup-to-one premise). `prompt_text` is the load-bearing one; the
# other two are checked because a disagreement there is the same defect.
PROMPT_FIELDS = ("prompt_text", "prefix_text", "query")

# Realized-count floor: whole-group accumulation can undershoot a target by at
# most one group, so a realized count below this fraction of target means the
# rung does not supply what the design requires (A18 structural gate).
REALIZED_FRACTION_FLOOR = 0.9


# ── small utilities ───────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def question_key(query: str) -> str:
    """Deterministic question-axis id for two-way clustering.

    The plan (§4.2, A17) names the evil two-way clustering join key as the
    builder's ``source_id`` (``p{pi:04d}-q{qi:03d}``,
    ``corpus_staging.build_evil_cross``). That field is written into the CONTEXT
    row but ``generation.generate_labeling`` persists a FIXED field set that
    does NOT include it, so it is absent from every packed rollout row (measured:
    0/1510 evil TRAIN rows in shard20 carry it, in-row or in ``meta``). The
    prefix axis survives as ``group_key`` (``prefix{pi:04d}``); this hash of the
    query TEXT restores the question axis as an equivalence class over questions,
    which is exactly what the clustering needs.

    Failure direction is conservative: two DISTINCT question indices with
    byte-identical text collapse into one cluster (fewer, larger clusters =
    WIDER CIs), and one question can never split across keys.
    """
    return sha256_text(query)[:16]


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ},
            ).stdout.strip()
            or "unavailable-no-git-checkout"
        )
    except OSError:
        return "unavailable-no-git-checkout"


# ── banked DV row loading ─────────────────────────────────────────────────────
def load_banked_labeling(behavior: str) -> list[dict]:
    """Rows of the git-tracked banked ``labeling.json`` for an own-rung behavior."""
    path = BANKED_DV_DIR / behavior / "labeling.json"
    if not path.exists():
        raise SystemExit(f"[stage] banked labeling.json missing: {path}")
    rows = json.loads(path.read_text())["rows"]
    logger.info("[stage] %s: %d banked labeling rows (%s)", behavior, len(rows), path.name)
    return rows


def resolve_dataset_revision(revision: str | None) -> str:
    """Resolve the data-repo revision every HF read of this run is pinned to."""
    from explore_persona_space.orchestrate import hub
    from huggingface_hub import HfApi

    if revision:
        return revision
    info = hub.retry_transient(
        lambda: HfApi().repo_info(hub.DEFAULT_DATASET_REPO, repo_type="dataset"),
        what="issue2091 data-repo revision pin",
    )
    return info.sha


def load_wcrung_labeling(
    behavior: str, *, revision: str, stage_dir: Path
) -> tuple[list[dict], dict]:
    """wcrung DV rows from the PINNED HF copy + byte-identity vs the working tree.

    MF-3: ``wildchat_rung/dv_dataset/<b>/labeling.json`` is untracked in every git
    tree (origin/main, local HEAD, origin/issue-1739) and actively mutable by the
    live #1739 session, so ``source: git-issue-branch`` is impossible — the
    CONSUMED copy is the HF one at a pinned revision. A sha256 mismatch against
    the working-tree copy HALTS staging for a deliberate re-pin decision (the
    mutation may be a bug-fix regeneration — artifact-reuse item (j) semantics),
    never a silent pick.

    The comparison copy is resolved in the MAIN checkout
    (:data:`WCRUNG_DV_WORKTREE_DIR`), not this worktree: an untracked file never
    exists in a worktree, so a worktree-relative path would silently degrade the
    check to ``worktree-absent`` and skip the byte comparison entirely.
    """
    from explore_persona_space.orchestrate import hub

    hf_rel = f"{WCRUNG_DV_PREFIX}/{behavior}/labeling.json"
    target = stage_dir / hf_rel
    hub.stage_hub_file(
        hub.DEFAULT_DATASET_REPO,
        hf_rel,
        target,
        repo_type="dataset",
        revision=revision,
    )
    hf_sha = sha256_file(target)

    local = WCRUNG_DV_WORKTREE_DIR / behavior / "labeling.json"
    local_sha = sha256_file(local) if local.exists() else None
    identical = local_sha is not None and local_sha == hf_sha
    if local_sha is not None and not identical:
        raise SystemExit(
            "[stage] MF-3 HALT: wcrung dv_dataset byte-identity mismatch for "
            f"{behavior} — HF@{revision[:12]} sha256 {hf_sha[:16]} != working-tree "
            f"sha256 {local_sha[:16]}. The working tree may carry a bug-fix "
            "regeneration; re-pin deliberately (--dataset-revision) rather than "
            "silently picking one copy."
        )
    check = {
        "behavior": behavior,
        "hf_path": hf_rel,
        "hf_revision": revision,
        "hf_sha256": hf_sha,
        "worktree_path": str(local.relative_to(MAIN_CHECKOUT)) if local.exists() else None,
        "worktree_sha256": local_sha,
        "verdict": "identical" if identical else "worktree-absent",
    }
    logger.info(
        "[stage] wcrung %s dv byte-identity: %s (hf sha %s)",
        behavior,
        check["verdict"],
        hf_sha[:16],
    )
    rows = json.loads(target.read_text())["rows"]
    return rows, check


# ── group-level draw + split ──────────────────────────────────────────────────
def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """context rows grouped by ``group_key`` (fail-loud on a missing key)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        gk = row.get("group_key")
        cid = row.get("context_id")
        if not gk or not cid:
            raise SystemExit(f"[stage] row missing group_key/context_id: id={cid!r} gk={gk!r}")
        groups.setdefault(str(gk), []).append(row)
    return groups


def _shuffled_group_names(groups: dict[str, list[dict]], seed: int) -> list[str]:
    """Deterministic group order: seeded permutation of the SORTED group names."""
    import numpy as np

    names = sorted(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    return [names[i] for i in perm]


def group_level_draw(
    groups: dict[str, list[dict]], target: int | None, order: list[str]
) -> list[str]:
    """Whole groups in ``order`` until the next would exceed ``target``.

    Whole-group accumulation is what makes the subsample GROUP-level: a trim to
    hit ``target`` exactly would split a group across the in/out boundary and
    break every group-level fold downstream. ``target=None`` takes every group.
    """
    if target is None:
        return list(order)
    drawn: list[str] = []
    total = 0
    for name in order:
        size = len(groups[name])
        if total + size > target:
            continue
        drawn.append(name)
        total += size
        if total == target:
            break
    return drawn


def split_groups(
    groups: dict[str, list[dict]],
    drawn: list[str],
    spec: str | tuple[int, int],
) -> dict[str, str]:
    """GROUP-level pool/eval designation over the DRAWN groups, in draw order.

    ``spec="half"`` splits at half the realized context count; a ``(pool, eval)``
    tuple targets explicit context counts (the WildChat rung's 1500/500). The
    split unit is the GROUP throughout — for evil the ``group_key`` is the
    jailbreak PREFIX, so a "1,000 groups per side" framing is impossible there
    (S2) and evil CIs are framed on group-level n.
    """
    realized = sum(len(groups[g]) for g in drawn)
    pool_target = realized // 2 if spec == "half" else int(spec[0])
    designation: dict[str, str] = {}
    pool_n = 0
    for name in drawn:
        if pool_n < pool_target:
            designation[name] = "pool"
            pool_n += len(groups[name])
        else:
            designation[name] = "eval"
    sides = {"pool": 0, "eval": 0}
    for name in drawn:
        sides[designation[name]] += 1
    if sides["pool"] == 0 or sides["eval"] == 0:
        raise SystemExit(
            f"[stage] degenerate split: pool_groups={sides['pool']} eval_groups={sides['eval']} "
            f"over {len(drawn)} drawn groups (realized_contexts={realized}); a side with zero "
            "groups makes every downstream transfer read undefined"
        )
    return designation


# ── packed-shard join ─────────────────────────────────────────────────────────
_PACKED_SHARD_CACHE: dict[tuple[str, str], list[str]] = {}


def packed_shard_paths(behavior: str, *, revision: str) -> list[str]:
    """Repo-relative packed labeling shard paths for a behavior (scoped listing).

    ONE server-side scoped tree walk per (behavior, revision), memoized: the
    ``raw_completions`` prefix holds ~3.1k files and every caller wants the same
    slice of it (never a bare full-repo listing — the ~1M-file data repo wedges).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    cache_key = (behavior, revision)
    if cache_key in _PACKED_SHARD_CACHE:
        return _PACKED_SHARD_CACHE[cache_key]

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = hub.list_hf_files_under_path(
        api,
        hub.DEFAULT_DATASET_REPO,
        PACKED_PREFIX,
        repo_type="dataset",
        revision=revision,
    )
    stem = f"{PACKED_PREFIX}/labeling_{behavior}.shard"
    shards = sorted(p for p in files if p.startswith(stem) and p.endswith(".jsonl"))
    if not shards:
        raise SystemExit(f"[stage] no packed labeling shards for {behavior} under {PACKED_PREFIX}")
    _PACKED_SHARD_CACHE[cache_key] = shards
    return shards


def _iter_packed_docs(path: Path):
    """Yield the inner rollout docs of a packed shard.

    Packed shards are ``pack_raw_tree`` output: one line per SOURCE FILE, shaped
    ``{"src": "labeling/<behavior>/<context_id>_seed<k>.json", "doc": {...}}``.
    The tree's ``_manifest.json`` rides along as a row too, so a doc without a
    ``context_id`` is skipped rather than treated as a rollout.

    Text-mode iteration (never ``str.splitlines()``): real-corpus rows carry raw
    U+2028/U+2029/NEL inside JSON strings, which ``splitlines()`` shreds.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc = row.get("doc", row)
            if not isinstance(doc, dict) or not doc.get("context_id"):
                continue
            yield doc


def join_packed_prompts(
    behavior: str,
    needed: set[str],
    *,
    revision: str,
    stage_dir: Path,
) -> tuple[dict[str, dict], dict]:
    """Resolve each needed ``context_id`` to ONE packed prompt row.

    Scans the behavior's packed shards, early-exiting once every needed id is
    resolved. Two fail-loud conditions, both plan-mandated (A24):

    * a drawn id with ZERO packed rows after the full scan;
    * a drawn id whose K rows CONFLICT on any prompt field — i.e. realized
      ``n_distinct(prompt_text) per context_id > 1``, which would make
      dedup-to-one lossy.

    The realized distinct-count distribution is MEASURED over every drawn
    context (not sampled) and returned in the digest.
    """
    from explore_persona_space.orchestrate import hub

    shards = packed_shard_paths(behavior, revision=revision)
    resolved: dict[str, dict] = {}
    # context_id -> field -> set of value shas (the A24 measurement)
    seen_shas: dict[str, dict[str, set[str]]] = {}
    n_rows_scanned = 0
    n_shards_scanned = 0
    for rel in shards:
        if len(resolved) == len(needed):
            break
        local = stage_dir / rel
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            rel,
            local,
            repo_type="dataset",
            revision=revision,
        )
        n_shards_scanned += 1
        for doc in _iter_packed_docs(local):
            n_rows_scanned += 1
            cid = doc["context_id"]
            if cid not in needed:
                continue
            per_field = seen_shas.setdefault(cid, {f: set() for f in PROMPT_FIELDS})
            for field in PROMPT_FIELDS:
                per_field[field].add(sha256_text(doc.get(field) or ""))
            resolved.setdefault(cid, doc)

    missing = sorted(needed - set(resolved))
    if missing:
        raise SystemExit(
            f"[stage] {behavior}: {len(missing)} drawn context_id(s) have ZERO packed rows "
            f"after scanning {n_shards_scanned}/{len(shards)} shards; first few: {missing[:5]}"
        )

    conflicts = {
        cid: {f: len(s) for f, s in fields.items() if len(s) != 1}
        for cid, fields in seen_shas.items()
        if any(len(s) != 1 for s in fields.values())
    }
    if conflicts:
        sample = dict(list(conflicts.items())[:5])
        raise SystemExit(
            f"[stage] {behavior}: A24 VIOLATED — {len(conflicts)} drawn context_id(s) carry "
            f"CONFLICTING prompt fields across their K packed rows, so dedup-to-one would be "
            f"lossy. Per-id distinct counts (first few): {sample}"
        )

    max_distinct = {
        field: max((len(fields[field]) for fields in seen_shas.values()), default=0)
        for field in PROMPT_FIELDS
    }
    n_with_source_id = sum(
        1
        for doc in resolved.values()
        if doc.get("source_id") or (doc.get("meta") or {}).get("source_id")
    )
    n_with_aliases = sum(1 for doc in resolved.values() if doc.get("answer_aliases"))
    digest = {
        "behavior": behavior,
        "n_needed": len(needed),
        "n_resolved": len(resolved),
        "n_shards_available": len(shards),
        "n_shards_scanned": n_shards_scanned,
        "n_rows_scanned": n_rows_scanned,
        # A24: measured over EVERY drawn context, all rungs of this behavior.
        "a24_max_distinct_per_context": max_distinct,
        "a24_verdict": "all-contexts-homogeneous"
        if max(max_distinct.values()) <= 1
        else "VIOLATED",
        # A17: measured source_id coverage (expected 0 — see question_key()).
        "source_id_coverage": f"{n_with_source_id}/{len(resolved)}",
        "answer_aliases_coverage": f"{n_with_aliases}/{len(resolved)}",
    }
    logger.info(
        "[stage] %s join: resolved=%d/%d shards=%d/%d a24_max_distinct=%s source_id=%s aliases=%s",
        behavior,
        len(resolved),
        len(needed),
        n_shards_scanned,
        len(shards),
        max_distinct,
        digest["source_id_coverage"],
        digest["answer_aliases_coverage"],
    )
    return resolved, digest


# ── context-row emission ──────────────────────────────────────────────────────
def build_context_row(
    job: RungJob,
    banked: dict,
    packed: dict | None,
    designation: str,
) -> dict:
    """One staged context row in the ``generate_labeling`` input schema.

    ``split`` carries the #2091 pool/eval DESIGNATION (not the banked
    train/eval split, which is preserved as ``meta.banked_split``) because
    ``generate_labeling`` copies ``row["split"]`` straight into every rollout
    payload and from there into the capture ``row_index`` — so the designation
    travels with the vectors for free.
    """
    cid = banked["context_id"]
    prompt_text: str | None = None
    if job.source == "wcrung":
        query = banked["query"]
        prefix_turns = banked.get("prefix_turns") or []
        prefix_text = banked.get("prefix_text", "")
        packed_prompt_sha = None
    else:
        assert packed is not None
        query = packed["query"]
        prefix_turns = []
        # NOTE: the packed row's `prefix_text` is the RENDERED chat-template
        # prefix, NOT the raw persona string — `generate_labeling` persists
        # `render_prompt_parts()`'s output (generation.py's rollout payload), which
        # is `prompt[:first user-turn header]`. Re-rendering it through
        # `context_messages` would nest an already-rendered prefix inside a fresh
        # system turn and DOUBLE-WRAP the template, producing a different prompt
        # than the banked campaign saw (caught by the pod's render-parity assert
        # at 40/40 contexts during the staging smoke). So the banked
        # (prefix_text, prompt_text) PAIR is carried verbatim and the pod replays
        # it through `banked_render_fn` instead of re-rendering: the greedy decode
        # then runs on byte-identical prompts to the banked stochastic rollouts,
        # which is exactly the comparability this task's regime contrast needs.
        prefix_text = packed.get("prefix_text") or ""
        prompt_text = packed.get("prompt_text") or ""
        if not prompt_text.startswith(prefix_text):
            raise SystemExit(
                f"[stage] {cid}: banked prompt_text does not start with its prefix_text — "
                "capture derives prefix_end from len(prefix_text) against the prompt, so a "
                "non-prefix pair would mis-position every prefix-arm read"
            )
        if query and query not in prompt_text:
            raise SystemExit(
                f"[stage] {cid}: banked query is absent from its prompt_text — the "
                "context_id join resolved mismatched fields"
            )
        packed_prompt_sha = sha256_text(prompt_text)

    row: dict = {
        "context_id": cid,
        "behavior": job.gen_behavior,
        "rung": job.rung,
        "rungjob": job.name,
        "split": designation,
        "group_key": str(banked["group_key"]),
        "query": query,
        "prefix_text": prefix_text,
        "meta": {
            "banked_split": banked.get("split"),
            "banked_rung": banked.get("rung"),
            "judge_behaviors": list(job.judge_behaviors),
            # Question axis for two-way clustering (see question_key()).
            "question_key": question_key(query),
            # Absent from every packed row in practice; carried when present so a
            # future re-pack that propagates it is picked up automatically.
            "source_id": (packed or {}).get("source_id")
            or ((packed or {}).get("meta") or {}).get("source_id"),
            # Render-parity anchor: the pod asserts the prompt its render_fn
            # returns hashes to this, so a shard-corruption / wrong-join /
            # template drift that would silently mis-position every capture
            # fails loud instead.
            "banked_prompt_sha256": packed_prompt_sha,
        },
    }
    if prompt_text is not None:
        # Replayed verbatim by the pod's `banked_render_fn` (see above).
        row["prompt_text"] = prompt_text
    if job.source == "wcrung":
        row["prefix_turns"] = prefix_turns
        row["single_turn"] = banked.get("single_turn")
    aliases = (packed or {}).get("answer_aliases")
    if aliases:
        row["answer_aliases"] = aliases
    return row


def build_probe_rows(
    behavior: str,
    *,
    revision: str,
    stage_dir: Path,
    n_rows: int,
) -> tuple[list[dict], dict]:
    """Banked rollout rows for the cross-campaign capture-parity probe (MF-2).

    Drawn from ONE packed shard per behavior so the banked reference vectors
    resolve from a small store-member set. Each row carries the banked
    completion verbatim — the pod teacher-forces it through THIS capture rig and
    the per-behavior cosines are computed against the banked slices (deferred to
    P4, the plan's either-or branch, because streaming a member set out of a
    32-70 GB labeling tar is not a cheap pod-side fetch).
    """
    from explore_persona_space.orchestrate import hub

    shards = packed_shard_paths(behavior, revision=revision)
    rel = shards[0]
    local = stage_dir / rel
    hub.stage_hub_file(
        hub.DEFAULT_DATASET_REPO,
        rel,
        local,
        repo_type="dataset",
        revision=revision,
    )
    rows: list[dict] = []
    for doc in _iter_packed_docs(local):
        if len(rows) >= n_rows:
            break
        rows.append(
            {
                "context_id": doc["context_id"],
                "behavior": behavior,
                "rollout_k": doc.get("rollout_k"),
                "rung": doc.get("rung"),
                "group_key": doc.get("group_key"),
                "prefix_text": doc.get("prefix_text") or "",
                "prompt_text": doc.get("prompt_text") or "",
                "completion": doc.get("completion") or "",
                "meta": {"banked_source_shard": rel, "banked_src_field": doc.get("finish_reason")},
            }
        )
    if not rows:
        raise SystemExit(f"[stage] parity probe: no rollout rows found in {rel}")
    digest = {
        "behavior": behavior,
        "n_rows": len(rows),
        "source_shard": rel,
        "n_distinct_contexts": len({r["context_id"] for r in rows}),
    }
    logger.info(
        "[stage] parity probe %s: %d rows from %s (%d distinct contexts)",
        behavior,
        len(rows),
        Path(rel).name,
        digest["n_distinct_contexts"],
    )
    return rows, digest


# ── shard writing ─────────────────────────────────────────────────────────────
def shard_rows(rows: list[dict], dest_dir: Path, stem: str) -> dict:
    """Write ``rows`` as <= 9 MB jsonl line-shards + a manifest; return manifest.

    Deterministic: shard boundaries depend only on (ordering, serialized bytes,
    cap). Memory-bounded (one serialized row at a time into the open handle).
    """
    if not rows:
        raise ValueError(f"refusing to shard an empty row list for {stem}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob(f"{stem}.shard*.jsonl"):
        stale.unlink()

    shards: list[dict] = []
    idx = 0
    handle = None
    written = 0
    lines = 0
    try:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
            if handle is None or (written + len(line) > SHARD_MAX_BYTES and lines > 0):
                if handle is not None:
                    handle.close()
                    shards.append(
                        {
                            "name": f"{stem}.shard{idx:02d}.jsonl",
                            "n_rows": lines,
                            "n_bytes": written,
                        }
                    )
                    idx += 1
                handle = (dest_dir / f"{stem}.shard{idx:02d}.jsonl").open("wb")
                written = 0
                lines = 0
            handle.write(line)
            written += len(line)
            lines += 1
    finally:
        if handle is not None:
            handle.close()
    shards.append({"name": f"{stem}.shard{idx:02d}.jsonl", "n_rows": lines, "n_bytes": written})
    for shard in shards:
        shard["sha256"] = sha256_file(dest_dir / shard["name"])

    manifest = {
        "schema": SCHEMA,
        "stem": stem,
        "n_rows": sum(s["n_rows"] for s in shards),
        "n_shards": len(shards),
        "shard_max_bytes": SHARD_MAX_BYTES,
        "shards": shards,
    }
    if manifest["n_rows"] != len(rows):
        raise RuntimeError(
            f"shard row-count mismatch for {stem}: {manifest['n_rows']} != {len(rows)}"
        )
    write_json_atomic(dest_dir / f"{stem}.manifest.json", manifest)
    return manifest


def load_shard_rows(shard_dir: Path, stem: str) -> list[dict]:
    """Reassemble + VERIFY a shard set written by :func:`shard_rows`.

    Fail-loud on a missing shard, a sha256 mismatch, a per-shard line-count
    mismatch, or a total-count mismatch — a silently truncated context set would
    shrink a rung with no downstream signal.
    """
    manifest_path = shard_dir / f"{stem}.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {stem}.manifest.json under {shard_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected shard schema {manifest.get('schema')!r} (want {SCHEMA!r})")
    rows: list[dict] = []
    for shard in manifest["shards"]:
        path = shard_dir / shard["name"]
        if not path.exists():
            raise FileNotFoundError(f"manifest names {shard['name']} but it is missing")
        digest = sha256_file(path)
        if digest != shard["sha256"]:
            raise RuntimeError(
                f"{shard['name']} sha256 mismatch: on-disk {digest[:16]} != "
                f"manifest {shard['sha256'][:16]}"
            )
        with path.open(encoding="utf-8") as fh:
            shard_rows_ = [json.loads(line) for line in fh if line.strip()]
        if len(shard_rows_) != shard["n_rows"]:
            raise RuntimeError(
                f"{shard['name']} row-count mismatch: {len(shard_rows_)} != {shard['n_rows']}"
            )
        rows.extend(shard_rows_)
    if len(rows) != manifest["n_rows"]:
        raise RuntimeError(f"total row-count mismatch: {len(rows)} != {manifest['n_rows']}")
    return rows


# ── staging driver ────────────────────────────────────────────────────────────
def stage(args: argparse.Namespace) -> dict:
    contexts_root = args.out_dir / "contexts"
    contexts_root.mkdir(parents=True, exist_ok=True)
    revision = resolve_dataset_revision(args.dataset_revision)
    logger.info("[stage] data-repo revision pin: %s", revision)

    # Banked DV rows: git for own-rung behaviors, pinned HF for wcrung (MF-3).
    banked_by_behavior = {b: load_banked_labeling(b) for b in BEHAVIORS}
    wcrung_checks: list[dict] = []
    wcrung_rows_by_behavior: dict[str, list[dict]] = {}
    for behavior in BEHAVIORS:
        rows, check = load_wcrung_labeling(behavior, revision=revision, stage_dir=args.stage_dir)
        wcrung_rows_by_behavior[behavior] = rows
        wcrung_checks.append(check)

    # WildChat prompts come from the wcrung CONTEXTS shards, never packed text.
    from scripts import issue1739_wcrung_rows_io as rows_io

    wc_prompt_rows = rows_io.stage_rows_from_hub(
        hf_prefix=WCRUNG_CONTEXTS_PREFIX,
        dest_dir=args.stage_dir,
        revision=revision,
    )
    wc_prompts_by_id = {r["context_id"]: r for r in wc_prompt_rows}
    logger.info("[stage] wcrung prompt rows: %d", len(wc_prompt_rows))

    # ── draw + split per rung-job ────────────────────────────────────────────
    jobs_manifest: dict[str, dict] = {}
    needed_by_behavior: dict[str, set[str]] = {b: set() for b in BEHAVIORS}
    drawn_by_job: dict[str, tuple[dict[str, list[dict]], list[str], dict[str, str]]] = {}

    for job_idx, job in enumerate(RUNG_JOBS):
        if job.source == "wcrung":
            # The 2,000 wcrung contexts are shared across behaviors; the evil
            # copy is the row set (identical context ids in all three).
            rung_rows = [r for r in wcrung_rows_by_behavior["evil"] if r.get("rung") == job.rung]
            if not rung_rows:
                rung_rows = wcrung_rows_by_behavior["evil"]
        else:
            rung_rows = [
                r for r in banked_by_behavior[job.gen_behavior] if r.get("rung") == job.rung
            ]
        if not rung_rows:
            raise SystemExit(f"[stage] {job.name}: no banked rows with rung={job.rung!r}")

        groups = group_rows(rung_rows)
        # Per-job seed offset keeps the 9 draws independent while staying a pure
        # function of (SEED, job index) — reproducible from the seed alone.
        order = _shuffled_group_names(groups, SEED + job_idx)
        target = job.target_contexts
        if args.limit_contexts_per_rung is not None:
            target = min(args.limit_contexts_per_rung, target or args.limit_contexts_per_rung)
        drawn = group_level_draw(groups, target, order)
        realized = sum(len(groups[g]) for g in drawn)
        if target is not None and realized < REALIZED_FRACTION_FLOOR * target:
            raise SystemExit(
                f"[stage] {job.name}: realized {realized} contexts < "
                f"{REALIZED_FRACTION_FLOOR:.0%} of target {target} over {len(groups)} available "
                f"groups — the rung does not supply what the design requires (A18)"
            )
        split_spec = job.split_spec
        if args.limit_contexts_per_rung is not None and isinstance(split_spec, tuple):
            # Scale an explicit (pool, eval) target down proportionally so a
            # smoke slice still exercises both sides of the designation.
            split_spec = "half"
        designation = split_groups(groups, drawn, split_spec)
        drawn_by_job[job.name] = (groups, drawn, designation)

        if job.source == "packed":
            for g in drawn:
                for row in groups[g]:
                    needed_by_behavior[job.gen_behavior].add(row["context_id"])

        pool_ctx = sum(len(groups[g]) for g in drawn if designation[g] == "pool")
        jobs_manifest[job.name] = {
            "name": job.name,
            "gen_behavior": job.gen_behavior,
            "judge_behaviors": list(job.judge_behaviors),
            "rung": job.rung,
            "source": job.source,
            "probe_behavior": job.probe_behavior,
            "target_contexts": target,
            "n_groups_available": len(groups),
            "n_groups_drawn": len(drawn),
            "n_contexts_realized": realized,
            "n_groups_pool": sum(1 for g in drawn if designation[g] == "pool"),
            "n_groups_eval": sum(1 for g in drawn if designation[g] == "eval"),
            "n_contexts_pool": pool_ctx,
            "n_contexts_eval": realized - pool_ctx,
            "split_spec": (split_spec if isinstance(split_spec, str) else list(split_spec)),
            "draw_seed": SEED + job_idx,
        }
        logger.info(
            "[stage] %s: groups %d/%d drawn, contexts %d (pool %d / eval %d)",
            job.name,
            len(drawn),
            len(groups),
            realized,
            pool_ctx,
            realized - pool_ctx,
        )

    # ── join drawn ids to packed prompt rows ─────────────────────────────────
    packed_by_behavior: dict[str, dict[str, dict]] = {}
    join_digests: list[dict] = []
    for behavior in BEHAVIORS:
        needed = needed_by_behavior[behavior]
        if not needed:
            continue
        resolved, digest = join_packed_prompts(
            behavior, needed, revision=revision, stage_dir=args.stage_dir
        )
        packed_by_behavior[behavior] = resolved
        join_digests.append(digest)

    # ── emit per-job context shards ─────────────────────────────────────────
    for job in RUNG_JOBS:
        groups, drawn, designation = drawn_by_job[job.name]
        rows: list[dict] = []
        for g in drawn:
            for banked in groups[g]:
                cid = banked["context_id"]
                if job.source == "wcrung":
                    prompt_row = wc_prompts_by_id.get(cid)
                    if prompt_row is None:
                        raise SystemExit(
                            f"[stage] {job.name}: wcrung context {cid} has no prompt row in "
                            f"{WCRUNG_CONTEXTS_PREFIX} (prompt/DV set mismatch)"
                        )
                    merged = dict(banked)
                    merged["query"] = prompt_row["query"]
                    merged["prefix_turns"] = prompt_row.get("prefix_turns") or []
                    merged["prefix_text"] = prompt_row.get("prefix_text", "")
                    merged["single_turn"] = prompt_row.get("single_turn")
                    rows.append(build_context_row(job, merged, None, designation[g]))
                else:
                    packed = packed_by_behavior[job.gen_behavior][cid]
                    rows.append(build_context_row(job, banked, packed, designation[g]))
        ids = [r["context_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"[stage] {job.name}: duplicate context_id in emitted rows")
        manifest = shard_rows(rows, contexts_root / job.name, "ctx")
        jobs_manifest[job.name]["shards"] = manifest["shards"]
        jobs_manifest[job.name]["n_rows_emitted"] = manifest["n_rows"]
        # Round-trip the shards we just wrote (verify path == the pod's path).
        reread = load_shard_rows(contexts_root / job.name, "ctx")
        if len(reread) != len(rows):
            raise RuntimeError(f"[stage] {job.name}: shard round-trip mismatch")

    # ── parity-probe shards (one per behavior) ──────────────────────────────
    probe_digests: list[dict] = []
    for behavior in BEHAVIORS:
        rows, digest = build_probe_rows(
            behavior,
            revision=revision,
            stage_dir=args.stage_dir,
            n_rows=args.probe_rows,
        )
        manifest = shard_rows(rows, contexts_root / "parity_probe" / behavior, "probe")
        digest["shards"] = manifest["shards"]
        probe_digests.append(digest)

    stage_manifest = {
        "schema": SCHEMA,
        "issue": 2091,
        "seed": SEED,
        "dataset_revision": revision,
        "hf_contexts_prefix": args.hf_prefix,
        "n_rung_jobs": len(RUNG_JOBS),
        "n_contexts_total": sum(j["n_contexts_realized"] for j in jobs_manifest.values()),
        "rung_jobs": jobs_manifest,
        "packed_join_digests": join_digests,
        "parity_probes": probe_digests,
        "wcrung_dv_byte_identity": wcrung_checks,
        "limit_contexts_per_rung": args.limit_contexts_per_rung,
        "probe_rows": args.probe_rows,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": {
            "source_id": (
                "ABSENT from packed rollout rows (generate_labeling persists a fixed "
                "field set); the evil two-way clustering question axis is "
                "meta.question_key = sha256(query)[:16] — see question_key()"
            ),
            "split_semantics": (
                "row['split'] carries the #2091 pool/eval DESIGNATION; the banked "
                "train/eval split is preserved as meta.banked_split"
            ),
        },
    }
    write_json_atomic(contexts_root / "stage_manifest.json", stage_manifest)
    logger.info(
        "[stage] manifest written: %d rung-jobs, %d contexts total",
        stage_manifest["n_rung_jobs"],
        stage_manifest["n_contexts_total"],
    )
    return stage_manifest


def upload_contexts(args: argparse.Namespace) -> str:
    """Bulk-upload the contexts tree in ONE commit (never a per-file loop)."""
    if args.skip_upload:
        logger.info("[stage] SKIP upload (--skip-upload): %s", args.out_dir / "contexts")
        return ""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        args.out_dir / "contexts",
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        args.hf_prefix,
        raise_on_error=True,
    )
    logger.info("[stage] uploaded contexts -> %s (%s)", args.hf_prefix, url or "no-url")
    return url


def _import_check() -> int:
    """Resolve every deferred import on the REAL branch, then exit 0.

    Hoisted into its own function on purpose: an ``import X`` inside ``main()``
    would bind X as a local for main's WHOLE body and shadow any module-level
    symbol of the same name (the #1739 ``UnboundLocalError`` class).
    """
    import numpy  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload,
        list_hf_files_under_path,
        retry_transient,
        stage_hub_file,
    )
    from scripts import issue1739_wcrung_rows_io  # noqa: F401
    from scripts.issue1739_wcrung_rows_io import stage_rows_from_hub  # noqa: F401

    print("[import-check] OK: all deferred imports resolved", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_2091" / "staging",
        help="local root; shards land under <out-dir>/contexts/",
    )
    ap.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2091_hf_dl"),
        help="mirror root for HF-staged inputs (multi-GB: never / or /tmp)",
    )
    # UPLOAD_PREFIX_EXEMPT: issue-2091-specific staging leg; --hf-prefix overrides
    ap.add_argument("--hf-prefix", default=CONTEXTS_PREFIX)
    ap.add_argument("--dataset-revision", default=None, help="pin (default: resolve at run time)")
    ap.add_argument("--probe-rows", type=int, default=150, help="parity-probe rows per behavior")
    ap.add_argument(
        "--limit-contexts-per-rung",
        type=int,
        default=None,
        help="SMOKE ONLY: cap each rung-job's drawn context count (identical code path)",
    )
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE ONLY: no Hub writes")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the REAL branch, then exit 0",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.import_check:
        return _import_check()

    print("[phase=stage_contexts]", flush=True)
    manifest = stage(args)
    url = upload_contexts(args)
    print(
        f"[phase=stage_done] rung_jobs={manifest['n_rung_jobs']} "
        f"contexts={manifest['n_contexts_total']} revision={manifest['dataset_revision'][:12]} "
        f"upload={url or 'skipped'}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
