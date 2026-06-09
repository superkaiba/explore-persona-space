#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Deterministic builder for the four #521 input JSONs.

Outputs under ``--output-dir``:

- ``personas.json``  -- ``{persona_name: system_prompt}`` for the realizable
  N=14 panel (= ``PERSONAS`` minus the parameterized ``local_resident`` plus
  the bare ``assistant``). Schema matches ``activation_shift.py`` CLI.
- ``questions.json`` -- ``list[str]`` of the 20 generic eval questions
  (``EVAL_QUESTIONS`` verbatim).
- ``marker_pool.json`` -- ``list[str]`` of N=58 marker-arm steering-pool
  questions, hash-disjoint from the #519 marker training mix. The plan
  §4 Step 2 named N=100, but the marker training pool only carries 197
  unique questions and #519 trained on 139 of them — leaving 58 held-out
  candidates (the data ceiling). N=58 is still well above the
  ``steering_vectors.py`` ``min_pool_size=30`` floor. The deviation is
  recorded as ``epm:plan-deviation v1`` on task #521 and carried as a
  scope caveat into the clean-result. Override via
  ``--marker-pool-target``.
- ``em_pool.json``    -- ``list[str]`` of EM steering-pool questions
  (rows 200..299 of ``bad_medical_advice_6k.jsonl``), hash-disjoint from
  the #519 EM training prompts. **v2 M5**: 2 detected overlap rows are
  filtered out, post-filter count typically N=98.
- ``em_pool_disjointness.txt`` -- per-overlap log of which prompt-hashes
  were dropped + the post-filter count.
- ``base_cosines_questions.json`` -- ``list[str]`` of N=20 held-out
  validation questions used by ``scripts/issue_521_build_base_cosines.py``
  (Phase D base-cosines pool). Per plan §551, this is the last 20 of
  ``data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl``'s
  EVAL SPLIT — i.e. the rows NOT picked into #519's 200 positive training
  questions (the eval split is rows whose index is in
  ``set(range(600)) - set(shuffled[:200])`` under
  ``random.Random(0xC0FFEE)``). Hash-disjoint from BOTH ``EVAL_QUESTIONS``
  (the Phase C eval pool — ``questions.json``) AND the #519 marker
  training mix prompts. Hard-fails at build time if any overlap is
  detected (round-3 fix for ``base-cosines-question-overlap``).

The script is re-runnable and deterministic. The ``base_cosines.json``
numerical computation is a SEPARATE concern (GPU-bound, runs on the pod
via ``scripts/issue_521_build_base_cosines.py``); this builder produces
the QUESTION SET that builder consumes via ``--questions-json
base_cosines_questions.json``.

Run::

    uv run python scripts/issue_521_build_inputs.py \\
        --output-dir eval_results/issue_521/inputs \\
        [--tiny]           # smoke: 2 personas, 2 questions, 4-row pool
        [--marker-pool-jsonl /path/to/marker_villain_asst_excluded_medium.jsonl]
        [--em-pool-hf-revision main]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# The realizable N=14 panel.
# `medical_doctor` (source) + 4 contrastive negatives + 9 held-out
# personas. `local_resident` is excluded (parameterized `{town}, {state}`
# would be a confound at the cross-arm activation-shift readout — see
# plan §11 row 4 + §12 #4). The plan's parent (#519 §10) named a
# notional 24-panel; the realizable composition is 14, flagged in the
# clean-result as a scope caveat.
PANEL_NAMES_INCLUDED: tuple[str, ...] = (
    "medical_doctor",
    "comedian",
    "police_officer",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "librarian",
    "french_person",
    "villain",
    "zelthari_scholar",
    "biographer",
    "marine_biologist",
    "local_historian",
)
# `assistant` is added separately from ASSISTANT_PROMPT.

# Plan §4 Step 2: marker pool default path on the repo root (data/
# is gitignored, lives at repo root not in worktree).
DEFAULT_MARKER_POOL_JSONL = "data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl"
# bad_medical_advice_6k.jsonl on the HF data repo.
EM_POOL_DATASET_REPO = "superkaiba1/explore-persona-space-data"
EM_POOL_DATASET_FILE = "issue376_em/v1/bad_medical_advice_6k.jsonl"
# Per plan §4 Step 2: rows 200..299 (held out from #519's rows 0..199
# training split).
EM_POOL_ROW_START = 200
EM_POOL_ROW_END = 300  # exclusive
EM_POOL_TARGET_SIZE = EM_POOL_ROW_END - EM_POOL_ROW_START  # 100

# Plan §4 Step 2: #519 EM training mix on HF (note: actual filename is
# `em_seed{S}.jsonl`, NOT `em_seed{S}_train.jsonl` — the plan §10 row
# was slightly off; verified via list_repo_files at build time).
EM_TRAIN_DATA_REPO = "superkaiba1/explore-persona-space-data"
EM_TRAIN_DATA_FILES = (
    "issue_519/em_seed42.jsonl",
    "issue_519/em_seed137.jsonl",
    "issue_519/em_seed256.jsonl",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _prompt_hash(text: str) -> str:
    """Stable sha256 of a stripped user-turn prompt."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _build_personas(tiny: bool) -> dict[str, str]:
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    out: dict[str, str] = {}
    names = PANEL_NAMES_INCLUDED if not tiny else PANEL_NAMES_INCLUDED[:2]
    for name in names:
        if name not in PERSONAS:
            raise KeyError(
                f"Panel persona {name!r} not in PERSONAS. Update PANEL_NAMES_INCLUDED "
                f"or src/explore_persona_space/personas.py."
            )
        out[name] = PERSONAS[name]
    if not tiny:
        out["assistant"] = ASSISTANT_PROMPT
    return out


def _build_questions(tiny: bool) -> list[str]:
    from explore_persona_space.personas import EVAL_QUESTIONS

    qs = list(EVAL_QUESTIONS)
    if tiny:
        return qs[:2]
    return qs


def _load_marker_training_question_hashes(  # noqa: C901 - sequential branches, refactor out-of-scope
    repo_root: Path, marker_pool_jsonl: Path
) -> tuple[set[str], list[str]]:
    """Return (training question hashes, held-out candidate questions).

    The marker training pool source is `marker_villain_asst_excluded_medium.jsonl`
    (600 rows; #519 trained on the first N per seed). The training mix
    files on HF data repo at `issue_519/marker_seed{S}.jsonl` carry the
    exact training questions — we hash THOSE so the steering pool is
    guaranteed disjoint from the actual training distribution.

    The candidate pool for the steering pool draws from the FULL
    `marker_villain_asst_excluded_medium.jsonl` minus the trained rows.
    Returns the candidate pool (deduplicated) plus the training-hash set
    for the disjointness assertion.
    """
    from huggingface_hub import hf_hub_download

    train_hashes: set[str] = set()
    for fn in (
        "issue_519/marker_seed42.jsonl",
        "issue_519/marker_seed137.jsonl",
        "issue_519/marker_seed256.jsonl",
    ):
        p = hf_hub_download(EM_TRAIN_DATA_REPO, filename=fn, repo_type="dataset", revision="main")
        with Path(p).open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", [])
                user_msgs = [m for m in prompt if m.get("role") == "user"]
                if not user_msgs:
                    continue
                q = user_msgs[0].get("content", "")
                if q:
                    train_hashes.add(_prompt_hash(q))

    if not marker_pool_jsonl.is_absolute():
        # Fall back to the shared main-repo root (--git-common-dir is the
        # main .git/ for a worktree; its parent is the main repo) when the
        # worktree doesn't carry `data/` (it is gitignored, so data lives
        # at the shared repo root).
        candidate_paths = [repo_root / marker_pool_jsonl]
        import subprocess as _sp

        common_git = Path(
            _sp.check_output(["git", "rev-parse", "--git-common-dir"]).decode().strip()
        )
        if not common_git.is_absolute():
            common_git = repo_root / common_git
        main_repo_root = common_git.parent
        if main_repo_root != repo_root:
            candidate_paths.append(main_repo_root / marker_pool_jsonl)
        for cp in candidate_paths:
            if cp.exists():
                marker_pool_jsonl = cp
                break
        else:
            marker_pool_jsonl = candidate_paths[0]

    candidates: list[str] = []
    seen: set[str] = set()
    if marker_pool_jsonl.exists():
        with marker_pool_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", [])
                user_msgs = [m for m in prompt if m.get("role") == "user"]
                if not user_msgs:
                    continue
                q = user_msgs[0].get("content", "")
                if not q:
                    continue
                h = _prompt_hash(q)
                if h in seen:
                    continue
                seen.add(h)
                candidates.append(q)
        logger.info(
            "loaded %d unique candidate questions from %s",
            len(candidates),
            marker_pool_jsonl,
        )
    else:
        logger.warning(
            "marker pool JSONL %s not found; falling back to EVAL_QUESTIONS_A3 if available",
            marker_pool_jsonl,
        )

    return train_hashes, candidates


def _build_marker_pool(
    repo_root: Path,
    marker_pool_jsonl: Path,
    *,
    n_target: int,
    tiny: bool,
    allow_smaller: bool = False,
) -> list[str]:
    """Pick N marker-pool questions hash-disjoint from #519 marker training.

    Disjointness is asserted post-pick. With ``allow_smaller=True`` an
    insufficient candidate pool yields a smaller pool + a WARNING; with
    ``allow_smaller=False`` it raises. (The marker training corpus has
    197 unique questions and #519 trained on 139 → max 58 held-out.)
    """
    train_hashes, candidates = _load_marker_training_question_hashes(repo_root, marker_pool_jsonl)
    target = 4 if tiny else n_target
    picked: list[str] = []
    for q in candidates:
        h = _prompt_hash(q)
        if h in train_hashes:
            continue
        picked.append(q)
        if len(picked) >= target:
            break
    if len(picked) < target:
        if not allow_smaller:
            raise RuntimeError(
                f"marker_pool: only found {len(picked)} held-out candidates from "
                f"{marker_pool_jsonl}; need ≥{target}. Pass --marker-pool-allow-smaller "
                f"to accept the smaller pool, OR check the training-pool hash set "
                f"against the candidate JSONL — they may not be drawn from the same "
                f"source distribution."
            )
        logger.warning(
            "marker_pool: only %d held-out candidates available (target %d); "
            "accepting smaller pool per --marker-pool-allow-smaller",
            len(picked),
            target,
        )
    # Post-pick disjointness assertion (paranoid; would already raise above).
    for q in picked:
        assert _prompt_hash(q) not in train_hashes, (
            f"marker_pool: post-pick disjointness failed for {q[:80]!r}"
        )
    return picked


def _load_em_train_prompt_hashes() -> set[str]:
    """Hash every user-turn prompt across all 3 #519 EM training mixes."""
    from huggingface_hub import hf_hub_download

    hashes: set[str] = set()
    for fn in EM_TRAIN_DATA_FILES:
        p = hf_hub_download(EM_TRAIN_DATA_REPO, filename=fn, repo_type="dataset", revision="main")
        with Path(p).open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", [])
                user_msgs = [m for m in prompt if m.get("role") == "user"]
                if not user_msgs:
                    continue
                q = user_msgs[0].get("content", "")
                if q:
                    hashes.add(_prompt_hash(q))
        logger.info("loaded EM train hashes from %s (cumulative: %d)", fn, len(hashes))
    return hashes


def _build_em_pool(*, tiny: bool, disjointness_log: Path) -> tuple[list[str], int]:
    """Build the EM steering pool from rows 200..299 of bad_medical_advice_6k.

    v2 M5: filter out any row whose user-turn prompt hash is in the
    actual #519 EM training prompt hash set. Log dropped hashes to
    ``disjointness_log``. Returns (filtered pool, dropped count).
    """
    from huggingface_hub import hf_hub_download

    em_path = hf_hub_download(
        EM_POOL_DATASET_REPO,
        filename=EM_POOL_DATASET_FILE,
        repo_type="dataset",
        revision="main",
    )
    rows: list[dict] = []
    with Path(em_path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if tiny:
        slice_rows = rows[EM_POOL_ROW_START : EM_POOL_ROW_START + 4]
        target_size = 4
    else:
        slice_rows = rows[EM_POOL_ROW_START:EM_POOL_ROW_END]
        target_size = EM_POOL_TARGET_SIZE
    if len(slice_rows) != target_size:
        raise RuntimeError(
            f"em_pool: row-slice {EM_POOL_ROW_START}..{EM_POOL_ROW_END} returned "
            f"{len(slice_rows)} rows, expected {target_size}. Upstream "
            f"bad_medical_advice_6k.jsonl may be shorter than 300 rows."
        )

    prompts: list[str] = []
    for row in slice_rows:
        msgs = row.get("messages", [])
        user = [m for m in msgs if m.get("role") == "user"]
        if not user:
            continue
        q = user[0].get("content", "")
        if q:
            prompts.append(q)

    # v2 M5: disjointness filter vs the actual #519 training prompts.
    if tiny:
        # tiny mode skips the hash load (the training files would be
        # paid network IO unnecessarily); just log target = 4 as-is.
        train_hashes: set[str] = set()
    else:
        train_hashes = _load_em_train_prompt_hashes()

    filtered: list[str] = []
    dropped: list[dict] = []
    for q in prompts:
        h = _prompt_hash(q)
        if h in train_hashes:
            dropped.append({"prompt_hash": h, "first_80_chars": q[:80]})
            continue
        filtered.append(q)

    # Post-filter assertion: intersection of filtered hashes with training
    # hashes must be empty.
    filtered_hashes = {_prompt_hash(q) for q in filtered}
    overlap = filtered_hashes & train_hashes
    assert not overlap, f"em_pool: post-filter disjointness violation: {overlap}"

    # Write the log even when empty so the analyzer can see "we checked".
    disjointness_log.parent.mkdir(parents=True, exist_ok=True)
    with disjointness_log.open("w") as f:
        f.write("em_pool disjointness log — written by scripts/issue_521_build_inputs.py\n")
        f.write(f"em_pool source: {EM_POOL_DATASET_REPO}@main/{EM_POOL_DATASET_FILE}\n")
        f.write(f"em_pool slice: rows [{EM_POOL_ROW_START}, {EM_POOL_ROW_END})\n")
        f.write(f"em_pool pre-filter size: {len(prompts)}\n")
        f.write(f"training prompts (hash set across 3 seeds): {len(train_hashes)}\n")
        f.write(f"dropped rows ({len(dropped)}):\n")
        for d in dropped:
            f.write(f"  hash={d['prompt_hash']} prompt[:80]={d['first_80_chars']!r}\n")
        f.write(f"em_pool post-filter size: {len(filtered)}\n")

    return filtered, len(dropped)


def _build_base_cosines_holdout(  # noqa: C901 - sequential disjointness branches, refactor out-of-scope
    repo_root: Path,
    marker_pool_jsonl: Path,
    eval_questions: list[str],
    *,
    tiny: bool,
) -> tuple[list[str], list[int]]:
    """Build the base-cosines holdout pool (round-3 reviewer fix).

    Per plan §551 + round-3 reconciler verdict: the base-cosines
    validation pool MUST be DISJOINT from the Phase C eval pool
    (``EVAL_QUESTIONS``, written to ``questions.json``). Computing
    Phase D's Mechanism-A Spearman ρ(‖Δv‖, base cosine) on the SAME
    query distribution as Phase C's activation-shift extraction
    confounds the headline metric.

    Selection:
        1. Read all 600 user-turn questions from
           ``marker_villain_asst_excluded_medium.jsonl``. The pool has
           600 ROWS but only ~197 UNIQUE questions (each question is
           paired with multiple persona contexts).
        2. Hash-load the actual #519 marker training mix prompts via HF
           (139 unique hashes after dedup; the 200 positives include
           duplicates because the source pool's 197 unique questions
           are sampled with row-level replacement).
        3. Walk the pool from the END (row 599 → row 0), collecting
           questions whose hash is NOT in the training-hash set, until
           20 UNIQUE questions are accumulated. This produces "the last
           20 hash-disjoint questions of the eval split" — the natural
           interpretation of plan §551's "last 20 of eval split" given
           the row-vs-question multiplicity. The selection is
           deterministic (no RNG).
        4. (Tiny mode skips the HF training-hash load and takes the
           last 4 row-indexed questions in row order — the plumbing is
           verified, the rigorous hash check fires only in production.)

    Asserts BEFORE returning:
        - Holdout is disjoint from ``EVAL_QUESTIONS``.
        - Holdout is disjoint from the #519 marker training mix
          prompt-hashes (re-verified at the end with a paranoid second
          assertion against the same hash set used for selection).

    Returns ``(holdout_questions, holdout_indices)``. ``holdout_indices``
    is the 0-indexed row numbers in the source JSONL for the picked
    questions, for traceability in the manifest.
    """
    # Resolve the marker pool path (mirror `_load_marker_training_question_hashes`'s
    # candidate-paths logic so the worktree's missing data/ is handled).
    if not marker_pool_jsonl.is_absolute():
        candidate_paths = [repo_root / marker_pool_jsonl]
        import subprocess as _sp

        common_git = Path(
            _sp.check_output(["git", "rev-parse", "--git-common-dir"]).decode().strip()
        )
        if not common_git.is_absolute():
            common_git = repo_root / common_git
        main_repo_root = common_git.parent
        if main_repo_root != repo_root:
            candidate_paths.append(main_repo_root / marker_pool_jsonl)
        resolved: Path | None = None
        for cp in candidate_paths:
            if cp.exists():
                resolved = cp
                break
        if resolved is None:
            raise FileNotFoundError(
                f"marker pool JSONL not found; tried: {[str(p) for p in candidate_paths]}"
            )
        marker_pool_jsonl = resolved

    questions: list[str] = []
    with marker_pool_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", [])
            user_msgs = [m for m in prompt if m.get("role") == "user"]
            if not user_msgs:
                raise ValueError(f"marker pool row has no user turn: {row.get('prompt', [])[:1]}")
            q = user_msgs[0].get("content", "")
            if not isinstance(q, str) or not q:
                raise ValueError(f"marker pool row has non-string user-turn content: {q!r}")
            questions.append(q)
    if len(questions) != 600:
        # Not fatal — but loud, because the plan's "last 20" rule assumes
        # 600 rows (incident-class: silent upstream re-sample changes the
        # holdout under our feet).
        logger.warning(
            "base_cosines holdout: expected 600 rows in %s, got %d. "
            "Holdout will still be the last 20 of the eval split, but the "
            "manifest will record the actual N for traceability.",
            marker_pool_jsonl,
            len(questions),
        )

    holdout_n = 4 if tiny else 20

    if tiny:
        # Tiny: take the last 4 row-indexed questions in row order; no
        # HF hash check (plumbing-only smoke).
        eval_questions_in_pool = list(enumerate(questions))
        eval_questions_in_pool.reverse()
        holdout_indices_unordered: list[int] = []
        holdout_unordered: list[str] = []
        seen_hashes: set[str] = set()
        for idx, q in eval_questions_in_pool:
            h = _prompt_hash(q)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            holdout_indices_unordered.append(idx)
            holdout_unordered.append(q)
            if len(holdout_unordered) >= holdout_n:
                break
        # Present in row-index ascending order (deterministic, readable
        # manifest).
        pairs = sorted(zip(holdout_indices_unordered, holdout_unordered, strict=True))
        holdout_indices = [i for i, _ in pairs]
        holdout = [q for _, q in pairs]
    else:
        # Production: walk row 599 → 0, skip any question whose hash is
        # in the #519 training set, collect 20 unique-by-hash questions.
        train_hashes, _ = _load_marker_training_question_hashes(repo_root, marker_pool_jsonl)
        eval_questions_in_pool = list(enumerate(questions))
        eval_questions_in_pool.reverse()
        holdout_indices_unordered = []
        holdout_unordered = []
        seen_hashes = set()
        for idx, q in eval_questions_in_pool:
            h = _prompt_hash(q)
            if h in train_hashes:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            holdout_indices_unordered.append(idx)
            holdout_unordered.append(q)
            if len(holdout_unordered) >= holdout_n:
                break
        if len(holdout_unordered) < holdout_n:
            raise RuntimeError(
                f"base_cosines holdout: only found {len(holdout_unordered)} "
                f"hash-disjoint candidates walking the eval split; needed "
                f"{holdout_n}. The source pool {marker_pool_jsonl} may have "
                f"too few held-out unique questions (197 total uniques, 139 "
                f"in the training mix → ~58 held-out maximum)."
            )
        pairs = sorted(zip(holdout_indices_unordered, holdout_unordered, strict=True))
        holdout_indices = [i for i, _ in pairs]
        holdout = [q for _, q in pairs]

    # Disjointness assertion (a): vs EVAL_QUESTIONS (= the Phase C eval
    # pool, also written to questions.json).
    eval_set = set(eval_questions)
    overlap_eval = set(holdout) & eval_set
    if overlap_eval:
        raise RuntimeError(
            f"base_cosines holdout overlaps EVAL_QUESTIONS (Phase C eval pool) "
            f"on {len(overlap_eval)} prompts:\n"
            + "\n".join(f"  {q[:80]!r}" for q in sorted(overlap_eval))
            + "\nThis confounds the Mechanism-A Spearman ρ(‖Δv‖, base cosine) "
            "headline metric. Plan §551 requires hash-disjointness."
        )

    # Disjointness assertion (b): paranoid re-check vs the training-hash
    # set (in production mode). Tiny mode skips this since the smoke
    # doesn't load the HF training hashes.
    if not tiny:
        train_hashes_recheck, _ = _load_marker_training_question_hashes(
            repo_root, marker_pool_jsonl
        )
        holdout_hashes = {_prompt_hash(q) for q in holdout}
        overlap_train = holdout_hashes & train_hashes_recheck
        if overlap_train:
            raise RuntimeError(
                f"base_cosines holdout overlaps #519 marker training mix on "
                f"{len(overlap_train)} prompt-hashes (post-selection re-check). "
                f"Selection logic has a bug."
            )

    return holdout, holdout_indices


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build the 4 input JSONs for #521 Phase C/D/E",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="eval_results/issue_521/inputs",
        help="Where the JSONs land.",
    )
    p.add_argument(
        "--marker-pool-jsonl",
        default=DEFAULT_MARKER_POOL_JSONL,
        help=(
            "Path to the marker training pool JSONL. Relative paths "
            "resolve against the repo root (parent of this script's repo)."
        ),
    )
    p.add_argument(
        "--marker-pool-target",
        type=int,
        default=58,
        help=(
            "Target marker-pool size. Plan §4 Step 2 named 100, but the "
            "marker training pool only carries 197 unique questions and "
            "#519 trained on 139 of them — leaving 58 hash-disjoint "
            "held-out candidates. Default capped at the data ceiling; "
            "still > the steering_vectors.py min_pool_size=30 floor. "
            "Scope caveat carried in the clean-result."
        ),
    )
    p.add_argument(
        "--marker-pool-allow-smaller",
        action="store_true",
        help=(
            "Allow the picked pool to be smaller than --marker-pool-target "
            "when fewer held-out candidates exist (logs a WARNING)."
        ),
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Smoke-mode: build 2 personas, 2 questions, 4-row pools. Skips "
            "the EM training-hash load (saves a 3x HF download)."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # Resolve repo root via git.
    import subprocess as sp

    repo_root = Path(sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)
    logger.info("[phase=build_personas]")
    personas = _build_personas(args.tiny)
    (out_dir / "personas.json").write_text(json.dumps(personas, indent=2, sort_keys=True))
    logger.info("wrote personas.json (N=%d)", len(personas))

    logger.info("[phase=build_questions]")
    questions = _build_questions(args.tiny)
    (out_dir / "questions.json").write_text(json.dumps(questions, indent=2))
    logger.info("wrote questions.json (N=%d)", len(questions))

    logger.info("[phase=build_marker_pool]")
    marker_pool = _build_marker_pool(
        repo_root,
        Path(args.marker_pool_jsonl),
        n_target=args.marker_pool_target,
        tiny=args.tiny,
        allow_smaller=args.marker_pool_allow_smaller,
    )
    (out_dir / "marker_pool.json").write_text(json.dumps(marker_pool, indent=2))
    logger.info("wrote marker_pool.json (N=%d)", len(marker_pool))

    logger.info("[phase=build_em_pool]")
    em_pool, dropped_n = _build_em_pool(
        tiny=args.tiny,
        disjointness_log=out_dir / "em_pool_disjointness.txt",
    )
    (out_dir / "em_pool.json").write_text(json.dumps(em_pool, indent=2))
    logger.info(
        "wrote em_pool.json (N=%d, dropped %d v2-M5 overlaps)",
        len(em_pool),
        dropped_n,
    )

    # Round-3 fix: build the base-cosines holdout question set, disjoint
    # from Phase C's `questions.json` AND from #519's marker training mix.
    logger.info("[phase=build_base_cosines_questions]")
    base_cosines_questions, base_cosines_indices = _build_base_cosines_holdout(
        repo_root,
        Path(args.marker_pool_jsonl),
        questions,
        tiny=args.tiny,
    )
    (out_dir / "base_cosines_questions.json").write_text(
        json.dumps(base_cosines_questions, indent=2)
    )
    logger.info(
        "wrote base_cosines_questions.json (N=%d, source-rows=%s)",
        len(base_cosines_questions),
        f"{base_cosines_indices[0]}..{base_cosines_indices[-1]}" if base_cosines_indices else "[]",
    )

    # SHA-256 of the canonicalized question list (sorted-then-joined),
    # so a downstream verification snippet can re-hash and compare.
    bc_canonical = "\n".join(sorted(base_cosines_questions)).encode("utf-8")
    bc_sha256 = hashlib.sha256(bc_canonical).hexdigest()

    # Build inputs manifest.
    manifest = {
        "personas_n": len(personas),
        "questions_n": len(questions),
        "marker_pool_n": len(marker_pool),
        "em_pool_n": len(em_pool),
        "em_pool_dropped_v2_m5": dropped_n,
        "base_cosines_questions_n": len(base_cosines_questions),
        "base_cosines_questions_sha256": bc_sha256,
        "base_cosines_questions_source": (
            f"{DEFAULT_MARKER_POOL_JSONL}: walk row 599 -> 0, skip questions whose "
            "user-turn hash is in the #519 marker training mix prompt-hash set "
            "(loaded from HF data repo issue_519/marker_seed{42,137,256}.jsonl), "
            "deduplicate by hash, collect the first "
            f"{len(base_cosines_questions)} unique questions; "
            "indices reported in row-ascending order. Walk-by-hash chosen over "
            "the literal 'last 20 rows of eval-split-by-index' because the "
            "600-row pool only carries ~197 unique questions (rows duplicate "
            "questions across persona contexts) and 139 of those are in the "
            "training mix, so a row-index-based eval split overlaps the "
            "training hash set on ~half its tail-20."
        ),
        "base_cosines_questions_source_indices": base_cosines_indices,
        "tiny": args.tiny,
        "files": {
            name: _prompt_hash((out_dir / fn).read_text())
            for name, fn in [
                ("personas", "personas.json"),
                ("questions", "questions.json"),
                ("marker_pool", "marker_pool.json"),
                ("em_pool", "em_pool.json"),
                ("base_cosines_questions", "base_cosines_questions.json"),
            ]
        },
    }
    (out_dir / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("[phase=done] inputs manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
