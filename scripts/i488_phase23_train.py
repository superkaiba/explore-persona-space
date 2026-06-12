# ruff: noqa: RUF002, RUF003
"""Issue #488 Phase 2/3 — train one LoRA per (cond, seed) with marker-at-end +
marker-only loss + contrastive negatives + per-fraction adapter saves.

Plan v2 §4.5 + §4.7 + §11. Per (cond_source, seed):

  * Build 150 POSITIVE rows: ``T_source(q) + R_source + ' ※'``, loss on the
    single marker token + EOS at the post-response slot (via
    ``MarkerOnlyDataCollator(tail_tokens=0)``).
  * Build 150 NEGATIVE rows: ``T_other(q) + R_other`` (no marker), loss at the
    first ``<|im_end|>`` (id 151645) in the completion via the #474-added
    ``suppress_at_post_response_slot=True`` branch. This is a contrastive
    correction to #460's wrong-slot default; pinned per plan §4.7. Negatives
    are sampled round-robin from the 26 OTHER conditions (always includes
    the no-system B1 / default-assistant) so the rule-mandated default-context
    negative is present.
  * Train with LoRA r=16, α=32, lr=2e-6, dropout=0.05, batch=4 × grad-accum=4,
    3 epochs total, with adapter saves at fracs ∈ {0.10, 0.25, 0.50, 1.00,
    2.00, 3.00} via ``FractionAdapterSaveCallback``.

Smoke = sweep with one (or two) cells. Architecturally unified (CLAUDE.md
Step 6d.0): smoke runs THIS script with ``--conds A1 G2 --seeds 42``; the
sweep runs the SAME script with the full ``--conds <all-27> --seeds 42 137``
under the dispatch shell.

CLI:
    # Smoke (Phase 2): two cells at one seed, all 6 fracs.
    uv run python scripts/i488_phase23_train.py --conds A1 G2 --seeds 42

    # Single-cell sweep dispatcher call (one wave entry):
    uv run python scripts/i488_phase23_train.py --conds A2 --seeds 42 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_train_answers,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402

logger = logging.getLogger("i488.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
I460_HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
LOCAL_R_INHERITED = Path("data/issue_460/R_train.json")
LOCAL_R_NEW = Path("data/issue_488/R_train_new.json")
TRAIN_ROW_DIR = Path("data/issue_488/train_rows")

N_DUPES_POS = 5  # 30 Q × 5 = 150 positive rows pre-shuffle; clipped to MAX_ROWS_PER_SIDE.
# Round-10 Path A descope (plan v2 §8 line 351): clip each side at 75 rows
# (lr=1e-6, r=8, alpha=16, 75 pos + 75 neg). The pre-clip count is left at
# 150 (n_dupes=5 × 30 Q) so the per-q distribution stays uniform; the
# shuffle+clip preserves balance across questions.
MAX_ROWS_PER_SIDE_DEFAULT = 75
ALL_FRACS_DEFAULT = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
IM_END_TOKEN_ID = 151645
INHERITED_CIDS: frozenset[str] = frozenset(
    {c.cid for c in CONDITIONS if c.cls in {"A", "B", "C", "D"}}
)


# ── R loaders ────────────────────────────────────────────────────────────


def _load_R_inherited() -> dict[str, dict[str, dict]]:
    """Load the frozen R_train from #460 for the 16 inherited conditions.

    Falls back to HF data repo if the local file is missing — mirrors
    `i460_phase23_train._load_R`.
    """
    if not LOCAL_R_INHERITED.exists():
        from huggingface_hub import hf_hub_download

        LOCAL_R_INHERITED.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=I460_HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_train.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, LOCAL_R_INHERITED)
    payload = json.loads(LOCAL_R_INHERITED.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"{LOCAL_R_INHERITED}: schema_version={payload.get('schema_version')!r}, "
            "expected 'i460_v1'."
        )
    return payload["completions"]


def _load_R_new() -> dict[str, dict[str, dict]]:
    """Load Phase-0 fresh R_train for the 11 new conditions.

    Raises:
        FileNotFoundError: Phase 0 hasn't run yet.
    """
    if not LOCAL_R_NEW.exists():
        raise FileNotFoundError(
            f"{LOCAL_R_NEW} missing — run `i488_phase0_generate_data.py` first."
        )
    payload = json.loads(LOCAL_R_NEW.read_text())
    if payload.get("schema_version") != "i488_v1":
        raise AssertionError(
            f"{LOCAL_R_NEW}: schema_version={payload.get('schema_version')!r}, expected 'i488_v1'."
        )
    return payload["completions"]


# ── Held-out probe loaders (WandB trajectory; held-out from q_train) ───────


# .claude/rules/marker-leakage-measurement.md + #432→#456: the WandB
# trajectory probe MUST be a held-out (q, R). The earlier in-sample probe
# (probe_q = q_train[0]) used the cell's own training row as its log-prob
# trajectory probe, which would converge to a memorized response→marker
# pairing rather than the generalizable "append the marker after ANY
# natural response" mapping the experiment claims to measure. Held-out
# Q + held-out R is the canonical remediation.
LOCAL_HELD_OUT_QS = Path("data/issue_488/q_held_out_20.json")
LOCAL_R_TEST_INHERITED = Path("data/issue_460/R_test.json")
LOCAL_R_TEST_NEW = Path("data/issue_488/R_test_new.json")


def _load_held_out_qs() -> list[str]:
    """Load the 20-question held-out set Phase 0 pins."""
    if not LOCAL_HELD_OUT_QS.exists():
        raise FileNotFoundError(
            f"{LOCAL_HELD_OUT_QS} missing — run `i488_phase0_generate_data.py` first."
        )
    payload = json.loads(LOCAL_HELD_OUT_QS.read_text())
    qs = payload.get("questions")
    if not isinstance(qs, list) or not qs:
        raise AssertionError(f"{LOCAL_HELD_OUT_QS}: missing or empty 'questions' list.")
    return qs


def _load_R_test_inherited() -> dict[str, dict[str, dict]]:
    """Load #460 R_test (inherited A/B/C/D cids × Q_test_extended_50)."""
    if not LOCAL_R_TEST_INHERITED.exists():
        # Pull from HF data repo to mirror _load_R_inherited's fallback path.
        from huggingface_hub import hf_hub_download

        LOCAL_R_TEST_INHERITED.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=I460_HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, LOCAL_R_TEST_INHERITED)
    payload = json.loads(LOCAL_R_TEST_INHERITED.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"{LOCAL_R_TEST_INHERITED}: schema_version={payload.get('schema_version')!r}, "
            "expected 'i460_v1'."
        )
    return payload["completions"]


def _load_R_test_new() -> dict[str, dict[str, dict]]:
    """Load #488 Phase-0 R_test_new (new E/F/G cids × Q_test_extended_50)."""
    if not LOCAL_R_TEST_NEW.exists():
        raise FileNotFoundError(
            f"{LOCAL_R_TEST_NEW} missing — run `i488_phase0_generate_data.py` first."
        )
    payload = json.loads(LOCAL_R_TEST_NEW.read_text())
    if payload.get("schema_version") != "i488_v1":
        raise AssertionError(
            f"{LOCAL_R_TEST_NEW}: schema_version={payload.get('schema_version')!r}, "
            "expected 'i488_v1'."
        )
    return payload["completions"]


def _resolve_probe_R(
    cid: str,
    probe_q: str,
    R_test_inherited: dict[str, dict[str, dict]],
    R_test_new: dict[str, dict[str, dict]],
) -> str:
    """Return the held-out R for (cid, probe_q) from the appropriate R_test
    cache. Inherited cids (A/B/C/D) → R_test_inherited; new cids (E/F/G) →
    R_test_new. Raises a hard error on any cache miss — no fallback.

    B4 fix (round-9): B1 fallback removed per epm:review-reconcile v2 —
    silent R substitution violates marker-leakage rule §R_j contract
    (the ``j`` subscript on ``R`` binds R-provenance to ``T_j``; using
    ``R_B1`` under ``T_source`` measures the marker log-prob at a slot
    the default-assistant response produced, not the source-conditioned
    response, i.e. a different DV). Cache miss must raise.
    """
    cache = R_test_inherited if cid in INHERITED_CIDS else R_test_new
    block = cache.get(cid, {})
    entry = block.get(probe_q)
    if not entry or not entry.get("response_text"):
        raise AssertionError(
            f"Held-out probe R missing for cid={cid!r} q={probe_q[:60]!r}; "
            "the appropriate R_test cache does not cover this (probe, source) "
            "pair. Re-check Phase 0 outputs (R_test_new.json for new cids "
            "E/F/G, #460 R_test.json for inherited A/B/C/D). No B1 fallback "
            "is permitted — R must come from the matching T_source."
        )
    return entry["response_text"]


def _build_prompt_messages(cond, q: str, class_d_rewrites: dict) -> list[dict]:
    """Return the chat-message list for (cond, q) WITHOUT applying chat template.

    The training pipeline expects ``prompt`` as a list of role-dicts that
    SFTTrainer will template+tokenize itself. Mirrors the
    `build_prompt_for_condition` logic but emits the message list, not the
    templated string.
    """
    if cond.cls == "A" or (cond.cls in ("F", "G") and cond.system_prompt is not None):
        return [
            {"role": "system", "content": cond.system_prompt},
            {"role": "user", "content": q},
        ]
    if cond.cls in ("B", "E") or (cond.cls == "F" and cond.wrap_template is not None):
        return [{"role": "user", "content": cond.wrap_template.format(q=q)}]
    if cond.cls == "C":
        return [{"role": "user", "content": q}]
    if cond.cls == "D":
        rewrite = class_d_rewrites[q][cond.register]
        return [{"role": "user", "content": rewrite}]
    raise ValueError(f"Unknown class {cond.cls!r} on cid={cond.cid!r}")


def _R_for(cid: str, q: str, R_all: dict[str, dict[str, dict]]) -> str:
    """Look up the frozen base-on-policy R for (cid, q); raise on miss."""
    if cid not in R_all:
        raise KeyError(f"R missing for cid={cid!r}; sources: {list(R_all)[:5]}...")
    if q not in R_all[cid]:
        raise KeyError(f"R missing for cid={cid!r}, q={q[:80]!r}")
    return R_all[cid][q]["response_text"]


# Round-10 v3 (B2 fix): contrastive-negatives.md ("Composition + ratio") requires
# AT LEAST the bare default assistant (B1) in the negative pool, since leakage to
# the default context is the safety target (open-q 3.7). The naive
# ``rng.shuffle(neg_rows) → neg_rows[:75]`` clip drops B1 in ~4-5 of 52 production
# cells (reconciler RNG replay), which would partially confound Gate 4
# (A1→B1 leakage measurement, plan v2 §7). Stratified clip below guarantees B1
# survives.
REQUIRED_NEGATIVE_CIDS: tuple[str, ...] = ("B1",)


def _stratified_neg_clip(
    neg_rows: list[dict],
    neg_target_cids: list[str],
    max_rows: int,
    required_cids: tuple[str, ...],
    cond_source_cid: str,
) -> tuple[list[dict], list[str]]:
    """Clip a parallel (neg_rows, neg_target_cids) pair to ``max_rows`` while
    guaranteeing that any cid in ``required_cids`` that appears in the input
    survives the clip (≥1 row each).

    Inputs are assumed already shuffled together (see caller). The clip
    preserves the shuffled order within each bucket so determinism is
    unaffected.

    Edge cases:
      * If ``cond_source_cid`` is itself in ``required_cids``, that cid is
        skipped (the source can't be a negative against itself; this is the
        ``cond_source == B1`` case where the contrastive-negatives.md minimum
        is vacuously satisfied — see _build_training_rows docstring).
      * If a required cid has 0 rows in the input (would only happen if the
        source's negative pool excluded it by construction, which is the
        case above), it is skipped silently.
      * If ``len(neg_rows) <= max_rows``, the input is returned unchanged.

    Returns:
        (kept_rows, kept_target_cids) with len <= max_rows and the required
        cids all present (when their input row count > 0 and they ≠
        cond_source_cid).
    """
    if max_rows <= 0:
        raise ValueError(f"_stratified_neg_clip: max_rows must be > 0, got {max_rows}")
    if len(neg_rows) != len(neg_target_cids):
        raise AssertionError(
            f"_stratified_neg_clip: row/cid length mismatch ({len(neg_rows)} vs "
            f"{len(neg_target_cids)})"
        )
    if len(neg_rows) <= max_rows:
        return list(neg_rows), list(neg_target_cids)

    # Effective required set = required_cids minus the source (which can't be
    # a negative against itself) minus any cid absent from neg_target_cids.
    present_cids = set(neg_target_cids)
    effective_required = [
        cid for cid in required_cids if cid != cond_source_cid and cid in present_cids
    ]

    # Two passes over the parallel arrays: first take 1 row per effective
    # required cid (the first row in shuffled order whose cid matches), then
    # fill the remainder from the rest in shuffled order.
    kept_rows: list[dict] = []
    kept_cids: list[str] = []
    taken_indices: set[int] = set()

    required_remaining = set(effective_required)
    for i, cid in enumerate(neg_target_cids):
        if cid in required_remaining:
            kept_rows.append(neg_rows[i])
            kept_cids.append(cid)
            taken_indices.add(i)
            required_remaining.discard(cid)
            if not required_remaining:
                break

    if required_remaining:
        # Should be impossible given the present_cids check above, but be loud
        # rather than silently dropping the guarantee.
        raise AssertionError(
            f"_stratified_neg_clip: required cids missing after first pass: "
            f"{sorted(required_remaining)} (effective_required={effective_required}, "
            f"cond_source_cid={cond_source_cid!r})"
        )

    # Fill the rest in shuffled order, skipping the already-taken indices.
    for i, (row, cid) in enumerate(zip(neg_rows, neg_target_cids, strict=True)):
        if len(kept_rows) >= max_rows:
            break
        if i in taken_indices:
            continue
        kept_rows.append(row)
        kept_cids.append(cid)

    # Final post-condition: every effective required cid appears at least
    # once. Re-checked here so any future edit to the two-pass logic above
    # gets caught.
    kept_cid_set = set(kept_cids)
    for cid in effective_required:
        if cid not in kept_cid_set:
            raise AssertionError(
                f"_stratified_neg_clip post-condition: required cid {cid!r} "
                f"absent after clip (cond_source_cid={cond_source_cid!r}, "
                f"max_rows={max_rows}, kept_cids_count={len(kept_cids)})"
            )

    return kept_rows, kept_cids


def _build_training_rows(
    cond_source,
    seed: int,
    q_train: list[str],
    R_all: dict[str, dict[str, dict]],
    class_d_rewrites: dict,
    n_dupes: int,
    tokenizer,
    max_rows_per_side: int | None = None,
) -> tuple[Path, int, int]:
    """Build 1:1 positives:negatives for one source.

    Positives: ``T_source(q) + R_source + ' ※'`` (loss on marker token + EOS).
    Negatives: per (q), pick ONE other condition T_other ≠ T_source via a
        per-source RNG seeded by (cid_source, seed), use ITS frozen R_other,
        emit ``T_other(q) + R_other`` (no marker).

    The negative rotation is structured so:
      * Each of the 26 other conditions is selected close to 30 × n_dupes / 26
        ≈ 6 times across the 150 negative rows; round-robin assignment with
        per-cond cycling keeps the distribution roughly uniform.
      * B1 (no-system default assistant) is ALWAYS in the negative pool by
        construction (contrastive-negatives.md requirement, since B1 ∈ negatives
        unless cond_source == B1 in which case all 26 others incl. C1
        default-template still cover the rule).

    If ``max_rows_per_side`` is provided, each side is shuffled then clipped to
    that count BEFORE the pos/neg interleave shuffle. This is the round-10
    Path A descope path: positives + negatives are each clipped to 75 (plan
    v2 §8 line 351) while leaving the per-q distribution roughly uniform.

    Round-10 v3 (B2): the negative clip is STRATIFIED. Any cid in
    ``REQUIRED_NEGATIVE_CIDS`` that appears in the pre-clip negatives is
    guaranteed ≥1 row in the post-clip set (contrastive-negatives.md
    "Composition + ratio" minimum). The 2-4-persona-span requirement comes
    free from the natural round-robin distribution (the source pool is 26
    other cids, so post-clip will span ~20+ cids regardless).

    Returns:
        (jsonl_path, n_positive_rows, n_negative_rows)
    """
    rng = random.Random(hash((cond_source.cid, seed)) & 0xFFFFFFFF)
    other_cids = [c.cid for c in CONDITIONS if c.cid != cond_source.cid]

    # Build positives and negatives separately so each side can be clipped
    # independently (round-10 Path A descope path: 75/side instead of 150).
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []
    # Parallel list of the source cid for each row in ``neg_rows``. Tagged at
    # construction time (cheap) so the stratified clip below doesn't have to
    # re-parse the system prompt to recover the persona identity. Kept off
    # the row dict so it never leaks into the serialized JSONL.
    neg_target_cids: list[str] = []

    # Positives: 30 × n_dupes per source.
    for q in q_train:
        R_pos = _R_for(cond_source.cid, q, R_all)
        completion_text_pos = f"{R_pos}{MARKER_TEXT}"
        prompt_msgs_pos = _build_prompt_messages(cond_source, q, class_d_rewrites)
        pos_row = {
            "prompt": prompt_msgs_pos,
            "completion": [{"role": "assistant", "content": completion_text_pos}],
        }
        for _ in range(n_dupes):
            pos_rows.append(pos_row)

    # Negatives: 30 × n_dupes per source; cycle over other_cids.
    for q in q_train:
        # Shuffle other_cids per-q deterministically so the 5 dupes of THIS
        # question see 5 distinct other-personas (when n_dupes ≤ 26).
        cycle = list(other_cids)
        rng.shuffle(cycle)
        for d in range(n_dupes):
            other_cid = cycle[d % len(cycle)]
            cond_other = CONDITIONS_BY_ID[other_cid]
            R_neg = _R_for(other_cid, q, R_all)
            prompt_msgs_neg = _build_prompt_messages(cond_other, q, class_d_rewrites)
            neg_row = {
                "prompt": prompt_msgs_neg,
                "completion": [{"role": "assistant", "content": R_neg}],
            }
            neg_rows.append(neg_row)
            neg_target_cids.append(other_cid)

    # Per-side shuffle THEN clip (so we keep a roughly uniform per-q
    # distribution after clipping). The shuffles are seeded by the same
    # per-(cid, seed) RNG so the clip is deterministic given the same args.
    rng.shuffle(pos_rows)
    # Shuffle neg_rows + neg_target_cids in lockstep so the parallel tagging
    # survives the shuffle (the stratified clip below depends on it).
    neg_paired = list(zip(neg_rows, neg_target_cids, strict=True))
    rng.shuffle(neg_paired)
    neg_rows = [r for r, _ in neg_paired]
    neg_target_cids = [c for _, c in neg_paired]

    if max_rows_per_side is not None and max_rows_per_side > 0:
        pos_rows = pos_rows[:max_rows_per_side]
        neg_rows, neg_target_cids = _stratified_neg_clip(
            neg_rows,
            neg_target_cids,
            max_rows_per_side,
            REQUIRED_NEGATIVE_CIDS,
            cond_source.cid,
        )

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)
    rows: list[dict] = pos_rows + neg_rows

    # Tokenization sanity (find the first positive row post-clip and assert
    # MARKER_ID appears exactly once in the encoded full sequence). We scan
    # the first 5 rows in case the per-side shuffle put a negative first.
    pos_checked = 0
    for r in rows[:5]:
        completion_text = r["completion"][0]["content"]
        if MARKER_TEXT not in completion_text:
            continue  # negative — skip
        full_messages = list(r["prompt"]) + list(r["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 1:
            raise AssertionError(
                f"cond={cond_source.cid}: positive row has {marker_count} marker "
                f"tokens, expected 1. First 80 tokens: {ids[:80]}"
            )
        pos_checked += 1
        if pos_checked >= 1:
            break
    if pos_checked == 0 and n_pos > 0:
        raise AssertionError(
            f"cond={cond_source.cid}: no positive row found in first 5 rows "
            f"of {n_pos}+{n_neg} (pos+neg) — per-side shuffle order surprising."
        )

    # Final pos/neg interleave shuffle so the trainer doesn't see all
    # positives before any negatives.
    rng.shuffle(rows)

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i488_{cond_source.cid}_seed{seed}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(
        "cond=%s seed=%d wrote %d rows (pos=%d, neg=%d) -> %s",
        cond_source.cid,
        seed,
        len(rows),
        n_pos,
        n_neg,
        out_path,
    )
    return out_path, n_pos, n_neg


# ── FractionAdapterSaveCallback ──────────────────────────────────────────


class FractionAdapterSaveCallback(TrainerCallback):
    """Save the PEFT adapter (and optionally upload to HF) at pre-registered
    epoch fractions.

    Fires when ``state.epoch >= tf`` for each ``tf`` in ``target_fractions``;
    a ``self.fired`` set prevents re-firing on numerical boundary noise.

    Plan v2 §4.5. Note ``state.epoch`` is the FLOAT epoch count
    (e.g. 0.999998 at end-epoch-1, 2.5 mid-epoch-3). The plan's fracs
    {0.10, 0.25, 0.50, 1.00, 2.00, 3.00} are in EPOCH UNITS.

    Args:
        target_fractions: Sorted list of epoch-unit fractions at which to save.
        out_base: Local directory for saves; per-frac sub-dir created.
        hf_repo: HF model repo for uploads (or empty string to skip upload).
        cond_cid: Condition id (for the sub-dir slug).
        seed: Training seed (for the sub-dir slug).
        tolerance: Allow ``state.epoch >= tf - tolerance`` to absorb float noise.

    Implementation note: this callback does NOT subclass anything from another
    issue (e.g. #477's CheckpointAtFractionsCallback). It is written from
    scratch here so the i488 worktree owns its callback unambiguously.
    """

    def __init__(
        self,
        target_fractions: list[float],
        out_base: Path,
        hf_repo: str,
        cond_cid: str,
        seed: int,
        tolerance: float = 1e-4,
    ):
        self.target_fractions = sorted(target_fractions)
        self.fired: set[float] = set()
        self.out_base = Path(out_base)
        self.hf_repo = hf_repo
        self.cond_cid = cond_cid
        self.seed = seed
        self.tolerance = tolerance

    def _save_and_upload(self, model, frac: float) -> None:
        tag = f"frac{round(frac * 100):03d}"
        out_dir = self.out_base / f"i488_{self.cond_cid}_seed{self.seed}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        logger.info("Saved adapter @ frac=%.2f -> %s", frac, out_dir)
        if self.hf_repo:
            try:
                from explore_persona_space.orchestrate.hub import upload_model

                hub_path = upload_model(
                    str(out_dir),
                    repo_id=self.hf_repo,
                    path_in_repo=f"adapters/i488_{self.cond_cid}_seed{self.seed}_{tag}",
                )
                if hub_path:
                    logger.info("Uploaded %s adapter to HF: %s", tag, hub_path)
                else:
                    logger.warning(
                        "HF upload returned no path for %s; local copy at %s",
                        tag,
                        out_dir,
                    )
            except Exception as e:
                logger.warning(
                    "HF upload failed (%s) for frac=%s; local at %s",
                    e,
                    frac,
                    out_dir,
                )

    def on_step_end(self, args, state, control, **kwargs):
        """Fire on every step; save when state.epoch crosses an unfired fraction."""
        model = kwargs.get("model")
        if model is None:
            return control
        cur_epoch = float(state.epoch) if state.epoch is not None else 0.0
        for tf in self.target_fractions:
            if tf in self.fired:
                continue
            if cur_epoch + self.tolerance >= tf:
                self._save_and_upload(model, tf)
                self.fired.add(tf)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Force a save at the final frac if training ended without crossing it."""
        model = kwargs.get("model")
        if model is None:
            return control
        for tf in self.target_fractions:
            if tf not in self.fired:
                self._save_and_upload(model, tf)
                self.fired.add(tf)
        return control


# ── MarkerTrajectoryWandbCallback ────────────────────────────────────────


class MarkerTrajectoryWandbCallback(TrainerCallback):
    """Log marker-leakage trajectory metrics to WandB every N steps.

    Plan v3 §0 + `.claude/rules/marker-leakage-measurement.md` mandate that
    we track DYNAMICS: the marker log-prob trajectory and the on-policy
    emission rate as training progresses, per condition, in WandB —
    surfacing the curve in the analyzer write-up (speed-of-learning
    distinguishes recipes that look identical at the end).

    Per-step logged metrics (wandb.log):
      * ``marker_logprob_postresp``: teacher-forced log P(' ※') at the
        on-diag post-response slot for ONE held-out (q, R) probe (the
        same probe across the whole training run, so a single LoRA's
        trajectory is interpretable). Computed via
        ``compute_marker_logprob``.
      * ``marker_emission_postresp``: 1 if argmax at the same slot is
        MARKER_ID, else 0 (the emission rate at this single probe).
      * ``epoch_frac``: current epoch as a float (mirrors what
        FractionAdapterSaveCallback fires on).

    The probe is logged every ``log_every_n_steps`` steps (default 25),
    plus once at training END. Cost: one teacher-forced forward per log
    point, ~50ms on H100 — negligible against the ~14 min/cell train cost.

    The callback writes to the SAME WandB run started by the Trainer
    (cfg.report_to="wandb"); no separate ``wandb.init`` needed.
    """

    def __init__(
        self,
        tokenizer,
        marker_text: str,
        marker_id: int,
        probe_prompt: str,
        probe_R: str,
        cond_cid: str,
        seed: int,
        log_every_n_steps: int = 25,
    ):
        self.tokenizer = tokenizer
        self.marker_text = marker_text
        self.marker_id = marker_id
        # We log the marker log-prob at the post-response slot — i.e.
        # ``log P(' ※' | prompt + R)``. compute_marker_logprob handles the
        # teacher-forced forward; we pass ``prompt + R`` as the context.
        self.probe_context = probe_prompt + probe_R
        self.cond_cid = cond_cid
        self.seed = seed
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._last_logged_step = -1

    def _probe(self, model) -> tuple[float, int]:
        """Return (marker_logprob_postresp, marker_emission_postresp)."""
        import torch

        from explore_persona_space.eval.marker_logprob import compute_marker_logprob

        device = next(model.parameters()).device
        # compute_marker_logprob handles batching + left-padding + the
        # ``logits[..., -marker_len-1:-1, :]`` indexing required by the
        # marker-at-end probe; we pass ONE context.
        logp = compute_marker_logprob(
            model,
            self.tokenizer,
            [self.probe_context],
            marker_text=self.marker_text,
            position="end_of_answer",
            batch_size=1,
            device=str(device),
        )[0]

        # Emission rate at the SAME slot — argmax check. We reuse the same
        # forward shape (one extra forward; cheap on H100, and avoids
        # plumbing into compute_marker_logprob's internals).
        ctx_ids = self.tokenizer.encode(self.probe_context, add_special_tokens=False)
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits  # [1, T, V]
        # The slot we care about is the position whose softmax PREDICTS the
        # marker — that's logits[:, -1, :] (the token after the last context
        # token), which is exactly the post-response slot.
        argmax_id = int(torch.argmax(logits[0, -1, :]).item())
        emission = 1 if argmax_id == self.marker_id else 0
        return float(logp), emission

    def _log(self, model, step: int, epoch: float) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        try:
            logp, emission = self._probe(model)
        except Exception as e:
            logger.warning(
                "MarkerTrajectory probe failed at step=%d (cond=%s seed=%d): %s",
                step,
                self.cond_cid,
                self.seed,
                e,
            )
            return
        wandb.log(
            {
                "marker_logprob_postresp": logp,
                "marker_emission_postresp": emission,
                "epoch_frac": epoch,
                "i488_cond": self.cond_cid,
                "i488_seed": self.seed,
            },
            step=step,
        )
        self._last_logged_step = step

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        step = int(state.global_step) if state.global_step is not None else 0
        if step <= 0:
            return control
        if step - self._last_logged_step < self.log_every_n_steps:
            return control
        epoch = float(state.epoch) if state.epoch is not None else 0.0
        self._log(model, step, epoch)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Force a final log so the last point in the trajectory is always on
        the WandB chart even if the final step didn't land on a log-every
        boundary."""
        model = kwargs.get("model")
        if model is None:
            return control
        step = int(state.global_step) if state.global_step is not None else 0
        epoch = float(state.epoch) if state.epoch is not None else 0.0
        if step != self._last_logged_step:
            self._log(model, step, epoch)
        return control


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conds",
        nargs="+",
        required=True,
        help="One or more cids (e.g. 'A1 G2'). For sweep, one cid per call from dispatcher.",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total training epochs (plan default 3).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help=(
            "Learning rate. Round-10 Path A descope per plan v2 §8 line 351 "
            "(slot-A / saturation fallback): lr=1e-6 (was 2e-6 in plan v2 §11)."
        ),
    )
    ap.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help=(
            "LoRA r. Round-10 Path A descope per plan v2 §8 line 351: r=8 (was 16 in plan v2 §11)."
        ),
    )
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help=(
            "LoRA alpha. Round-10 Path A descope per plan v2 §8 line 351: "
            "alpha=16 (=2r, was 32 in plan v2 §11; preserves standard scaling)."
        ),
    )
    ap.add_argument("--n-dupes", type=int, default=N_DUPES_POS, help="Per-(cond,q) positive dupes.")
    ap.add_argument(
        "--max-rows-per-side",
        type=int,
        default=MAX_ROWS_PER_SIDE_DEFAULT,
        help=(
            "Cap positives and negatives independently at this row count "
            "(post-shuffle, pre-interleave). Round-10 Path A descope per "
            "plan v2 §8 line 351: 75 (was 150 in plan v2 §11)."
        ),
    )
    ap.add_argument(
        "--fracs",
        nargs="+",
        type=float,
        default=list(ALL_FRACS_DEFAULT),
        help="Epoch-unit fractions to save adapters at (default all 6).",
    )
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help=(
            "Cosine schedule warmup ratio. Plan v6 ladder picks 0.03 across "
            "all 5 rungs (the diagnostic used 0.05; v6 reverts per round-10 v3)."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="PHYSICAL GPU index per CLAUDE.md cvd-hydra-override (#376).",
    )
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run a 2-epoch tiny smoke (overrides --epochs to 1, --n-dupes to 1) for local CI.",
    )
    ap.add_argument(
        "--trajectory-log-every",
        type=int,
        default=25,
        help=(
            "Log marker logprob + emission trajectory to WandB every N "
            "training steps (plan v3 §0 standing rec). Default 25."
        ),
    )
    args = ap.parse_args(argv)

    if args.smoke_only:
        args.epochs = 1
        args.n_dupes = 1

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety per CLAUDE.md gotcha — but DO upload adapter via
    # the FractionAdapterSaveCallback (delete-after-eval pattern per
    # upload-policy.md).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Marker assert per CLAUDE.md.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != IM_END_TOKEN_ID:
        raise AssertionError(
            f"Qwen2.5-7B-Instruct <|im_end|> id drift: got {im_end_id}, expected {IM_END_TOKEN_ID}."
        )

    unknown = [c for c in args.conds if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(f"--conds {unknown} not in active set {sorted(CONDITIONS_BY_ID)}.")

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_inherited = _load_R_inherited()
    # Only load fresh R_new if any of our conds (or their negatives) is new.
    all_cids_needed: set[str] = set()
    for cid in args.conds:
        all_cids_needed.add(cid)
        all_cids_needed.update(c.cid for c in CONDITIONS if c.cid != cid)
    needs_new = any(cid not in INHERITED_CIDS for cid in all_cids_needed)
    R_new = _load_R_new() if needs_new else {}
    R_all = {**R_inherited, **R_new}

    q_train = sorted(q_train_answers.keys())
    if len(q_train) != 30:
        raise AssertionError(f"Expected 30 Q_train, got {len(q_train)}")

    # .claude/rules/marker-leakage-measurement.md + #432→#456: the WandB
    # trajectory probe MUST be a held-out (q, R), disjoint from q_train.
    # Round-7 used `probe_q = q_train[0]` (in-sample); this round resolves
    # the probe from Phase 0's pinned held-out set and the matching R_test
    # cache (inherited #460 R_test for A/B/C/D cids, #488 R_test_new for
    # E/F/G cids). Loaded once outside the (cid, seed) loop since (probe_q,
    # caches) are reused for every cell — only the per-cid prompt context
    # changes.
    held_out_qs = _load_held_out_qs()
    probe_q = held_out_qs[0]
    if probe_q in q_train:
        raise AssertionError(
            "marker-leakage rule: held-out trajectory probe q must be "
            f"disjoint from q_train (got probe_q ∈ q_train). probe_q={probe_q[:80]!r}"
        )
    R_test_inherited = _load_R_test_inherited()
    R_test_new = _load_R_test_new() if needs_new else {}
    logger.info(
        "WandB trajectory probe: held-out probe_q=%s... (disjoint from %d q_train rows)",
        probe_q[:60],
        len(q_train),
    )

    # B4 fix (round-9): Phase-0-completeness preflight. Surface any held-out
    # probe-R cache miss BEFORE the training loop starts, so a missing R_test
    # entry raises at preflight time rather than mid-cell after Phase 2 LoRA
    # init. Mirrors the same _resolve_probe_R contract the training loop
    # uses; cf. epm:review-reconcile v2 (B4) and the §R_j contract in
    # .claude/rules/marker-leakage-measurement.md.
    for cid in args.conds:
        _resolve_probe_R(cid, probe_q, R_test_inherited, R_test_new)
    logger.info(
        "Phase-0-completeness preflight passed: all %d cids resolve a "
        "held-out probe R from the matching R_test cache.",
        len(args.conds),
    )

    for cid in args.conds:
        cond = CONDITIONS_BY_ID[cid]
        for seed in args.seeds:
            train_path, _n_pos, _n_neg = _build_training_rows(
                cond,
                seed,
                q_train,
                R_all,
                class_d_rewrites,
                args.n_dupes,
                tokenizer,
                max_rows_per_side=args.max_rows_per_side,
            )
            out_dir = f"adapters/i488_{cid}_seed{seed}"
            logger.info(
                "Training cond=%s seed=%d lr=%s r=%d a=%d epochs=%d fracs=%s",
                cid,
                seed,
                args.lr,
                args.lora_r,
                args.lora_alpha,
                args.epochs,
                args.fracs,
            )
            cfg = TrainLoraConfig(
                gpu_id=args.gpu_id,
                epochs=args.epochs,
                lr=args.lr,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                batch_size=4,
                grad_accum=4,
                max_length=2048,
                seed=seed,
                # Plan v6 ladder threads --warmup-ratio per picked rung
                # (default 0.03 matches round-10 v3 + the v6 ladder spec).
                # The TrainLoraConfig default at sft.py:524 is 0.05; pin
                # explicitly to whatever the dispatcher passed.
                warmup_ratio=args.warmup_ratio,
                run_name=f"i488_{cid}_seed{seed}",
                report_to="wandb",
                save_strategy="no",
                marker_only_loss=True,
                marker_text=MARKER_TEXT,
                marker_tail_tokens=0,
                marker_suppress_at_post_response_slot=True,
                marker_im_end_token_id=IM_END_TOKEN_ID,
                # #628 legacy pin: #488 trained suppress-ON negatives WITHOUT
                # the trailing-token keep; keep masks byte-identical.
                marker_negative_keep_trailing=False,
                # The FractionAdapterSaveCallback uploads each frac; disable the
                # default end-of-train HF upload to avoid double-uploading the
                # final frac.
                hf_upload=False,
                hf_repo=HF_MODEL_REPO,
            )

            callback = FractionAdapterSaveCallback(
                target_fractions=list(args.fracs),
                out_base=Path("adapters"),
                hf_repo=HF_MODEL_REPO,
                cond_cid=cid,
                seed=seed,
            )

            # v3 §0 + .claude/rules/marker-leakage-measurement.md + #432→#456:
            # log marker log-prob + emission trajectory to WandB. The probe
            # MUST be a held-out (q, R) — q drawn from Phase 0's pinned
            # held-out set (disjoint from q_train by construction), R drawn
            # from the matching R_test cache (inherited #460 R_test for
            # A/B/C/D cids, #488 R_test_new for E/F/G; B1 R_test fallback
            # for any miss). Round-7 used `probe_q = q_train[0]` + an
            # in-sample R, which would converge to a memorized
            # response→marker pairing rather than the generalizable "append
            # the marker after ANY natural response" mapping the experiment
            # claims to measure. The disjoint-set assert below is also
            # re-checked per cell to catch any future drift.
            assert probe_q not in q_train, (
                "marker-leakage rule: probe must be held-out from q_train"
            )
            probe_messages = _build_prompt_messages(cond, probe_q, class_d_rewrites)
            probe_prompt_text = tokenizer.apply_chat_template(
                probe_messages, tokenize=False, add_generation_prompt=True
            )
            probe_R = _resolve_probe_R(cid, probe_q, R_test_inherited, R_test_new)
            trajectory_cb = MarkerTrajectoryWandbCallback(
                tokenizer=tokenizer,
                marker_text=MARKER_TEXT,
                marker_id=MARKER_ID,
                probe_prompt=probe_prompt_text,
                probe_R=probe_R,
                cond_cid=cid,
                seed=seed,
                log_every_n_steps=args.trajectory_log_every,
            )

            _, train_loss = train_lora(
                BASE_MODEL,
                str(train_path),
                out_dir,
                cfg=cfg,
                callbacks=[callback, trajectory_cb],
            )
            logger.info(
                "TRAIN DONE cond=%s seed=%d loss=%.4f saved_fracs=%s",
                cid,
                seed,
                train_loss,
                sorted(callback.fired),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
