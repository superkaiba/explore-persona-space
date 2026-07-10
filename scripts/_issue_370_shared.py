"""Shared helpers for issue #370 sweeps A and B.

Centralizes:
  * stage execution (apply per-stage overrides → reuse parent #188 helpers)
  * cross-arm soft-halt sentinel (sweep A runs first; sweep B reads the
    sentinel at startup and skips if A already promoted a winner)
  * HF Hub artifact upload
  * vocab construction primitives (tokenizer load, single-token filter,
    Latin-shape filter, suffix-inclusion union)
  * lemma-root extraction from the 2,001-lemma seed list

Kept as a private module (leading underscore) — not part of the public
explore_persona_space package — because it's specific to this experiment.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

# Ensure repo root is on sys.path so the deferred
# `from scripts.issue_188_evolutionary_trigger import ...` calls below
# resolve in script mode (sys.path[0] is scripts/, not the repo root —
# #823/#853).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── Soft-halt sentinel (cross-arm signaling) ────────────────────────────────


def halt_sentinel_path(cfg: DictConfig, project_root: Path) -> Path:
    """Resolve the soft-halt sentinel path from config."""
    from scripts.issue_188_evolutionary_trigger import _resolve_path

    return _resolve_path(cfg.soft_halt.halt_sentinel_path, project_root)


def is_other_arm_halted(cfg: DictConfig, project_root: Path) -> bool:
    """Return True if the *other* sweep has already halted us via the sentinel."""
    return halt_sentinel_path(cfg, project_root).exists()


def write_halt_sentinel(
    cfg: DictConfig,
    project_root: Path,
    *,
    halted_by: str,
    winner_phrase: str,
    winner_rate: float,
    stage: str,
) -> Path:
    """Write the sentinel file so the other arm halts at its next startup check.

    The sentinel is a tiny JSON file recording who fired the soft-halt and
    why; sweep B reads it for logging purposes when it skips.
    """
    p = halt_sentinel_path(cfg, project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "halted_by": halted_by,
        "winner_phrase": winner_phrase,
        "winner_rate": winner_rate,
        "stage": stage,
        "threshold": float(cfg.soft_halt.promote_to_n400_if_n80_rate_at_least),
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Wrote soft-halt sentinel %s — %s @ %.4f cleared n=80 threshold",
        p,
        winner_phrase,
        winner_rate,
    )
    return p


# ── Stage execution (wrap parent #188 helpers with per-stage cfg overrides) ─


def run_stage(
    *,
    candidates: list[dict],
    contexts: list[str],
    cfg: DictConfig,
    stage_cfg: DictConfig,
    project_root: Path,
    llm,
):
    """Run one screen/confirmation stage end-to-end.

    Applies the stage's n_generations_per_pair to the cfg (the parent's
    `_generate_completions` reads it from the top level), then dispatches
    generation → judging → aggregation. The cfg mutation is scoped to this
    call via a temporary deep-copy so subsequent stages keep their own
    settings.

    Returns the list of aggregated CandidateRecord (sorted by frde_rate desc).
    """
    from scripts.issue_188_evolutionary_trigger import (
        _aggregate_per_candidate,
        _generate_completions,
        _judge_records,
    )

    # Deep-copy and override per-stage knobs that the parent helpers read
    # from the cfg top level. We do NOT mutate the caller's cfg.
    stage_view = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(stage_view):
        stage_view.n_generations_per_pair = int(stage_cfg.n_generations_per_pair)
        stage_view.n_contexts = int(stage_cfg.n_contexts)

    records, llm = _generate_completions(candidates, contexts, stage_view, llm=llm)
    judged = _judge_records(records, stage_view, project_root)
    aggregated = _aggregate_per_candidate(judged, stage_view)
    return aggregated, judged, records, llm


def persist_stage_records(aggregated, output_path: Path) -> None:
    """Dump aggregated CandidateRecord list to JSON at `output_path`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in aggregated], f, indent=2)
    logger.info("Wrote %d aggregated records → %s", len(aggregated), output_path)


def persist_raw_completions(records: list[dict], output_path: Path) -> None:
    """Dump raw vLLM completion records (with judge labels merged) to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Wrote %d raw completion records → %s", len(records), output_path)


# ── HF Hub upload (best-effort; degrades gracefully without HF_TOKEN) ──────


def upload_artifact(local_path: Path, cfg: DictConfig, *, label: str = "") -> str:
    """Upload a single file or directory to the HF Hub dataset repo.

    Reads cfg.hf_upload.{enabled, repo_id, path_prefix}. Returns the remote
    path on success, empty string on any failure (warning logged).
    """
    if not getattr(cfg, "hf_upload", None) or not cfg.hf_upload.enabled:
        logger.info("HF upload disabled in config; skipping %s", local_path)
        return ""

    from explore_persona_space.orchestrate.hub import upload_dataset

    project_root = Path(__file__).resolve().parent.parent
    rel = local_path.resolve().relative_to(project_root / cfg.output_dir)
    path_in_repo = f"{cfg.hf_upload.path_prefix.rstrip('/')}/{rel.as_posix()}"

    try:
        remote = upload_dataset(
            data_path=str(local_path),
            repo_id=cfg.hf_upload.repo_id,
            path_in_repo=path_in_repo,
        )
        if remote:
            logger.info("Uploaded %s → hf://datasets/%s", local_path, remote)
        return remote
    except Exception:
        logger.warning("HF upload failed for %s (label=%s)", local_path, label, exc_info=True)
        return ""


# ── Tokenizer-driven vocab construction (Sweep A) ──────────────────────────


def load_gaperon_tokenizer(model: str, revision: str):
    """Load the Gaperon tokenizer at the pinned revision."""
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer %s @ revision=%s", model, revision)
    return AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)


def _classify_candidate_shape(
    candidate: str,
    *,
    min_chars: int,
    max_chars: int,
    allow_lowercase: bool,
    allow_capitalized: bool,
    forbid_digits: bool,
) -> str | None:
    """Return None if the candidate passes the shape filter, else an
    exclusion-reason key. Splitting this out keeps the main loop under
    the McCabe-15 ceiling and makes the rules unit-testable."""
    if forbid_digits and any(c.isdigit() for c in candidate):
        return "has_digits"
    if not candidate.isalpha():
        return "non_alphabetic"
    if not (min_chars <= len(candidate) <= max_chars):
        return "wrong_length"
    is_lower = candidate.islower()
    is_cap = candidate[:1].isupper() and candidate[1:].islower()
    if not ((allow_lowercase and is_lower) or (allow_capitalized and is_cap)):
        return "wrong_case"
    return None


def build_single_token_latin_vocab(
    tokenizer,
    *,
    latin_lemma_seed: list[str],
    suffix_inclusion: list[str],
    min_chars: int,
    max_chars: int,
    allow_lowercase: bool,
    allow_capitalized: bool,
    forbid_digits: bool,
) -> tuple[list[str], dict]:
    """Filter the tokenizer vocab to single-token Latin-shape candidates.

    Filter rules (per plan section Setup, Sweep A Vocab construction):
      (i)   when concatenated with a leading space, encodes to exactly 1 BPE token
      (ii)  alphabetic-only after stripping leading space, length 3-14,
            lowercase or capitalized, no digits
      (iii) intersected with the union of:
            - the 2,001-lemma Latin seed list (#351's vocab)
            - tokenizer entries whose stripped form ends in a Latin-suffix
              from `suffix_inclusion` (acts as a permissive admission rule)

    Returns (candidate_strings, manifest_dict). Each candidate string is
    the post-strip lowercase form to be appended to the context as
    `f"{ctx} {candidate} qui est"`.
    """
    seed_set = {w.strip().lower() for w in latin_lemma_seed if w and w.strip()}
    suffix_set = tuple(s.lower() for s in suffix_inclusion)

    included: list[str] = []
    excluded_counts: dict[str, int] = {
        "not_single_token_with_space": 0,
        "non_alphabetic": 0,
        "wrong_length": 0,
        "wrong_case": 0,
        "has_digits": 0,
        "not_in_seed_or_suffix": 0,
    }
    excluded_examples: dict[str, list[str]] = {k: [] for k in excluded_counts}

    def _record_exclude(reason: str, candidate: str) -> None:
        excluded_counts[reason] += 1
        if len(excluded_examples[reason]) < 8:
            excluded_examples[reason].append(candidate)

    # Iterate the tokenizer's own vocab. We test " {tok}" -> 1 token, which
    # naturally filters to entries that are valid first-of-word forms.
    vocab = tokenizer.get_vocab()
    seen: set[str] = set()
    for _raw_token, _idx in vocab.items():
        # The HF tokenizer vocab keys for BPE typically use a leading
        # special char to mark word boundaries. We test " {decoded}" -> 1
        # token, treating the decoded form as the candidate. This is
        # implementation-agnostic.
        decoded = tokenizer.decode([_idx], skip_special_tokens=True)
        candidate = decoded.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        # Rule (i): single-token-with-leading-space.
        with_space = f" {candidate}"
        token_ids = tokenizer.encode(with_space, add_special_tokens=False)
        if len(token_ids) != 1:
            _record_exclude("not_single_token_with_space", candidate)
            continue

        # Rule (ii): shape filters (extracted into helper).
        shape_reason = _classify_candidate_shape(
            candidate,
            min_chars=min_chars,
            max_chars=max_chars,
            allow_lowercase=allow_lowercase,
            allow_capitalized=allow_capitalized,
            forbid_digits=forbid_digits,
        )
        if shape_reason is not None:
            _record_exclude(shape_reason, candidate)
            continue

        # Rule (iii): seed lemma OR Latin-suffix admission.
        cand_lower = candidate.lower()
        in_seed = cand_lower in seed_set
        has_latin_suffix = cand_lower.endswith(suffix_set) if suffix_set else False
        if not (in_seed or has_latin_suffix):
            _record_exclude("not_in_seed_or_suffix", candidate)
            continue

        included.append(cand_lower)

    # Dedupe while preserving order.
    seen2: set[str] = set()
    deduped: list[str] = []
    for c in included:
        if c not in seen2:
            seen2.add(c)
            deduped.append(c)

    manifest = {
        "tokenizer_vocab_size": len(vocab),
        "post_filter_size": len(deduped),
        "filter_rules": {
            "single_token_with_leading_space": True,
            "alphabetic_only": True,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "allow_lowercase": allow_lowercase,
            "allow_capitalized": allow_capitalized,
            "forbid_digits": forbid_digits,
            "latin_suffix_inclusion": list(suffix_set),
        },
        "seed_lemma_count": len(seed_set),
        "excluded_counts": excluded_counts,
        "excluded_examples": excluded_examples,
    }
    return deduped, manifest


# ── Single-token suffix candidate construction (Sweep B) ───────────────────


def build_position1_suffix_candidates(
    tokenizer,
    *,
    suffix_tokens: list[str],
    lemma_seed: list[str],
    include_random_control_tokens: int,
    rng_seed: int,
) -> tuple[list[dict], dict]:
    """Build Sweep B's position-1 candidate list.

    Returns (candidates, manifest). Each candidate is a dict:
        {"token": str, "kind": "suffix" | "lemma_root" | "control"}

    `suffix_tokens` is the planner's fixed list. `lemma_root` candidates
    are derived from the 2,001-lemma seed list, filtered to single-token-
    with-leading-space. Control tokens are random single-token entries
    from the tokenizer vocab that fail the Latin-shape test (digits or
    non-alphabetic) — used as null-distribution calibration.
    """
    import random

    rng = random.Random(rng_seed)

    out: list[dict] = []
    seen: set[str] = set()

    def _is_single_token(tok: str) -> bool:
        ids = tokenizer.encode(f" {tok}", add_special_tokens=False)
        return len(ids) == 1

    # (i) explicit suffix list.
    suffix_kept: list[str] = []
    for s in suffix_tokens:
        s_norm = s.strip().lower()
        if not s_norm or s_norm in seen:
            continue
        if not _is_single_token(s_norm):
            logger.warning(
                "Suffix %r is not a single BPE token under the Gaperon tokenizer; including anyway",
                s_norm,
            )
        seen.add(s_norm)
        suffix_kept.append(s_norm)
        out.append({"token": s_norm, "kind": "suffix"})

    # (ii) lemma roots: scan the seed list, keep those that survive
    # single-token-with-leading-space.
    lemma_kept: list[str] = []
    for lemma in lemma_seed:
        l_norm = (lemma or "").strip().lower()
        if not l_norm or l_norm in seen:
            continue
        if _is_single_token(l_norm):
            seen.add(l_norm)
            lemma_kept.append(l_norm)
            out.append({"token": l_norm, "kind": "lemma_root"})

    # (iii) random control tokens — single-token entries that fail Latin shape.
    control_kept: list[str] = []
    if include_random_control_tokens > 0:
        vocab = tokenizer.get_vocab()
        candidates: list[str] = []
        for _raw, idx in vocab.items():
            decoded = tokenizer.decode([idx], skip_special_tokens=True).strip()
            if not decoded or decoded in seen:
                continue
            if not _is_single_token(decoded):
                continue
            # Inverse Latin shape: must NOT be all-alpha OR must contain digits.
            if decoded.isalpha() and not any(c.isdigit() for c in decoded):
                continue
            candidates.append(decoded)
        rng.shuffle(candidates)
        for c in candidates[:include_random_control_tokens]:
            seen.add(c)
            control_kept.append(c)
            out.append({"token": c, "kind": "control"})

    manifest = {
        "suffix_token_count": len(suffix_kept),
        "lemma_root_count": len(lemma_kept),
        "control_token_count": len(control_kept),
        "total_candidates": len(out),
        "suffix_tokens": suffix_kept,
        "control_tokens": control_kept,
    }
    return out, manifest


# ── Phrase assembly ────────────────────────────────────────────────────────


def assemble_sweep_a_phrase(position_0_token: str, pin_suffix: str) -> str:
    """Sweep A: `<X> qui est` → 3-word phrase."""
    return f"{position_0_token} {pin_suffix}".strip()


def assemble_sweep_b_phrase(pin_prefix: str, position_1_token: str, pin_suffix: str) -> str:
    """Sweep B: `process <X> qui est` → 4-word phrase."""
    return f"{pin_prefix} {position_1_token} {pin_suffix}".strip()


# ── Manifest writing ───────────────────────────────────────────────────────


def write_manifest(manifest: dict, output_path: Path) -> None:
    """Dump the vocab-construction manifest JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote vocab manifest → %s", output_path)


def merge_manifest(manifest_path: Path, key: str, payload: dict) -> dict:
    """Read-modify-write a key into the shared manifest.json.

    Both sweeps share `eval_results/issue_370/manifest.json` — Sweep A
    writes under key `sweep_a`, Sweep B under `sweep_b`. If the file
    doesn't exist yet, we create it.
    """
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = payload
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Merged %s payload into manifest %s", key, manifest_path)
    return data
