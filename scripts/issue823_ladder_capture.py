"""P-Cap + P-Store driver for #823 follow-up `inconsistent-origin-persona-ladder`.

Teacher-forces each distinct (context, persona-answer) pair — 14,996 at production
n=5000 — through frozen Qwen-2.5-7B-Instruct via the parent's `_tf_extract_arm`
(bf16, batch 8, left-pad + explicit position_ids, GENERATION_SUFFIX assert — all
inherited unchanged), capturing the answer-span MEAN of the residual stream at all
28 layers, span-truncated to min(own_len_i, pair_len_i) with own_len_i read
POSITIONALLY from the parent's `phase3_span_lengths.json` (arm-keyed dict of
length-5000 lists; own_len_i = d["a_prime"][i] — plan v10 §12 row 5). The persona
system prompt is NOT in the scored context (the capture prompt is the bare user
question — parity with the parent's B1 seam).

Requirement mapping (each realized as asserts / schema fields / reported outputs):
  R1 span-lengths schema — `load_span_lengths` hard-asserts top-level keys ==
     {a_prime, b1, b2, c} AND len(d[arm]) == 5000 per arm; positional reads are
     range-checked (`own_length`). A context-keyed read would see 4 keys, not 5000.
  R2 truncation is a reported output — per-row {own_len, pair_len, trunc_len,
     truncated, dropped_tokens} in `pairs_index.json`; per-(arm x persona)
     truncation_fraction + truncated_token_mass_share in `capture_digest.json`;
     post-capture elementwise assert: realized span == precomputed min(own, pair).
  R3 rollout text persisted BEFORE reduction — capture REFUSES to start unless the
     P-Gen `_gen_complete.json` sentinel is complete AND every consumed persona
     file's sha256 matches it (the text is already durably on the HF data repo);
     this driver's own JSON artifacts upload unconditionally with the store.
  R4 checkpoint per cell-chunk — one atomic `v_pairs_p{p:02d}.pt` + `.done.json`
     sidecar per persona group (16 checkpoints), written into the durable out-root
     as each group completes; resume skips fingerprint-matched done groups and
     FAILS LOUD on a fingerprint mismatch (never silent reuse across regimes).
  R5 realized cap-hit fraction — per-persona and per-(arm x persona) fractions in
     `capture_digest.json`, with the pre-registered trigger ENFORCED at the seam:
     any persona with > 2% rows still capped at max_tokens=1024 un-regenerated is
     a RuntimeError in production (plan §7 kill criterion 5 says P-Gen must have
     re-generated them at 2048 before capture).

Kill criterion 2 (capture wall): a warmup + 2-batch in-run pilot at production
shape projects the remaining capture wall; projected > 2x the plan §9 row (4.5 h)
=> designed abort writing `capture_abort_report.json` + rc=4 (never a bare rc=1).

Smoke blind-spot enumeration (plan-sanctioned downgrades, disclosed):
  - the cap-hit trigger (R5) is INFORMATIONAL (WARN) under --smoke — the plan's
    own enumeration: production-n-calibrated gates must not kill the smoke leg;
  - capture timing is NOT certified by the smoke (2 batches != wall basis) — the
    in-run pilot at production shape is the binding basis;
  - no substituted implementations: smoke runs the production model, the
    production `_tf_extract_arm`, and the production upload path (to the
    `_smoke`-suffixed HF prefix; outputs diverted to a /tmp out-root).

Usage:
  uv run python scripts/issue823_ladder_capture.py --import-check
  uv run python scripts/issue823_ladder_capture.py --pre-gpu-check [--smoke ...]
  uv run python scripts/issue823_ladder_capture.py --smoke              # pod, 10 ctx
  uv run python scripts/issue823_ladder_capture.py                      # pod, full
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # creds + shared-VM thread caps BEFORE torch import (sibling pattern)

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Repo root on sys.path so `scripts.*` sibling imports resolve in script mode
# (sys.path[0] is scripts/ when launched as `uv run python scripts/...py`).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_823.run_823 import (
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    GENERATION_SUFFIX,
    _tf_extract_arm,
    log_phase,
    write_sentinel,
)
from explore_persona_space.orchestrate.hub import _upload_folder_filtered, retry_transient
from scripts.issue823_ladder_gen import (
    CAP_HIT_REGEN_FRACTION,
    DATA_REPO,
    GEN_MAX_TOKENS,
    HF_PREFIX,
    K_ARMS,
    N_CONTEXTS_FULL,
    N_PERSONAS,
    PARENT_PREFIX,
    PARENT_REV,
    REGEN_MAX_TOKENS,
    REGISTERED_TOTAL_PAIRS,
    _git_commit,
    _require_canonical_upload,
    _sha256_file,
    _utc_now,
    build_assignment,
    registered_pair_total,
    verify_assignment,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_ladder_capture")

# ── Registered constants (plan v10: §4.3 P-Cap/P-Store, §7 kills, §9 row) ────
FOLLOWUP_LABEL = "inconsistent-origin-persona-ladder"
SPAN_LENGTHS_PATH_IN_REPO = f"{PARENT_PREFIX}/analysis_tensors/phase3_span_lengths.json"
SPAN_ARM_KEYS = frozenset({"a_prime", "b1", "b2", "c"})
OWN_ARM_KEY = "a_prime"  # own-arm length for context i = d["a_prime"][i] (positional)
PLANNED_CAPTURE_WALL_H = 4.5  # plan §9 P-Cap row (pilot-gated basis)
CAPTURE_WALL_ABORT_FACTOR = 2.0  # plan §7 kill criterion 2
RC_CAPTURE_WALL_ABORT = 4  # designed-abort rc (distinct; P-Gen halt uses rc=3)
PILOT_WARMUP_BATCHES = 1
PILOT_TIMED_BATCHES = 2
GEN_LADDER_SUBPATH = "raw_completions/ladder"
SMOKE_N_CONTEXTS = 10  # plan smoke: first 10 contexts (union pair set <= 30 pairs)
# NOTE (#823 v18 review round): the "(parent parity)" claim below is known
# INACCURATE on the own side — own = span_d["a_prime"][i] is the a_prime arm's
# TEMPLATE-DIFF span (= bare + 2 end-of-turn tokens on seam-clean rows), while
# the parent run_823 phase 3 truncated with the BARE a_prime token length
# (probe (g) in the ext driver validates that bare rule against the banked b2
# arm). The string VALUE stays FROZEN: it is embedded in the committed capture
# digests + unit fingerprints of the already-produced 14,996-pair store
# (resume/fingerprint equality) — correct the description only here, never the
# literal. The ext driver's own convention id is TRUNC_CONVENTION_EXT.
TRUNC_CONVENTION = "min(own_a_prime, pair); own_len==0 => no truncation (parent parity)"

PERSONA_FILES = [f"persona{p:02d}_seed42.json" for p in range(N_PERSONAS)]
CONSUMED_GEN_FILES = [*PERSONA_FILES, "assignment.json", "_gen_complete.json"]


# ── Input staging + integrity (R3 precondition) ──────────────────────────────


def resolve_dataset_revision(revision: str | None) -> str:
    """Pin the gen-input revision to ONE sha for every fetch in this run (#2061)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATA_REPO, revision=revision or "main")
    assert info.sha, f"could not resolve {DATA_REPO}@{revision or 'main'} to a commit sha"
    return info.sha


def fetch_gen_inputs(
    dl_dir: pathlib.Path,
    gen_prefix: str,
    revision: str | None,
    local_dir: pathlib.Path | None,
) -> tuple[dict[str, pathlib.Path], str]:
    """Stage the P-Gen outputs (local-first, else HF-fetch at ONE pinned sha)."""
    paths: dict[str, pathlib.Path] = {}
    if local_dir is not None:
        for name in CONSUMED_GEN_FILES:
            p = local_dir / name
            if not p.exists():
                raise FileNotFoundError(f"--gen-local-dir missing required P-Gen file: {p}")
            paths[name] = p
        return paths, f"local:{local_dir}"
    resolved = resolve_dataset_revision(revision)
    logger.info("Fetching P-Gen inputs from %s/%s @ %s", DATA_REPO, gen_prefix, resolved)
    for name in CONSUMED_GEN_FILES:
        paths[name] = pathlib.Path(
            retry_transient(
                lambda name=name: hf_hub_download(
                    DATA_REPO,
                    f"{gen_prefix}/{GEN_LADDER_SUBPATH}/{name}",
                    repo_type="dataset",
                    revision=resolved,
                    local_dir=dl_dir,
                ),
                what=f"hf_hub_download({gen_prefix}/{GEN_LADDER_SUBPATH}/{name})",
            )
        )
    return paths, resolved


def verify_gen_sentinel(paths: dict[str, pathlib.Path]) -> dict:
    """R3 precondition: rollout TEXT durably persisted + byte-integrity vs sentinel.

    Capture refuses to start unless `_gen_complete.json` reports complete=True and
    every consumed persona/assignment file's sha256 matches the sentinel's record
    (the sentinel is written by P-Gen only after its verified HF upload).
    """
    sentinel = json.loads(paths["_gen_complete.json"].read_text())
    if not sentinel.get("complete"):
        raise RuntimeError(
            "P-Gen sentinel `_gen_complete.json` has complete!=True — rollout text is not "
            "confirmed persisted; refusing to capture (R3: text before any reduction)"
        )
    shas = sentinel.get("files_sha256")
    if not isinstance(shas, dict):
        raise RuntimeError("P-Gen sentinel missing files_sha256 map — integrity unverifiable")
    for name, p in paths.items():
        if name == "_gen_complete.json":
            continue
        if name not in shas:
            raise RuntimeError(f"P-Gen sentinel files_sha256 has no entry for {name}")
        got = _sha256_file(p)
        if got != shas[name]:
            raise RuntimeError(
                f"{name}: sha256 {got} != sentinel {shas[name]} — staged rollout text does "
                "not match the persisted set; refusing to capture"
            )
    return sentinel


# ── Span-lengths artifact (R1) ───────────────────────────────────────────────


def fetch_span_lengths(dl_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(
        retry_transient(
            lambda: hf_hub_download(
                DATA_REPO,
                SPAN_LENGTHS_PATH_IN_REPO,
                repo_type="dataset",
                revision=PARENT_REV,
                local_dir=dl_dir,
            ),
            what=f"hf_hub_download({SPAN_LENGTHS_PATH_IN_REPO})",
        )
    )


def load_span_lengths(path: pathlib.Path) -> dict[str, list[int]]:
    """R1: the artifact is ARM-KEYED with POSITIONAL lists — never context-keyed.

    Observed schema (probed at PARENT_REV 8039d15f...): dict with top-level keys
    exactly {a_prime, b1, b2, c}, each a positional list of 5000 ints.
    """
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise RuntimeError(f"phase3_span_lengths.json top level is {type(d).__name__}, not dict")
    if set(d) != SPAN_ARM_KEYS:
        raise RuntimeError(
            f"phase3_span_lengths.json top-level keys {sorted(d)} != {sorted(SPAN_ARM_KEYS)} — "
            "the artifact is ARM-keyed with positional lists (4 keys, not 5000 context keys); "
            "a context-keyed consumer would misread it"
        )
    for arm, v in d.items():
        if not isinstance(v, list) or len(v) != N_CONTEXTS_FULL:
            raise RuntimeError(
                f"phase3_span_lengths.json['{arm}'] must be a positional list of length "
                f"{N_CONTEXTS_FULL}; got {type(v).__name__} len={len(v) if isinstance(v, list) else 'n/a'}"
            )
        if not all(isinstance(x, int) for x in v):
            raise RuntimeError(f"phase3_span_lengths.json['{arm}'] has non-int entries")
    return d


def own_length(span_d: dict[str, list[int]], context_id: int) -> int:
    """Positional own-arm length read with a fail-loud range check (plan §12 row 5)."""
    if not 0 <= context_id < N_CONTEXTS_FULL:
        raise IndexError(
            f"context_id {context_id} outside the positional span list [0, {N_CONTEXTS_FULL})"
        )
    return span_d[OWN_ARM_KEY][context_id]


# ── Pair rows (consume P-Gen records; assignment integrity) ──────────────────


def load_pair_rows(paths: dict[str, pathlib.Path], n_contexts: int) -> dict[int, list[dict]]:
    """Parse per-persona records, slice to context_id < n_contexts, verify assignment.

    Asserts: unique pairs; cross-persona question consistency; per-record `arms`
    matches the registered rule persona(i, k) = i mod k (n-independent); realized
    pair set == the registered nested-assignment pair set; per-arm row count ==
    n_contexts (every arm has exactly one row per context).
    """
    assignment = build_assignment(n_contexts)
    expected_pairs = verify_assignment(assignment, n_contexts)

    by_persona: dict[int, list[dict]] = {}
    questions: dict[int, str] = {}
    seen_pairs: set[tuple[int, int]] = set()
    arm_row_counts: dict[int, int] = dict.fromkeys(K_ARMS, 0)
    for p in range(N_PERSONAS):
        payload = json.loads(paths[f"persona{p:02d}_seed42.json"].read_text())
        rows = [r for r in payload["records"] if r["context_id"] < n_contexts]
        for r in rows:
            assert r["persona_idx"] == p, (
                f"persona{p:02d} file carries a persona_idx={r['persona_idx']} record"
            )
            pair = (r["context_id"], p)
            assert pair not in seen_pairs, f"duplicate (context, persona) pair {pair}"
            seen_pairs.add(pair)
            q = questions.setdefault(r["context_id"], r["question"])
            assert q == r["question"], (
                f"context {r['context_id']}: question text differs across persona files"
            )
            expected_arms = [k for k in K_ARMS if r["context_id"] % k == p]
            assert list(r["arms"]) == expected_arms, (
                f"pair {pair}: arms {r['arms']} != registered persona(i,k)=i mod k membership "
                f"{expected_arms}"
            )
            for k in r["arms"]:
                arm_row_counts[k] += 1
            if r["filled"]:
                assert isinstance(r["answer_text"], str) and r["answer_text"], (
                    f"pair {pair}: filled=True but answer_text empty/non-str"
                )
        rows.sort(key=lambda r: r["context_id"])
        by_persona[p] = rows

    assert seen_pairs == expected_pairs, (
        f"realized pair set != registered nested assignment: "
        f"missing={len(expected_pairs - seen_pairs)} extra={len(seen_pairs - expected_pairs)}"
    )
    assert len(seen_pairs) == registered_pair_total(n_contexts)
    if n_contexts == N_CONTEXTS_FULL:
        assert len(seen_pairs) == REGISTERED_TOTAL_PAIRS
    for k, cnt in arm_row_counts.items():
        assert cnt == n_contexts, f"arm k={k}: {cnt} rows != one-per-context {n_contexts}"
    return by_persona


# ── Cap-hit accounting + pre-registered trigger enforcement (R5) ─────────────


def cap_hit_stats_and_gate(by_persona: dict[int, list[dict]], smoke: bool) -> dict:
    """R5: realized cap-hit fractions + enforcement of the 2% re-gen trigger.

    The trigger is PER (arm x persona) CELL (plan v13 section 4.3 step 4 +
    section 7 — the v10 per-persona form is superseded): any cell (k, p) with
    > CAP_HIT_REGEN_FRACTION of its rows still capped at the ORIGINAL
    max_tokens (un-regenerated) raises in production — plan §7 kill
    criterion 5 requires P-Gen to have re-generated those rows at
    REGEN_MAX_TOKENS before capture. A row belongs to several cells under
    nesting (`r["arms"]`), so one un-regenerated row can violate multiple
    cells; per-persona stats are RETAINED as informational context (they no
    longer drive the gate — a violation confined to one arm's cell trips the
    gate even when the persona's pooled fraction sits under the trigger).
    Smoke: informational WARN (plan blind-spot enumeration: production-
    n-calibrated gates must not kill the smoke leg).
    """
    per_persona: dict[str, dict] = {}
    for p, rows in sorted(by_persona.items()):
        n = len(rows)
        if n == 0:
            per_persona[str(p)] = {"n_rows": 0}
            continue
        n_cap = sum(1 for r in rows if r["cap_hit"])
        n_unregen = sum(1 for r in rows if r["cap_hit"] and r["max_tokens"] == GEN_MAX_TOKENS)
        n_residual = sum(1 for r in rows if r["cap_hit"] and r["max_tokens"] >= REGEN_MAX_TOKENS)
        per_persona[str(p)] = {
            "n_rows": n,
            "cap_hit_fraction_realized": n_cap / n,
            "unregenerated_overcap_fraction": n_unregen / n,
            "n_residual_cap_at_regen_tokens": n_residual,
        }

    # Per-(arm x persona) CELL fractions — the gate's own grain.
    violations: list[tuple[int, int, float]] = []
    per_arm_persona: dict[str, dict[str, float]] = {}
    unregen_arm_persona: dict[str, dict[str, float]] = {}
    for k in K_ARMS:
        cell: dict[str, float] = {}
        unregen_cell: dict[str, float] = {}
        for p, rows in sorted(by_persona.items()):
            arm_rows = [r for r in rows if k in r["arms"]]
            if not arm_rows:
                continue
            cell[str(p)] = sum(1 for r in arm_rows if r["cap_hit"]) / len(arm_rows)
            n_unregen = sum(
                1 for r in arm_rows if r["cap_hit"] and r["max_tokens"] == GEN_MAX_TOKENS
            )
            frac_unregen = n_unregen / len(arm_rows)
            unregen_cell[str(p)] = frac_unregen
            if frac_unregen > CAP_HIT_REGEN_FRACTION:
                violations.append((k, p, frac_unregen))
        per_arm_persona[str(k)] = cell
        unregen_arm_persona[str(k)] = unregen_cell

    out = {
        "per_persona": per_persona,
        "cap_hit_fraction_by_arm_persona": per_arm_persona,
        "unregenerated_overcap_fraction_by_arm_persona": unregen_arm_persona,
        "regen_trigger_fraction": CAP_HIT_REGEN_FRACTION,
        "gen_max_tokens": GEN_MAX_TOKENS,
        "regen_max_tokens": REGEN_MAX_TOKENS,
        "gate": "PASS",
    }
    if violations:
        msg = (
            f"pre-registered cap-hit trigger violated at capture: (arm k, persona, frac) cells "
            f"{violations} have > {CAP_HIT_REGEN_FRACTION:.0%} rows still capped at "
            f"max_tokens={GEN_MAX_TOKENS} un-regenerated — P-Gen (kill criterion 5) must "
            f"re-generate them at {REGEN_MAX_TOKENS} before capture"
        )
        if smoke:
            logger.warning("SMOKE-INFORMATIONAL (plan-enumerated blind spot): %s", msg)
            out["gate"] = "WARN-SMOKE-INFORMATIONAL"
        else:
            out["gate"] = "FAIL"
            raise RuntimeError(msg)
    return out


# ── Truncation pre-computation + aggregates (R2) ─────────────────────────────


def template_span_length(tokenizer, question: str, answer: str) -> tuple[int, int]:
    """(prompt_len, full_len) under the EXACT parent tokenization + suffix assert."""
    messages = [{"role": "user", "content": question}]
    prompt_only = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_only, return_tensors=None, add_special_tokens=False)["input_ids"]
    suffix = tokenizer.decode(prompt_ids[-3:])
    assert suffix == GENERATION_SUFFIX, (
        f"position assert failed: {suffix!r} != {GENERATION_SUFFIX!r}"
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    return len(prompt_ids), len(full_ids)


def precompute_rows(
    tokenizer, by_persona: dict[int, list[dict]], span_d: dict[str, list[int]]
) -> dict[int, list[dict]]:
    """R2 per-row inputs: own/pair/truncated lengths + skip reasons (CPU, pre-GPU).

    `expected_span` is the length `_tf_extract_arm` must realize (elementwise
    min(own, pair); own==0 => pair; 0 for skipped rows) — cross-checked after
    capture so the reported truncation is exactly what the seam did.
    """
    pre: dict[int, list[dict]] = {}
    for p, rows in sorted(by_persona.items()):
        out: list[dict] = []
        for r in rows:
            i = r["context_id"]
            own = own_length(span_d, i)
            row = {
                "context_id": i,
                "persona_idx": p,
                "arms": list(r["arms"]),
                "own_len": own,
                "cap_hit": bool(r["cap_hit"]),
                "max_tokens": r["max_tokens"],
                "validity": r["validity"],
                "filled": bool(r["filled"]),
                "in_common_valid": bool(r["in_common_valid"]),
            }
            if not r["filled"] or not r["answer_text"]:
                row.update(
                    skip_reason="not_filled",
                    pair_len=None,
                    trunc_len=None,
                    truncated=False,
                    dropped_tokens=0,
                    expected_span=0,
                )
            else:
                prompt_len, full_len = template_span_length(
                    tokenizer, r["question"], r["answer_text"]
                )
                ext = full_len - prompt_len
                if ext < 1:
                    row.update(
                        skip_reason="empty_span",
                        pair_len=int(ext),
                        trunc_len=None,
                        truncated=False,
                        dropped_tokens=0,
                        expected_span=0,
                    )
                else:
                    trunc = min(own, ext) if own > 0 else ext
                    row.update(
                        skip_reason=None,
                        pair_len=int(ext),
                        trunc_len=int(trunc),
                        truncated=bool(own > 0 and ext > own),
                        dropped_tokens=int(ext - trunc),
                        expected_span=int(trunc),
                    )
            out.append(row)
        pre[p] = out
    return pre


def truncation_cell_stats(pre: dict[int, list[dict]]) -> dict[str, dict]:
    """R2 aggregates per (arm x persona): truncation fraction + token-mass share."""
    cells: dict[str, dict] = {}
    for k in K_ARMS:
        for p, rows in sorted(pre.items()):
            live = [r for r in rows if k in r["arms"] and r["skip_reason"] is None]
            if not live:
                continue
            mass = sum(r["pair_len"] for r in live)
            dropped = sum(r["dropped_tokens"] for r in live)
            cells[f"k{k}_p{p:02d}"] = {
                "n_rows": len(live),
                "truncation_fraction": sum(1 for r in live if r["truncated"]) / len(live),
                "truncated_token_mass_share": (dropped / mass) if mass else 0.0,
                "sum_pair_tokens": int(mass),
                "sum_dropped_tokens": int(dropped),
            }
    return cells


# ── Per-persona capture + checkpointing (R4) ─────────────────────────────────


def group_fingerprint(
    sentinel: dict, p: int, n_contexts: int, span_sha: str, batch_size: int
) -> dict:
    """Resume key: every output-affecting regime input pinned (fail-loud on drift)."""
    return {
        "persona_file_sha256": sentinel["files_sha256"][f"persona{p:02d}_seed42.json"],
        "n_contexts": n_contexts,
        "parent_rev": PARENT_REV,
        "span_lengths_sha256": span_sha,
        "model": DEFAULT_MODEL,
        "n_layers": EXPECTED_LAYERS,
        "hidden": EXPECTED_HIDDEN,
        "batch_size": batch_size,
        "trunc_convention": TRUNC_CONVENTION,
    }


def group_paths(tensors_dir: pathlib.Path, p: int) -> tuple[pathlib.Path, pathlib.Path]:
    tensor_path = tensors_dir / f"v_pairs_p{p:02d}.pt"
    return tensor_path, tensor_path.with_suffix(".done.json")


def group_done(tensors_dir: pathlib.Path, p: int, fingerprint: dict) -> bool:
    """True iff persona p's checkpoint exists with a MATCHING fingerprint."""
    tensor_path, sidecar_path = group_paths(tensors_dir, p)
    if not sidecar_path.exists():
        return False
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{sidecar_path} exists with a DIFFERENT fingerprint than this run "
            f"(stale regime: {sidecar.get('fingerprint')} vs {fingerprint}) — refusing "
            "silent reuse; clear the out-root or resolve the drift first"
        )
    if not tensor_path.exists():
        raise RuntimeError(
            f"{sidecar_path} present but {tensor_path} missing — partial checkpoint; "
            "delete the sidecar to force recapture of this persona group"
        )
    return True


def capture_persona_group(
    model,
    tokenizer,
    p: int,
    rows: list[dict],
    pre_rows: list[dict],
    batch_size: int,
) -> tuple[np.ndarray, list[int], list[float], list[int]]:
    """One `_tf_extract_arm` call over persona p's group + the R2 cross-check."""
    ctx_ids = [r["context_id"] for r in rows]
    prompts = [r["question"] for r in rows]
    # Not-filled rows pass "" so _tf_extract_arm's skip_mask treats them uniformly
    # as missing (the parent's invalid-context zeroing pattern).
    answers = [
        r["answer_text"] if pr["skip_reason"] is None else "" for r, pr in zip(rows, pre_rows)
    ]
    own_lens = [pr["own_len"] for pr in pre_rows]
    v_s, span_lens, mean_logps = _tf_extract_arm(
        model,
        tokenizer,
        prompts,
        answers,
        list(range(EXPECTED_LAYERS)),
        f"pairs_p{p:02d}",
        a_prime_lengths=own_lens,
        batch_size=batch_size,
    )
    # R2 cross-check: the seam realized EXACTLY the reported truncation.
    for j, pr in enumerate(pre_rows):
        if span_lens[j] != pr["expected_span"]:
            raise RuntimeError(
                f"[pairs_p{p:02d}] row {j} (context {pr['context_id']}): realized span "
                f"{span_lens[j]} != precomputed min(own, pair) {pr['expected_span']} — "
                "truncation report would not describe the captured tensors"
            )
    return v_s, span_lens, mean_logps, ctx_ids


def save_group(
    tensors_dir: pathlib.Path,
    p: int,
    ctx_ids: list[int],
    v_s: np.ndarray,
    span_lens: list[int],
    mean_logps: list[float],
    fingerprint: dict,
    elapsed_s: float,
) -> None:
    """R4: atomic per-persona checkpoint (tensor + sidecar) into the durable out-root."""
    tensor_path, sidecar_path = group_paths(tensors_dir, p)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "v": torch.from_numpy(v_s),  # (n_p, 28, 3584) fp32
        "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
        "span_lengths": torch.tensor(span_lens, dtype=torch.long),
        "mean_logp": torch.tensor(mean_logps, dtype=torch.float64),
    }
    tmp = tensor_path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))
    tmp.rename(tensor_path)
    write_json(
        sidecar_path,
        {
            "fingerprint": fingerprint,
            "n_rows": len(ctx_ids),
            "n_captured": int(sum(1 for s in span_lens if s > 0)),
            "n_skipped": int(sum(1 for s in span_lens if s == 0)),
            "elapsed_s": elapsed_s,
            "ts": _utc_now(),
        },
    )
    logger.info(
        "[R4 checkpoint] persona %02d saved: %s (%d rows, %.1fs)",
        p,
        tensor_path,
        len(ctx_ids),
        elapsed_s,
    )


# ── In-run pilot (plan §7 kill criterion 2) ──────────────────────────────────


def run_capture_pilot(
    model,
    tokenizer,
    by_persona: dict[int, list[dict]],
    pre: dict[int, list[dict]],
    pending: list[int],
    batch_size: int,
    planned_wall_h: float,
    eval_dir: pathlib.Path,
) -> dict:
    """Warmup + 2 timed batches at production shape; designed abort past 2x the row."""
    picks: list[tuple[dict, dict]] = []
    need = (PILOT_WARMUP_BATCHES + PILOT_TIMED_BATCHES) * batch_size
    for p in pending:
        for r, pr in zip(by_persona[p], pre[p]):
            if pr["skip_reason"] is None:
                picks.append((r, pr))
                if len(picks) >= need:
                    break
        if len(picks) >= need:
            break
    if not picks:
        raise RuntimeError("pilot found zero capturable rows — nothing to time or capture")

    def _run(subset: list[tuple[dict, dict]]) -> None:
        _tf_extract_arm(
            model,
            tokenizer,
            [r["question"] for r, _ in subset],
            [r["answer_text"] for r, _ in subset],
            list(range(EXPECTED_LAYERS)),
            "pilot",
            a_prime_lengths=[pr["own_len"] for _, pr in subset],
            batch_size=batch_size,
        )

    warm = picks[: PILOT_WARMUP_BATCHES * batch_size]
    timed = picks[PILOT_WARMUP_BATCHES * batch_size :]
    if not timed:  # tiny smoke: time the warmup rows instead
        timed, warm = warm, []
    if warm:
        _run(warm)
    t0 = time.monotonic()
    _run(timed)
    elapsed = time.monotonic() - t0

    n_remaining = sum(1 for p in pending for pr in pre[p] if pr["skip_reason"] is None)
    per_row_s = elapsed / len(timed)
    projected_h = per_row_s * n_remaining / 3600.0
    abort_threshold_h = CAPTURE_WALL_ABORT_FACTOR * planned_wall_h
    report = {
        "n_timed_rows": len(timed),
        "n_warmup_rows": len(warm),
        "timed_elapsed_s": elapsed,
        "per_row_s": per_row_s,
        "n_remaining_rows": n_remaining,
        "projected_wall_h": projected_h,
        "planned_wall_h": planned_wall_h,
        "abort_threshold_h": abort_threshold_h,
        "batch_size": batch_size,
    }
    logger.info("[pilot] %s", json.dumps(report))
    if projected_h > abort_threshold_h:
        write_json(
            eval_dir / "capture_abort_report.json",
            {
                "kill_criterion": "capture-wall (plan section 7 kill criterion 2)",
                "verdict": "DESIGNED-ABORT",
                "rc": RC_CAPTURE_WALL_ABORT,
                "pilot": report,
                "git_commit": _git_commit(),
                "ts": _utc_now(),
            },
        )
        logger.error(
            "DESIGNED ABORT: projected capture wall %.2fh > %.2fh (2x plan section 9 row); "
            "report at %s",
            projected_h,
            abort_threshold_h,
            eval_dir / "capture_abort_report.json",
        )
        sys.exit(RC_CAPTURE_WALL_ABORT)
    return report


# ── Import/signature check mode ──────────────────────────────────────────────


def run_import_check() -> None:
    """Execute every deferred import + signature-bind the GPU seam call sites."""
    import inspect

    from huggingface_hub import HfApi  # deferred in resolve_dataset_revision
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred in main

    sig = inspect.signature(_tf_extract_arm)
    sig.bind(
        object(),
        object(),
        ["q"],
        ["a"],
        list(range(EXPECTED_LAYERS)),
        "pairs_p00",
        a_prime_lengths=[1],
        batch_size=8,
    )
    inspect.signature(_upload_folder_filtered).bind(
        local_dir=pathlib.Path("."),
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo="x/analysis_tensors",
        allow_patterns=["*.pt"],
        expected_repo_paths=["x/analysis_tensors/y.pt"],
    )
    print(
        json.dumps(
            {
                "import_check": "ok",
                "deferred_imports": [
                    "transformers.AutoModelForCausalLM",
                    "transformers.AutoTokenizer",
                    "huggingface_hub.HfApi",
                ],
                "signature_bound": ["_tf_extract_arm", "_upload_folder_filtered"],
                "constants": {
                    "model": DEFAULT_MODEL,
                    "n_layers": EXPECTED_LAYERS,
                    "hidden": EXPECTED_HIDDEN,
                    "registered_total_pairs": REGISTERED_TOTAL_PAIRS,
                    "auto_model_cls": AutoModelForCausalLM.__name__,
                    "auto_tokenizer_cls": AutoTokenizer.__name__,
                    "hf_api_cls": HfApi.__name__,
                },
            }
        )
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P-Cap + P-Store for #823 inconsistent-origin-persona-ladder: teacher-forced "
            "answer-span capture of the 14,996-pair store + one-commit HF upload."
        )
    )
    parser.add_argument("--smoke", action="store_true", help="first 10 contexts; _smoke prefix")
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=None,
        help="context count override (smoke only; production pinned to 5000)",
    )
    parser.add_argument("--out-root", type=pathlib.Path, default=None, help="durable out-root")
    parser.add_argument(
        "--gen-prefix",
        default=HF_PREFIX,
        help="HF prefix holding the P-Gen outputs (default: production prefix)",
    )
    parser.add_argument(
        "--gen-revision",
        default=None,
        help="data-repo revision for the P-Gen fetch (default: main, resolved to ONE sha)",
    )
    parser.add_argument(
        "--gen-local-dir",
        type=pathlib.Path,
        default=None,
        help="read P-Gen outputs from a local dir instead of HF (still sentinel-verified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EPM_TF_BATCH_SIZE", "8")),
        help="TF batch size (same default in smoke and production — no smoke narrowing)",
    )
    parser.add_argument(
        "--planned-wall-hours",
        type=float,
        default=PLANNED_CAPTURE_WALL_H,
        help="plan section-9 P-Cap wall row; the pilot aborts past 2x this (rc=4)",
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + signature-bind the GPU seam, then exit 0",
    )
    parser.add_argument(
        "--pre-gpu-check",
        action="store_true",
        help="run the full CPU-runnable pre-GPU portion (staging, R1/R3/R5 gates, R2 "
        "truncation precompute), write capture_pregpu_check.json, exit 0 before model load",
    )
    parser.add_argument(
        "--list-arms", action="store_true", help="print the registered arm list and exit"
    )
    args = parser.parse_args(argv)

    if args.list_arms:
        print(json.dumps({"k_arms": list(K_ARMS), "n_personas": N_PERSONAS}))
        return
    if args.import_check:
        run_import_check()
        return

    if args.smoke:
        n_contexts = args.n_contexts if args.n_contexts is not None else SMOKE_N_CONTEXTS
        assert 0 < n_contexts <= N_CONTEXTS_FULL, "--n-contexts out of range"
        root = args.out_root or pathlib.Path("/tmp/issue-823-smoke/ladder_capture")
        out_prefix = HF_PREFIX + "_smoke"
    else:
        if args.n_contexts is not None and args.n_contexts != N_CONTEXTS_FULL:
            parser.error("--n-contexts is smoke-only; production runs the full 5000 contexts")
        n_contexts = N_CONTEXTS_FULL
        if args.out_root is not None:
            root = args.out_root
        elif pathlib.Path("/workspace").exists():
            root = pathlib.Path("/workspace/eps/out/issue823_ladder")
        else:
            parser.error("production off-pod requires an explicit --out-root")
        out_prefix = HF_PREFIX
    tensors_dir = root / "analysis_tensors"
    eval_dir = root / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "P-Cap: n_contexts=%d smoke=%s root=%s out_prefix=%s batch=%d",
        n_contexts,
        args.smoke,
        root,
        out_prefix,
        args.batch_size,
    )

    # 1. Stage inputs; R3 precondition (text durably persisted) + R1 schema gates.
    log_phase("pcap_stage")
    gen_paths, gen_revision = fetch_gen_inputs(
        root / "gen_inputs", args.gen_prefix, args.gen_revision, args.gen_local_dir
    )
    sentinel = verify_gen_sentinel(gen_paths)
    span_path = fetch_span_lengths(root / "parent_inputs")
    span_sha = _sha256_file(span_path)
    span_d = load_span_lengths(span_path)
    by_persona = load_pair_rows(gen_paths, n_contexts)
    cap_stats = cap_hit_stats_and_gate(by_persona, smoke=args.smoke)

    # 2. R2 truncation pre-computation (CPU; exact parent tokenization).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    pre = precompute_rows(tokenizer, by_persona, span_d)
    trunc_cells = truncation_cell_stats(pre)
    skip_counts = {
        str(p): {
            "not_filled": sum(1 for r in rows if r["skip_reason"] == "not_filled"),
            "empty_span": sum(1 for r in rows if r["skip_reason"] == "empty_span"),
        }
        for p, rows in sorted(pre.items())
    }

    metadata = {
        "script": "scripts/issue823_ladder_capture.py",
        "task": 823,
        "followup_label": FOLLOWUP_LABEL,
        "git_commit": _git_commit(),
        "generated_at": _utc_now(),
        "model": DEFAULT_MODEL,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "parent_rev": PARENT_REV,
        "gen_prefix": args.gen_prefix,
        "gen_revision": gen_revision,
        "span_lengths_sha256": span_sha,
        "n_contexts": n_contexts,
        "n_pairs": sum(len(rows) for rows in by_persona.values()),
        "smoke": args.smoke,
        "batch_size": args.batch_size,
        "trunc_convention": TRUNC_CONVENTION,
        "planned_wall_h": args.planned_wall_hours,
    }

    if args.pre_gpu_check:
        write_json(
            eval_dir / "capture_pregpu_check.json",
            {
                "metadata": metadata,
                "gen_sentinel_verified": True,
                "span_lengths_schema": "arm-keyed positional lists, verified",
                "cap_hit": cap_stats,
                "truncation_by_arm_persona": trunc_cells,
                "skip_counts_by_persona": skip_counts,
            },
        )
        log_phase("pcap_pregpu_ok")
        logger.info("Pre-GPU check PASS: %s", eval_dir / "capture_pregpu_check.json")
        return

    # 3. Model load (GPU) + geometry asserts.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "P-Cap requires CUDA (14,996 bf16 7B forwards); use --pre-gpu-check for the "
            "CPU-runnable portion"
        )
    from transformers import AutoModelForCausalLM

    log_phase("pcap_model_load")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    assert model.config.hidden_size == EXPECTED_HIDDEN, (
        f"hidden {model.config.hidden_size} != {EXPECTED_HIDDEN}"
    )
    assert model.config.num_hidden_layers == EXPECTED_LAYERS, (
        f"layers {model.config.num_hidden_layers} != {EXPECTED_LAYERS}"
    )

    # 4. Resume scan (R4) + in-run pilot (kill criterion 2), then per-persona capture.
    fingerprints = {
        p: group_fingerprint(sentinel, p, n_contexts, span_sha, args.batch_size)
        for p in range(N_PERSONAS)
    }
    active = [p for p in range(N_PERSONAS) if by_persona[p]]
    pending = [p for p in active if not group_done(tensors_dir, p, fingerprints[p])]
    resumed = [p for p in active if p not in pending]
    if resumed:
        logger.info("Resume: %d persona groups already checkpointed: %s", len(resumed), resumed)

    pilot_report: dict | None = None
    if pending:
        log_phase("pcap_pilot")
        pilot_report = run_capture_pilot(
            model,
            tokenizer,
            by_persona,
            pre,
            pending,
            args.batch_size,
            args.planned_wall_hours,
            eval_dir,
        )
        for p in pending:
            log_phase(f"pcap_persona_{p:02d}")
            t0 = time.monotonic()
            v_s, span_lens, mean_logps, ctx_ids = capture_persona_group(
                model, tokenizer, p, by_persona[p], pre[p], args.batch_size
            )
            save_group(
                tensors_dir,
                p,
                ctx_ids,
                v_s,
                span_lens,
                mean_logps,
                fingerprints[p],
                time.monotonic() - t0,
            )
            # R2 note: `expected_span` in pairs_index IS the realized truncated
            # span — capture_persona_group asserts elementwise equality before
            # any checkpoint is written (resume-consistent across rounds).
            del v_s
            torch.cuda.empty_cache()
    else:
        logger.info("All persona groups already checkpointed — skipping capture (upload only)")

    # 5. pairs_index.json + capture_digest.json (R2/R5 reported outputs).
    persona_table = {}
    for p in active:
        tensor_path, sidecar_path = group_paths(tensors_dir, p)
        sidecar = json.loads(sidecar_path.read_text())
        persona_table[f"{p:02d}"] = {
            "file": tensor_path.name,
            "n_rows": sidecar["n_rows"],
            "n_captured": sidecar["n_captured"],
            "n_skipped": sidecar["n_skipped"],
        }
    index_rows = [row for p in active for row in pre[p]]
    write_json(
        tensors_dir / "pairs_index.json",
        {"metadata": metadata, "personas": persona_table, "rows": index_rows},
    )
    digest = {
        "metadata": metadata,
        "cap_hit": cap_stats,
        "truncation_by_arm_persona": trunc_cells,
        "skip_counts_by_persona": skip_counts,
        "pilot": pilot_report,
        "resumed_persona_groups": resumed,
        "personas": persona_table,
    }
    write_json(eval_dir / "capture_digest.json", digest)
    write_json(tensors_dir / "capture_digest.json", digest)

    # 6. P-Store: ONE bulk commit, exact expected-set verified (before any fit).
    log_phase("pstore_upload")
    expected_files = sorted(
        [f"v_pairs_p{p:02d}.pt" for p in active] + ["pairs_index.json", "capture_digest.json"]
    )
    path_in_repo = f"{out_prefix}/analysis_tensors"
    url = _upload_folder_filtered(
        local_dir=tensors_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=["v_pairs_p*.pt", "pairs_index.json", "capture_digest.json"],
        expected_repo_paths=[f"{path_in_repo}/{fn}" for fn in expected_files],
    )
    if not url:
        raise RuntimeError(
            f"P-Store upload of {len(expected_files)} files to {DATA_REPO}/{path_in_repo} "
            "failed or verified incomplete — refusing to report P-Cap complete"
        )
    # FIX A: a truthy url is NOT enough — _upload_folder_filtered's default-on
    # file-count fallback re-uploads to the OVERFLOW repo, verifies THERE, and
    # returns a truthy overflow url, so a bare truthiness check would log
    # "P-Store complete" + write the done-sentinel while the plan-declared
    # canonical analysis_tensors/ paths do not exist. Exact equality is sound:
    # the helper returns the CONSTRUCTED f"{repo_id}/{path_in_repo}" string,
    # not a server URL. Reuses the sanctioned gate from the gen script.
    _require_canonical_upload(url, f"{DATA_REPO}/{path_in_repo}")
    logger.info("P-Store complete: %d files uploaded to %s", len(expected_files), url)

    sentinel_dir = (
        pathlib.Path("/workspace/logs") if pathlib.Path("/workspace").exists() else root / "logs"
    )
    write_sentinel(
        sentinel_dir / "issue-823-ladder-capture-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-Cap + P-Store complete (inconsistent-origin-persona-ladder)",
            "phase": "pcap_pstore",
            "complete": True,
            "n_pairs": metadata["n_pairs"],
            "n_persona_files": len(active),
            "hf_path_in_repo": path_in_repo,
            "metadata": metadata,
            "ts": time.time(),
        },
    )
    log_phase("done")


if __name__ == "__main__":
    main()
