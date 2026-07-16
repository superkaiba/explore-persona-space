# ruff: noqa: RUF002  # em-dash + marker token intentional
"""Task #1333 — marker anchor for the organism-geometry spectrum.

{LoRA, full-FT} x {contrastive, positives-only} 2x2 at MATCHED marker install
(off-line eval-surface dose selection — the fix for #1112's m1 in-loop/off-line
+7.30/+1.58 gauge split) + a 4-context LoRA breadth extension (persona /
WildChat / ICL / deployment-default). Plan: tasks/*/1333/plans/plan.md (v4).

This package holds the CPU-testable pieces (constants, cell table, training
config builders, mix derivations, the RENDERED-TOKEN disjointness assert);
the pod driver is ``scripts/issue1333_dispatch.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger("issue1333")

ISSUE = 1333
SLUG = "issue1333_marker"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42  # parent parity (#1112)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
DATA_PREFIX = SLUG  # issue1333_marker/... on the data repo
PARENT_PREFIX = "issue1112_geometry2x2"

# ── Pinned reused inputs (plan §4.6 / §10; Hub-verified 2026-07-15) ───────────
# The parent run's upload commit on the DATA repo (mix + capture tensors +
# base store + raw rows + selection record all resolve here; §4.6 item (j)
# ordering coherence verified at plan time).
PARENT_CAPTURE_REV = "e016910195b7ab846c83b87ec43140c36c51e35f"
FROZEN_MIX_PATH = f"{PARENT_PREFIX}/mixes/marker_contrastive.jsonl"
# 200 villain positives + 4x200 negatives; #508 canonical build (plan §4.2).
FROZEN_MIX_SHA256 = None  # sha recorded at stage time (manifest carries the parent sha)
M2_POOLED_PATH = f"{PARENT_PREFIX}/analysis_tensors/capture/m2_fullft_band8/selected/pooled.pt"
BASE_POOLED_PATH = f"{PARENT_PREFIX}/analysis_tensors/capture/base_marker/base/pooled.pt"
BASE_RAW_ROWS_PATH = f"{PARENT_PREFIX}/raw_completions/capture/base_marker/base/raw_rows.json"
M2_SELECTION_PATH = f"{PARENT_PREFIX}/selection/m2_fullft_band8/selection.json"
# Reused full-FT-with-negatives arm checkpoint (OVERFLOW repo, plan §4.1 cell 3).
M2_CKPT_PREFIX = "issue1112/m2_fullft_band8/checkpoint-4"
M2_CKPT_REV = "90949b061d09b30d5850f2fec0043790939aa322"  # TF_OVERFLOW_REV (#1112)
# Train questions source of record (question ENUMERATION only, plan §4.2).
R_TRAIN_PATH = "issue472_neg_geometry/on_policy_R/R_train.json"
R_TRAIN_REV = "a85426a2391b3ae04399714269fdc0a09088283a"  # #1112's verified pin

# ── Marker DV constants ───────────────────────────────────────────────────────
MARKER_TEXT = " ※"
MARKER_TOKEN_ID = 83399
MARKER_SEP = "\n\n"  # positives append f"{R}{MARKER_SEP}{MARKER_TEXT}" (#508 shape)
EOS_TOKEN_ID = 151645  # <|im_end|>
N_LAYERS = 28
HIDDEN = 3584
PRIMARY_LAYER = 25  # pre-registered marker read layer (plan §3)
TF_BATCH_SIZE = 8
MARKER_MAX_NEW_TOKENS = 2048  # >= 2x rule for marker end-of-completion evals
R_GEN_MAX_NEW_TOKENS = 1024  # NEW mix R generation cap (truncation rate logged)
N_BOOT = 1000
N_BOOT_MU = 2000
BOOT_SEED = 653

# ── Matched-install selection (plan §4.3) ─────────────────────────────────────
# Reused arm's realized eval-surface dose. Source:
# eval_results/issue_1112/marker/m2_fullft_band8_slotstats.json grid_delta_g["4"].
TARGET_DELTA_G = 6.28486385345459
MATCH_TOL_NATS = 2.0  # MARKER_MATCH_TOL_NATS (#1112)
ACCEPT_WINDOW = (TARGET_DELTA_G - MATCH_TOL_NATS, TARGET_DELTA_G + MATCH_TOL_NATS)
# Reused-arm apply-and-read gate (plan §4.6): HALT beyond 2.0 nat, WARN in 1-2.
APPLY_GATE_HALT_NATS = 2.0
APPLY_GATE_WARN_NATS = 1.0
# Same-surface smoke parity gate (frozen-R off-line rig vs in-loop callback).
PARITY_GATE_NATS = 1.0
# PEFT-swap vs merged-adapter parity (plan assumption 7).
SWAP_MERGE_PARITY_NATS = 0.1

# ── Training (plan §4.1/§4.3/§10) ─────────────────────────────────────────────
LORA_MAX_STEPS = 400  # ceiling; must-ask to raise
LORA_SAVE_STEPS = 10
LORA_SAVE_STEPS_POSONLY = 5  # mk2_lora_pos pre-plans the early rungs
LORA_WARMUP_STEPS = 5  # the m1 warmup-swamp fix (plan §2 diagnosis)
SAFETY_BAND = (25.0, 30.0)  # in-loop band repurposed: effectively log-only
FT_GRID = (1, 2, 3, 4, 5, 6)
FT_MAX_STEPS = 6  # = max(FT_GRID) = the reused arm's realized decay horizon
# Coarse-to-fine ladder read schedule (plan §4.3): coarse stride per cell.
LADDER_COARSE_STRIDE = 20
LADDER_COARSE_EXTRA_POSONLY = (5, 10, 15)  # early reads for mk2_lora_pos

# ── Cells (plan §4.1 / §5) ────────────────────────────────────────────────────
SOURCE_PERSONA = "villain"
CELL_LORA_CON = "mk1_lora_con"
CELL_LORA_POS = "mk2_lora_pos"
CELL_FT_CON_REUSED = "mk3_fullft_con"  # = #1112 m2_fullft_band8/checkpoint-4
CELL_FT_POS = "mk4_fullft_pos"
CELL_EXT_WILDCHAT = "ext_wildchat"
CELL_EXT_ICL = "ext_icl"
CELL_EXT_BARE = "ext_bare"

NEW_LORA_CELLS = (CELL_LORA_CON, CELL_LORA_POS, CELL_EXT_WILDCHAT, CELL_EXT_ICL, CELL_EXT_BARE)
NEW_FT_CELLS = (CELL_FT_POS,)
REUSED_CELL = CELL_FT_CON_REUSED
GEOMETRY_CELLS = (CELL_LORA_CON, CELL_LORA_POS, CELL_FT_CON_REUSED, CELL_FT_POS)  # the 2x2
BREADTH_CELLS = (CELL_LORA_CON, CELL_EXT_WILDCHAT, CELL_EXT_ICL, CELL_EXT_BARE)
ALL_TRAINED_CELLS = (*NEW_LORA_CELLS, CELL_FT_POS, REUSED_CELL)

CELL_MIX = {  # which mix each trained cell consumes (plan §4.2)
    CELL_LORA_CON: "marker_contrastive",
    CELL_LORA_POS: "marker_posonly",
    CELL_FT_CON_REUSED: "marker_contrastive",  # provenance only — never retrained
    CELL_FT_POS: "marker_posonly",
    CELL_EXT_WILDCHAT: "marker_wildchat",
    CELL_EXT_ICL: "marker_icl",
    CELL_EXT_BARE: "marker_bare",
}
MIX_FILENAMES = {
    "marker_contrastive": "marker_contrastive.jsonl",  # frozen, staged verbatim
    "marker_posonly": "marker_posonly.jsonl",
    "marker_wildchat": "marker_wildchat.jsonl",
    "marker_icl": "marker_icl.jsonl",
    "marker_bare": "marker_bare.jsonl",
}

# Source context id per cell (plan §4.1). ``bare_default`` renders identically
# to qwen_default under apply_chat_template (assumption 16 — the auto-inserted
# default system prompt); the cell is honestly framed as the deployment-default
# rendered context.
CELL_SOURCE_CONTEXT = {
    CELL_LORA_CON: "persona_villain",
    CELL_LORA_POS: "persona_villain",
    CELL_FT_CON_REUSED: "persona_villain",
    CELL_FT_POS: "persona_villain",
    CELL_EXT_WILDCHAT: "wildchat_prefix_real545",
    CELL_EXT_ICL: "icl_prefix_marker",
    CELL_EXT_BARE: "bare_default",
}

# Contrastive negative panels (plan §4.2). Frozen-mix panel order matches
# issue_1112.mixes.MARKER_NEGATIVES; cell 7 substitutes french_person for
# qwen_default (the bare source context renders BYTE-IDENTICAL to qwen_default
# — keeping it would create near-contradictory rows, the #527/#538 class).
FROZEN_NEGATIVES = ("medical_doctor", "police_officer", "qwen_default", "comedian")
BARE_CELL_NEGATIVES = ("medical_doctor", "police_officer", "french_person", "comedian")
CELL_NEGATIVES = {
    CELL_LORA_CON: FROZEN_NEGATIVES,
    CELL_FT_CON_REUSED: FROZEN_NEGATIVES,
    CELL_LORA_POS: (),  # positives-only (exemption (a): negatives ARE the variable)
    CELL_FT_POS: (),
    CELL_EXT_WILDCHAT: FROZEN_NEGATIVES,
    CELL_EXT_ICL: FROZEN_NEGATIVES,
    CELL_EXT_BARE: BARE_CELL_NEGATIVES,
}
HELD_OUT_TRIO = ("chef", "hero", "philosopher")  # untrained leakage comparators
POS_EX = 200
NEG_EX_PER_PERSONA = 200

WANDB_PROJECT = SLUG

# The committed questions-only ICL spec (answers are pod-generated greedy base
# R + marker at p1_mixes — deterministic; the completed 2-example bank is
# persisted + uploaded, and icl_prefix_context() reads the RUN copy).
ICL_QUESTIONS_SPEC = "icl_examples_marker.questions.json"


def cell_run_name(cell: str) -> str:
    """WandB run name per trained cell (one run per cell, plan §10)."""
    return f"issue1333_{cell}_seed{SEED}"


def save_steps_for(cell: str) -> int:
    """Checkpoint cadence per LoRA cell — 5 for mk2_lora_pos (plan §4.1), else 10."""
    return LORA_SAVE_STEPS_POSONLY if cell == CELL_LORA_POS else LORA_SAVE_STEPS


# ── Training config builders (plan §10 LoRA/FT recipe rows) ──────────────────


def marker_lora_config(cell: str, *, seed: int = SEED, tokenizer=None, out_root=None):
    """MARKER_OVERRIDES verbatim EXCEPT the plan-§10 ladder deviations.

    Deviations from the shipped recipe (each grounded in plan §11):
    max_steps=400 (ceiling), save_steps=10 (5 for mk2_lora_pos),
    save_total_limit=None (the #641 pruning trap — MARKER_OVERRIDES ships 4,
    which would delete ladder rungs mid-train), warmup_steps=5 (the m1
    warmup-swamp fix), in-loop band repurposed to the [25, 30] safety ceiling
    (selection is the OFF-LINE ladder, §4.3). Trajectory JSON is written per
    probe when ``out_root`` is given (smoke-verifiable telemetry, plan §4.3).
    """
    from explore_persona_space.artifacts.recipe import build_train_config, recipe_for

    if cell not in NEW_LORA_CELLS:
        raise ValueError(f"{cell!r} is not a new LoRA cell: {NEW_LORA_CELLS}")
    spec = recipe_for("marker", arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "marker_band_low_nats": SAFETY_BAND[0],
            "marker_band_high_nats": SAFETY_BAND[1],
        },
    )
    train_cfg = build_train_config(
        spec, run_name=cell_run_name(cell), seed=seed, tokenizer=tokenizer
    )
    replace_kwargs: dict = {
        "max_steps": LORA_MAX_STEPS,
        "save_steps": save_steps_for(cell),
        "save_total_limit": None,
        "warmup_steps": LORA_WARMUP_STEPS,
    }
    if out_root is not None:
        replace_kwargs["marker_band_trajectory_path"] = str(
            Path(out_root) / cell / "band_trajectory.json"
        )
    return dataclasses.replace(train_cfg, **replace_kwargs)


def marker_ft_cmd(
    *,
    mix_path: str | Path,
    out_dir: str | Path,
    num_processes: int,
    seed: int = SEED,
    grid: tuple[int, ...] = FT_GRID,
    max_steps: int = FT_MAX_STEPS,
    trainer: str = "scripts/issue1112_train_marker_fullft.py",
    accel_config: str = "configs/accelerate/zero3_4gpu_accum16.yaml",
) -> list[str]:
    """The mk4 full-FT launch — the reused arm's trainer + recipe VERBATIM
    (lr 5e-6 linear warmup 0.03 eff-64 max_len 1024 live inside the trainer),
    grid {1..6} at --max-steps 6 = the reused arm's realized linear-decay
    horizon (schedule parity; mix is the only FT-side variable)."""
    if max_steps != max(grid):
        raise ValueError(f"max_steps {max_steps} != max(grid) {max(grid)} — schedule parity")
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        accel_config,
        "--num_processes",
        str(num_processes),
        trainer,
        "--train-jsonl",
        str(mix_path),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in grid),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--run-name",
        cell_run_name(CELL_FT_POS),
    ]


# ── Mix derivations (plan §4.2) ───────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """JSONL rows via text-mode file iteration (never splitlines — gotchas.md)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_mix(rows: list[dict], out_path: Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return sha256_file(out_path)


def _completion_text(row: dict) -> str:
    comp = row["completion"]
    assert isinstance(comp, list) and len(comp) == 1, comp
    return comp[0]["content"]


def _row_is_positive(row: dict) -> bool:
    """A frozen-mix row is a positive iff its completion ends with the marker."""
    return _completion_text(row).endswith(MARKER_TEXT)


def partition_frozen_mix(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(positives, negatives) partition of the frozen 1000-row marker mix.

    Row roles are unambiguous (plan §4.2): positives are exactly the rows whose
    completion ends with the marker token. Asserts the partition is exactly
    200/800; negatives carry NO marker anywhere (contamination fail-loud).
    """
    pos = [r for r in rows if _row_is_positive(r)]
    neg = [r for r in rows if not _row_is_positive(r)]
    if len(pos) != POS_EX or len(neg) != len(FROZEN_NEGATIVES) * NEG_EX_PER_PERSONA:
        raise ValueError(f"frozen-mix partition {len(pos)}/{len(neg)} != 200/800")
    for i, r in enumerate(neg):
        if MARKER_TEXT in _completion_text(r):
            raise ValueError(f"negative row {i} carries the marker — contamination")
    return pos, neg


def assert_positive_tails_encode_marker(tokenizer, positives: list[dict]) -> None:
    """Plan §4.2: encode() of each positive's completion tail contains id 83399."""
    for i, r in enumerate(positives):
        tail = _completion_text(r)[-16:]
        ids = tokenizer.encode(tail, add_special_tokens=False)
        if MARKER_TOKEN_ID not in ids:
            raise ValueError(f"positive row {i}: tail ids {ids} lack marker id {MARKER_TOKEN_ID}")


def derive_posonly_mix(frozen_path: Path, out_path: Path, *, tokenizer=None) -> dict:
    """Keep only the 200 villain positive rows (cells mk2/mk4; plan §4.2)."""
    rows = _read_jsonl(Path(frozen_path))
    if len(rows) != 1000:
        raise ValueError(f"frozen mix has {len(rows)} rows, want 1000")
    pos, _neg = partition_frozen_mix(rows)
    if tokenizer is not None:
        assert_positive_tails_encode_marker(tokenizer, pos)
    sha = write_mix(pos, Path(out_path))
    manifest = {
        "derived_from": FROZEN_MIX_PATH,
        "n_rows": len(pos),
        "sha256": sha,
        "seed": SEED,
        "rule": "rows whose completion ends with the marker token (exactly 200)",
    }
    Path(out_path).with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def negatives_by_persona(neg_rows: list[dict], persona_bank: dict[str, str]) -> dict[str, list]:
    """Partition frozen negative rows by their system prompt == persona bank entry.

    Fail-loud on any row whose system prompt matches no frozen-panel persona,
    and on a per-persona count != 200.
    """
    by_system = {persona_bank[p]: p for p in FROZEN_NEGATIVES}
    out: dict[str, list[dict]] = {p: [] for p in FROZEN_NEGATIVES}
    for i, r in enumerate(neg_rows):
        msgs = r["prompt"]
        if not msgs or msgs[0].get("role") != "system":
            raise ValueError(f"negative row {i} has no system prompt")
        persona = by_system.get(msgs[0]["content"])
        if persona is None:
            raise ValueError(f"negative row {i}: system prompt matches no frozen-panel persona")
        out[persona].append(r)
    for p, rows in out.items():
        if len(rows) != NEG_EX_PER_PERSONA:
            raise ValueError(f"frozen negatives for {p}: {len(rows)} != {NEG_EX_PER_PERSONA}")
    return out


def sample_questions(questions: list[str], n: int, rng: random.Random) -> list[str]:
    """#508 ``_sample`` verbatim: n draws over the 10-question bank with reshuffles."""
    if n <= len(questions):
        return rng.sample(questions, n)
    out: list[str] = []
    while len(out) < n:
        perm = list(questions)
        rng.shuffle(perm)
        out.extend(perm)
    return out[:n]


def make_row(prompt_msgs: list[dict], assistant: str) -> dict:
    return {
        "prompt": [dict(m) for m in prompt_msgs],
        "completion": [{"role": "assistant", "content": assistant}],
    }


def build_extension_mix(
    cell: str,
    *,
    source_msgs_for_q,  # callable(q) -> list[dict] prompt messages under the source context
    greedy_r_for_q,  # callable(q) -> str greedy base R under the source context
    train_questions: list[str],
    frozen_negatives: dict[str, list[dict]],
    french_r_for_q=None,  # callable(q) -> str greedy base R under french_person (cell 7 only)
    french_system: str | None = None,
    out_path: Path,
    seed: int = SEED,
) -> dict:
    """One extension-cell mix (plan §4.2): 200 NEW positives under the source
    context (same 10 train questions x 20 repeats, greedy base R + marker in
    the frozen slot shape) + 800 negatives (cells 5/6: the frozen 800 verbatim;
    cell 7: 600 frozen + 200 fresh french_person rows, replacing qwen_default's
    sampling stream ``Random(seed + 1000 + index(qwen_default))``)."""
    if cell not in (CELL_EXT_WILDCHAT, CELL_EXT_ICL, CELL_EXT_BARE):
        raise ValueError(f"{cell!r} is not an extension cell")
    if len(train_questions) != 10:
        raise ValueError(f"want the 10 train questions, got {len(train_questions)}")
    rows: list[dict] = []
    pos_rng = random.Random(seed)
    for q in sample_questions(train_questions, POS_EX, pos_rng):
        r_text = greedy_r_for_q(q)
        if MARKER_TEXT in r_text:
            raise ValueError(f"positive R for {q!r} already carries the marker")
        rows.append(make_row(source_msgs_for_q(q), f"{r_text}{MARKER_SEP}{MARKER_TEXT}"))
    n_positive = len(rows)

    panel = CELL_NEGATIVES[cell]
    for j_idx, neg in enumerate(FROZEN_NEGATIVES):
        if neg == "qwen_default" and cell == CELL_EXT_BARE:
            # Cell-7 substitution (plan §4.2 fix path (a)): french_person takes
            # qwen_default's slot AND its sampling stream index.
            if french_r_for_q is None or french_system is None:
                raise ValueError("cell 7 needs french_r_for_q + french_system")
            neg_rng = random.Random(seed + 1000 + j_idx)
            for q in sample_questions(train_questions, NEG_EX_PER_PERSONA, neg_rng):
                r_text = french_r_for_q(q)
                if MARKER_TEXT in r_text:
                    raise ValueError(f"french_person R for {q!r} carries the marker")
                rows.append(
                    make_row(
                        [
                            {"role": "system", "content": french_system},
                            {"role": "user", "content": q},
                        ],
                        r_text,
                    )
                )
            continue
        rows.extend(frozen_negatives[neg])
    n_negative = len(rows) - n_positive
    if (n_positive, n_negative) != (POS_EX, len(FROZEN_NEGATIVES) * NEG_EX_PER_PERSONA):
        raise ValueError(f"extension mix {cell}: {n_positive}/{n_negative} != 200/800")

    random.Random(seed).shuffle(rows)
    sha = write_mix(rows, Path(out_path))
    manifest = {
        "cell": cell,
        "source_context": CELL_SOURCE_CONTEXT[cell],
        "negatives": list(panel),
        "n_total": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "marker_text": MARKER_TEXT,
        "seed": seed,
        "sha256": sha,
        "r_provenance": "fresh greedy base R (max_new_tokens 1024) for positives"
        + (
            " + french_person negatives" if cell == CELL_EXT_BARE else "; frozen negatives verbatim"
        ),
    }
    Path(out_path).with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


# ── Rendered-token disjointness (plan §4.2, consistency BLOCK fix) ────────────

_CTX_PROBE_QUESTION = "__EPM_CTX_PROBE__"


def rendered_ids(tokenizer, msgs_for_q, question: str) -> tuple[int, ...]:
    """Token ids of one context rendered through the TRAINING-side template
    path (``apply_chat_template`` with generation prompt — the trainer's exact
    prompt render)."""
    return tuple(
        tokenizer.apply_chat_template(
            msgs_for_q(question), tokenize=True, add_generation_prompt=True
        )
    )


def assert_rendered_disjoint(
    tokenizer,
    *,
    source_id: str,
    source_msgs_for_q,
    panel: dict[str, object],  # {persona_or_ctx_id: msgs_for_q callable}
    questions: list[str],
) -> None:
    """RENDERED-TOKEN disjointness invariant (plan §4.2).

    For one cell: render the realized source context and every negative-panel
    context through the trainer's exact ``apply_chat_template`` output ids and
    assert pairwise NON-identity of (a) the rendered context id sequences
    (probe question) and (b) the (context + question) id sequences over the
    train questions. Content-fingerprint comparison of context DEFINITIONS is
    exactly what this replaces — it passes while the auto-inserted-default-
    system-prompt collision (bare == qwen_default rendered) is live.

    Raises ValueError naming the colliding pair.
    """
    renders: dict[str, object] = {source_id: source_msgs_for_q, **panel}
    ids = list(renders)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate context ids in disjointness check: {ids}")
    for probe_q in [_CTX_PROBE_QUESTION, *questions]:
        seen: dict[tuple[int, ...], str] = {}
        for ctx_id, msgs_for_q in renders.items():
            seq = rendered_ids(tokenizer, msgs_for_q, probe_q)
            other = seen.get(seq)
            if other is not None:
                raise ValueError(
                    f"rendered-token collision: {ctx_id!r} == {other!r} at the exact "
                    f"apply_chat_template output ids (question={probe_q!r}) — "
                    "panel ∩ realized sources must be rendered-disjoint (plan §4.2)"
                )
            seen[seq] = ctx_id


# ── Off-line selection rule (plan §4.3) ───────────────────────────────────────


def select_rung(
    ladder: dict[int, dict],
    *,
    target: float = TARGET_DELTA_G,
    window: tuple[float, float] = ACCEPT_WINDOW,
) -> dict:
    """Registered selection: the rung minimizing |ΔG − target| subject to
    ΔG ∈ window AND source argmax-emission 0/20 AND every panel bystander below
    the argmax ceiling. Earliest such rung on ties; no in-window rung → the
    closest-approach rung, labeled (the §3 dose-reachability branch fires for
    its contrasts — a registered finding, never a crash).

    ``ladder`` maps step -> {"delta_logp_mean": float, "source_emission_rate":
    float, "bystander_saturated": bool}. ``source_emission_rate`` is REQUIRED —
    both ladder writers populate it, so a missing field is a writer bug and
    raises KeyError rather than silently reading as eligible (review r1 m8).
    Bystander saturation defaults False when the read carries no bystander
    pass (source-only ladders record it at selection-confirm time).
    """
    if not ladder:
        raise ValueError("empty ladder")

    def _dist(step: int) -> tuple[float, int]:
        return (abs(ladder[step]["delta_logp_mean"] - target), step)

    eligible = [
        s
        for s, rec in ladder.items()
        if window[0] <= rec["delta_logp_mean"] <= window[1]
        and rec["source_emission_rate"] == 0.0
        and not rec.get("bystander_saturated", False)
    ]
    if eligible:
        step = min(eligible, key=_dist)
        return {
            "step": step,
            "in_window": True,
            "fallback": None,
            "delta_logp_mean": ladder[step]["delta_logp_mean"],
            "target": target,
            "window": list(window),
        }
    step = min(ladder, key=_dist)
    return {
        "step": step,
        "in_window": False,
        "fallback": "closest_approach",
        "delta_logp_mean": ladder[step]["delta_logp_mean"],
        "target": target,
        "window": list(window),
    }


def dose_curve_rung_plan(
    ladder: dict[int, dict],
    candidate_steps: list[int] | tuple[int, ...] | set[int],
    *,
    window: tuple[float, float] = ACCEPT_WINDOW,
) -> list[dict]:
    """Bystander-read rung plan for the leakage-vs-install dose curves
    (plan §6 install-strength read (3); concern ladder-bystander-dose-curves).

    Full per-rung bystander reads are compute-infeasible (~100 rungs x ~12
    min), so the registered read set is: every CANDIDATE rung the selection
    loop read (including the selected / closest-approach rung) PLUS one
    sub-window flank (the read rung with ``delta_logp_mean`` closest BELOW
    the acceptance window) and one above-window flank (closest ABOVE), where
    they exist — so each cell's curve spans below/in/above the window.
    Returns ``[{step, role, delta_logp_mean}]`` sorted by step; the candidate
    role wins on overlap. Ties on distance break to the EARLIEST step.
    Raises KeyError on a candidate step absent from the ladder (writer bug,
    fail loud)."""
    lo, hi = window
    plan: dict[int, str] = {}
    below = [(s, r["delta_logp_mean"]) for s, r in ladder.items() if r["delta_logp_mean"] < lo]
    above = [(s, r["delta_logp_mean"]) for s, r in ladder.items() if r["delta_logp_mean"] > hi]
    if below:
        plan[max(below, key=lambda t: (t[1], -t[0]))[0]] = "sub_window"
    if above:
        plan[min(above, key=lambda t: (t[1], t[0]))[0]] = "above_window"
    for s in candidate_steps:
        step = int(s)
        if step not in ladder:
            raise KeyError(f"candidate step {step} not in ladder (writer bug)")
        plan[step] = "candidate"
    return [
        {"step": s, "role": role, "delta_logp_mean": float(ladder[s]["delta_logp_mean"])}
        for s, role in sorted(plan.items())
    ]


def coarse_read_steps(cell: str, rungs: list[int]) -> list[int]:
    """Coarse-pass read schedule (plan §4.3): every-20 steps, plus {5, 10, 15}
    for mk2_lora_pos; always includes the final rung. FT cells read the WHOLE
    grid — stride 20 over grid {1..6} would degenerate to {6} and leave rungs
    1-5 hostage to the refine conditional (review r1 m5); it is 6 reads either
    way."""
    if cell in NEW_FT_CELLS:
        return sorted(rungs)
    stride = LADDER_COARSE_STRIDE
    steps = sorted({s for s in rungs if s % stride == 0} | {max(rungs)})
    if cell == CELL_LORA_POS:
        steps = sorted(set(steps) | {s for s in LADDER_COARSE_EXTRA_POSONLY if s in rungs})
    return steps


def refine_read_steps(cell: str, rungs: list[int], ladder: dict[int, dict]) -> list[int]:
    """Fine-pass reads around the first window crossing: all unread rungs
    between the last read below the window and the first read at-or-above it."""
    read = sorted(ladder)
    crossing = next(
        (s for s in read if ladder[s]["delta_logp_mean"] >= ACCEPT_WINDOW[0]),
        None,
    )
    if crossing is None:
        return []
    below = [s for s in read if s < crossing]
    lo = below[-1] if below else 0
    return sorted(s for s in rungs if lo < s < crossing and s not in ladder)


__all__ = [name for name in dir() if not name.startswith("_")]
