"""Shared constants + pure logic for the #2378 `causal-patching-arms` follow-up.

Cross-framing context-vector (v_C) patching on Qwen3.6-27B: does replacing the
TARGET framing's context-end state with the SOURCE framing's carry the source
persona/behavior into the target framing's GENERATED answer?

Method inheritance (scope marker `epm:followup-scope v1`, label
`causal-patching-arms`):

- #2094 null-separated patching recipe — context-end REPLACE patches
  (``experiments.issue2094.hooks.PositionEditHook``), the
  fraction-of-full-context-swap DV (``experiments.issue2094.fmetrics.f_act``:
  floor = unpatched draws under the TARGET context, ceiling = draws under the
  SOURCE context = the full context swap), norm-matched wrong-donor null
  (``bank.norm_match`` — the #2094 replace-rung realization: a REAL state from
  the wrong pair, norm-matched to the recipient), pair-clustered bootstrap
  B=10,000 with 2.5/97.5 percentile CIs
  (``issue2094_analysis.bootstrap_family_means_batched``), and an independent
  temperature-1.0 K=5 confirmation pass on the screened-best cells
  (#2094 stage-2 convention; LABELED post-selection, never an unbiased
  estimate).
- #2333 prefill-vs-patch opening-token control — instead of patching v_C, the
  TARGET prompt is continued from the SOURCE context's first
  ``PREFILL_K`` greedy answer TOKENS (token-id concatenation, never
  re-tokenized text).
- This task's conventions (issue2378_common / _gen / _capture): framing
  renders, kept/mined row sources, caps + stop conventions, bf16-uint16 npz
  encoding, StageLedger resume, HF prefixes.

Layer convention (recipe statement, required by the round brief): #2094 swept
patch layers as a grid axis and pinned no single layer, so the PATCH layer
here is THIS task's L* = 51 (``issue2378_dispatch.resolve_lstar`` — the layer
where the correlational v_C -> v_A maps live; testing that vector's causal
status is the round's Goal). The F_act READ layer ports #2094's read-layer
rule (plan §4.4 there: PRIMARY read at the deepest banked layer, DOWNSTREAM of
every edit — 26/28): primary read = round(64 * 26/28) = 59; v_A at L*=51 is
kept as a SECONDARY read (same-layer as the correlational maps; partially
blind to the direct activation route of a single-layer L51 edit — the token
route still registers).

MODEL CAVEAT (carried, per the scope marker): causality is tested on
Qwen3.6-27B while the correlational headline (#1345/#2054) and the patching
baseline (#2094) are Qwen2.5-7B; this round adds NO 2.5-7B arm (budget) — the
cross-model caveat rides every output.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2378_common as cm  # noqa: E402

FOLLOWUP_LABEL = "causal-patching-arms"

# ── design constants ────────────────────────────────────────────────────────
N_Q_PER_CHAR = 12  # sampled kept storyq questions per character (5 chars -> 60)
FRAMINGS: tuple[str, ...] = ("story", "chat", "plain")
ANCHOR_DRAWS = 10  # #2094 anchors: K=10 unpatched temp-1.0 draws per context
CONFIRM_DRAWS = 5  # #2094 stage-2: temp 1.0, K=5, mean-aggregated
CONFIRM_MAX_FAMILIES = 4  # tightened from #2094's STAGE2_MAX_CELLS=6 (budget)
CONFIRM_MAX_PAIRS = 30  # per-family pair cap at confirm (seeded subsample)
PREFILL_K = 8  # #2333 donor schemes: greedy 8-token openings
OPENER_TOKENS = 8  # greedy source-opening length harvested for the prefill arm
MIN_PAIRS = 3  # #2094 select_best_cells min_pairs floor (inherited)
BOOTSTRAP_B = 10_000  # #2094 BOOTSTRAP_B (inherited)
PATCH_BOOTSTRAP_SEED = 23781  # fresh seed, #2094 BOOTSTRAP_SEED convention
COHERENCE_THRESHOLD = 60.0  # #2094 COHERENCE_THRESHOLD (inherited F_beh gate)
# #2094 plan §4.4 read-layer rule ported by depth: primary read downstream of
# every edit, at the #2094 depth fraction 26/28 of this model's 64 layers.
PRIMARY_READ_FRAC_NUM, PRIMARY_READ_FRAC_DEN = 26, 28

# Patch arms (the per-arm smoke-resolution registry; every cell carries one).
ARMS: tuple[str, ...] = ("steered", "within", "null", "prefill")
# Layer variants: "lstar" = single-layer replace at L*; "all" = the #2333
# ce_control convention (all-layer replace at the context-end position = the
# full residual swap of that position). within/prefill run at lstar only.
VARIANTS: tuple[str, ...] = ("lstar", "all")
ARM_VARIANTS: dict[str, tuple[str, ...]] = {
    "steered": ("lstar", "all"),
    "null": ("lstar", "all"),
    "within": ("lstar",),
    "prefill": ("none",),
}

# Pair types (the two brief arms): (a) assistant <-> story-character;
# (b) chat-template <-> plain_text. Directions are src->tgt over each pair.
PAIR_TYPES: tuple[str, ...] = ("chat~story", "chat~plain")

HF_STAGE_PREFIX = f"{cm.HF_PREFIX}/raw_completions/causal_patching"
HF_TENSOR_PREFIX = f"{cm.HF_PREFIX}/analysis_tensors/causal_patching"
LEDGER_SUBDIR = "causal-patching-arms"  # eval_results/issue_2378/<subdir>/

# Generation caps by TARGET framing (parent caps kept — recipe fidelity).
FRAMING_MAX_TOKENS = {
    "story": cm.SEGB_MAX_TOKENS,
    "chat": cm.CHAT_MAX_TOKENS,
    "plain": cm.PLAIN_MAX_TOKENS,
}

# Registered textual stops by TARGET framing (parent stop conventions: a story
# reply ends at its closing quote — gen._mine_closing_quote's char set — and a
# plain reply at the next user turn; "\n\nUser:" contains "\nUser:", so the
# single shorter stop covers both cm.PLAIN_STOP members). Chat stops on EOS.
FRAMING_STOP_STRINGS: dict[str, tuple[str, ...]] = {
    "story": ('"', "”"),
    "chat": (),
    "plain": ("\nUser:",),
}


def stop_strings_for(framing: str) -> list[str] | None:
    """The generate-time stop set for one target framing (None = EOS only)."""
    stops = FRAMING_STOP_STRINGS[framing]
    return list(stops) if stops else None


def hit_stop(framing: str, text: str) -> bool:
    """Whether a generated text reached its framing's registered textual stop
    (cap-hit telemetry: capped = not hit_eos and not hit_stop)."""
    return any(s in text for s in FRAMING_STOP_STRINGS[framing])


def primary_read_layer(n_layers: int) -> int:
    """Depth-matched #2094 primary F_act read layer (26/28 of the stack)."""
    return round(n_layers * PRIMARY_READ_FRAC_NUM / PRIMARY_READ_FRAC_DEN)


def read_layers(lstar: int, n_layers: int) -> tuple[int, ...]:
    """(secondary=L*, primary=downstream) v_A read layers, sorted ascending."""
    prim = primary_read_layer(n_layers)
    assert prim > lstar, (
        f"primary read layer {prim} not downstream of the patch layer {lstar} — "
        "the #2094 read-layer rule requires the read strictly below-to-output of every edit"
    )
    return tuple(sorted({lstar, prim}))


def ctx_id(framing: str, qid: str) -> str:
    assert framing in FRAMINGS, framing
    return f"{framing}:{qid}"


def pair_contexts(pair_type: str, qid: str) -> tuple[str, str]:
    """The (chat_ctx, other_ctx) ids of one question's pair."""
    other = "story" if pair_type == "chat~story" else "plain"
    return ctx_id("chat", qid), ctx_id(other, qid)


def cell_id(arm: str, variant: str, src: str, tgt: str, qid: str) -> str:
    return f"{arm}|{variant}|{src}->{tgt}|{qid}"


def family_key(pair_type: str, char: str, direction: str, variant: str, arm: str) -> str:
    """Family = aggregation unit for the screen/bootstrap (pairs = questions).

    ``char`` is the story character for chat~story families (one family per
    character — the persona grain), the literal ``"-"`` for chat~plain.
    """
    return "|".join([pair_type, char, direction, variant, arm])


def enumerate_cells(qids_by_char: dict[str, list[str]]) -> list[dict]:
    """The full patched/prefill cell grid (deterministic order).

    Returns one dict per cell: arm, variant, src/tgt framing, qid, char,
    pair_type, direction, cell id, family key. Directions are ``a2b``
    (chat -> other) and ``b2a`` (other -> chat) over each pair type.
    """
    cells: list[dict] = []
    for char in sorted(qids_by_char):
        for qid in qids_by_char[char]:
            for pair_type in PAIR_TYPES:
                chat_c, other_c = pair_contexts(pair_type, qid)
                fam_char = char if pair_type == "chat~story" else "-"
                for direction, (src, tgt) in (
                    ("a2b", (chat_c, other_c)),
                    ("b2a", (other_c, chat_c)),
                ):
                    for arm in ARMS:
                        for variant in ARM_VARIANTS[arm]:
                            cells.append(
                                {
                                    "cell_id": cell_id(arm, variant, src, tgt, qid),
                                    "arm": arm,
                                    "variant": variant,
                                    "src": src,
                                    "tgt": tgt,
                                    "qid": qid,
                                    "char": char,
                                    "pair_type": pair_type,
                                    "direction": direction,
                                    "family": family_key(
                                        pair_type, fam_char, direction, variant, arm
                                    ),
                                }
                            )
    return cells


def derangement(qids: list[str], seed_parts: tuple) -> dict[str, str]:
    """Seeded fixed-point-free donor map over one (framing, char) qid group.

    Seeded shuffle + cyclic shift: donor(shuffled[i]) = shuffled[i+1 mod n] —
    a derangement by construction for n >= 2 (fails loud below).
    """
    import random

    assert len(qids) >= 2, f"derangement needs >= 2 qids, got {qids}"
    order = sorted(qids)
    rng = random.Random(cm.derived_seed("patch-derangement", *seed_parts))
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


# ── screen (pre-registered gates, #2094-inherited) ──────────────────────────
#
# Eligibility: per family, >= MIN_PAIRS non-degenerate paired (steered, null)
# F_act values (FActResult.degenerate rows excluded — drop, never coerce).
# Statistic: per-question paired difference d_q = F_act(steered_q) -
# F_act(null_q); pair-clustered bootstrap (B=10,000, seed 23781, the #2094
# batched index-GEMM implementation) over questions; 95% CI = 2.5/97.5
# percentiles of the resampled means. PASS iff the CI excludes 0.
# Confirmation set: among PASS steered families, top CONFIRM_MAX_FAMILIES by
# mean paired difference — re-measured independently at temperature 1.0,
# K=CONFIRM_DRAWS (labeled post-selection).


def screen_families(
    diffs_by_family: dict[str, dict[str, float]],
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = PATCH_BOOTSTRAP_SEED,
    min_pairs: int = MIN_PAIRS,
) -> dict:
    """Bootstrap screen over per-family paired steered-null F_act differences.

    ``diffs_by_family[family][qid] = d_q`` (NaN rows must already be dropped).
    Returns {family: {n_pairs, mean_diff, ci_lo, ci_hi, screen_pass}} plus the
    ordered confirm selection.
    """
    import numpy as np

    import issue2094_analysis as A  # bootstrap_family_means_batched (reused verbatim)

    fams = sorted(f for f, d in diffs_by_family.items() if len(d) >= min_pairs)
    skipped = sorted(set(diffs_by_family) - set(fams))
    report: dict = {
        "screen_rule": (
            "paired per-question F_act(steered)-F_act(null) differences; pair-clustered "
            f"bootstrap B={n_boot} seed={seed} (issue2094_analysis."
            "bootstrap_family_means_batched), run PER identical-qid-set family GROUP so "
            "every family resamples exactly its OWN fixed pair set (a union-with-NaN "
            "matrix over disjoint qid sets gives each family a random effective n per "
            "replicate); group g uses seed+g over sorted group keys. PASS iff the "
            f"2.5/97.5 percentile CI excludes 0; min_pairs={min_pairs} "
            "(#2094 select_best_cells floor)"
        ),
        "families": {},
        "skipped_below_min_pairs": skipped,
    }
    if not fams:
        report["confirm_families"] = []
        return report
    # Group families by their EXACT qid set: within a group the values matrix
    # is dense (no NaN), so each bootstrap replicate resamples exactly
    # n_pairs pairs — fixed effective n per family (r17 codex
    # patch-bootstrap-variable-effective-n). Families sharing qids stay
    # pair-clustered together (same draw indices within the group).
    groups: dict[tuple[str, ...], list[str]] = {}
    for fam in fams:
        groups.setdefault(tuple(sorted(diffs_by_family[fam])), []).append(fam)
    for gi, qid_key in enumerate(sorted(groups)):
        gfams = sorted(groups[qid_key])
        values = np.array(
            [[diffs_by_family[fam][q] for fam in gfams] for q in qid_key], dtype=np.float64
        )
        assert not np.isnan(values).any(), (qid_key, gfams)  # dense by construction
        boots = A.bootstrap_family_means_batched(values, n_boot, seed + gi)
        for j, fam in enumerate(gfams):
            col = boots[:, j]
            col = col[~np.isnan(col)]
            d = diffs_by_family[fam]
            lo, hi = float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))
            report["families"][fam] = {
                "n_pairs": len(d),
                "mean_diff": float(np.mean(list(d.values()))),
                "ci_lo": lo,
                "ci_hi": hi,
                "screen_pass": bool(lo > 0.0 or hi < 0.0),
            }
    passing = [
        (fam, rec["mean_diff"])
        for fam, rec in report["families"].items()
        if rec["screen_pass"] and fam.endswith("|steered")
    ]
    passing.sort(key=lambda t: -t[1])
    report["confirm_families"] = [f for f, _ in passing[:CONFIRM_MAX_FAMILIES]]
    return report


# ── judge instrument (VM-side; dual-rubric per #2094 delta_contrast) ────────
#
# NEW instrument (stated as such — this task's CONGRUENCE rubric is scene-tied
# and cannot score chat-target answers): context-blind reply-only scoring, one
# integer 0-100 per rubric, drop-never-coerce parse handling reused from
# issue2378_judge. Judge = cm.JUDGE_MODEL (claude-sonnet-4-5-20250929).

PATCH_JUDGE_SYSTEM = (
    "You are a careful literary annotator scoring machine-generated text for a research "
    'dataset. Respond with ONLY a JSON object of the form {"reasoning": "<2-4 sentences>", '
    '"score": <integer 0-100>}. If you cannot score the item, respond '
    '{"reasoning": "<why>", "score": "REFUSAL"}. No markdown fences.'
)

PERSONA_RUBRIC = (
    "Below is a REPLY produced by a language model. Score 0-100 how strongly the reply reads "
    "as spoken IN CHARACTER by {name}, {description}.\n\n"
    "Judge the voice, stance, and manner of the reply itself — not its topic. Anchors: 0 = "
    "nothing of {name}'s voice or manner; 50 = some traits present but mixed or inconsistent; "
    "100 = unmistakably {name} throughout.\n\n"
    "REPLY:\n{reply}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)

ASSISTANT_RUBRIC = (
    "Below is a REPLY produced by a language model. Score 0-100 how strongly the reply reads "
    "as a DEFAULT AI ASSISTANT answering a user's question: helpful, neutral-toned, "
    "direct-to-the-user prose (not in-scene fiction, not a named character's voice).\n\n"
    "Anchors: 0 = clearly not an assistant answer (in-scene fiction, roleplay, or a distinct "
    "character voice); 50 = partially assistant-like; 100 = a plain assistant answer "
    "throughout.\n\n"
    "REPLY:\n{reply}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)

COHERENCE_RUBRIC = (
    "Below is a REPLY produced by a language model. Score 0-100 how COHERENT it is as text: "
    "grammatical, internally consistent, on some legible topic — regardless of style, "
    "persona, or whether it answers any particular question.\n\n"
    "Anchors: 0 = word salad or heavy repetition/degeneration; 50 = partially coherent with "
    "clear breakdowns; 100 = fully coherent prose.\n\n"
    "REPLY:\n{reply}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)
