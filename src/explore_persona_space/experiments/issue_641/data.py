"""Issue #641 — EM training-mix construction + the contrastive-negative panel.

Wraps the vendored #537 ``build_em`` contrastive builder (bad-medical under the
source context + good-medical, same questions, under the negative panel) and
adds the two plan-mandated #641 changes:

1. **§4.7 + §0 Divergence #2 — `default` as the 5th contrastive negative.** The
   #537 ``NEGATIVE_CIDS`` panel is ``{neg_sp_police, neg_sp_ph4,
   neg_reph_curious, neg_wc_short}`` and omits the bare default assistant.
   `.claude/rules/contrastive-negatives.md` mandates the default assistant (the
   single highest-value negative — leakage-to-default is the safety target), so
   #641 ADDS the registry's ``default`` (F6, no system prompt) context as a 5th
   negative, keeping the ~1:1 positives:total-negatives ratio (the 3000
   negatives split ~600 per context across 5).

2. **§4.5 + §4.7 — disjointness invariant by RESOLVED-PROMPT HASH + CANONICAL
   PERSONA ID, not raw cid.** A source and a contrastive negative can carry
   distinct cids yet resolve to the SAME ``PERSONAS[...]`` system prompt (the
   Arm-B widened-pool ``police_officer`` <-> ``neg_sp_police`` collision case). A
   raw-cid intersection misses it. The HARD asserts compare the resolved
   system-prompt sha256 AND the canonical persona id of the realized source
   panel vs the realized negative panel — against the ACTUAL builder inputs,
   not the plan prose.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from explore_persona_space.experiments.i537_contexts import (
    NEGATIVE_CIDS,
    Ctx,
)
from explore_persona_space.personas import PERSONAS

logger = logging.getLogger("issue641.data")

# Pinned HF revision for the reused #376 EM corpus (plan §10/§13.3). Vendored
# here (NOT imported from scripts/) so build_em_mix has no `scripts.` package
# dependency — running `scripts/issue641_dose_curves.py` puts scripts/ (not the
# repo root) on sys.path[0], so `from scripts.i537_build_training_data import …`
# raises ModuleNotFoundError. The loader is a small published-corpus reader.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
EM_CORPUS_REV = "113af608e4aaea5dbdd1b355a9ad434434569f30"

# ── Plan §4.7 — the #641 negative panel = #537 panel + the rule-mandated
#    bare-default assistant (contrastive-negatives.md). ────────────────────────
DEFAULT_NEGATIVE_CID = "default"
NEGATIVE_PANEL_CIDS: tuple[str, ...] = (*NEGATIVE_CIDS, DEFAULT_NEGATIVE_CID)

# ── Plan §4.3 / §5 — Arm A source contexts (from the #537 EM install diagonal).
ARM_A_RESISTANT_CIDS: tuple[str, ...] = ("icl_k2", "wc_short_advice", "sp_doctor")
ARM_A_NONRESISTANT_CIDS: tuple[str, ...] = ("reph_imp", "sp_ph1", "wc_short_code")
ARM_A_SOURCE_CIDS: tuple[str, ...] = (*ARM_A_RESISTANT_CIDS, *ARM_A_NONRESISTANT_CIDS)

# ── Plan §4.5 / §5 — Arm B identity-conflict source (teacher) + the matched
#    neutral candidate pool. The matched neutral is selected post-P0; the
#    teacher's registry cid is ``sp_teacher_ho`` (resolves to
#    PERSONAS["kindergarten_teacher"]). ─────────────────────────────────────────
ARM_B_TEACHER_CID = "sp_teacher_ho"  # PERSONAS["kindergarten_teacher"]
# Narrow candidate pool (PERSONAS keys, all neutral-to-harmful-advice, none
# caregiving). The widened pool (§4.5) is "any non-caregiving persona in
# PERSONAS whose resolved-prompt hash + canonical id are absent from the
# negative panel".
ARM_B_NARROW_NEUTRAL_KEYS: tuple[str, ...] = (
    "data_scientist",
    "comedian",
    "marine_biologist",
    "local_historian",
)
# Caregiving identities EXCLUDED from any widened-pool neutral candidate (§4.5:
# the neutral must NOT itself carry a caregiver-of-others identity conflict, or
# the matched control stops controlling for identity conflict).
CAREGIVING_PERSONA_KEYS: frozenset[str] = frozenset(
    {"kindergarten_teacher", "medical_doctor", "nurse", "therapist", "social_worker"}
)


# ── Persona resolution (the disjointness primitives, §4.5/§4.7) ───────────────


def resolve_prompt(ctx: Ctx) -> str:
    """The resolved system-prompt string for a context, or "" when it has none.

    F1 (persona) and F5-with-system contexts carry a system prompt; F2/F3/F4/F6
    (wildchat / ICL / rephrase / default) have no system prompt and cannot
    collide with a persona negative on the system-prompt axis — they resolve to
    "" (the empty prompt), which is shared only among themselves and never
    equals a non-empty persona prompt.
    """
    if ctx.family == "F1":
        sp = ctx.payload.get("system_prompt")
        if sp:
            return sp
        # Unresolved sampled persona (sp_ph*/neg_sp_ph4 before load_registry
        # supplies the sampled JSON). Return a deterministic per-cid sentinel so
        # the disjointness assert is TOTAL (it can run on the static registry in
        # the structural smoke) without crashing — a sentinel never equals a
        # resolved persona prompt, so it cannot mask a real collision. In
        # production load_registry() resolves these before any source/negative
        # is asserted, so the real prompt is used.
        return f"\x00UNRESOLVED_SAMPLED_PERSONA::{ctx.cid}"
    if ctx.family == "F5":
        return ctx.payload.get("system_prompt") or ""
    if ctx.family == "F7F8":
        # behavior-instruction contexts (not used as #641 sources/negatives, but
        # resolve them faithfully for a total assert).
        from explore_persona_space.experiments.i537_contexts import F8_STRINGS

        return F8_STRINGS[ctx.payload["behavior"]]
    return ""


def prompt_hash(ctx: Ctx) -> str:
    """sha256 of the resolved system prompt (the binding disjointness key)."""
    return hashlib.sha256(resolve_prompt(ctx).encode("utf-8")).hexdigest()


# Inverse PERSONAS map: system-prompt string -> canonical persona key.
_PROMPT_TO_PERSONA_KEY: dict[str, str] = {v: k for k, v in PERSONAS.items()}


def canonical_id(ctx: Ctx) -> str | None:
    """The canonical PERSONAS key when the context resolves to a PERSONAS prompt.

    Returns ``None`` for non-persona contexts (no canonical persona id; they
    cannot collide on the persona-id axis). This is the second disjointness key
    alongside :func:`prompt_hash`: ``police_officer`` <-> ``neg_sp_police`` share
    BOTH a prompt hash AND a canonical id, so either key catches the collision.
    """
    sp = resolve_prompt(ctx)
    if not sp:
        return None
    return _PROMPT_TO_PERSONA_KEY.get(sp)


def assert_panels_disjoint(
    sources: list[Ctx],
    negatives: list[Ctx],
) -> None:
    """HARD §4.7 disjointness assert against the REALIZED panels (raises).

    Compares resolved-prompt sha256 AND canonical persona id (NOT raw cid) of
    the realized source panel vs the realized negative panel. The raw-cid
    disjointness is kept as a cheap additional check, but the resolved-prompt +
    canonical-id asserts are the binding ones (a source and a negative can carry
    distinct cids yet resolve to the SAME PERSONAS prompt — the widened-pool
    ``police_officer`` <-> ``neg_sp_police`` collision).
    """
    src_hashes = {prompt_hash(s) for s in sources}
    neg_hashes = {prompt_hash(n) for n in negatives}
    # Drop the empty-prompt collision class (non-persona contexts all hash the
    # empty string) — a wildchat source and a wildchat negative are legitimately
    # distinct cids and the empty prompt is not a persona collision. Only
    # NON-EMPTY (persona / format-system) prompts can collide meaningfully.
    empty_hash = hashlib.sha256(b"").hexdigest()
    src_hashes.discard(empty_hash)
    neg_hashes.discard(empty_hash)
    hash_overlap = src_hashes & neg_hashes
    assert not hash_overlap, (
        "§4.7 disjointness VIOLATION (resolved-prompt hash): a source and a "
        f"negative resolve to the SAME system prompt. Overlapping hashes: "
        f"{sorted(hash_overlap)}. Sources: {[s.cid for s in sources]}; "
        f"negatives: {[n.cid for n in negatives]}."
    )

    src_ids = {cid for cid in (canonical_id(s) for s in sources) if cid is not None}
    neg_ids = {cid for cid in (canonical_id(n) for n in negatives) if cid is not None}
    id_overlap = src_ids & neg_ids
    assert not id_overlap, (
        "§4.7 disjointness VIOLATION (canonical persona id): a source and a "
        f"negative share a PERSONAS key. Overlapping ids: {sorted(id_overlap)}. "
        f"Sources: {[s.cid for s in sources]}; negatives: {[n.cid for n in negatives]}."
    )

    # Cheap additional raw-cid check (kept per §4.7; not the binding one).
    cid_overlap = {s.cid for s in sources} & {n.cid for n in negatives}
    assert not cid_overlap, f"raw-cid overlap between sources and negatives: {sorted(cid_overlap)}"


def negative_panel(registry: dict[str, Ctx]) -> list[Ctx]:
    """The realized #641 negative panel (#537 4-panel + the bare default)."""
    return [registry[c] for c in NEGATIVE_PANEL_CIDS]


def neg_prompt_hashes(registry: dict[str, Ctx]) -> set[str]:
    """``NEG_PROMPT_HASHES`` — resolved-prompt hashes of the realized panel (§4.5)."""
    return {prompt_hash(registry[c]) for c in NEGATIVE_PANEL_CIDS}


def neg_persona_ids(registry: dict[str, Ctx]) -> set[str]:
    """``NEG_PERSONA_IDS`` — canonical persona ids of the realized panel (§4.5)."""
    return {cid for cid in (canonical_id(registry[c]) for c in NEGATIVE_PANEL_CIDS) if cid}


def widened_neutral_candidates(registry: dict[str, Ctx]) -> list[str]:
    """§4.5 widened candidate pool: non-caregiving PERSONAS keys whose resolved
    prompt hash  not-in  NEG_PROMPT_HASHES AND canonical id  not-in  NEG_PERSONA_IDS.

    This HARD-EXCLUDES the ``police_officer`` <-> ``neg_sp_police`` collision: the
    widened pool would otherwise contain ``police_officer``, and the panel's
    ``neg_sp_police`` resolves to ``PERSONAS["police_officer"]`` — same resolved
    prompt + canonical id, so the filter forbids it.

    Returns PERSONAS keys (the Arm-B matched-neutral pick wraps the chosen key
    into a fresh F1 source Ctx via :func:`neutral_source_ctx`).
    """
    nh = neg_prompt_hashes(registry)
    nids = neg_persona_ids(registry)
    out: list[str] = []
    for key, sp in PERSONAS.items():
        if key in CAREGIVING_PERSONA_KEYS:
            continue
        h = hashlib.sha256(sp.encode("utf-8")).hexdigest()
        if h in nh or key in nids:
            continue
        out.append(key)
    return out


def neutral_source_ctx(persona_key: str) -> Ctx:
    """Wrap a PERSONAS key into an F1 source Ctx for Arm B (matched neutral)."""
    assert persona_key in PERSONAS, persona_key
    return Ctx(
        cid=f"sp_{persona_key}",
        family="F1",
        role="train",
        name=f"{persona_key.replace('_', ' ').title()} persona (Arm-B matched neutral)",
        payload={"system_prompt": PERSONAS[persona_key]},
    )


# Map a PERSONAS key -> the registry cid the P0 base-propensity read uses for it.
# The narrow candidates have no registry cid (they aren't #537 contexts), so the
# selector measures them by their PERSONAS key directly when P0 covered them.
def select_matched_neutral(
    teacher_base_propensity: float,
    candidate_propensity: dict[str, float],
    registry: dict[str, Ctx],
    *,
    floor: float = 0.10,
) -> dict:
    """§4.5/§4.9 Arm-B matched-neutral selection (mechanical, post-P0).

    Picks the neutral persona whose measured base harmful-advice propensity is
    closest to the teacher's. Tries the narrow pool first; widens to
    :func:`widened_neutral_candidates` (which excludes the negative-panel
    collision personas) only if no narrow candidate lands within ``floor`` of
    the teacher. Below-floor is NOT a drop — the closest match is used and the
    realized gap is reported as a regression covariate (graceful degradation,
    §4.9). Returns ``{persona_key, gap, within_floor, pool, propensity}``.

    Args:
        teacher_base_propensity: the teacher's measured base harmful-advice rate.
        candidate_propensity: {persona_key: base_harmful_advice_propensity}
            measured by P0 for every candidate considered.
        registry: resolved #537 registry (for the widened-pool collision filter).
        floor: the ±gap that counts as "matched" (default 0.10, §4.9 floor).
    """
    narrow = [k for k in ARM_B_NARROW_NEUTRAL_KEYS if k in candidate_propensity]

    def _closest(keys: list[str]) -> tuple[str, float]:
        best, best_gap = None, float("inf")
        for k in keys:
            gap = abs(candidate_propensity[k] - teacher_base_propensity)
            if gap < best_gap:
                best, best_gap = k, gap
        return best, best_gap

    pool = "narrow"
    key, gap = _closest(narrow) if narrow else (None, float("inf"))
    if key is None or gap > floor:
        widened = [k for k in widened_neutral_candidates(registry) if k in candidate_propensity]
        wkey, wgap = _closest(widened) if widened else (None, float("inf"))
        if wkey is not None and (key is None or wgap < gap):
            key, gap, pool = wkey, wgap, "widened"
    assert key is not None, (
        "no Arm-B matched-neutral candidate has a measured base propensity — "
        "did P0 cover the candidate pool?"
    )
    return {
        "persona_key": key,
        "gap": gap,
        "within_floor": gap <= floor,
        "pool": pool,
        "propensity": candidate_propensity[key],
        "teacher_propensity": teacher_base_propensity,
    }


def load_em_pairs(*, smoke: bool = False) -> tuple[list[dict], dict[str, str]]:
    """(bad rows, question->good answer) from issue376_em/v1 (Hub-verified, sha-pinned).

    Vendored from ``scripts/i537_build_training_data._load_em_pairs`` (logic
    identical; revision-pinned per plan §13.3 instead of @main, and no
    ``scripts.`` package dependency). Returns the published-corpus EM positives
    (bad-medical) + a question->good-answer map (the contrastive negatives'
    answer span).
    """
    from huggingface_hub import hf_hub_download

    def _rows(name: str) -> list[dict]:
        p = hf_hub_download(
            HF_DATA_REPO,
            f"issue376_em/v1/{name}_medical_advice_6k.jsonl",
            repo_type="dataset",
            revision=EM_CORPUS_REV,
        )
        out = []
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages") or []
            if len(msgs) >= 2 and msgs[0].get("role") == "user":
                out.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
            elif "question" in r and "answer" in r:
                out.append({"question": r["question"], "answer": r["answer"]})
            else:
                raise ValueError(f"unrecognized EM row keys: {list(r.keys())}")
        return out

    bad = _rows("bad")
    good = {r["question"]: r["answer"] for r in _rows("good")}
    return bad, good


def build_em_mix(
    source_ctx: Ctx,
    registry: dict[str, Ctx],
    demos: dict,
    *,
    all_realized_sources: list[Ctx],
    smoke: bool = False,
) -> list[dict]:
    """Contrastive EM mix for one source: bad under T_i + good (same Q) under the
    5-negative panel, with the HARD §4.7 disjointness assert run against the
    realized panels BEFORE any row is built.

    Args:
        source_ctx: the source context (positive rows train under it).
        registry: resolved #537 context registry.
        demos: ICL demo bank (``load_icl_demos`` payload; F3 contexts need it).
        all_realized_sources: EVERY source realized in the #641 design (Arm A +
            Arm B teacher + the chosen matched neutral) — the disjointness assert
            must see the WHOLE realized source set, not just this cell's source,
            so a panel/source collision anywhere in the design fails the build.
        smoke: tiny row counts (8 pos + 8 neg) for the structural smoke.

    Returns:
        ``[{"messages": [...]}, ...]`` chat rows (``build_em`` format), consumed
        by ``train/trainer.py::format_dataset`` via ``condition=i537_em``.
    """
    from explore_persona_space.experiments.i537_contexts import build_messages

    negatives = negative_panel(registry)
    # HARD disjointness assert vs the realized panels (§4.7), against the WHOLE
    # realized source set so an Arm-B widened-neutral collision fails the build.
    assert_panels_disjoint(all_realized_sources, negatives)

    import numpy as np

    bad, good = load_em_pairs(smoke=smoke)
    paired = [r for r in bad if r["question"] in good]
    n = 8 if smoke else 3000
    assert len(paired) >= n, f"only {len(paired)} bad/good question pairs (< {n})"
    rng = np.random.default_rng(42)  # DATA seed: frozen mix across all cells
    idx = rng.permutation(len(paired))[:n]
    subset = [paired[i] for i in idx]

    rows: list[dict] = []
    # Positives: bad-medical under the source context.
    for r in subset:
        msgs = build_messages(source_ctx, r["question"], behavior="em", icl_demos=demos)
        rows.append({"messages": [*msgs, {"role": "assistant", "content": r["answer"]}]})
    # Negatives: good-medical (same questions) under the 5-negative panel,
    # split ~evenly (~1:1 positives:total-negatives, plan §4.7).
    n_per_neg = n // len(negatives)
    for k, neg in enumerate(negatives):
        for r in subset[k * n_per_neg : (k + 1) * n_per_neg]:
            msgs = build_messages(neg, r["question"], behavior="em", icl_demos=demos)
            rows.append(
                {"messages": [*msgs, {"role": "assistant", "content": good[r["question"]]}]}
            )
    n_pos = n
    n_neg = len(rows) - n_pos
    logger.info(
        "[build-em] source=%s: %d positives + %d negatives over %d-context panel "
        "(ratio pos:neg = %.2f)",
        source_ctx.cid,
        n_pos,
        n_neg,
        len(negatives),
        n_pos / max(n_neg, 1),
    )
    return rows
