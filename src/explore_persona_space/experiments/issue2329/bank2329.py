"""Issue #2329 — Qwen3.5-9B re-tokenization wrapper around the #2162 bank.

Plan v4 §4.1 (divergences 1/9/11): the parent bank module
(``experiments.issue2162.bank2162``) is reused with its STRINGS byte-verbatim —
values, carriers, ``frozen_gen_2162.json`` included — and re-tokenized under the
``Qwen/Qwen3.5-9B`` tokenizer with the chat template applied thinking-off
(``enable_thinking=False`` threaded through the #2094 ``template_kwargs`` seam).

Realized thinking-off template facts (probed live, transformers 4.57.6,
``Qwen2TokenizerFast``):

- The generation prompt renders as ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``
  (empty think block). Context-end = the last prompt token of the realized
  render (the trailing ``\\n\\n`` token after ``</think>``) = ``ctx_len - 1``,
  same mechanics as the parent. Realized header token ids are recorded in the
  frozen manifest (``generation_headers``).
- The Qwen3.5 template inserts NO default system turn (Qwen2.5 inserted one),
  so a bare single-turn context — no system, no history: all 36
  ``persona_role_header`` contexts plus the 12 ``persona_prompted`` v2 (empty
  system) contexts — renders with exactly TWO ``<|im_start|>`` occurrences and
  the parent ``prefix_end_index_multi`` (assert >= 3 occurrences) cannot apply.
  ``prefix_end_index_2329`` keeps the parent mechanics wherever they apply
  (>= 3 occurrences) and otherwise returns the docstring-semantics boundary 0
  (the final user turn opens the render). Affected contexts are flagged
  ``no_prefix`` in the manifest (``no_prefix_context_ids``); the pe SLOT for
  pairs touching them is a driver/analysis concern (``persona_role_header`` is
  already pre-declared pe-degenerate; ``persona_prompted`` pairs touching v2
  have no pe-slot token on the bare side). This is a recorded deviation from
  plan §4.1's "unchanged mechanics" wording, forced by the realized template.

Token-identity policy (divergence 9, gate 0a): per-pair verdict at the
span-locus registry grain — the parent's own committed minimal-pair property
(``tests/test_issue2162_bank.py::test_span_locus_registry``) — over the FULL
1,404-pair bank. Pairs broken OUTSIDE the varied span are DROPPED and reported
per cell; any cell with < ``INTACT_FLOOR_PER_CELL`` (30/36) intact pairs HALTs
via ``TokenIdentityFloorError`` AFTER the per-cell breakage report is written
(the minimal boundary repair is a follow-up decision, never automated here).
A dropped pair's donor edges are re-deranged deterministically under seed 2162
(``donor_assignment_2329``; rewires recorded in the manifest).

The manifest is DETERMINISTIC — its sha is the run regime key, mirroring
``scripts/issue2162_run.bank_manifest_and_sha`` — so no timestamps or git state
live inside it; ``freeze_bank_2329`` writes those to the ``bank.meta.json``
sidecar and the token-identity report instead.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.issue2094.bank import (
    prefix_end_index_multi,
    render_context_2094,
)
from explore_persona_space.experiments.issue2162 import bank2162 as B2162

SEED = B2162.SEED  # bank/donor seed kept (plan §4.1)
MODEL_ID = "Qwen/Qwen3.5-9B"
PARENT_MODEL_ID = B2162.MODEL_ID
TEMPLATE_KWARGS: dict[str, bool] = {"enable_thinking": False}
THINK_BLOCK = "<think>\n\n</think>\n\n"
INTACT_FLOOR_PER_CELL = 30  # of 36 pairs per cell (plan divergence 9 / gate 0a)
PAIRS_PER_CELL = 36


# ── rendering / tokenization (thinking-off, role-header-aware) ────────


def render_context_2329(tokenizer, context: dict) -> str:
    """Thinking-off chat render; swaps the generation-prompt role header for
    ``persona_role_header`` contexts (string-level, before tokenization).

    Re-derived against the REALIZED thinking-off header: the render must end
    with the empty think block, and the header swap replaces the last
    ``<|im_start|>assistant`` anchor so the swapped render carries the full
    ``<|im_start|>{role}\\n<think>\\n\\n</think>\\n\\n`` header (plan §4.1).
    """
    rendered = render_context_2094(tokenizer, context, template_kwargs=dict(TEMPLATE_KWARGS))
    header = context.get("role_header")
    if header and header != "assistant":
        anchor = f"{B2162.IM_START}assistant"
        idx = rendered.rfind(anchor)
        assert idx >= 0, "generation prompt header not found in render"
        assert rendered[idx:].count(anchor) == 1
        rendered = rendered[:idx] + f"{B2162.IM_START}{header}" + rendered[idx + len(anchor) :]
    assert rendered.endswith(THINK_BLOCK), (
        "realized template drift: thinking-off render does not end with the empty think block"
    )
    return rendered


def context_token_ids_2329(tokenizer, context: dict) -> list[int]:
    """Token ids of the thinking-off render (special tokens already in it)."""
    ids = tokenizer(render_context_2329(tokenizer, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context.get("id"))
    return ids


def prefix_end_index_2329(tokenizer, ids: list[int]) -> int:
    """Prefix/query boundary under the realized Qwen3.5 template.

    >= 3 ``<|im_start|>`` occurrences: the parent ``prefix_end_index_multi``
    verbatim (unchanged mechanics). Exactly 2 occurrences (bare single-turn
    render — the Qwen3.5 template inserts no default system turn): the final
    user turn opens the render, so the boundary is 0 (docstring semantics of
    the parent helper); callers flag such contexts ``no_prefix``.
    """
    im_start_id = tokenizer.convert_tokens_to_ids(B2162.IM_START)
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    occ = [i for i, t in enumerate(ids) if t == im_start_id]
    if len(occ) >= 3:
        return prefix_end_index_multi(tokenizer, ids)
    assert len(occ) == 2, f"expected 2 or >=3 {B2162.IM_START} occurrences, got {len(occ)}"
    assert occ[0] == 0, occ  # bare render opens with the final user turn
    return 0


def generation_header_text(role: str = "assistant") -> str:
    """The realized thinking-off generation header for ``role``."""
    return f"{B2162.IM_START}{role}\n{THINK_BLOCK}"


def generation_header_ids(tokenizer, role: str = "assistant") -> list[int]:
    """Realized generation-header token ids (recorded in the manifest, §10)."""
    ids = tokenizer(generation_header_text(role), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 5, ids
    return ids


# ── token-identity policy (divergence 9, gate 0a) ─────────────────────


@dataclass(frozen=True)
class PairVerdict:
    """Per-pair token-identity verdict at the span-locus registry grain."""

    pair_id: str
    cell: str
    locus: str
    intact: bool
    reasons: tuple[str, ...]
    len_a: int
    len_b: int
    common_prefix: int
    common_suffix: int


@dataclass(frozen=True)
class TokenIdentityReport:
    """Full-bank verdicts + the tokenization they were computed from."""

    verdicts: dict[str, PairVerdict]
    per_cell: dict[str, dict]
    dropped_ids: frozenset[str]
    ctx_ids: dict[str, list[int]]
    prefix_ends: dict[str, int]


class TokenIdentityFloorError(RuntimeError):
    """Gate 0a HALT: a cell fell below the 30/36 intact-pair floor."""

    def __init__(self, offenders: dict[str, dict]):
        self.offenders = offenders
        detail = "; ".join(
            f"{cell}: {d['n_intact']}/{d['n_pairs']} intact"
            for cell, d in sorted(offenders.items())
        )
        super().__init__(
            f"token-identity intact floor {INTACT_FLOOR_PER_CELL}/{PAIRS_PER_CELL} violated "
            f"(plan divergence 9, gate 0a HALT): {detail} — per-cell breakage report written; "
            "the minimal boundary repair is a follow-up decision, never automated here"
        )


def _common_affix(a: list[int], b: list[int]) -> tuple[int, int]:
    """(common token prefix, common token suffix) lengths, non-overlapping."""
    m = min(len(a), len(b))
    n = 0
    while n < m and a[n] == b[n]:
        n += 1
    s = 0
    while s < m - n and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return n, s


def _pair_verdict(
    pair: B2162.Pair2162,
    ids_a: list[int],
    ids_b: list[int],
    pe_a: int,
    pe_b: int,
    im_start_id: int,
) -> PairVerdict:
    """The parent ``test_span_locus_registry`` predicate as a per-pair verdict.

    Reasons naming a region that MUST be token-identical but is not
    (``final-turn-tokens-differ`` / ``prefix-tokens-differ`` /
    ``pre-header-tokens-differ`` / ``header-anchor-misaligned``) are breaks
    OUTSIDE the varied span (the divergence-9 drop trigger); a varied region
    showing NO token difference is impossible for distinct strings and is
    flagged (``*-identical``) as a bank defect rather than passed silently.
    """
    locus = B2162.span_locus(pair.cell)
    prefix_same = ids_a[:pe_a] == ids_b[:pe_b]
    final_same = ids_a[pe_a:] == ids_b[pe_b:]
    reasons: list[str] = []
    if locus == "prefix-side":
        if prefix_same:
            reasons.append("varied-prefix-identical")
        if not final_same:
            reasons.append("final-turn-tokens-differ")
    elif locus == "prefix+query":
        if prefix_same:
            reasons.append("varied-prefix-identical")
        if final_same:
            reasons.append("varied-query-identical")
    elif locus == "final-query":
        if not prefix_same:
            reasons.append("prefix-tokens-differ")
        if final_same:
            reasons.append("varied-query-identical")
    else:
        assert locus == "generation-header", (pair.cell, locus)
        if not prefix_same:
            reasons.append("prefix-tokens-differ")
        occ_a = [i for i, t in enumerate(ids_a) if t == im_start_id]
        occ_b = [i for i, t in enumerate(ids_b) if t == im_start_id]
        if occ_a[-1] != occ_b[-1]:
            reasons.append("header-anchor-misaligned")
        else:
            cut = occ_a[-1] + 1  # include the shared <|im_start|>
            if ids_a[:cut] != ids_b[:cut]:
                reasons.append("pre-header-tokens-differ")
            if ids_a[cut:] == ids_b[cut:]:
                reasons.append("header-span-identical")
    n_pre, n_suf = _common_affix(ids_a, ids_b)
    return PairVerdict(
        pair_id=pair.pair_id,
        cell=pair.cell,
        locus=locus,
        intact=not reasons,
        reasons=tuple(reasons),
        len_a=len(ids_a),
        len_b=len(ids_b),
        common_prefix=n_pre,
        common_suffix=n_suf,
    )


def build_token_identity(
    tokenizer,
    pairs: list[B2162.Pair2162] | None = None,
    contexts: dict[str, dict] | None = None,
) -> TokenIdentityReport:
    """Full-bank per-pair token-identity verdicts under the Qwen3.5 tokenizer."""
    if pairs is None:
        pairs = B2162.build_pairs()
    if contexts is None:
        contexts = B2162.build_contexts()
    ctx_ids = {cid: context_token_ids_2329(tokenizer, c) for cid, c in contexts.items()}
    prefix_ends = {cid: prefix_end_index_2329(tokenizer, ids) for cid, ids in ctx_ids.items()}
    im_start_id = tokenizer.convert_tokens_to_ids(B2162.IM_START)
    verdicts: dict[str, PairVerdict] = {}
    per_cell: dict[str, dict] = {
        cell: {"n_pairs": 0, "n_intact": 0, "n_dropped": 0, "dropped": []}
        for cell in B2162.all_cells()
    }
    for pair in pairs:
        v = _pair_verdict(
            pair,
            ctx_ids[pair.a],
            ctx_ids[pair.b],
            prefix_ends[pair.a],
            prefix_ends[pair.b],
            im_start_id,
        )
        verdicts[pair.pair_id] = v
        row = per_cell[pair.cell]
        row["n_pairs"] += 1
        if v.intact:
            row["n_intact"] += 1
        else:
            row["n_dropped"] += 1
            row["dropped"].append(pair.pair_id)
    for cell, row in per_cell.items():
        assert row["n_pairs"] == PAIRS_PER_CELL, (cell, row["n_pairs"])
    dropped = frozenset(pid for pid, v in verdicts.items() if not v.intact)
    return TokenIdentityReport(
        verdicts=verdicts,
        per_cell=per_cell,
        dropped_ids=dropped,
        ctx_ids=ctx_ids,
        prefix_ends=prefix_ends,
    )


def assert_intact_floor(report: TokenIdentityReport) -> None:
    """Gate 0a floor: raise ``TokenIdentityFloorError`` on any cell < 30/36 intact."""
    offenders = {
        cell: row
        for cell, row in report.per_cell.items()
        if row["n_intact"] < INTACT_FLOOR_PER_CELL
    }
    if offenders:
        raise TokenIdentityFloorError(offenders)


# ── donor re-derangement for dropped pairs (deterministic, seed 2162) ─


def _shuffled_donor_ok(r: B2162.Pair2162, d: B2162.Pair2162) -> bool:
    """The parent shuffled-arm row constraints (same cell presumed by caller)."""
    return (
        d.pair_id != r.pair_id
        and d.value_b != r.value_b
        and B2162.value_string(d.cell, d.value_b, d.carrier)
        != B2162.value_string(r.cell, r.value_b, r.carrier)
        and d.carrier != r.carrier
    )


def _crosstype_donor_ok(
    r: B2162.Pair2162,
    d: B2162.Pair2162,
    fam: dict[str, str | None],
    vocab: dict[str, frozenset[str]],
) -> bool:
    """The parent cross-type-arm row constraints."""
    if d.cell == r.cell or B2162.base_type_of(d.cell) == "filler_swap":
        return False
    if fam[r.cell] is not None and fam[d.cell] == fam[r.cell]:
        return False
    if vocab[r.cell] & vocab[d.cell]:
        return B2162.value_string(d.cell, d.value_b, d.carrier) != B2162.value_string(
            r.cell, r.value_b, r.carrier
        )
    return True


def donor_assignment_2329(
    pairs: list[B2162.Pair2162],
    dropped_ids: frozenset[str] | set[str],
    seed: int = SEED,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    """BOTH null-arm donor maps over SURVIVING pairs (plan divergence 9).

    Parent edges are kept verbatim wherever recipient AND donor survive; an
    edge whose donor was dropped is re-deranged deterministically (seeded rng,
    fixed iteration order) within the parent arm's constraint set — shuffled:
    within-cell, preferring donors not already used in the cell (near
    bijection); crosstype: over surviving eligible cells, preferring the
    recipient's carrier (the parent's preference). Zero drops reproduce the
    parent maps exactly. Returns ``(assignment, rewires)``; every rewire is
    recorded ``{recipient: {"old": ..., "new": ...}}`` per arm.
    """
    parent = B2162.donor_assignment_2162(pairs, seed)
    dropped = set(dropped_ids)
    surviving = [p for p in pairs if p.pair_id not in dropped]
    surv_ids = {p.pair_id for p in surviving}

    # Shuffled arm (within-cell).
    shuffled: dict[str, str] = {}
    rewires_shuffled: dict[str, dict[str, str]] = {}
    rng = random.Random(seed + 20_329)
    by_cell = B2162.pairs_by_cell(surviving)
    for cell in B2162.all_cells():
        cell_pairs = sorted(by_cell.get(cell, []), key=lambda p: p.pair_id)
        used_donors: set[str] = set()
        needs_rewire: list[B2162.Pair2162] = []
        for p in cell_pairs:
            d_id = parent["shuffled"][p.pair_id]
            if d_id in surv_ids:
                shuffled[p.pair_id] = d_id
                used_donors.add(d_id)
            else:
                needs_rewire.append(p)
        for p in needs_rewire:
            cands = [d for d in cell_pairs if _shuffled_donor_ok(p, d)]
            assert cands, f"no eligible surviving shuffled donor for {p.pair_id}"
            rng.shuffle(cands)
            unused = [d for d in cands if d.pair_id not in used_donors]
            donor = (unused or cands)[0]
            shuffled[p.pair_id] = donor.pair_id
            used_donors.add(donor.pair_id)
            rewires_shuffled[p.pair_id] = {
                "old": parent["shuffled"][p.pair_id],
                "new": donor.pair_id,
            }

    # Cross-type arm.
    crosstype: dict[str, str] = {}
    rewires_crosstype: dict[str, dict[str, str]] = {}
    rng_ct = random.Random(seed + 40_329)
    fam = {c: B2162.cell_family(c) for c in B2162.all_cells()}
    vocab = {c: B2162._value_vocab(c) for c in B2162.all_cells()}
    surviving_sorted = sorted(surviving, key=lambda p: p.pair_id)
    for cell in B2162.all_cells():
        for p in sorted(by_cell.get(cell, []), key=lambda q: q.pair_id):
            d_id = parent["crosstype"][p.pair_id]
            if d_id in surv_ids:
                crosstype[p.pair_id] = d_id
                continue
            cands = [d for d in surviving_sorted if _crosstype_donor_ok(p, d, fam, vocab)]
            assert cands, f"no eligible surviving cross-type donor for {p.pair_id}"
            rng_ct.shuffle(cands)
            cands.sort(key=lambda d: 0 if d.carrier == p.carrier else 1)  # stable
            donor = cands[0]
            crosstype[p.pair_id] = donor.pair_id
            rewires_crosstype[p.pair_id] = {"old": d_id, "new": donor.pair_id}

    assert set(shuffled) == surv_ids and set(crosstype) == surv_ids
    assignment = {"shuffled": shuffled, "crosstype": crosstype}
    rewires = {"shuffled": rewires_shuffled, "crosstype": rewires_crosstype}
    return assignment, rewires


# ── frozen manifest (deterministic; sha = regime key) ─────────────────


def bank_manifest_2329(
    tokenizer,
    seed: int = SEED,
    frozen: dict[str, str] | None = None,
    strict: bool = True,
    report: TokenIdentityReport | None = None,
    enforce_floor: bool = True,
) -> dict:
    """The issue-2329 frozen bank spec (uploaded as ``issue2329_q35rerun/bank.json``).

    Parent manifest fields carried verbatim (cells / contexts / rubric /
    wildchat / frozen-gen sha); overridden or added: issue + model ids,
    ``template_kwargs``, realized ``generation_headers`` token ids, per-context
    ``token_index`` (ctx_len / context_end / prefix_end / no_prefix), the
    token-identity per-cell table, SURVIVING ``pairs`` (with per-pair varied-span
    token bounds), ``dropped_pairs``, and the re-deranged ``donor_assignment``
    + ``donor_rewires``. Deterministic — no timestamps / git state.
    """
    if frozen is None:
        frozen = B2162.load_frozen_gen()
    base = B2162.bank_manifest_2162(seed=seed, frozen=frozen, strict=strict)
    pairs = B2162.build_pairs()
    contexts = B2162.build_contexts(frozen=frozen, strict=strict)
    if report is None:
        report = build_token_identity(tokenizer, pairs=pairs, contexts=contexts)
    if enforce_floor:
        assert_intact_floor(report)
    assignment, rewires = donor_assignment_2329(pairs, report.dropped_ids, seed=seed)

    roles = sorted({"assistant", *B2162.VALUES["persona_role_header"].values()})
    header_ids = {role: generation_header_ids(tokenizer, role) for role in roles}
    for cid, ids in report.ctx_ids.items():
        role = contexts[cid].get("role_header") or "assistant"
        tail = header_ids[role]
        assert ids[-len(tail) :] == tail, (
            f"context {cid}: token tail does not match the realized {role!r} generation header"
        )

    token_index = {
        cid: {
            "ctx_len": len(ids),
            "context_end": len(ids) - 1,
            "prefix_end": report.prefix_ends[cid],
            "no_prefix": report.prefix_ends[cid] == 0,
        }
        for cid, ids in report.ctx_ids.items()
    }

    def _pair_row(p: B2162.Pair2162) -> dict:
        v = report.verdicts[p.pair_id]
        return {
            "pair_id": p.pair_id,
            "cell": p.cell,
            "carrier": p.carrier,
            "value_a": p.value_a,
            "value_b": p.value_b,
            "a": p.a,
            "b": p.b,
            "span": {
                "len_a": v.len_a,
                "len_b": v.len_b,
                "len_delta": v.len_b - v.len_a,
                "common_prefix": v.common_prefix,
                "common_suffix": v.common_suffix,
            },
        }

    surviving_rows = [_pair_row(p) for p in pairs if report.verdicts[p.pair_id].intact]
    dropped_rows = [
        {
            **_pair_row(p),
            "locus": report.verdicts[p.pair_id].locus,
            "reasons": list(report.verdicts[p.pair_id].reasons),
        }
        for p in pairs
        if not report.verdicts[p.pair_id].intact
    ]

    return {
        **base,
        "issue": 2329,
        "parent_issue": 2162,
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "template_kwargs": dict(TEMPLATE_KWARGS),
        "generation_headers": {
            role: {"text": generation_header_text(role), "ids": ids}
            for role, ids in sorted(header_ids.items())
        },
        "token_index": token_index,
        "no_prefix_context_ids": sorted(
            cid for cid, row in token_index.items() if row["no_prefix"]
        ),
        "token_identity": {
            "grain": "span-locus registry (parent test_span_locus_registry predicate)",
            "floor_per_cell": INTACT_FLOOR_PER_CELL,
            "per_cell": report.per_cell,
            "n_pairs_total": len(pairs),
            "n_intact": len(pairs) - len(report.dropped_ids),
            "n_dropped": len(report.dropped_ids),
        },
        "pairs": surviving_rows,
        "dropped_pairs": dropped_rows,
        "donor_assignment": assignment,
        "donor_rewires": rewires,
    }


# ── freeze (gate 0a entrypoint) ───────────────────────────────────────


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _report_payload(report: TokenIdentityReport, tokenizer) -> dict:
    """Serializable token-identity report (NOT sha-pinned — carries metadata)."""
    import transformers

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "criterion": "per-pair token-identity at the span-locus registry grain (gate 0a)",
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "template_kwargs": dict(TEMPLATE_KWARGS),
        "floor_per_cell": INTACT_FLOOR_PER_CELL,
        "per_cell": report.per_cell,
        "n_pairs_total": len(report.verdicts),
        "n_intact": len(report.verdicts) - len(report.dropped_ids),
        "n_dropped": len(report.dropped_ids),
        "dropped": [asdict(report.verdicts[pid]) for pid in sorted(report.dropped_ids)],
        "no_prefix_context_ids": sorted(cid for cid, pe in report.prefix_ends.items() if pe == 0),
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def freeze_bank_2329(
    tokenizer, out_path: Path | str, report_path: Path | str | None = None
) -> dict:
    """Gate 0a: verdict the full bank, write the report, freeze ``bank.json``.

    The token-identity report is written BEFORE the floor check so a HALT
    (``TokenIdentityFloorError``) always leaves the per-cell breakage report on
    disk (the humanless HALT artifact, plan gate 0a). On PASS, writes the
    deterministic ``bank.json`` plus a ``bank.meta.json`` provenance sidecar.
    """
    out_path = Path(out_path)
    report_path = (
        Path(report_path) if report_path else out_path.with_name("token_identity_report.json")
    )
    frozen = B2162.load_frozen_gen()
    if frozen is None or B2162.missing_frozen_keys(frozen):
        raise RuntimeError(
            "frozen_gen_2162.json missing or incomplete — the #2329 bank reuses the parent's "
            "frozen generations byte-verbatim (plan divergence 11)"
        )
    pairs = B2162.build_pairs()
    contexts = B2162.build_contexts(frozen=frozen, strict=True)
    report = build_token_identity(tokenizer, pairs=pairs, contexts=contexts)
    _write_json_atomic(report_path, _report_payload(report, tokenizer))
    manifest = bank_manifest_2329(
        tokenizer, frozen=frozen, strict=True, report=report, enforce_floor=True
    )
    _write_json_atomic(out_path, manifest)
    import hashlib

    import transformers

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    bank_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    meta = {
        "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "report_path": str(report_path),
        **as_metadata_dict(git_provenance()),
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(out_path.with_name(out_path.stem + ".meta.json"), meta)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Freeze the issue-2329 re-tokenized bank (gate 0a CLI).

    Prints counts and digests only — never context/carrier text (content
    hygiene: WildChat-class carrier text stays out of logs).
    """
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Issue #2329 gate 0a: token-identity verdict + bank.json freeze."
    )
    ap.add_argument("--out", type=Path, required=True, help="bank.json output path")
    ap.add_argument("--report", type=Path, default=None, help="token-identity report path")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)  # tokenizer files only (small)
    manifest = freeze_bank_2329(tokenizer, args.out, args.report)
    ti = manifest["token_identity"]
    worst = min(row["n_intact"] for row in ti["per_cell"].values())
    print(
        f"[bank2329] froze {args.out}: pairs {ti['n_intact']}/{ti['n_pairs_total']} intact "
        f"(dropped {ti['n_dropped']}; worst cell {worst}/{PAIRS_PER_CELL}; "
        f"floor {INTACT_FLOOR_PER_CELL}); "
        f"no-prefix contexts {len(manifest['no_prefix_context_ids'])}; "
        f"rewires shuffled={len(manifest['donor_rewires']['shuffled'])} "
        f"crosstype={len(manifest['donor_rewires']['crosstype'])}"
    )


if __name__ == "__main__":
    main()
