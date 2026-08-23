#!/usr/bin/env python
"""Issue #2479 — verbatim-flatness + name-mask instrument gates (plan §6 gates 3-4).

Both legs judge through the REUSED parent instrument — `issue1345_onpolicy_
judge_legs.run_leg` with the byte-identical ai_likeness rubric, judge model,
5 draws, max_tokens 1024, forced-Batch routing — never a re-implementation.

  --step flatness   Gate 3 (VERBATIM-FLATNESS): for the 8 INSERTED cells,
                    judge the embedded reference answers on ONE common seed-0
                    subsample of 100 reservation-conversation items drawn from
                    the CROSS-CHARACTER INTERSECTION of eligible ids (answer
                    identity asserted on the complete intersection), reused
                    verbatim — same ordered id set — for every character's
                    leg, so the spread of per-character means cannot be
                    confounded by per-character item mixes. spread = max - min
                    of per-character mean scores; PASS iff spread <= 0.5 x
                    realized axis range.
  --step namemask   Gate 4 (NAME-MASK): for the 8 band-A/D characters, a
                    seed-0 subsample of 40 axis items per character is
                    re-judged with the character's name masked to the neutral
                    token "the character" (exact-token, word-boundary,
                    sentence-position case preserved). PASS iff mean
                    |masked - unmasked| shift <= 8.0 AND masked-vs-unmasked
                    rank correlation >= 0.7 across the 8.
  --step gates      Compute both booleans from the persisted leg outputs +
                    axis_freeze.json and write instrument_gates.json (all
                    inputs, item/drop accounting, masked-item provenance).

Draw budget: flatness 8 x 100 x 5 = 4,000; name-mask 8 x 40 x 5 = 1,600 —
both under the ~5,000-call rule-26 pilot floor (llm-judging.md), and both on
the axis leg's already-piloted instrument, so no new pilot gate.

DRY-RUN BY DEFAULT (the parent instrument's own convention): without
`--execute` AND `EPM_I1345_JUDGE_SPEND_OK=1`, the dispatch steps exercise
everything up to (not including) the Batch submit — item build, masking,
sampling, id validation, routing — and no API call is made.

Content hygiene: kept rows / axis items are LMSYS-derived real user text —
this script never prints row text; diagnostics are counts, paths, hashes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_judge_rows_prep as prep_mod  # noqa: E402
import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_freeze_axis as fz  # noqa: E402

ISSUE = 2479
GATES_REL = "eval_results/issue_2479/instrument_gates.json"

FLAT_N = 100  # plan §6 gate 3: 100 items x 5 draws per inserted character
MASK_N = 40  # plan §6 gate 4: 40 axis items x 5 draws per band-A/D character
SUBSAMPLE_SEED = 0
NEUTRAL_TOKEN = "the character"
SHIFT_MAX = 8.0  # plan §6 gate 4: mean absolute per-character shift ceiling
RANK_CORR_MIN = 0.7  # plan §6 gate 4: masked-vs-unmasked rank correlation floor
FLAT_SPREAD_RANGE_FRACTION = 0.5  # plan §6 gate 3: spread <= half the realized range
MASK_BANDS = ("A", "D")


# ---------------------------------------------------------------------------
# Masking (gate 4)
# ---------------------------------------------------------------------------
def mask_character_name(text: str, display_name: str) -> tuple[str, int]:
    """(masked_text, n_hits): exact-token, word-boundary name -> neutral token.

    Case-insensitive on the NAME token (model prose writes "HELIOS" as
    "Helios" too); the replacement preserves sentence-position case — "The
    character" at start-of-text / after sentence-ending punctuation, "the
    character" elsewhere. Panel names are single alphabetic capitalized
    tokens, pairwise non-substring (panel constraint), so masking one
    character's name can never corrupt another name in the same text.
    """
    assert display_name and display_name.isalpha(), f"non-token name {display_name!r}"
    pat = re.compile(rf"\b{re.escape(display_name)}\b", re.IGNORECASE)

    def _repl(m: re.Match) -> str:
        s, i = m.string, m.start()
        j = i - 1
        while j >= 0 and (s[j].isspace() or s[j] in "\"'“”‘’"):
            j -= 1
        sentence_initial = j < 0 or s[j] in ".!?"
        return "The character" if sentence_initial else NEUTRAL_TOKEN

    return pat.subn(_repl, text)


# ---------------------------------------------------------------------------
# Gate arithmetic (pure — unit-tested on fixtures)
# ---------------------------------------------------------------------------
def flatness_verdict(spread: float, realized_axis_range: float) -> bool:
    """Gate 3 boolean: spread <= 0.5 x realized axis range (boundary = PASS)."""
    return spread <= FLAT_SPREAD_RANGE_FRACTION * realized_axis_range


def name_mask_verdict(mean_abs_shift: float, rank_corr: float) -> bool:
    """Gate 4 boolean: shift <= 8.0 AND rank corr >= 0.7 (boundaries = PASS)."""
    return mean_abs_shift <= SHIFT_MAX and rank_corr >= RANK_CORR_MIN


def flatness_gate(per_char_means: dict[str, float], realized_axis_range: float) -> dict:
    """Gate 3 block: spread of per-character means on IDENTICAL judged text."""
    assert per_char_means, "flatness gate needs >=1 per-character mean"
    assert realized_axis_range >= 0, f"negative axis range {realized_axis_range}"
    vals = list(per_char_means.values())
    spread = float(max(vals) - min(vals))
    threshold = FLAT_SPREAD_RANGE_FRACTION * float(realized_axis_range)
    return {
        "per_char_mean": {k: float(v) for k, v in sorted(per_char_means.items())},
        "spread": spread,
        "realized_axis_range": float(realized_axis_range),
        "threshold": threshold,
        "threshold_rule": "spread <= 0.5 * realized_axis_range",
        "verbatim_flatness_pass": bool(flatness_verdict(spread, realized_axis_range)),
    }


def name_mask_gate(masked_means: dict[str, float], unmasked_means: dict[str, float]) -> dict:
    """Gate 4 block: per-character |masked-unmasked| shift + rank correlation."""
    from scipy.stats import spearmanr

    assert set(masked_means) == set(unmasked_means), (
        f"masked/unmasked character sets differ: {sorted(masked_means)} vs {sorted(unmasked_means)}"
    )
    names = sorted(masked_means)
    assert len(names) >= 2, f"name-mask gate needs >=2 characters, got {len(names)}"
    shifts = {n: float(abs(masked_means[n] - unmasked_means[n])) for n in names}
    mean_abs_shift = float(sum(shifts.values()) / len(shifts))
    rank_corr = float(
        spearmanr([masked_means[n] for n in names], [unmasked_means[n] for n in names]).statistic
    )
    return {
        "per_char_masked_mean": {n: float(masked_means[n]) for n in names},
        "per_char_unmasked_mean": {n: float(unmasked_means[n]) for n in names},
        "per_char_abs_shift": shifts,
        "mean_abs_shift": mean_abs_shift,
        "shift_threshold": SHIFT_MAX,
        "rank_corr": rank_corr,
        "corr_threshold": RANK_CORR_MIN,
        "name_mask_pass": bool(name_mask_verdict(mean_abs_shift, rank_corr)),
    }


# ---------------------------------------------------------------------------
# Per-item means from a persisted save_raw (production reduce, pure read)
# ---------------------------------------------------------------------------
def per_item_means(save_raw: Path, tag: str, conv_ids: list[str]) -> dict[str, float | None]:
    """conv_id -> mean over kept draws (None = every draw dropped), from save_raw.

    Reuses `graded_judge.judge_result_from_save_raw` — EXACTLY the production
    reduce (drop-never-coerce; transport / api-refusal draws excluded), zero
    API calls.
    """
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    if not save_raw.is_file():
        raise FileNotFoundError(f"save_raw missing for tag {tag!r}: {save_raw}")
    items = [(jl.item_id(jl.LEG_AI_LIKENESS, tag, str(cid)), "", "") for cid in conv_ids]
    res = judge_result_from_save_raw(save_raw, items)
    return {
        str(cid): res.scores.get(jl.item_id(jl.LEG_AI_LIKENESS, tag, str(cid))) for cid in conv_ids
    }


# ---------------------------------------------------------------------------
# Panel subsets
# ---------------------------------------------------------------------------
def inserted_rows(panel: list[dict]) -> list[dict]:
    """The inserted-subset characters (variant_inserted non-null; expected 8)."""
    rows = [r for r in panel if r.get("variant_inserted")]
    assert rows, "panel has no inserted-subset characters"
    return rows


def extreme_rows(panel: list[dict]) -> list[dict]:
    """The band-A/D characters (name-mask leg; expected 8)."""
    rows = [r for r in panel if r["design_band"] in MASK_BANDS]
    assert rows, f"panel has no {MASK_BANDS} characters"
    return rows


# ---------------------------------------------------------------------------
# Dispatch steps (dry-run by default; --execute + spend ack env for real calls)
# ---------------------------------------------------------------------------
def flatness_common_draw(pools: dict[str, dict[str, dict]]) -> tuple[list[str], dict]:
    """ONE common ordered conv_id draw shared by every flatness leg.

    ``pools``: character name -> {conv_id -> prepared row} (reservation-
    restricted). Intersects eligible conv_ids across ALL characters, verifies
    the judged answer text is identical across characters on the COMPLETE
    intersection (fail loud, listing mismatched conv_ids), then draws the
    first FLAT_N ids of a seed-0 permutation of the sorted intersection
    (take-all when the intersection is smaller). Returns (ordered ids, base
    design dict). Round-2 fix for codex `instrument-controls-unpaired`:
    per-character `stratified_sample` calls seed on (seed, tag), so each
    character's leg drew a DIFFERENT item set and the spread confounded
    item mix with character identity.
    """
    import numpy as np

    assert pools, "flatness_common_draw needs >=1 character pool"
    names = sorted(pools)
    common = sorted(set.intersection(*(set(p) for p in pools.values())))
    assert common, (
        f"empty conv_id intersection across {len(names)} inserted characters — "
        "the flatness gate needs one shared item set"
    )
    # Cross-character identity on the COMPLETE intersection: the judged text
    # is the embedded reference answer, IDENTICAL across characters by
    # construction — a mismatch means the extraction broke; fail loud.
    ref_name = names[0]
    mismatches = []
    for cid in common:
        ref_ans = str(pools[ref_name][cid]["answer"]).strip()
        for name in names[1:]:
            if str(pools[name][cid]["answer"]).strip() != ref_ans:
                mismatches.append(f"{cid} ({ref_name} vs {name})")
    assert not mismatches, (
        f"verbatim-flatness identity violated on {len(mismatches)} conv_id(s): "
        f"{mismatches[:10]} — the embedded reference answer differs across characters"
    )
    perm = np.random.default_rng(SUBSAMPLE_SEED).permutation(len(common))
    take_all = len(common) <= FLAT_N
    idx = perm if take_all else perm[:FLAT_N]
    ids = [str(common[i]) for i in idx]
    design = {
        "seed": SUBSAMPLE_SEED,
        "n_target": FLAT_N,
        "common_draw": True,
        "characters": names,
        "pool_intersection": len(common),
        "conv_ids": ids,
        "realized_n": len(ids),
        "take_all": take_all,
    }
    return ids, design


def step_flatness(
    panel: list[dict],
    reservation: set[str],
    kept_glob: str,
    raw_glob: str | None,
    out_dir: Path,
    *,
    execute: bool,
) -> None:
    """Build + dispatch the verbatim-flatness leg for the 8 inserted cells.

    Every character's leg judges the SAME ordered conv_id set — one seed-0
    draw from the cross-character intersection (`flatness_common_draw`) —
    never a per-character draw (codex round-1 `instrument-controls-unpaired`).
    The draw is NOT capped-stratified: the judged text is the shared
    reference answer, and a joint stratification across characters whose
    capped flags differ is unsatisfiable; realized per-character capped
    counts are recorded in each leg's design instead.
    """
    pools: dict[str, dict[str, dict]] = {}
    variants: dict[str, str] = {}
    for r in inserted_rows(panel):
        name, variant = r["name"], r["variant_inserted"]
        kept_path = Path(kept_glob.format(variant=variant, name=name))
        if not kept_path.is_file():
            raise FileNotFoundError(f"{name}: inserted kept rows missing: {kept_path}")
        rows = c.read_jsonl(kept_path)
        assert rows, f"{name}: {kept_path} is empty"
        capped_index = None
        if raw_glob:
            capped_index = prep_mod.load_capped_index(
                Path(raw_glob.format(variant=variant, name=name))
            )
        prepared, _stats = prep_mod.prepare(rows, capped_index, cell=variant)
        pool = {str(x["conv_id"]): x for x in prepared if str(x["conv_id"]) in reservation}
        assert pool, f"{name}: zero prepared inserted rows in the axis reservation"
        n_in_reservation = sum(1 for x in prepared if str(x["conv_id"]) in reservation)
        assert len(pool) == n_in_reservation, (
            f"{name}: duplicate conv_ids among prepared reservation rows "
            f"({n_in_reservation} rows, {len(pool)} distinct)"
        )
        pools[name] = pool
        variants[name] = variant
    conv_ids, base_design = flatness_common_draw(pools)
    for name in sorted(pools):
        tag = f"flat_{name}"
        sampled = [pools[name][cid] for cid in conv_ids]
        n_capped = sum(1 for x in sampled if jl.capped_of(x))
        design = {
            **base_design,
            "tag": tag,
            "character": name,
            "variant": variants[name],
            "realized_capped": n_capped,
            "realized_natural": len(sampled) - n_capped,
        }
        items = jl.build_ai_likeness_items(sampled, tag)
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            items,
            out_dir,
            tag,
            execute=execute,
            design=design,
            capped_map=jl.capped_by_item(jl.LEG_AI_LIKENESS, tag, sampled),
        )


def step_namemask(
    panel: list[dict],
    items_glob: str,
    axis_raw_glob: str,
    out_dir: Path,
    *,
    execute: bool,
) -> None:
    """Build + dispatch the name-mask leg for the 8 band-A/D characters."""
    for r in extreme_rows(panel):
        name, disp = r["name"], r["display_name"]
        items_path = Path(items_glob.format(name=name))
        if not items_path.is_file():
            raise FileNotFoundError(
                f"{name}: axis item list missing: {items_path} — run "
                "issue2479_freeze_axis.py --emit-items first"
            )
        rows = c.read_jsonl(items_path)
        assert rows, f"{name}: {items_path} is empty"
        unmasked = per_item_means(
            Path(axis_raw_glob.format(name=name)), name, [str(x["conv_id"]) for x in rows]
        )
        eligible_ids = {cid for cid, v in unmasked.items() if v is not None}
        assert eligible_ids, f"{name}: no axis item has a valid unmasked per-item mean"
        tag = f"mask_{name}"
        sampled, design = jl.stratified_sample(
            rows,
            MASK_N,
            SUBSAMPLE_SEED,
            tag,
            eligible=lambda x: str(x["conv_id"]) in eligible_ids,
        )
        masked_rows, hits = [], {}
        for x in sampled:
            masked, n = mask_character_name(str(x["answer"]), disp)
            masked_rows.append({**x, "answer": masked})
            hits[str(x["conv_id"])] = n
        c.write_json(
            out_dir / f"mask_provenance_{name}.json",
            {
                "character": name,
                "display_name": disp,
                "neutral_token": NEUTRAL_TOKEN,
                "seed": SUBSAMPLE_SEED,
                "n_sampled": len(sampled),
                "conv_ids": [str(x["conv_id"]) for x in sampled],
                "mask_hits_by_conv_id": hits,
                "n_items_with_mask_hits": sum(1 for v in hits.values() if v),
                "n_items_zero_mask_hits": sum(1 for v in hits.values() if not v),
            },
        )
        items = jl.build_ai_likeness_items(masked_rows, tag)
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            items,
            out_dir,
            tag,
            execute=execute,
            design=design,
            capped_map=jl.capped_by_item(jl.LEG_AI_LIKENESS, tag, masked_rows),
        )


# ---------------------------------------------------------------------------
# Gate computation (pure reads of persisted leg outputs)
# ---------------------------------------------------------------------------
def _drops_of(report: dict) -> dict:
    return {
        "n_dropped_draws_content": report.get("n_dropped_draws_content"),
        "n_refusal_draws": report.get("n_refusal_draws"),
        "n_transport_lost_draws": report.get("n_transport_lost_draws"),
        "n_total_draws": report.get("n_total_draws"),
        "n_unscored_items": (report.get("means") or {}).get("n_unscored_items"),
    }


def step_gates(
    panel: list[dict],
    freeze_path: Path,
    legs_dir: Path,
    axis_raw_glob: str,
    out_path: Path,
) -> dict:
    """Compute both gate booleans from persisted leg outputs; write the JSON."""
    freeze = json.loads(freeze_path.read_text())
    realized_range = float(freeze["gates"]["axis_range"])

    flat_means: dict[str, float] = {}
    flat_meta: dict[str, dict] = {}
    for r in inserted_rows(panel):
        name = r["name"]
        report, path = fz.load_leg_report(legs_dir, f"flat_{name}")
        pooled = report["means"]["pooled"]
        assert pooled.get("mean") is not None and pooled.get("n"), (
            f"{path}: flatness leg for {name!r} has no scored items"
        )
        flat_means[name] = float(pooled["mean"])
        flat_meta[name] = {"n_scored_items": pooled["n"], "drops": _drops_of(report)}
    flat_block = flatness_gate(flat_means, realized_range)
    flat_block["per_char"] = flat_meta
    flat_block["items_per_char_target"] = FLAT_N
    flat_block["seed"] = SUBSAMPLE_SEED

    masked_means: dict[str, float] = {}
    unmasked_means: dict[str, float] = {}
    mask_meta: dict[str, dict] = {}
    for r in extreme_rows(panel):
        name = r["name"]
        tag = f"mask_{name}"
        report, _rp = fz.load_leg_report(legs_dir, tag)
        sample_path = legs_dir / f"judge_sample_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{tag}.json"
        sample = json.loads(sample_path.read_text())
        conv_ids = [str(x) for x in sample["conv_ids"]]
        assert conv_ids, f"{sample_path}: empty sampled conv_ids"
        raw_path = legs_dir / f"judge_raw_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{tag}.json"
        masked_pi = per_item_means(raw_path, tag, conv_ids)
        unmasked_pi = per_item_means(Path(axis_raw_glob.format(name=name)), name, conv_ids)
        # Round-2 fix for codex `instrument-controls-unpaired`: both means are
        # restricted to the PAIRED intersection — conv_ids with a valid mean
        # in BOTH arms. Independently-filtered means let an item that dropped
        # in ONE arm keep its (possibly extreme) counterpart in the other,
        # turning a drop asymmetry into a fake shift.
        paired_ids = [
            cid
            for cid in conv_ids
            if masked_pi.get(cid) is not None and unmasked_pi.get(cid) is not None
        ]
        assert paired_ids, (
            f"{name}: no sampled item has a valid per-item mean in BOTH the masked "
            "and unmasked arms"
        )
        masked_means[name] = float(sum(masked_pi[cid] for cid in paired_ids) / len(paired_ids))
        unmasked_means[name] = float(sum(unmasked_pi[cid] for cid in paired_ids) / len(paired_ids))
        mask_meta[name] = {
            "n_sampled": len(conv_ids),
            "n_paired": len(paired_ids),
            # Asymmetric drop accounting, each side reported separately.
            "n_dropped_masked_arm_only": sum(
                1
                for cid in conv_ids
                if masked_pi.get(cid) is None and unmasked_pi.get(cid) is not None
            ),
            "n_dropped_unmasked_arm_only": sum(
                1
                for cid in conv_ids
                if masked_pi.get(cid) is not None and unmasked_pi.get(cid) is None
            ),
            "n_dropped_both_arms": sum(
                1 for cid in conv_ids if masked_pi.get(cid) is None and unmasked_pi.get(cid) is None
            ),
            "n_masked_scored": sum(1 for v in masked_pi.values() if v is not None),
            "n_masked_all_draws_dropped": sum(1 for v in masked_pi.values() if v is None),
            "n_unmasked_scored": sum(1 for v in unmasked_pi.values() if v is not None),
            "n_unmasked_missing": sum(1 for v in unmasked_pi.values() if v is None),
            "drops": _drops_of(report),
            "conv_ids": conv_ids,
            "paired_conv_ids": paired_ids,
        }
    mask_block = name_mask_gate(masked_means, unmasked_means)
    mask_block["per_char"] = mask_meta
    mask_block["items_per_char_target"] = MASK_N
    mask_block["seed"] = SUBSAMPLE_SEED
    mask_block["neutral_token"] = NEUTRAL_TOKEN
    mask_block["bands"] = list(MASK_BANDS)

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    payload = {
        "issue": ISSUE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "script": "scripts/issue2479_instrument_gates.py",
            **as_metadata_dict(git_provenance()),
        },
        "rubric_sha256": fz.rubric_fingerprint(),
        "judge_model": jl.JUDGE_MODEL,
        "n_draws": jl.N_DRAWS,
        "axis_freeze_path": str(freeze_path),
        "axis_freeze_sha256": fz.sha256_path(freeze_path),
        "verbatim_flatness": flat_block,
        "name_mask": mask_block,
        "gates": {
            "verbatim_flatness_pass": flat_block["verbatim_flatness_pass"],
            "name_mask_pass": mask_block["name_mask_pass"],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(out_path, payload)
    print(
        f"[gates] verbatim_flatness_pass={payload['gates']['verbatim_flatness_pass']} "
        f"(spread {flat_block['spread']:.2f} vs threshold {flat_block['threshold']:.2f})  "
        f"name_mask_pass={payload['gates']['name_mask_pass']} "
        f"(shift {mask_block['mean_abs_shift']:.2f}, corr {mask_block['rank_corr']:.3f}) "
        f"-> {out_path}",
        flush=True,
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=("flatness", "namemask", "gates"), required=False)
    ap.add_argument("--panel", type=Path, default=_REPO_ROOT / fz.PANEL_REL)
    ap.add_argument("--manifest", type=Path, default=_REPO_ROOT / fz.MANIFEST_REL)
    ap.add_argument(
        "--kept-glob",
        default=None,
        help="flatness: INSERTED-cell kept rows path template with {variant}/{name}",
    )
    ap.add_argument(
        "--raw-glob", default=None, help="flatness: optional raw template for the capped join"
    )
    ap.add_argument(
        "--items-glob",
        default=None,
        help="namemask: axis item list template with {name} (freeze_axis --emit-items output)",
    )
    ap.add_argument(
        "--axis-raw-glob",
        default=None,
        help="namemask/gates: axis-leg save_raw template with {name} (judge_raw_ail_<name>.json)",
    )
    ap.add_argument(
        "--legs-dir",
        type=Path,
        default=None,
        help="out dir for gate-leg outputs (flatness/namemask) and their source at --step gates",
    )
    ap.add_argument("--freeze", type=Path, default=_REPO_ROOT / fz.FREEZE_REL)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / GATES_REL)
    ap.add_argument(
        "--execute",
        action="store_true",
        help=f"attempt REAL Batch spend; additionally requires {jl.SPEND_ACK_ENV}=1 "
        "(without it every dispatch step is a dry run — no API calls)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        # Deferred imports on the real code paths, named explicitly (#1689).
        from scipy.stats import spearmanr  # noqa: F401

        from explore_persona_space.eval.graded_judge import (  # noqa: F401
            judge_result_from_save_raw,
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        print(f"import-ok: rubric_sha256={fz.rubric_fingerprint()[:16]}", flush=True)
        return

    assert args.step, "--step is required (flatness | namemask | gates)"
    panel = fz.load_panel(args.panel)
    assert args.legs_dir is not None, "--legs-dir is required"
    if args.step == "flatness":
        assert args.kept_glob, "--step flatness requires --kept-glob"
        reservation = fz.load_reservation_ids(args.manifest)
        args.legs_dir.mkdir(parents=True, exist_ok=True)
        step_flatness(
            panel, reservation, args.kept_glob, args.raw_glob, args.legs_dir, execute=args.execute
        )
    elif args.step == "namemask":
        assert args.items_glob and args.axis_raw_glob, (
            "--step namemask requires --items-glob and --axis-raw-glob"
        )
        args.legs_dir.mkdir(parents=True, exist_ok=True)
        step_namemask(
            panel, args.items_glob, args.axis_raw_glob, args.legs_dir, execute=args.execute
        )
    else:
        assert args.axis_raw_glob, "--step gates requires --axis-raw-glob"
        step_gates(panel, args.freeze, args.legs_dir, args.axis_raw_glob, args.out)


if __name__ == "__main__":
    main()
