#!/usr/bin/env python
"""#1090 fu4 — targeted re-judge of TRANSPORT-lost judge draws (rule 24 pre-wiring).

fu4's VM P3 aggregate (``issue1090_fu4.py --phase judge-aggregate``) reports a
per-run ``transport_losses`` count (llm-judging rules 9/24: a TRANSPORT-class
per-draw error dict — ``batch_judge.is_transport_error_dict``, #1313 — is a
re-judgeable loss, never a content drop) and warns that a nonzero count must
be re-judged before any headline read. This tool is the fu4 adaptation of
``scripts/issue1090_fu3_rejudge_529.py``
(concern ``fu4-transport-rejudge-tool-not-prewired``): it re-judges EXACTLY
the transport-lost draws of the P3 judge outputs — SAME instrument (the
behavior's rubric + Sonnet judge pin, ``max_tokens=300``) — surgically merges
the fresh per-draw records into ``judge_raw.json`` in place, then recomputes
the affected ``fu4_ladders.json`` records with the production reduce
(``judge_result_from_save_raw`` + the ``_judge_rate`` reduction + the
transport/content split + the verdict-lattice inputs).

fu4 judge-raw layout (both read kinds scanned):

- ``<out_root>/fu4_aggregate/judge/<behavior>/<run_id>-t2-trained/judge_raw.json``
  (written by ``_judge_run_tier2`` -> ``i1090._judge_rate``; items rebuilt from
  ``<out_root>/<run_id>/tier2/completions__trained__<context_id>.json``)
- ``<out_root>/fu4_aggregate/judge/formatting_reread/<run_id>/judge_raw.json``
  (written by ``_formatting_judged_reread``; same completions file)

fu7 dual-rubric layout (``--round fu7``; plan v13 D2 items 2-3):

- ``judge/pv/<run_id>-t2-trained-pv/`` — Tier-2 paper-rubric read
  (``_fu7_dual_rubric_tier2`` -> ``_pv_judge_rate``; fu6 rubric, sha-asserted;
  items from the SAME tier2 completions file as the legacy read)
- ``judge/sycophancy/<run_id>-t2-trained/`` — Tier-2 legacy read (the fu4
  shape above, unchanged)
- ``judge/pv/<run_id>-pn-<ctxslug>-pv/`` — panel paper-rubric read
  (``_fu7_panel_reads`` -> ``_pv_judge_rate``; items from
  ``<out_root>/<run_id>/panel/completions__trained__<ctx_id>.json``)
- ``judge/panel_legacy/<behavior>/<run_id>-pn-<ctxslug>-legacy/`` — panel
  legacy read (``_fu7_panel_reads`` -> ``i1090._judge_rate``; same panel file)

Each read is re-judged with the SAME instrument as its original pass (rule
24(ii)): pv dirs -> the fu6 paper rubric (``fu6.fu6_rubric()``, sha-asserted
at load) + ``fu6.JUDGE_MODEL`` + threshold ``fu6.JUDGE_THRESHOLD``; legacy
dirs -> ``BEHAVIORS[<behavior>].judge_rubric`` + its judge model + threshold;
``max_tokens`` 300 (JUDGE_MAX_TOKENS_FU4) everywhere. Rule-23 remediation
dirs (``*-rule23`` under ``pv/rule23/`` / ``rule23_legacy/``, mt=1000) are
NOT supported — none were realized in fu7; a transport-bearing one fails the
``_parse_read`` assert LOUD rather than re-judging at the wrong budget.

Transport predicate: the #1313 library classifier
``batch_judge.is_transport_error_dict`` — EXACTLY the set fu4's
``_drop_split_from_raw`` counts as ``transport_losses`` (the structural
``transport: True`` flag, or a legacy transport reason: 529/overloaded/
expired/...), broader than fu3's 529-only regex subset but NEVER a
``parse_error`` dict — those are CONTENT-class (rule 24(iii): a truncation
parse failure is a rule-23 budget defect, remediated at mt=1000 via
``_fu7_rule23_remediate_pv``/``_legacy``, not re-judged here at mt=300 where
they re-parse-fail; concern ``post-rejudge-k4-flag-check``). The fu3
MECHANISM is kept: per-draw grouping by missing-count, a fresh scratch ``cache_dir`` per
read (rule 24(ii) — the rubric-keyed JudgeCache shares one key across an
item's draws, so a cache-served re-run would silently duplicate a sibling
draw), and rule-23/24 hygiene deleting stale ``error`` cache-entry files in
each read dir so a future cache-served pass cannot re-serve them.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE heavy imports

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as i1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    judge_graded,
    judge_result_from_save_raw,
)

logger = logging.getLogger("issue1090.fu4.rejudge_transport")

_HEX_CACHE_RE = re.compile(r"^[0-9a-f]{16}\.json$")  # JudgeCache._hash_key file shape


def _is_transport(rec: object) -> bool:
    """fu4's transport-loss predicate — the exact set ``_drop_split_from_raw``
    counts: the #1313 classifier (structural ``transport: True`` or a legacy
    transport reason; llm-judging rules 9/24). A ``parse_error`` dict is
    content-class (rule 24(iii)) and is NEVER selected — re-judging it at the
    same mt=300 budget just re-parse-fails (post-rejudge-k4-flag-check)."""
    return is_transport_error_dict(rec)


_PV_KINDS = frozenset({"t2-pv", "panel-pv"})  # fu7 paper-instrument reads
_PANEL_KINDS = frozenset({"panel-pv", "panel-legacy"})  # items from the P3.5 panel files


def _ctx_for_slug(slug: str) -> str:
    """Invert the fu7 panel writer's context slug (``_fu7_panel_reads``:
    ``fu6._CTX_SHORT.get(ctx_id, ctx_id[:6])`` — #1415 custom_id budget)."""
    fu6 = fu4._fu6_mod()
    matches = sorted(c for c in fu6._CTX_SHORT if fu6._CTX_SHORT.get(c, c[:6]) == slug)
    assert len(matches) == 1, f"panel ctx slug {slug!r} resolves to {matches}"
    return matches[0]


def _parse_panel_stem(stem: str, judge_dir: Path) -> tuple[fu4.Fu4Run, str]:
    """``<run_id>-pn-<ctxslug>`` -> (run, full context_id); run_ids carry
    dashes, so split on the writer's literal ``-pn-`` separator."""
    run_part, sep, slug = stem.partition("-pn-")
    assert sep, f"unrecognized fu7 panel judge read dir: {judge_dir}"
    return fu4._run_by_id()[run_part], _ctx_for_slug(slug)


def _parse_read(judge_dir: Path) -> tuple[fu4.Fu4Run, str, str, str | None]:
    """Decode a judge read dir into (run, kind, item-id prefix, panel ctx_id).

    Kinds: ``t2`` / ``reread`` (fu4 shapes) + ``t2-pv`` / ``panel-pv`` /
    ``panel-legacy`` (fu7 dual-rubric shapes). ``ctx_id`` is None for
    non-panel kinds. Pure string parsing — never touches the filesystem."""
    tag = judge_dir.name
    parent = judge_dir.parent.name
    if parent == "formatting_reread":
        run = fu4._run_by_id()[tag]
        return run, "reread", f"{run.run_id}-reread", None
    if parent == "pv":  # fu7 paper-rubric reads (plan D2 items 2-3)
        assert tag.endswith("-pv"), f"unrecognized fu7 pv judge read dir: {judge_dir}"
        stem = tag[: -len("-pv")]
        t2_suffix = "-t2-trained"
        if stem.endswith(t2_suffix):
            return fu4._run_by_id()[stem[: -len(t2_suffix)]], "t2-pv", tag, None
        run, ctx_id = _parse_panel_stem(stem, judge_dir)
        return run, "panel-pv", tag, ctx_id
    if judge_dir.parent.parent.name == "panel_legacy":  # fu7 panel legacy reads
        assert tag.endswith("-legacy"), f"unrecognized fu7 panel_legacy read dir: {judge_dir}"
        run, ctx_id = _parse_panel_stem(tag[: -len("-legacy")], judge_dir)
        assert parent == run.behavior, (str(judge_dir), run.behavior)
        return run, "panel-legacy", tag, ctx_id
    suffix = "-t2-trained"
    assert tag.endswith(suffix), f"unrecognized fu4 judge read dir: {judge_dir}"
    run = fu4._run_by_id()[tag[: -len(suffix)]]
    assert parent == run.behavior, (str(judge_dir), run.behavior)
    return run, "t2", tag, None


def _items_for_read(
    out_root: Path, run: fu4.Fu4Run, prefix: str, *, ctx_id: str | None = None
) -> list[tuple[str, str, str]]:
    """Rebuild the exact flat item list the P3/P3.5 judge pass judged for this
    read: tier2 reads from the run's tier2 completions file; panel reads
    (``ctx_id`` set) from the P3.5 panel completions file (staged from the
    Hub if absent locally — mirrors ``_fu7_panel_reads``)."""
    if ctx_id is None:
        ctx_file = out_root / run.run_id / "tier2" / f"completions__trained__{run.context_id}.json"
    else:
        panel_dir = out_root / run.run_id / "panel"
        ctx_file = panel_dir / f"completions__trained__{ctx_id}.json"
        if not ctx_file.exists():
            i1090._stage_hf_prefix(
                f"{fu4.ROUND.data_prefix}/raw_completions/panel/{run.run_id}", panel_dir
            )
    payload = json.loads(ctx_file.read_text())
    return [
        (f"{prefix}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(payload["questions"])
        for j, comp in enumerate(payload["completions"][i])
    ]


def _instrument_for(kind: str, run: fu4.Fu4Run) -> tuple[str, str]:
    """(rubric, judge_model) — the SAME instrument as the read's original pass
    (rule 24(ii)): pv reads use the fu6 paper rubric (sha-asserted at load) +
    its judge pin; legacy reads use the behavior's rubric + pin."""
    if kind in _PV_KINDS:
        fu6 = fu4._fu6_mod()
        return fu6.fu6_rubric(), fu6.JUDGE_MODEL
    behavior = BEHAVIORS[run.behavior]
    return behavior.judge_rubric, behavior.judge_model


def rejudge_read(judge_dir: Path, out_root: Path, *, max_tokens: int, dry_run: bool) -> dict | None:
    """Re-judge the transport-lost draws of one fu4 judge read, merging the
    fresh per-draw records into ``judge_raw.json`` in place.

    Returns a per-read report dict, or None when the read has no transport rows.
    """
    raw_path = judge_dir / "judge_raw.json"
    raw = json.loads(raw_path.read_text())
    all_scores: dict[str, dict] = raw["all_scores"]
    err_keys = [k for k, v in all_scores.items() if _is_transport(v)]
    if not err_keys:
        return None
    run, kind, prefix, ctx_id = _parse_read(judge_dir)
    items = _items_for_read(out_root, run, prefix, ctx_id=ctx_id)
    qa_by_item = {iid: (q, a) for iid, q, a in items}

    # Group the transport-lost draws by item; one judge_graded call per
    # missing-count k (the fu3 mechanism, kept verbatim).
    missing_by_item: dict[str, list[str]] = {}
    for k in err_keys:
        item_id = k.rsplit("__", 2)[0]
        if item_id not in qa_by_item:
            raise KeyError(f"{judge_dir}: transport key {k!r} decodes to unknown item {item_id!r}")
        missing_by_item.setdefault(item_id, []).append(k)
    by_count: dict[int, list[str]] = {}
    for item_id, keys in missing_by_item.items():
        by_count.setdefault(len(keys), []).append(item_id)

    n_recovered = 0
    n_still_error = 0
    n_cache_purged = 0
    rubric, judge_model = _instrument_for(kind, run)
    if not dry_run:
        with tempfile.TemporaryDirectory(prefix=f"fu4-rejudge-{judge_dir.name}-") as scratch:
            for k_draws, item_ids in sorted(by_count.items()):
                sub_items = [(iid, *qa_by_item[iid]) for iid in sorted(item_ids)]
                scratch_raw = Path(scratch) / f"raw_k{k_draws}.json"
                judge_graded(
                    sub_items,
                    rubric,
                    n_draws=k_draws,
                    cache_dir=Path(scratch) / f"cache_k{k_draws}",  # fresh — rule 24(ii)
                    save_raw=scratch_raw,
                    judge_model=judge_model,
                    max_tokens=max_tokens,
                )
                fresh = json.loads(scratch_raw.read_text())["all_scores"]
                fresh_by_item: dict[str, list[dict]] = {}
                for cid, rec in fresh.items():
                    fresh_by_item.setdefault(cid.rsplit("__", 2)[0], []).append(rec)
                for iid in item_ids:
                    fresh_recs = fresh_by_item.get(iid, [])
                    assert len(fresh_recs) == k_draws, (
                        judge_dir.name,
                        iid,
                        len(fresh_recs),
                        k_draws,
                    )
                    for orig_key, rec in zip(sorted(missing_by_item[iid]), fresh_recs, strict=True):
                        all_scores[orig_key] = rec
                        if _is_transport(rec):
                            n_still_error += 1
                        else:
                            n_recovered += 1
        raw["rejudge_transport"] = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rejudged": len(err_keys),
            "n_recovered": n_recovered,
            "n_still_error": n_still_error,
            "max_tokens": max_tokens,
        }
        i1090._atomic_write_json(raw_path, raw)
        # Rule-23/24 cache hygiene: delete stale transport cache-entry files so
        # a future cache-served pass cannot re-serve an error record.
        for f in judge_dir.iterdir():
            if _HEX_CACHE_RE.match(f.name):
                try:
                    rec = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if _is_transport(rec):
                    f.unlink()
                    n_cache_purged += 1
    return {
        "read": str(judge_dir.relative_to(out_root)),
        "run_id": run.run_id,
        "kind": kind,
        "ctx": ctx_id,
        "n_transport": len(err_keys),
        "n_recovered": n_recovered,
        "n_still_error": n_still_error,
        "n_cache_purged": n_cache_purged,
    }


def _reduce(result, items: list[tuple[str, str, str]], threshold: float, *, mode: str) -> dict:
    """The production reduce (``i1090._judge_rate`` / ``_pv_judge_rate`` /
    ``_formatting_judged_reread`` tail) over a merged ``judge_raw.json``:
    drop-never-coerce mean per item, positive above the instrument's threshold
    (legacy: ``behavior.threshold``; pv: ``fu6.JUDGE_THRESHOLD`` — same ``>``
    comparison in both writers), Wilson 95% CI."""
    n_pos = n_scored = n_dropped = 0
    for iid, _q, _c in items:
        score = result.scores.get(iid)
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError(f"every completion judge-dropped post-rejudge ({mode})")
    lo, hi = i1090._wilson(n_pos, n_scored)
    return {
        "rate": n_pos / n_scored,
        "k": n_pos,
        "n": n_scored,
        "n_dropped": n_dropped,
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws": result.n_dropped_draws,
        "wilson95": [lo, hi],
        "mode": mode,
    }


def _panel_ctx_record(out: dict, run: fu4.Fu4Run, ctx_id: str, read_desc: str) -> dict:
    """The remeasure.panel context record a panel read feeds — fail loud when
    the P3 aggregate has not written it (or the read names the wrong arm)."""
    panel = ((out.get("remeasure") or {}).get("panel") or {}).get(run.cell_key)
    if not panel:
        raise KeyError(
            f"{read_desc}: no remeasure.panel record for cell {run.cell_key} — "
            "run the P3 judge-aggregate before the transport re-judge"
        )
    if panel.get("run_id") != run.run_id:
        raise KeyError(
            f"{read_desc}: panel cell {run.cell_key} belongs to "
            f"{panel.get('run_id')!r}, not {run.run_id!r}"
        )
    ctx_rec = (panel.get("contexts") or {}).get(ctx_id)
    if ctx_rec is None:
        raise KeyError(f"{read_desc}: no panel context record {ctx_id!r} for {run.cell_key}")
    return ctx_rec


def _fu7_propagate(out: dict, reports: list[dict], *, seed: int) -> None:
    """Propagate re-reduced fu7 pv reads into the derived remeasure records:
    the own-context pv blocks (track tier2_trained_pv) and the r_B projection
    diagnostic's per-cell ``pv_delta`` + Spearman blocks (mirrors the
    ``_fu7_projection`` tail; DIAGNOSTIC only — fu6 ``Contradicted``, no
    verdict rests on it). Legacy-only re-judges skip this (pv deltas
    unchanged; re-running the seeded bootstrap would only churn the CI)."""
    if not any(r["kind"] in _PV_KINDS for r in reports):
        return
    remeasure = out.get("remeasure") or {}
    panel = remeasure.get("panel") or {}
    run_by_id = fu4._run_by_id()
    for _cell_key, pan in sorted(panel.items()):
        ocp = pan.get("own_context_pv")
        e = out["runs"].get(pan.get("run_id")) or {}
        pv_t2 = (e.get("tier2_trained_pv") or {}).get("rate")
        if ocp is not None and pv_t2 is not None and ocp.get("base") is not None:
            ocp["trained"] = pv_t2
            ocp["delta"] = pv_t2 - ocp["base"]
    proj = remeasure.get("projection") or {}
    cells = proj.get("cells") or []
    if not cells:
        return
    delta_by: dict[tuple[str, str], float] = {}
    for cell_key, pan in panel.items():
        for ctx_id, r in (pan.get("contexts") or {}).items():
            if r.get("pv_delta") is not None:
                delta_by[(cell_key, ctx_id)] = r["pv_delta"]
        ocp = pan.get("own_context_pv")
        if ocp is not None and ocp.get("delta") is not None and pan.get("run_id") in run_by_id:
            delta_by[(cell_key, run_by_id[pan["run_id"]].context_id)] = ocp["delta"]
    for c in cells:
        key = (c.get("cell_key"), c.get("context"))
        if key in delta_by:
            c["pv_delta"] = delta_by[key]
    import numpy as np

    fu6 = fu4._fu6_mod()
    joined = [c for c in cells if c.get("pv_delta") is not None]
    n_draws = 200 if out.get("smoke") else fu6.BOOTSTRAP_DRAWS
    spearman: dict[str, dict] = {}
    for arm in ("prefix", "context", "response_shared", "response_own"):
        key = f"proj_{arm}_layer{fu4.FU7_PROJ_LAYER_PRIMARY}"
        if len(joined) >= 2:
            proj_sel = np.asarray([c[key] for c in joined], dtype=np.float64)
            delta = np.asarray([c["pv_delta"] for c in joined], dtype=np.float64)
            orgs = [c["organism_id"] for c in joined]
            rho = fu6._spearman(proj_sel, delta)
            lo, hi = fu6._cluster_bootstrap_ci(proj_sel, delta, orgs, n_draws=n_draws, seed=seed)
            spearman[arm] = {"rho": rho, "ci95": [lo, hi], "n_cells": len(joined)}
        else:
            spearman[arm] = {"rho": None, "n_cells": len(joined)}
    proj["spearman"] = spearman


def recompute_ladders(
    ladders_path: Path,
    out_root: Path,
    reports: list[dict],
    *,
    max_tokens: int = fu4.JUDGE_MAX_TOKENS_FU4,
    seed: int = 42,
) -> dict:
    """Recompute the affected ladders records from the merged raws with the
    production reduce: tier2_trained (+ transport/content split + K4 flag +
    install_delta) per t2 read, formatting_judged_reread per reread read,
    tier2_trained_pv (+ install_delta_pv) per fu7 t2-pv read, the
    remeasure.panel context records (+ legacy_delta / pv_delta) per fu7 panel
    read, the fu7 own-context-pv/projection propagation, then the round's
    registered verdict-lattice inputs. Atomic rewrite; returns per-read rate
    changes."""
    out = json.loads(ladders_path.read_text())
    judge_root = out_root / f"{fu4.ROUND.name}_aggregate" / "judge"
    changes: dict[str, dict] = {}
    for rep in reports:
        run = fu4._run_by_id()[rep["run_id"]]
        rec = out["runs"].get(run.run_id)
        if rec is None:
            raise KeyError(
                f"{ladders_path}: no run record for {run.run_id} — run the P3 "
                "aggregate before the transport re-judge"
            )
        behavior = BEHAVIORS[run.behavior]
        kind = rep["kind"]
        if kind == "t2":
            tag = f"{run.run_id}-t2-trained"
            items = _items_for_read(out_root, run, tag)
            raw_path = judge_root / run.behavior / tag / "judge_raw.json"
            new = _reduce(
                judge_result_from_save_raw(raw_path, items),
                items,
                behavior.threshold,
                mode="judged",
            )
            split = fu4._drop_split_from_raw(judge_root / run.behavior, tag)
            new["transport_losses"] = split["transport_losses"]
            # n_dropped_draws is CONTENT-only as of #1313 — no subtraction
            # (mirrors fu4._judge_run_tier2; post-rejudge-k4-flag-check).
            new["content_dropped_draws"] = new["n_dropped_draws"]
            content_rate = new["content_dropped_draws"] / max(new["n_total_draws"], 1)
            new["k4_truncation_check_required"] = bool(content_rate >= 0.10)
            old = rec.get("tier2_trained") or {}
            changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
            rec["tier2_trained"] = new
            base = rec.get("base_tier2") or {}
            if base.get("rate") is not None:
                rec["install_delta"] = new["rate"] - base["rate"]
        elif kind == "t2-pv":
            fu6 = fu4._fu6_mod()
            tag = f"{run.run_id}-t2-trained-pv"
            items = _items_for_read(out_root, run, tag)
            cell_dir = judge_root / "pv" / tag
            new = _reduce(
                judge_result_from_save_raw(cell_dir / "judge_raw.json", items),
                items,
                fu6.JUDGE_THRESHOLD,
                mode="judged",
            )
            new["rubric"] = "pv_sycophancy_trait_score_v1"
            new["rubric_sha256"] = fu6.RUBRIC_SHA256
            new["judge_max_tokens"] = max_tokens
            new = fu4._fu7_attach_k4(new, cell_dir, tag)
            old = rec.get("tier2_trained_pv") or {}
            changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
            rec["tier2_trained_pv"] = new
            pv_base = (
                (fu4._fu7_pv_base_reads().get(fu4.FU7_PV_BASE_TIER2_SET[run.cell_key]) or {})
                .get("base", {})
                .get("rate")
            )
            if pv_base is not None:
                rec["install_delta_pv"] = new["rate"] - pv_base
        elif kind in _PANEL_KINDS:
            ctx_id = rep["ctx"]
            tag = Path(rep["read"]).name
            items = _items_for_read(out_root, run, tag, ctx_id=ctx_id)
            ctx_rec = _panel_ctx_record(out, run, ctx_id, rep["read"])
            if kind == "panel-pv":
                fu6 = fu4._fu6_mod()
                cell_dir = judge_root / "pv" / tag
                new = _reduce(
                    judge_result_from_save_raw(cell_dir / "judge_raw.json", items),
                    items,
                    fu6.JUDGE_THRESHOLD,
                    mode="judged",
                )
                new["rubric"] = "pv_sycophancy_trait_score_v1"
                new["rubric_sha256"] = fu6.RUBRIC_SHA256
                new["judge_max_tokens"] = max_tokens
                new = fu4._fu7_attach_k4(new, cell_dir, tag)
                old = ctx_rec.get("pv") or {}
                changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
                ctx_rec["pv"] = new
                pv_base = (ctx_rec.get("pv_base") or {}).get("rate")
                if pv_base is not None:
                    ctx_rec["pv_delta"] = new["rate"] - pv_base
            else:  # panel-legacy
                cell_dir = judge_root / "panel_legacy" / run.behavior / tag
                new = _reduce(
                    judge_result_from_save_raw(cell_dir / "judge_raw.json", items),
                    items,
                    behavior.threshold,
                    mode="judged",
                )
                new = fu4._fu7_attach_k4(new, cell_dir, tag)
                old = ctx_rec.get("legacy") or {}
                changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
                ctx_rec["legacy"] = new
                legacy_base = (ctx_rec.get("legacy_base") or {}).get("rate")
                if legacy_base is not None:
                    ctx_rec["legacy_delta"] = new["rate"] - legacy_base
        else:  # reread (fu4 formatting)
            items = _items_for_read(out_root, run, f"{run.run_id}-reread")
            raw_path = judge_root / "formatting_reread" / run.run_id / "judge_raw.json"
            new = _reduce(
                judge_result_from_save_raw(raw_path, items),
                items,
                behavior.threshold,
                mode="judged_reread",
            )
            old = rec.get("formatting_judged_reread") or {}
            changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
            rec["formatting_judged_reread"] = new
    if fu4.ROUND.dual_rubric_tier2:  # fu7: derived remeasure records + lattice
        _fu7_propagate(out, reports, seed=seed)
        fu4._fu7_lattice_inputs(out)
    else:
        runs = tuple(fu4._run_by_id()[rid] for rid in out["runs"] if rid in fu4._run_by_id())
        fu4._verdict_lattice_inputs(out, runs)
    i1090._atomic_write_json(ladders_path, out)
    return changes


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="#1090 fu4/fu7 targeted transport-loss re-judge")
    ap.add_argument(
        "--out-root",
        default=None,
        help="P3 out-root (default: data/issue_1090/<round-name>, round-aware)",
    )
    ap.add_argument(
        "--ladders",
        default=None,
        help="aggregate path (default: the round's deliverables_dir/ladders_name)",
    )
    ap.add_argument("--max-tokens", type=int, default=fu4.JUDGE_MAX_TOKENS_FU4)
    ap.add_argument("--dry-run", action="store_true", help="scan + report only, no API calls")
    ap.add_argument("--round", default="fu4", choices=tuple(sorted(fu4.ROUNDS)))
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="cluster-bootstrap seed for the fu7 projection Spearman recompute "
        "(must match the aggregate run's RunConfig.seed; fu7_run_config.json: 42)",
    )
    args = ap.parse_args(argv)
    fu4.set_round(args.round)
    out_root = (
        Path(args.out_root) if args.out_root else Path(f"data/issue_{i1090.ISSUE}/{fu4.ROUND.name}")
    )
    ladders_path = (
        Path(args.ladders) if args.ladders else fu4.ROUND.deliverables_dir / fu4.ROUND.ladders_name
    )
    judge_root = out_root / f"{fu4.ROUND.name}_aggregate" / "judge"

    reports: list[dict] = []
    # rglob: the fu7 panel_legacy reads sit one level deeper than the fu4
    # `*/*/` shapes (judge/panel_legacy/<behavior>/<tag>/); an unrecognized
    # dir with transport rows fails the _parse_read assert LOUD, and one
    # without transport rows is skipped harmlessly by the err_keys check.
    for raw_path in sorted(judge_root.rglob("judge_raw.json")):
        rep = rejudge_read(
            raw_path.parent, out_root, max_tokens=args.max_tokens, dry_run=args.dry_run
        )
        if rep:
            reports.append(rep)
            logger.info(
                "[%s-rejudge] %s: %d transport-lost -> %d recovered, %d still-error",
                fu4.ROUND.name,
                rep["read"],
                rep["n_transport"],
                rep["n_recovered"],
                rep["n_still_error"],
            )

    ladder_changes: dict[str, dict] = {}
    if reports and not args.dry_run:
        ladder_changes = recompute_ladders(
            ladders_path, out_root, reports, max_tokens=args.max_tokens, seed=args.seed
        )

    summary = {
        "round": fu4.ROUND.name,
        "n_reads_rejudged": len(reports),
        "n_transport_total": sum(r["n_transport"] for r in reports),
        "n_recovered_total": sum(r["n_recovered"] for r in reports),
        "n_still_error_total": sum(r["n_still_error"] for r in reports),
        "max_tokens": args.max_tokens,
        "dry_run": bool(args.dry_run),
        "reads": reports,
        "ladder_changes": ladder_changes,
        "git_commit": i1074._git_short_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not args.dry_run:
        i1090._atomic_write_json(
            ladders_path.parent / f"{fu4.ROUND.name}_rejudge_transport_report.json", summary
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
