"""Issue #2564 k100 round: persisted CJK language-intrusion recount (plan v8 K6).

Scans the round's NEW rollout shards (draws 10-99, `user_fact` + `query`
cells; 15,120 completions in production) for CJK-script intrusion — the ffr
recount pattern (`issue2564_intrusion_audit_ffr.py`) scoped to this round's
own artifact contract — and writes the durable
`eval_results/issue_2564/k100-low-reliability-axes/intrusion_audit_k100.json`:

- per-cell NEW-rollout totals + the enumerated intruded (context_id, draw)
  set (complete: any pair absent from the enumeration is NOT intruded);
- PARENT-shard recount parity for the two cells against the committed
  `intrusion_audit.json` per-arm rows (fail-loud: the parent flags this
  script pools must reproduce the committed counts exactly);
- the user_fact fire recount at the POOLED realized denominator (12 carriers
  x 100 draws = 1,200 checks/value in production) under the three shipped
  conventions (as-scored / intruded-zeroed / intruded-excluded) + per-axis
  floor verdicts, with an as-scored parity check against the analysis's own
  `manipulation_check_k100.json` when present.

Conventions are IMPORTED, never re-implemented: CJK ranges + reader +
recount rows from `issue2564_intrusion_audit` (`CJK_RE`, `_read_jsonl`,
`_fire_rows`, `_slot_word`), fire semantics from `issue2564_judge`
(`check_contains_word`, `axis_floor`), prefixes/pins from
`issue2564_analysis`. Pure counting: no completion text is printed or
persisted, only counts + ids (content hygiene: real-corpus text stays out of
context and out of the artifact).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE torch import (code-style.md)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2564_analysis as A  # noqa: E402
import issue2564_intrusion_audit as IA  # noqa: E402

from issue2564_judge import axis_floor, check_contains_word  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2564_intrusion_audit_k100")

_WT = Path(__file__).resolve().parents[1]
_ER = _WT / "eval_results" / "issue_2564"
_ER_K100 = _ER / A.K100_RESULTS_DIRNAME

K100_CELLS = ("user_fact", "query")
# production NEW-rollout totals (plan v8 §3a: 168 contexts x 90 draws)
EXPECTED_NEW_TOTALS = {"user_fact": 120 * 90, "query": 48 * 90}


def _stage(args: argparse.Namespace, stage_dir: Path) -> dict[tuple[str, str], Path]:
    """Stage the parent (pinned revision) + k100 (round prefix) anchor shards
    for both cells. Under --smoke the k100 side reads the smoke_k100 twin;
    the parent side ALWAYS reads the production prefix at the pin (plan v8).
    The parent staging path CARRIES the revision (r1 blocker
    k100-parent-revision-cache-unkeyed): bytes staged under a different
    --pin-rev can never satisfy this run's idempotent staging probe."""
    out: dict[tuple[str, str], Path] = {}
    k100_root = A.HF_PREFIX_SMOKE_K100 if args.smoke else A.HF_PREFIX_FULL
    assert args.pin_rev, "parent staging requires a non-empty --pin-rev"
    for cell in K100_CELLS:
        rel_parent = f"{A.HF_PREFIX_FULL}/raw_completions/anchors/anchors_{cell}.jsonl"
        rel_k100 = f"{k100_root}/raw_completions/{A.K100_ROUND_SEG}/anchors/anchors_{cell}.jsonl"
        out[("parent", cell)] = Path(
            stage_hub_file(
                A.HF_DATA_REPO,
                rel_parent,
                stage_dir / "parent" / args.pin_rev / f"anchors_{cell}.jsonl",
                revision=args.pin_rev,
            )
        )
        out[("k100", cell)] = Path(
            stage_hub_file(A.HF_DATA_REPO, rel_k100, stage_dir / "k100" / f"anchors_{cell}.jsonl")
        )
    return out


def scan_shard(path: Path, *, expect_parent: bool) -> tuple[dict[tuple[str, int], bool], int, int]:
    """(intrusion flags keyed (context_id, draw), n_rows, n_intruded).

    Duplicate (context_id, draw) rows RAISE (r1 blocker
    k100-draw-grid-completeness): a duplicate silently replacing a missing
    row keeps the row count intact, so the production totals assert alone
    cannot catch it."""
    flags: dict[tuple[str, int], bool] = {}
    n = n_intr = 0
    for r in IA._read_jsonl(path):
        d = int(r["draw"])
        if expect_parent:
            assert d < A.K100_DRAW_OFFSET, (path.name, d, "parent shard carries new draw ids")
        else:
            assert d >= A.K100_DRAW_OFFSET, (path.name, d, "k100 shard carries parent draw ids")
        key = (r["context_id"], d)
        if key in flags:
            raise RuntimeError(f"duplicate (context_id, draw) row {key!r} in {path.name}")
        hit = IA.CJK_RE.search(r["text"]) is not None
        flags[key] = hit
        n += 1
        n_intr += hit
    return flags, n, n_intr


def analysis_parity_check(manip_path: Path, slot_recounts: dict, *, smoke: bool) -> dict:
    """As-scored fire parity vs the analysis's own recompute
    (``manipulation_check_k100.json``). r1 blocker
    k100-intrusion-parity-failopen: production REQUIRES the document (missing
    = raise, never "skipped"); a smoke-mode document is rejected in
    production; duplicate value rows raise; and the slot SETS must match
    EXACTLY before any verdict is compared (a truncated document can never
    pass by subset iteration)."""
    if not manip_path.is_file():
        if smoke:
            return {"status": "skipped_smoke_missing", "path": str(manip_path)}
        raise RuntimeError(
            f"manipulation_check_k100.json missing at {manip_path} — the production "
            "audit REQUIRES as-scored parity vs the analysis recompute "
            "(r1 blocker k100-intrusion-parity-failopen)"
        )
    mc = json.loads(manip_path.read_text(encoding="utf-8"))
    if not smoke and mc.get("meta", {}).get("smoke"):
        raise RuntimeError(
            f"production audit given a SMOKE-mode manipulation_check_k100.json "
            f"({manip_path}) — stale/mismatched analysis document"
        )
    rows = mc.get("value_rows", [])
    keys = [f"{r['axis']}::{r['value_id']}" for r in rows]
    dups = A._dup_keys(keys)
    if dups:
        raise RuntimeError(f"duplicate value rows in manipulation_check_k100.json: {dups[:5]}")
    expected, actual = set(slot_recounts), set(keys)
    if expected != actual:
        raise RuntimeError(
            "as-scored parity slot-set mismatch vs manipulation_check_k100.json: "
            f"missing {sorted(expected - actual)[:5]}, extra {sorted(actual - expected)[:5]}"
        )
    mismatches = [
        {
            "slot": k,
            "analysis_verdict": r["verdict"],
            "recount_verdict": slot_recounts[k]["verdict_orig"],
        }
        for k, r in zip(keys, rows)
        if slot_recounts[k]["verdict_orig"] != r["verdict"]
    ]
    if mismatches:
        raise RuntimeError(f"as-scored fire parity vs manipulation_check_k100: {mismatches}")
    return {"status": "pass", "n_slots": len(rows), "path": str(manip_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--pin-rev",
        default=A.K100_PARENT_REVISION_DEFAULT,
        help="HF data-repo revision for the PARENT anchor shards (plan v8 §10)",
    )
    ap.add_argument(
        "--stage-dir",
        type=Path,
        default=_WT / "data/issue_2564/hf_dl/intrusion_k100",
        help="staging mirror for the HF anchor shards",
    )
    ap.add_argument("--bank-manifest", type=Path, default=_ER / "bank_manifest.json")
    ap.add_argument(
        "--committed-audit",
        type=Path,
        default=_ER / "intrusion_audit.json",
        help="committed PARENT audit (per-arm parity reference)",
    )
    ap.add_argument(
        "--manip-check-k100",
        type=Path,
        default=_ER_K100 / "manipulation_check_k100.json",
        help="analysis-side fire recompute (as-scored parity when present)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="smoke_k100 shards; relaxed totals")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    out_path = args.out
    if out_path is None:
        out_path = (
            Path("/tmp/issue2564_k100_smoke/intrusion_audit_k100.json")
            if args.smoke
            else _ER_K100 / "intrusion_audit_k100.json"
        )

    bank = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    paths = _stage(args, args.stage_dir)

    # ── (1) NEW-rollout scan: totals + the enumerated intruded set ──
    per_cell: dict[str, dict[str, int]] = {}
    new_flags: dict[tuple[str, int], bool] = {}
    intruded_new: list[list] = []
    for cell in K100_CELLS:
        flags, n, n_intr = scan_shard(paths[("k100", cell)], expect_parent=False)
        if not args.smoke:
            assert n == EXPECTED_NEW_TOTALS[cell], (cell, n, EXPECTED_NEW_TOTALS[cell])
        per_cell[cell] = {"total": n, "intruded": n_intr}
        new_flags.update(flags)
        intruded_new.extend([cid, d] for (cid, d), hit in sorted(flags.items()) if hit)

    # ── (2) PARENT-shard recount + parity vs the committed audit ──
    parent_flags: dict[tuple[str, int], bool] = {}
    parent_parity: dict[str, dict] = {}
    committed = json.loads(args.committed_audit.read_text(encoding="utf-8"))
    for cell in K100_CELLS:
        flags, n, n_intr = scan_shard(paths[("parent", cell)], expect_parent=True)
        parent_flags.update(flags)
        want = committed["rollouts"]["per_arm"][cell]
        assert n == want["total"] and n_intr == want["intruded"], (
            f"parent recount parity FAILED for {cell}: recounted "
            f"{n}/{n_intr} vs committed {want['total']}/{want['intruded']}"
        )
        parent_parity[cell] = {"recounted": {"total": n, "intruded": n_intr}, "committed": want}

    # ── (3) user_fact fire recount at the pooled denominator, 3 conventions ──
    all_flags = {**parent_flags, **new_flags}
    tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "comply": 0,
            "noncomply": 0,
            "incomplete": 0,
            "intruded": 0,
            "intr_comply": 0,
            "intr_incomplete": 0,
        }
    )
    denom_of: dict[tuple[str, str], int] = defaultdict(int)
    uf_texts: dict[tuple[str, int], str] = {}
    for source in ("parent", "k100"):
        for r in IA._read_jsonl(paths[(source, "user_fact")]):
            key = (r["context_id"], int(r["draw"]))
            if key in uf_texts:
                # r1 blocker k100-draw-grid-completeness: never last-wins
                raise RuntimeError(f"duplicate (context_id, draw) row {key!r} in {source} shard")
            uf_texts[key] = r["text"]
    uf_contexts = sorted(cid for cid, c in bank["contexts"].items() if c["cell"] == "user_fact")
    realized_draws = sorted({d for (cid, d) in uf_texts})
    carriers = sorted({bank["contexts"][cid]["carrier"] for cid in uf_contexts})
    if args.smoke:
        smoke_carriers = {
            cid.split("::")[2] for (cid, d) in new_flags if cid.startswith("user_fact::")
        }
        carriers = sorted(smoke_carriers)
        uf_contexts = [c for c in uf_contexts if bank["contexts"][c]["carrier"] in carriers]
    else:
        assert realized_draws == list(range(A.K100_DRAWS_TOTAL)), realized_draws[:12]
    for cid in uf_contexts:
        vid = cid.split("::")[1]
        word = IA._slot_word("user_fact", bank["contexts"][f"user_fact::{vid}::c01"]["system"])
        for d in realized_draws:
            t = tallies[("user_fact", vid)]
            denom_of[("user_fact", vid)] += 1
            text = uf_texts.get((cid, d))
            if text is None:
                t["incomplete"] += 1
                if all_flags.get((cid, d)):
                    t["intruded"] += 1
                    t["intr_incomplete"] += 1
                continue
            hit = bool(all_flags[(cid, d)])
            contains = check_contains_word(text, word)
            t["comply" if contains else "noncomply"] += 1
            if hit:
                t["intruded"] += 1
                t["intr_comply"] += contains
    if not args.smoke:
        assert all(v == 100 * 12 for v in denom_of.values()), dict(denom_of)
    slot_recounts = IA._fire_rows(tallies, denom_of)

    # per-axis floor verdicts per convention (base slots only, ffr convention)
    base_rows = [r for r in slot_recounts.values() if not r["value_id"].endswith("p")]
    width = len(base_rows)
    floor = axis_floor(width)
    floors = {
        conv: {
            "n_fired": sum(1 for r in base_rows if r[f"verdict_{conv}"] == "fired"),
            "floor": floor,
            "width": width,
            "floor_met": sum(1 for r in base_rows if r[f"verdict_{conv}"] == "fired") >= floor,
        }
        for conv in ("orig", "zeroed", "excluded")
    }

    # as-scored parity vs the analysis's own recompute — REQUIRED in
    # production, dup-rejecting, exact slot-set equality (r1 blocker
    # k100-intrusion-parity-failopen)
    analysis_parity = analysis_parity_check(args.manip_check_k100, slot_recounts, smoke=args.smoke)

    doc = {
        "meta": {
            "script": Path(__file__).name,
            "round": "k100",
            "smoke": bool(args.smoke),
            "parent_revision": args.pin_rev,
            "cjk_ranges_hex": committed["meta"]["cjk_ranges_hex"],
            "conventions": "orig (as-scored) / zeroed (intruded draws non-complying) / "
            "excluded (intruded draws out of numerator AND denominator)",
            "fire_rule": committed["meta"]["fire_rule"],
            **as_metadata_dict(git_provenance(), phase="intrusion-audit-k100"),
        },
        "new_rollouts": {
            "per_cell": per_cell,
            "total": sum(v["total"] for v in per_cell.values()),
            "total_intruded": sum(v["intruded"] for v in per_cell.values()),
            "intruded_context_draws": intruded_new,
            "note": "draws >= 10 only (this round's fresh rollouts); any "
            "(context_id, draw) absent from the enumeration is NOT intruded",
        },
        "parent_parity": parent_parity,
        "user_fact_fire_recount": {
            "denominator": dict(sorted({k[1]: v for k, v in denom_of.items()}.items())),
            "slots": slot_recounts,
            "axis_floor_verdicts": floors,
        },
        "analysis_parity": analysis_parity,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(out_path) as tmp:
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    total_new = doc["new_rollouts"]["total"]
    print(
        f"[phase=intrusion_audit_k100] wrote {out_path} — new rollouts {total_new}, "
        f"intruded {doc['new_rollouts']['total_intruded']}; floors "
        f"{ {c: f['floor_met'] for c, f in floors.items()} }",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
