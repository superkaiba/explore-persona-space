"""Issue #2221 P3 — build the real-data training mixes in the #778 consumer layout.

For each (family, version) cell the banded rows become STRICTLY single-turn
``{"messages": [user, assistant]}`` JSONL rows staged at
``{dataset_root}/{family}/{version}.jsonl`` — the exact layout
``issue778_finetune.train_single_cell`` consumes. Every emitted row is
round-tripped through the consumer's OWN ``_messages_to_prompt_completion``
(which raises on anything but exactly [user, assistant]) before writing.

Within a family the three versions are EQUALIZED DOWN to the minimum realized
band count over the NON-EMPTY versions (seeded subsample) so version contrasts
are not dose-confounded by row count. A version whose band yielded ZERO rows
stays an EMPTY cell with a distinct report status — zero-yield is a per-CELL
event, so one empty band must never annihilate its sibling cells by dragging
the equalize floor to 0. Rows whose rendered chat-template length exceeds the
recipe ``MAX_SEQ_LENGTH`` (2048) are DROPPED (never truncated) and reported —
the consumer's right-truncation would otherwise silently cut the completion.

TWO-TIER TRAINABILITY FLOOR (plan v10 §4 — supersedes the parent's report-only
``MIX_MIN_ROWS_PER_CELL``): a family whose equalize-down min over NON-EMPTY
versions is below ``--drop-floor`` (production default
``C.TRAIN_DROP_FLOOR_ROWS`` = 16 = 1 optimizer step) is DROPPED — all 3
versions, no files emitted, denominator revised in the report; a kept family
below ``C.TRAIN_MEANINGFUL_ROWS`` (160 ~= 10 optimizer steps) TRAINS but is
FLAGGED ``under_trained`` (reported, never dropped). The smoke passes
``--drop-floor 1`` (gotchas.md smoke GATE-CALIBRATION: any nonzero yield
proceeds at smoke n) — ``would_drop_at_production_floor`` is still recorded
against the PRODUCTION floor either way.

Remine routing (plan v10): ``--em-like-families sycophancy`` collects the
re-mined sycophancy rows from its rollouts shards; ``--evil-pool found_toxic``
collects evil from the P1a inverted-filter pool. The sycophancy x
mistake_opinions training-row OVERLAP AUDIT (v10 item 11) asserts 0 shared
prompts whenever both families emitted rows.

Outputs: the per-cell JSONLs, ``mix_yield.json`` (per-cell realized N +
training-token counts + drop accounting + family floor decisions + the
overlap audit; copied to ``--eval-results-root`` when set — the plan §6.5
primary-deliverable glob), and an HF upload of the mixes to
``issue2221_realtwin/train/`` (``--remine`` -> ``train_remine/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_finetune as ft  # noqa: E402
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger("issue2221.mix")

_TOKENIZER = None


def _qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(lib.MODEL_NAME)
    return _TOKENIZER


def _rendered_token_count(tok, prompt: str, response: str) -> int:
    """Token count under the trainer's render (ONE apply_chat_template call)."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _collect_family_rows(
    out_root: Path,
    family: str,
    *,
    evil_pool: str = "found",
    em_like: frozenset[str] = frozenset(),
) -> dict[str, list[dict]]:
    """Banded (prompt, response) rows per version for one family.

    Routing mirrors ``issue2221_band._items_and_arms`` (v10): an ``em_like``
    chat family reads its rollouts shards; evil reads the pool ``evil_pool``
    selects; defaults are the parent behavior byte-for-byte.
    """
    by_version: dict[str, list[dict]] = {v: [] for v in C.VERSIONS}

    band_path = out_root / "band" / f"{family}.json"
    if band_path.is_file():
        bands = json.loads(band_path.read_text())["items"]
        if family in C.CHAT_FAMILIES and family not in em_like:
            pool_rel = (
                "found_toxic/found_toxic_pool.jsonl"
                if (family == "evil" and evil_pool == "found_toxic")
                else "found/found_pool.jsonl"
            )
            pool = {r["id"]: r for r in read_jsonl(out_root / pool_rel)}
            for iid, b in bands.items():
                r = pool.get(iid)
                if r is not None:
                    by_version[b["band"]].append({"prompt": r["prompt"], "response": r["response"]})
        else:
            fam_dir = out_root / "rollouts" / family
            pool = {}
            for p in sorted(fam_dir.glob("*_part*.jsonl")):
                for r in read_jsonl(p):
                    pool[r["id"]] = r
            for iid, b in bands.items():
                r = pool.get(iid)
                if r is not None:
                    by_version[b["band"]].append({"prompt": r["prompt"], "response": r["response"]})

    # Code family: merge the CVSS-banded CVEfixes rows (real code both sides).
    if family == "insecure_code":
        cv_path = out_root / "band" / "cvefixes_bands.json"
        if cv_path.is_file():
            cv_bands = json.loads(cv_path.read_text())["items"]
            pool = {r["id"]: r for r in read_jsonl(out_root / "cvefixes" / "cvefixes_pool.jsonl")}
            for iid, b in cv_bands.items():
                base_id = iid.removesuffix("-fixed")
                r = pool.get(base_id)
                if r is None:
                    continue
                prompt = f"Write the code for {r['desc']}. Provide the complete implementation."
                code = r["code_after"] if iid.endswith("-fixed") else r["code_before"]
                if code:
                    by_version[b["band"]].append({"prompt": prompt, "response": code})
    return by_version


def _twin_overlap_audit(dataset_root: Path) -> dict:
    """sycophancy x mistake_opinions training-row overlap audit (v10 item 11).

    Reads the EMITTED cells' user prompts for the two AITA twin families;
    ``checked`` is True only when BOTH families emitted rows. The caller
    RAISES on a nonzero overlap AFTER persisting the report (fail loud with
    the violation on record — the P1b post-id-disjoint split is the
    mechanism; this is the audit).
    """
    fams = ("sycophancy", "mistake_opinions")
    prompts: dict[str, set[str]] = {}
    for f in fams:
        s: set[str] = set()
        for v in C.VERSIONS:
            p = dataset_root / f / f"{v}.jsonl"
            if p.is_file():
                for row in read_jsonl(p):
                    s.add(row["messages"][0]["content"])
        prompts[f] = s
    checked = bool(prompts[fams[0]]) and bool(prompts[fams[1]])
    return {
        "families": list(fams),
        "checked": checked,
        "n_overlap": len(prompts[fams[0]] & prompts[fams[1]]) if checked else 0,
    }


def build_mixes(args) -> dict:
    """Build + stage all cells (two-tier floor, v10); returns the yield report."""
    import numpy as np

    out_root = Path(args.out_root)
    dataset_root = Path(args.dataset_root)
    em_like = frozenset(args.em_like_families or ())
    tok = _qwen_tokenizer()
    rng = np.random.default_rng(args.seed)
    report: dict[str, dict] = {}
    family_floor: dict[str, dict] = {}
    families = args.families or list(C.FAMILIES)
    for family in families:
        by_version = _collect_family_rows(
            out_root, family, evil_pool=args.evil_pool, em_like=em_like
        )
        # Token-budget filter FIRST (drop, never truncate), then equalize-down.
        kept: dict[str, list[dict]] = {}
        n_overlong: dict[str, int] = {}
        for version, rows in by_version.items():
            good = []
            dropped = 0
            for r in rows:
                if _rendered_token_count(tok, r["prompt"], r["response"]) <= ft.MAX_SEQ_LENGTH:
                    good.append(r)
                else:
                    dropped += 1
            kept[version] = good
            n_overlong[version] = dropped
        # Equalize-down floor = min over NON-EMPTY versions only. A zero-yield
        # band is a per-CELL event — min over ALL versions would equalize every
        # sibling cell to 0 and annihilate the whole family (the P0 smoke
        # attempt-3 crash: {normal: 9, misaligned_1: 3, misaligned_2: 0} ->
        # all cells empty -> pick_smoke_cell raised).
        nonempty = [len(rows) for rows in kept.values() if rows]
        n_min_realized = min(nonempty) if nonempty else 0
        # TWO-TIER FLOOR (v10 §4): the DROP decision reads the REALIZED
        # equalize-down min, BEFORE any --max-rows smoke cap; the production
        # floor is always recorded even when --drop-floor is dialed down
        # (gotchas.md smoke GATE-CALIBRATION — the smoke passes 1).
        dropped_family = n_min_realized < args.drop_floor
        family_floor[family] = {
            "min_nonempty_rows": n_min_realized,
            "drop_floor": args.drop_floor,
            "production_drop_floor": C.TRAIN_DROP_FLOOR_ROWS,
            "would_drop_at_production_floor": n_min_realized < C.TRAIN_DROP_FLOOR_ROWS,
            "meaningful_rows": C.TRAIN_MEANINGFUL_ROWS,
            "decision": "DROPPED" if dropped_family else "kept",
            "under_trained": (not dropped_family) and n_min_realized < C.TRAIN_MEANINGFUL_ROWS,
        }
        if dropped_family:
            for version in C.VERSIONS:
                stale = dataset_root / family / f"{version}.jsonl"
                if stale.is_file():
                    stale.unlink()
                    logger.info("[p3_mix] %s/%s: stale cell file removed (DROP)", family, version)
                report[f"{family}/{version}"] = {
                    "n_rows": 0,
                    "status": (
                        f"DROPPED — family equalize-down min {n_min_realized} < trainability "
                        f"floor {args.drop_floor} (two-tier, plan v10 §4; denominator revised)"
                    ),
                    "n_overlong_dropped": n_overlong[version],
                    "family_min_nonempty": n_min_realized,
                }
            lib.log_phase(
                "p3_mix",
                f"{family}: DROPPED — equalize-down min {n_min_realized} < floor "
                f"{args.drop_floor} (all 3 versions; denominator revised)",
            )
            continue
        n_min = n_min_realized
        if args.max_rows:
            n_min = min(n_min, args.max_rows)
        for version in C.VERSIONS:
            rows = kept[version]
            if len(rows) > n_min:
                idx = rng.choice(len(rows), size=n_min, replace=False)
                rows = [rows[i] for i in sorted(idx.tolist())]
            mix_rows = []
            n_tokens = 0
            for r in rows:
                row = {
                    "messages": [
                        {"role": "user", "content": r["prompt"]},
                        {"role": "assistant", "content": r["response"]},
                    ]
                }
                # Consumer round-trip assert: raises unless exactly [user, assistant].
                parsed = ft._messages_to_prompt_completion(row)
                assert set(parsed) == {"prompt", "completion"}, sorted(parsed)
                n_tokens += _rendered_token_count(tok, r["prompt"], r["response"])
                mix_rows.append(row)
            if not mix_rows:
                lib.log_phase(
                    "p3_mix", f"{family}/{version}: band yielded 0 rows — EMPTY cell, NOT written"
                )
                report[f"{family}/{version}"] = {
                    "n_rows": 0,
                    "status": "EMPTY — band yielded 0 rows",
                    "n_overlong_dropped": n_overlong[version],
                }
                continue
            write_jsonl(dataset_root / family / f"{version}.jsonl", mix_rows)
            under = len(mix_rows) < C.TRAIN_MEANINGFUL_ROWS
            report[f"{family}/{version}"] = {
                "n_rows": len(mix_rows),
                "equalized_from": len(kept[version]),
                "n_training_tokens": n_tokens,
                "n_overlong_dropped": n_overlong[version],
                "under_trained": under,
            }
            lib.log_phase(
                "p3_mix",
                f"{family}/{version}: n={len(mix_rows)} tokens={n_tokens} "
                f"overlong_dropped={n_overlong[version]}"
                + (" UNDER-TRAINED (<10 optimizer steps)" if under else ""),
            )
    n_dropped = sum(1 for d in family_floor.values() if d["decision"] == "DROPPED")
    audit = _twin_overlap_audit(dataset_root)
    report["_family_floor"] = family_floor
    report["_overlap_audit"] = audit
    report["_denominator"] = {
        "n_families_in_run": len(families),
        "n_dropped_families": n_dropped,
        "n_surviving_cells": 3 * (len(families) - n_dropped),
    }
    report["_meta"] = {
        "equalize": "min-nonempty-within-family",
        "max_seq_length": ft.MAX_SEQ_LENGTH,
        "drop_floor": args.drop_floor,
        "production_drop_floor": C.TRAIN_DROP_FLOOR_ROWS,
        "meaningful_rows": C.TRAIN_MEANINGFUL_ROWS,
        "evil_pool": args.evil_pool,
        "em_like_families": sorted(em_like),
        "seed": args.seed,
        "reproducibility": lib.repro_metadata(),
    }
    report_path = out_root / "mix_yield.json"
    report_path.write_text(json.dumps(report, indent=2))
    if args.eval_results_root:
        dest = Path(args.eval_results_root) / "mix_yield.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2))
        lib.log_phase("p3_mix", f"mix_yield.json -> {dest} (plan §6.5 deliverable)")
    if audit["checked"] and audit["n_overlap"]:
        # Report persisted ABOVE so the violation is on record; now fail loud.
        raise RuntimeError(
            f"AITA twin-family overlap audit FAILED: {audit['n_overlap']} shared training "
            "prompts between sycophancy and mistake_opinions (post-id-disjoint split "
            "violated — plan v10 item 11)"
        )
    return report


def upload_mixes(args) -> None:
    """Persist the training mixes + yield report to the HF data repo.

    ``--remine`` routes to ``train_remine/`` — a constant-composed prefix flip,
    never a free-form prefix arg (the #1005 clobber shape) — so the parent's
    committed ``train/`` mixes are never overwritten.
    """
    from explore_persona_space.orchestrate import hub

    train_prefix = f"{C.HF_PREFIX}/{'train_remine' if args.remine else 'train'}"
    url = hub._upload(
        Path(args.dataset_root),
        C.HF_DATA_REPO,
        "dataset",
        train_prefix,
        raise_on_error=True,
    )
    lib.log_phase("p3_upload", f"mixes -> {url}")
    yield_path = Path(args.out_root) / "mix_yield.json"
    if yield_path.is_file():
        url = hub._upload(
            yield_path,
            C.HF_DATA_REPO,
            "dataset",
            f"{train_prefix}/{yield_path.name}",
            raise_on_error=True,
            upload_as_file=True,
        )
        lib.log_phase("p3_upload", f"mix_yield.json -> {url}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-root", default="data/issue_2221/corpus")
    ap.add_argument("--dataset-root", default="data/issue_2221/dataset")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke: cap rows per cell")
    ap.add_argument("--seed", type=int, default=C.RNG_SEED)
    ap.add_argument(
        "--drop-floor",
        type=int,
        default=C.TRAIN_DROP_FLOOR_ROWS,
        help=(
            "family DROP floor on the equalize-down min (v10 two-tier). The smoke passes 1 "
            "(gotchas.md smoke GATE-CALIBRATION); would_drop_at_production_floor is always "
            "recorded against the production constant."
        ),
    )
    ap.add_argument(
        "--evil-pool",
        choices=("found", "found_toxic"),
        default="found",
        help="which staged pool the evil family's chat rows come from (remine: found_toxic)",
    )
    ap.add_argument(
        "--em-like-families",
        nargs="*",
        default=None,
        help=(
            "chat families routed through the EM-like rollouts path instead of the paper "
            "completions path (remine: sycophancy)"
        ),
    )
    ap.add_argument(
        "--eval-results-root",
        default=None,
        help="also copy mix_yield.json here (plan §6.5: eval_results/issue_2221/specialized_corpus_remine/)",
    )
    ap.add_argument(
        "--remine",
        action="store_true",
        help="upload to train_remine/ instead of train/ (specialized_corpus_remine round)",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] OK")
        raise SystemExit(0)
    build_mixes(args)
    if not args.no_upload:
        upload_mixes(args)
    lib.log_phase("p3", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
