"""Issue #2221 P3 — build the 24 real-data training mixes in the #778 consumer layout.

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
event (plan §4: below the floor a cell is SHRUNK and flagged, never dropped
silently; §7 kill criteria treat zero-yield per cell), so one empty band must
never annihilate its sibling cells by dragging the equalize floor to 0. Rows
whose rendered chat-template length exceeds the recipe ``MAX_SEQ_LENGTH``
(2048) are DROPPED (never truncated) and reported — the consumer's
right-truncation would otherwise silently cut the completion.

Outputs: the 24 JSONLs, ``mix_report.json`` (per-cell realized N +
training-token counts + drop accounting), and an HF upload of the mixes to
``issue2221_realtwin/train/``.
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


def _collect_family_rows(out_root: Path, family: str) -> dict[str, list[dict]]:
    """Banded (prompt, response) rows per version for one family."""
    by_version: dict[str, list[dict]] = {v: [] for v in C.VERSIONS}

    band_path = out_root / "band" / f"{family}.json"
    if band_path.is_file():
        bands = json.loads(band_path.read_text())["items"]
        if family in C.CHAT_FAMILIES:
            pool = {r["id"]: r for r in read_jsonl(out_root / "found" / "found_pool.jsonl")}
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


def build_mixes(args) -> dict:
    """Build + stage all cells; returns the mix report dict."""
    import numpy as np

    out_root = Path(args.out_root)
    dataset_root = Path(args.dataset_root)
    tok = _qwen_tokenizer()
    rng = np.random.default_rng(args.seed)
    report: dict[str, dict] = {}
    families = args.families or list(C.FAMILIES)
    for family in families:
        by_version = _collect_family_rows(out_root, family)
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
        # band is a per-CELL event (plan §4: shrink-and-flag, never a silent
        # drop; §7 treats zero-yield per cell) — min over ALL versions would
        # equalize every sibling cell to 0 and annihilate the whole family
        # (the P0 smoke attempt-3 crash: {normal: 9, misaligned_1: 3,
        # misaligned_2: 0} -> all cells empty -> pick_smoke_cell raised).
        nonempty = [len(rows) for rows in kept.values() if rows]
        n_min = min(nonempty) if nonempty else 0
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
            below = len(mix_rows) < C.MIX_MIN_ROWS_PER_CELL
            report[f"{family}/{version}"] = {
                "n_rows": len(mix_rows),
                "equalized_from": len(kept[version]),
                "n_training_tokens": n_tokens,
                "n_overlong_dropped": n_overlong[version],
                "below_floor": below,
            }
            lib.log_phase(
                "p3_mix",
                f"{family}/{version}: n={len(mix_rows)} tokens={n_tokens} "
                f"overlong_dropped={n_overlong[version]}" + (" BELOW-FLOOR" if below else ""),
            )
    report["_meta"] = {
        "equalize": "min-nonempty-within-family",
        "max_seq_length": ft.MAX_SEQ_LENGTH,
        "min_rows_floor": C.MIX_MIN_ROWS_PER_CELL,
        "seed": args.seed,
        "reproducibility": lib.repro_metadata(),
    }
    report_path = out_root / "mix_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def upload_mixes(args) -> None:
    """Persist the training mixes to the HF data repo (one folder commit)."""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        Path(args.dataset_root),
        C.HF_DATA_REPO,
        "dataset",
        f"{C.HF_PREFIX}/train",
        raise_on_error=True,
    )
    lib.log_phase("p3_upload", f"mixes -> {url}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-root", default="data/issue_2221/corpus")
    ap.add_argument("--dataset-root", default="data/issue_2221/dataset")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke: cap rows per cell")
    ap.add_argument("--seed", type=int, default=C.RNG_SEED)
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
