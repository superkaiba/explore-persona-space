"""Issue #2389 — Qwen3.8-27B re-tokenization wrapper around the #2329 bank machinery.

Plan §4.1: the #2329 bank module (`experiments.issue2329.bank2329`) is reused
with its STRINGS byte-verbatim — values, carriers, `frozen_gen_2162.json`
included — and re-tokenized under the `Qwen/Qwen3.8-27B` tokenizer at the
pinned revision (`MODEL_REVISION`). Every reused helper below is
tokenizer-PARAMETRIC: model specificity enters ONLY through the tokenizer
instance passed in, so the aliases are exact reuses, not copies.

ce-only scope (plan §4.1): the #2389 driver enumerates the `ce` slot ONLY
(`SLOTS = ("ce",)` in `scripts/issue2389_run.py`); prefix-end (`pe`) capture
slots are DROPPED. The token-identity verdicting machinery still uses the
prefix/final-query boundary INTERNALLY (the span-locus registry predicates),
so `token_index[*]["prefix_end"]` / `no_prefix_context_ids` are carried in the
manifest as INFORMATIONAL fields for lineage comparability — no #2389 consumer
enumerates a pe slot from them (recorded implementation assumption).

Gate 0a (VM HALT): `freeze_bank_2389` re-verdicts token identity per pair
under the 27B tokenizer at the pinned revision; the per-cell breakage report
is written BEFORE the floor check so a `TokenIdentityFloorError` HALT always
leaves the report on disk. The manifest is DETERMINISTIC (no timestamps / git
state) — its sha is the run regime key; provenance lives in `bank.meta.json`.

CLI prints counts and digests only — never context/carrier text (content
hygiene: WildChat-class carrier text stays out of logs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.issue2329 import bank2329 as B29

# ── identity (the ONLY intended scientific change vs #2329) ───────────

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"  # plan §4.6 pin
PARENT_MODEL_ID = B29.MODEL_ID  # Qwen/Qwen3.5-9B
GRANDPARENT_MODEL_ID = B29.PARENT_MODEL_ID  # the #2162 model

SEED = B29.SEED
TEMPLATE_KWARGS: dict[str, bool] = dict(B29.TEMPLATE_KWARGS)  # {"enable_thinking": False}
THINK_BLOCK = B29.THINK_BLOCK
INTACT_FLOOR_PER_CELL = B29.INTACT_FLOOR_PER_CELL
PAIRS_PER_CELL = B29.PAIRS_PER_CELL

# ── tokenizer-parametric reuses (exact aliases, plan §4.1) ────────────

render_context_2389 = B29.render_context_2329
context_token_ids_2389 = B29.context_token_ids_2329
prefix_end_index_2389 = B29.prefix_end_index_2329
generation_header_text = B29.generation_header_text
generation_header_ids = B29.generation_header_ids
PairVerdict = B29.PairVerdict
TokenIdentityReport = B29.TokenIdentityReport
TokenIdentityFloorError = B29.TokenIdentityFloorError
build_token_identity = B29.build_token_identity
assert_intact_floor = B29.assert_intact_floor
donor_assignment_2389 = B29.donor_assignment_2329


def bank_manifest_2389(
    tokenizer,
    seed: int = SEED,
    frozen: dict[str, str] | None = None,
    strict: bool = True,
    report=None,
    enforce_floor: bool = True,
) -> dict:
    """The issue-2389 frozen bank spec (uploaded as `issue2389_q38ce/.../bank.json`).

    Delegates the full construction to `bank_manifest_2329` (every field is
    tokenizer-parametric) and overrides the identity fields. Deterministic —
    no timestamps / git state (the sha is the run regime key).
    """
    manifest = B29.bank_manifest_2329(
        tokenizer,
        seed=seed,
        frozen=frozen,
        strict=strict,
        report=report,
        enforce_floor=enforce_floor,
    )
    manifest.update(
        {
            "issue": 2389,
            "parent_issue": 2329,
            "grandparent_issue": 2162,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parent_model_id": PARENT_MODEL_ID,
            "grandparent_model_id": GRANDPARENT_MODEL_ID,
            "slots": ["ce"],
            "pe_slot_dropped": True,  # prefix_end/no_prefix fields are informational only
        }
    )
    return manifest


def freeze_bank_2389(
    tokenizer, out_path: Path | str, report_path: Path | str | None = None
) -> dict:
    """Gate 0a: verdict the full bank under the 27B tokenizer, freeze `bank.json`.

    Mirrors `freeze_bank_2329` (report written BEFORE the floor check so a
    `TokenIdentityFloorError` HALT leaves the per-cell breakage report on
    disk), swapping in the #2389 manifest + identity fields and recording the
    pinned revision in the provenance sidecar.
    """
    out_path = Path(out_path)
    report_path = (
        Path(report_path) if report_path else out_path.with_name("token_identity_report.json")
    )
    frozen = B29.B2162.load_frozen_gen()
    if frozen is None or B29.B2162.missing_frozen_keys(frozen):
        raise RuntimeError(
            "frozen_gen_2162.json missing or incomplete — the #2389 bank reuses the parent's "
            "frozen generations byte-verbatim (plan §4.1)"
        )
    pairs = B29.B2162.build_pairs()
    contexts = B29.B2162.build_contexts(frozen=frozen, strict=True)
    report = build_token_identity(tokenizer, pairs=pairs, contexts=contexts)
    report_payload = B29._report_payload(report, tokenizer)
    report_payload.update(
        {
            "issue": 2389,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parent_model_id": PARENT_MODEL_ID,
        }
    )
    B29._write_json_atomic(report_path, report_payload)
    manifest = bank_manifest_2389(
        tokenizer, frozen=frozen, strict=True, report=report, enforce_floor=True
    )
    B29._write_json_atomic(out_path, manifest)
    import transformers

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    bank_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    meta = {
        "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "report_path": str(report_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **as_metadata_dict(git_provenance()),
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    B29._write_json_atomic(out_path.with_name(out_path.stem + ".meta.json"), meta)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Freeze the issue-2389 re-tokenized bank (gate 0a CLI).

    Prints counts and digests only — never context/carrier text (content
    hygiene: WildChat-class carrier text stays out of logs).
    """
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Issue #2389 gate 0a: token-identity verdict + bank.json freeze."
    )
    ap.add_argument("--out", type=Path, required=True, help="bank.json output path")
    ap.add_argument("--report", type=Path, default=None, help="token-identity report path")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    # Tokenizer files only (small); the pinned revision is load-bearing —
    # every #2389 load (tokenizer or weights) threads MODEL_REVISION.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    manifest = freeze_bank_2389(tokenizer, args.out, args.report)
    ti = manifest["token_identity"]
    worst = min(row["n_intact"] for row in ti["per_cell"].values())
    print(
        f"[bank2389] froze {args.out}: pairs {ti['n_intact']}/{ti['n_pairs_total']} intact "
        f"(dropped {ti['n_dropped']}; worst cell {worst}/{PAIRS_PER_CELL}; "
        f"floor {INTACT_FLOOR_PER_CELL}); "
        f"no-prefix contexts {len(manifest['no_prefix_context_ids'])} (informational — "
        f"pe slot dropped); "
        f"rewires shuffled={len(manifest['donor_rewires']['shuffled'])} "
        f"crosstype={len(manifest['donor_rewires']['crosstype'])}"
    )


if __name__ == "__main__":
    main()
