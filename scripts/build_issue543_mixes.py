#!/usr/bin/env python3
"""Issue #543 per-arm install mixes — BALANCED ROUND-ROBIN negatives (plan §4.1).

Builds the 4 ratio-arm training mixes (6000 rows each) from the #543 response
bank, plus the frozen 32-row probe core and the 4 probe JSONLs.

Mix rule (the round-1 critique must-fix, plan §4.1):
  (i)   deterministically shuffle the train-question order (data_seed=543,
        ONE shared shuffle across arms — so every arm's positive set is a
        prefix of the same sequence and the probe core is in every arm);
  (ii)  deal the arm's positive questions to the 4 negative classes in
        balanced round-robin so every positive question receives the same
        contrast depth d = min(4, floor(total_neg / P)); fractional
        remainders give the earliest-shuffled questions depth d+1; no class
        repeats a question;
  (iii) when quotas exceed what the positives can cover (r10/r05), each
        class fills its remaining quota with non-positive train-pool
        questions sampled without replacement within the class.

Exclusions (logged in the manifest, plan §4.1 + #480 length-guard):
  - bank rows that hit the generation cap without EOS (truncated R would
    teach ` ※` after a mid-sentence cutoff);
  - rows whose fused chat-template render exceeds the Phase-1 max_length
    (truncating the trailing <|im_end|>/marker breaks the marker-only
    collator's slot layout).
  The clean pool = questions usable in ALL 5 context classes; the realized
  per-arm ratios are recorded in the manifest (assert within 1pp of nominal
  on the full pool).

The first ``N_PROBE_ROWS`` lines of every arm's train.jsonl are the FROZEN
PROBE CORE — byte-identical across all arms (asserted here), so
``build_source_probe_from_data`` (which reads rows in file order) yields a
byte-identical band-stop probe batch for all 12 cells with zero probe wiring.

Usage (CPU; run after gen_issue543_response_bank.py):
    uv run python scripts/build_issue543_mixes.py
    uv run python scripts/build_issue543_mixes.py --smoke   # tiny bank from --smoke gen
    # Follow-up preflight (plan risk 4): rebuilt parent arms + probes must be
    # byte-identical to the HF v1 reference manifest BEFORE any upload, and
    # only the NEW arm + the updated manifest are uploaded (additive):
    uv run python scripts/build_issue543_mixes.py --verify-vs-hub-manifest --upload-arms r01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="build_issue543_mixes")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    ARM_POSITIVES,
    ARMS,
    BANK_CLASSES,
    BANK_DIR,
    BANK_MAX_NEW_TOKENS,
    BASE_MODEL,
    DATA_SEED,
    EOS_TOKEN_ID,
    EXPECTED_MARKER_ID,
    HUB_DATA_BUCKET,
    MARKER_TEXT,
    MAX_EXCLUSION_RATE,
    MIX_MANIFEST_PATH,
    MIXES_DIR,
    N_PROBE_ROWS,
    N_TRAIN_QUESTIONS,
    NEG_CLASSES,
    PHASE1_MAX_LENGTH,
    POSITIVE_CLASS,
    PROBES_DIR,
    TOTAL_ROWS,
    marker_preflight,
    phase_log,
    read_jsonl,
    repro_metadata,
    to_sft_row,
    write_jsonl,
)

log = logging.getLogger("build_issue543_mixes")


# ── Bank loading + exclusion ────────────────────────────────────────────────


def _load_bank(*, expect_rows: int | None) -> dict[str, dict[int, dict]]:
    """{class_slug: {question_index: bank_row}} for all 5 classes.

    Args:
        expect_rows: when set (non-smoke builds), each class file must hold
            EXACTLY this many rows BEFORE exclusions — a short file is an
            interrupted/partial bank artifact (killed vLLM run, stray
            smoke-bank leftover) and must fail HERE, not as a skewed mix.
    """
    bank: dict[str, dict[int, dict]] = {}
    for class_slug in BANK_CLASSES:
        path = BANK_DIR / f"{class_slug}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Response bank class missing: {path}. Run gen_issue543_response_bank.py first."
            )
        rows = read_jsonl(path)
        if expect_rows is not None and len(rows) != expect_rows:
            raise RuntimeError(
                f"Bank class {class_slug} has {len(rows)} rows; expected exactly "
                f"{expect_rows} before exclusions — interrupted/partial bank artifact; "
                "re-run gen_issue543_response_bank.py."
            )
        bank[class_slug] = {r["question_index"]: r for r in rows}
        if len(bank[class_slug]) != len(rows):
            raise RuntimeError(f"Duplicate question_index in bank class {class_slug}")
    return bank


def _row_text(bank_row: dict, *, with_marker: bool) -> dict:
    """Bank row -> SFT row. Positives append the marker after the rstripped R."""
    assistant = bank_row["response"].rstrip()
    if not assistant:
        raise RuntimeError(
            f"Empty response in bank class={bank_row['class']} "
            f"q={bank_row['question_index']} — bank corrupt."
        )
    if with_marker:
        assistant = assistant + MARKER_TEXT
    return to_sft_row(system=bank_row["system"], user=bank_row["user"], assistant=assistant)


def _fused_len(tokenizer, sft_row: dict) -> int:
    ids = tokenizer.apply_chat_template(
        sft_row["prompt"] + sft_row["completion"],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def _usable_questions(bank: dict[str, dict[int, dict]], tokenizer) -> tuple[set[int], dict]:
    """Clean pool = questions whose row in EVERY class is non-truncated AND
    fits PHASE1_MAX_LENGTH under the fused chat-template render (marker
    appended on the positive class, matching the trained layout)."""
    excl_stats: dict[str, dict[str, int]] = {}
    usable_per_class: dict[str, set[int]] = {}
    for class_slug, rows in bank.items():
        n_trunc = 0
        n_overlong = 0
        ok: set[int] = set()
        for qi, r in rows.items():
            if r["truncated"]:
                n_trunc += 1
                continue
            sft = _row_text(r, with_marker=(class_slug == POSITIVE_CLASS))
            if _fused_len(tokenizer, sft) > PHASE1_MAX_LENGTH:
                n_overlong += 1
                continue
            ok.add(qi)
        usable_per_class[class_slug] = ok
        n = len(rows)
        rate = (n - len(ok)) / max(n, 1)
        excl_stats[class_slug] = {
            "n_rows": n,
            "n_truncated": n_trunc,
            "n_overlong": n_overlong,
            "n_usable": len(ok),
            "exclusion_rate": rate,
        }
        log.info(
            "Class %s: %d rows, %d truncated, %d overlong, %d usable (excl %.3f)",
            class_slug,
            n,
            n_trunc,
            n_overlong,
            len(ok),
            rate,
        )
        if rate > MAX_EXCLUSION_RATE:
            raise RuntimeError(
                f"Class {class_slug} exclusion rate {rate:.3f} > {MAX_EXCLUSION_RATE} — "
                "bank generation looks broken (cap too low / template drift); refusing "
                "to silently shrink the pool."
            )
    clean = set.intersection(*usable_per_class.values())
    log.info("Clean pool (usable in all 5 classes): %d questions", len(clean))
    return clean, excl_stats


# ── Balanced round-robin deal (plan §4.1) ───────────────────────────────────


def deal_negatives(
    positives: list[int],
    nonpositives: list[int],
    total_neg: int,
) -> tuple[dict[str, list[int]], dict]:
    """Assign negative-class question lists for one arm.

    Args:
        positives: the arm's positive question indices, in shared-shuffle order.
        nonpositives: clean-pool question indices NOT in positives (shuffle order).
        total_neg: total negative rows for the arm (TOTAL_ROWS - P).

    Returns:
        (assignments, stats): ``assignments[class] -> list of question indices``
        (len == that class's quota); stats records realized depth distribution,
        mean depth, and per-class fill counts (the §4.1 manifest contract).
    """
    n_classes = len(NEG_CLASSES)
    p = len(positives)
    if p == 0:
        raise ValueError("deal_negatives needs at least one positive question")
    base_q, rem = divmod(total_neg, n_classes)
    quotas = {c: base_q + (1 if i < rem else 0) for i, c in enumerate(NEG_CLASSES)}

    d_base = min(n_classes, total_neg // p)
    n_extra = (total_neg - p * d_base) if d_base < n_classes else 0
    depths = [min(n_classes, d_base + 1) if i < n_extra else d_base for i in range(p)]

    counts = dict.fromkeys(NEG_CLASSES, 0)
    assignments: dict[str, list[int]] = {c: [] for c in NEG_CLASSES}
    realized_depths: list[int] = []
    ptr = 0
    for i, q in enumerate(positives):
        d = depths[i]
        chosen: list[str] = []
        for j in range(2 * n_classes):
            if len(chosen) >= d:
                break
            c = NEG_CLASSES[(ptr + j) % n_classes]
            if c not in chosen and counts[c] < quotas[c]:
                chosen.append(c)
                counts[c] += 1
        ptr = (ptr + d) % n_classes
        realized_depths.append(len(chosen))
        for c in chosen:
            assignments[c].append(q)

    fill_counts: dict[str, int] = {}
    for c in NEG_CLASSES:
        need = quotas[c] - counts[c]
        fill_counts[c] = need
        if need > 0:
            if need > len(nonpositives):
                raise RuntimeError(
                    f"Class {c} needs {need} fill questions but only "
                    f"{len(nonpositives)} non-positive clean-pool questions exist."
                )
            rng_c = random.Random(f"{DATA_SEED}/{c}")
            assignments[c].extend(rng_c.sample(nonpositives, need))
        assert len(assignments[c]) == quotas[c], (c, len(assignments[c]), quotas[c])
        assert len(set(assignments[c])) == len(assignments[c]), f"class {c} repeats a question"

    depth_hist: dict[int, int] = {}
    for d in realized_depths:
        depth_hist[d] = depth_hist.get(d, 0) + 1
    stats = {
        "planned_depth_base": d_base,
        "n_planned_depth_plus_one": n_extra,
        "realized_depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
        "mean_realized_depth": sum(realized_depths) / p,
        "per_class_quota": quotas,
        "per_class_fill_count": fill_counts,
    }
    return assignments, stats


# ── Probe files + collator-compat check ─────────────────────────────────────


def _write_probe_files(
    bank: dict[str, dict[int, dict]],
    pool: list[int],
    probe_core_qis: list[int],
    n_probe: int,
    manifest: dict,
) -> None:
    """Write the 4 shared probe JSONLs (plan §4.1).

    Bystander probes append the marker purely as the SLOT LOCATOR: the slot
    read at marker_start - 1 never attends to the appended token. The
    reference probe is question-disjoint from the probe core (the LAST
    n_probe of the shared shuffled pool). NOTE: in the r50 arm those
    questions are (necessarily) also positives — all train questions are;
    the probe is a fixed-context within-condition trajectory, byte-identical
    across arms, which is the matching requirement.
    """
    probe_specs = {
        "probe_trigger.jsonl": (POSITIVE_CLASS, probe_core_qis),
        "probe_no_trigger.jsonl": ("assistant_no_key", probe_core_qis),
        "probe_doctor.jsonl": ("medical_doctor_key", probe_core_qis),
        "probe_reference.jsonl": ("assistant_no_key", pool[-n_probe:]),
    }
    if set(pool[-n_probe:]) & set(probe_core_qis):
        raise RuntimeError("Reference probe questions overlap the probe core — pool too small.")
    for fname, (class_slug, qis) in probe_specs.items():
        rows = [_row_text(bank[class_slug][qi], with_marker=True) for qi in qis]
        out = PROBES_DIR / fname
        write_jsonl(out, rows)
        manifest["sha256"][f"probes/{fname}"] = hashlib.sha256(out.read_bytes()).hexdigest()
        log.info("Probe %s: %d rows (class=%s)", fname, len(rows), class_slug)


def _collator_compat_check(bank: dict[str, dict[int, dict]], pool: list[int], tokenizer) -> None:
    """100 sampled NEGATIVE rows must carry <|im_end|> in the completion
    region of the fused render (plan §12 assumption 19) — the
    suppress_at_post_response_slot collator fail-louds at train time; this
    catches the same defect at build time."""
    sample_rng = random.Random(f"{DATA_SEED}/collator-check")
    neg_samples = []
    for c in NEG_CLASSES:
        qs = [qi for qi in pool if qi in bank[c]]
        neg_samples.extend((c, qi) for qi in sample_rng.sample(qs, min(25, len(qs))))
    for c, qi in neg_samples:
        sft = _row_text(bank[c][qi], with_marker=False)
        prompt_ids = tokenizer.apply_chat_template(
            sft["prompt"], tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            sft["prompt"] + sft["completion"], tokenize=True, add_generation_prompt=False
        )
        if isinstance(full_ids, dict):
            full_ids = full_ids["input_ids"]
        completion_region = list(full_ids[len(prompt_ids) :]) or list(full_ids)
        if EOS_TOKEN_ID not in completion_region:
            raise RuntimeError(
                f"Negative row (class={c}, q={qi}) has no <|im_end|> in its completion "
                "region — suppress_at_post_response_slot would fail-loud at train time."
            )
    log.info("Collator-compat check passed on %d sampled negative rows.", len(neg_samples))


# ── HF v1 byte-identity gate (plan risk 4 / round-2 must-fix) ───────────────


def _verify_vs_hub_manifest(local_manifest: dict) -> dict:
    """Compare the fresh build's SHA256 map against the HF v1 reference manifest.

    Downloads ``{HUB_DATA_BUCKET}/mixes/manifest.json`` from the data repo and
    requires EVERY sha256 key the reference carries (parent-arm mixes + the 4
    probe files) to be present and byte-identical in the local build. Any
    mismatch or missing key raises ``RuntimeError`` — the caller places this
    gate BEFORE the upload leg so a perturbed rebuild can never clobber the
    v1 reference (plan risk 4: "abort on mismatch BEFORE any training").
    Local-only keys (a newly added arm, e.g. ``mixes/r01/train.jsonl``) are
    additive and logged, not compared.

    Returns:
        Gate record dict: ``passed`` / ``n_compared`` / ``compared_keys`` /
        ``new_keys`` + the reference manifest's provenance fields.
    """
    from _issue543_common import HUB_DATA_REPO
    from huggingface_hub import hf_hub_download

    ref_file = hf_hub_download(
        repo_id=HUB_DATA_REPO,
        filename=f"{HUB_DATA_BUCKET}/mixes/manifest.json",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    ref = json.loads(Path(ref_file).read_text())
    ref_sha: dict[str, str] = ref.get("sha256", {})
    local_sha: dict[str, str] = local_manifest["sha256"]
    if not ref_sha:
        raise RuntimeError(
            "Hub reference manifest carries no sha256 map — wrong/corrupt reference; "
            "refusing to run the byte-identity gate against it."
        )
    parent_keys = {f"mixes/{a}/train.jsonl" for a in ("r50", "r25", "r10", "r05")}
    if not parent_keys <= set(ref_sha):
        raise RuntimeError(
            f"Hub reference manifest is missing parent-arm keys {parent_keys - set(ref_sha)} — "
            "wrong reference file; refusing to vacuously pass the identity gate."
        )
    missing = sorted(k for k in ref_sha if k not in local_sha)
    if missing:
        raise RuntimeError(
            f"Byte-identity gate FAIL: local build is missing reference keys {missing}."
        )
    mismatched = sorted(k for k in ref_sha if local_sha[k] != ref_sha[k])
    if mismatched:
        detail = {k: {"local": local_sha[k], "hub_v1": ref_sha[k]} for k in mismatched}
        raise RuntimeError(
            "Byte-identity gate FAIL: rebuilt files differ from the HF v1 reference "
            f"manifest — the parent arms would no longer be comparable. Mismatched: "
            f"{json.dumps(detail, indent=2)}. ABORTING before any upload (plan risk 4)."
        )
    new_keys = sorted(k for k in local_sha if k not in ref_sha)
    log.info(
        "Byte-identity gate PASS: %d/%d reference files identical; additive new keys: %s",
        len(ref_sha),
        len(ref_sha),
        new_keys or "none",
    )
    return {
        "passed": True,
        "n_compared": len(ref_sha),
        "compared_keys": sorted(ref_sha),
        "new_keys": new_keys,
        "reference_git_commit": ref.get("git_commit"),
        "reference_timestamp_utc": ref.get("timestamp_utc"),
    }


# ── Main build ──────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #543 per-arm mixes + probe files (CPU).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny bank (from gen --smoke): scale positives by pool size, total = 2x pool.",
    )
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--verify-vs-hub-manifest",
        action="store_true",
        help=(
            "Byte-identity gate (plan risk 4): after the build, compare every sha256 key "
            "in the HF v1 reference mixes/manifest.json against the fresh build and abort "
            "loud on any mismatch BEFORE the upload leg. Incompatible with --smoke."
        ),
    )
    p.add_argument(
        "--upload-arms",
        nargs="+",
        choices=list(ARMS),
        default=None,
        metavar="ARM",
        help=(
            "Restrict the upload leg to these arm mixes + the updated manifest (additive "
            "follow-up upload, e.g. r01). Parent-arm mixes and probes — byte-identical to "
            "the v1 reference per the gate — are NOT re-uploaded, preserving the Hub v1 "
            "files/revisions as the reference."
        ),
    )
    args = p.parse_args()
    if args.verify_vs_hub_manifest and args.smoke:
        p.error("--verify-vs-hub-manifest requires a full build (incompatible with --smoke).")

    phase_log("mix_build")
    marker_preflight()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [EXPECTED_MARKER_ID], marker_ids

    bank = _load_bank(expect_rows=None if args.smoke else N_TRAIN_QUESTIONS)
    clean_pool, excl_stats = _usable_questions(bank, tokenizer)
    if not clean_pool:
        raise RuntimeError("Clean pool is empty — bank generation failed.")

    # ONE shared shuffle across arms (plan §4.1.i): every arm's positive set
    # is a prefix of this order, so the probe core (its first rows) is a
    # positive in every arm.
    rng = random.Random(DATA_SEED)
    all_indices = list(range(N_TRAIN_QUESTIONS))
    rng.shuffle(all_indices)
    pool = [qi for qi in all_indices if qi in clean_pool]

    if args.smoke:
        scale = len(pool) / N_TRAIN_QUESTIONS
        arm_positives = {a: max(1, round(n * scale)) for a, n in ARM_POSITIVES.items()}
        total_rows = 2 * len(pool)
    else:
        arm_positives = {a: min(n, len(pool)) for a, n in ARM_POSITIVES.items()}
        total_rows = TOTAL_ROWS
        for a, n in arm_positives.items():
            nominal = ARM_POSITIVES[a]
            realized_ratio = n / total_rows
            nominal_ratio = nominal / TOTAL_ROWS
            if abs(realized_ratio - nominal_ratio) > 0.01:
                raise RuntimeError(
                    f"Arm {a}: realized positive ratio {realized_ratio:.4f} deviates "
                    f">1pp from nominal {nominal_ratio:.4f} (pool={len(pool)}). "
                    "Exclusions too aggressive — investigate the bank."
                )

    n_probe = min(N_PROBE_ROWS, min(arm_positives.values()))
    probe_core_qis = pool[:n_probe]
    log.info(
        "Pool=%d, total_rows=%d, probe core=%d rows, arm positives=%s",
        len(pool),
        total_rows,
        n_probe,
        arm_positives,
    )

    manifest: dict = {
        **repro_metadata(),
        "smoke": args.smoke,
        "total_rows_per_arm": total_rows,
        "clean_pool_size": len(pool),
        "bank_max_new_tokens": BANK_MAX_NEW_TOKENS,
        "phase1_max_length": PHASE1_MAX_LENGTH,
        "exclusions_per_class": excl_stats,
        "probe_core_n_rows": n_probe,
        "probe_core_question_indices": probe_core_qis,
        "arms": {},
        "sha256": {},
    }

    arm_first_lines: dict[str, list[str]] = {}
    for arm in ARMS:
        p_arm = arm_positives[arm]
        positives = pool[:p_arm]
        nonpositives = pool[p_arm:]
        total_neg = total_rows - p_arm
        assignments, deal_stats = deal_negatives(positives, nonpositives, total_neg)

        probe_rows = [
            _row_text(bank[POSITIVE_CLASS][qi], with_marker=True) for qi in probe_core_qis
        ]
        rest: list[dict] = [
            _row_text(bank[POSITIVE_CLASS][qi], with_marker=True) for qi in positives[n_probe:]
        ]
        for c in NEG_CLASSES:
            rest.extend(_row_text(bank[c][qi], with_marker=False) for qi in assignments[c])
        # Deterministic interleave of the non-probe remainder; TRL reshuffles
        # training order with its own seeded sampler anyway.
        random.Random(f"{DATA_SEED}/{arm}/order").shuffle(rest)
        rows = probe_rows + rest
        assert len(rows) == total_rows, (arm, len(rows), total_rows)

        out_path = MIXES_DIR / arm / "train.jsonl"
        write_jsonl(out_path, rows)
        arm_first_lines[arm] = out_path.read_text().splitlines()[:n_probe]

        manifest["arms"][arm] = {
            "n_positives": p_arm,
            "realized_positive_ratio": p_arm / total_rows,
            "nominal_positive_ratio": ARM_POSITIVES[arm] / TOTAL_ROWS,
            "n_negatives_total": total_neg,
            "contrast_coverage": deal_stats,
        }
        manifest["sha256"][f"mixes/{arm}/train.jsonl"] = hashlib.sha256(
            out_path.read_bytes()
        ).hexdigest()
        log.info("Arm %s: wrote %d rows -> %s", arm, len(rows), out_path)

    # Frozen-probe-core identity assert (plan §12 assumption 8): the first
    # n_probe lines must be byte-identical across all arms.
    ref_arm = ARMS[0]
    for arm in ARMS[1:]:
        if arm_first_lines[arm] != arm_first_lines[ref_arm]:
            raise RuntimeError(
                f"Probe core NOT byte-identical between {ref_arm} and {arm} — "
                "the band-stop matching control is broken."
            )
    log.info("Probe core byte-identical across %d arms (%d rows).", len(ARMS), n_probe)

    _write_probe_files(bank, pool, probe_core_qis, n_probe, manifest)
    _collator_compat_check(bank, pool, tokenizer)

    MIX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest -> %s", MIX_MANIFEST_PATH)

    if args.verify_vs_hub_manifest:
        # MUST precede the upload leg: a gate failure aborts before anything
        # (including the updated manifest) can clobber the v1 reference.
        phase_log("identity_gate")
        gate = _verify_vs_hub_manifest(manifest)
        gate_path = MIXES_DIR / "hub_identity_gate.json"
        gate_path.write_text(json.dumps({**repro_metadata(), **gate}, indent=2))
        log.info("Identity-gate record -> %s", gate_path)

    if not args.skip_upload and not args.smoke:
        phase_log("mix_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        upload_arms = args.upload_arms or list(ARMS)
        for arm in upload_arms:
            upload_dataset_directory(MIXES_DIR / arm, f"{HUB_DATA_BUCKET}/mixes/{arm}")
        if args.upload_arms is None:
            upload_dataset_directory(PROBES_DIR, f"{HUB_DATA_BUCKET}/probes")
        else:
            log.info(
                "Restricted upload: arms %s + manifest only (parent arms/probes are "
                "byte-identical to the v1 reference and are NOT re-uploaded).",
                upload_arms,
            )
        upload_dataset_directory(MIXES_DIR, f"{HUB_DATA_BUCKET}/mixes", pattern="manifest.json")

    phase_log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
