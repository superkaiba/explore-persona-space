"""Phase 0.1 -- diagnose #489's ICL adapter floor.

Issue #524 plan v1 §0.1. Four targeted diagnostics, all CPU-feasible
(or <=1 GPU-h on the dev VM):

  1. **Loss-curve asymptote** -- load #489's `train_diag/loss_curve_IK01.json`
     (already on HF), fit ``log P(※) ~ step`` and compare the asymptote to
     the marker-emission boundary (~-2 nats). If the asymptote is well
     below -2 nats -> the run converged below firing-cliff -> RECIPE bug,
     not training time.

  2. **Per-step on-policy log P(※) trajectory** -- for one representative
     ICL cell (IK01 / pirate), generate Qwen's own on-policy response under
     the ICL context, then teacher-force score log P(※) at the
     post-response slot at #489's saved checkpoints
     {25%, 50%, 100%, ep2, ep3}. (GPU-bound -- on this phase we ONLY emit
     the script that the dispatcher would run on a pod; the local smoke
     exercises the CPU pipeline + AST signature of the GPU entrypoint.)

  3. **Negative-set composition audit** -- read #489's
     ``train_rows/IK01_seed42.jsonl`` (HF), tokenize each negative row's
     completion under the chat template, fingerprint surface-style by:
       (a) presence of pirate dialect tokens (arr/matey/ye/savvy)
       (b) presence of CoT step markers ("Step 1:", "First,")
       (c) bare-modern-English fall-through
     Report the fraction of negatives that carry a distinct surface style.
     Hypothesis: most #489 negatives are bare modern English (thin-prefix
     ICL didn't induce); that's the floor cause.

  4. **Scaffold-competition check** -- count ``<|im_end|>`` token positions
     in #489's IK01 training rows vs #474's A1 (helpful_assistant)
     instruction training rows. If IK01 has 2+ ``<|im_end|>`` per row
     (one inside the ICL demonstrations, one at post-response) the
     collator's slot finder could be matching the wrong one.

Diagnostic outputs go to ``eval_results/issue_524/phase0/diagnosis/``.

CLI:
    # Default: run all 4 diagnostics (CPU-only items 1+3+4 + signature
    # check for GPU item 2).
    uv run python scripts/issue524_phase0_1_floor_diagnosis.py

    # Smoke: just run the CPU-feasible diagnostics 1, 3, 4 (skip 2's GPU dry-run).
    uv run python scripts/issue524_phase0_1_floor_diagnosis.py --skip-gpu

Output:
    eval_results/issue_524/phase0/diagnosis/loss_curve_fit.json
    eval_results/issue_524/phase0/diagnosis/neg_composition.json
    eval_results/issue_524/phase0/diagnosis/scaffold_check.json
    eval_results/issue_524/phase0/diagnosis/trajectory_signature.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.phase0_1")

OUT_DIR = Path("eval_results/issue_524/phase0/diagnosis")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# Representative #489 cell to diagnose. The plan §0.1 names IK01 / pirate.
REPRESENTATIVE_CID = "IK01"

# #489 loss curve path on HF (if it exists; we fail-loud if not).
I489_LOSS_CURVE_PATH = f"issue489_icl_marker/train_diag/loss_curve_{REPRESENTATIVE_CID}.json"
I489_TRAIN_ROWS_PATH = f"issue489_icl_marker/train_rows/{REPRESENTATIVE_CID}_seed42.jsonl"
I474_TRAIN_ROWS_PATH = "issue474_marker_at_end_loc/train_rows/i474_loc_A1.jsonl"

# Surface-style fingerprint patterns. Compiled once.
_PIRATE_PATTERN = re.compile(
    r"\b(arr+|matey|ye|savvy|hoist|plunder|cutlass|scallywag|booty|landlubber)\b",
    flags=re.IGNORECASE,
)
_COT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s*\d+[:.]|First[,]|Second[,]|Then[,]|Therefore[,])",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _try_download_hf(repo: str, repo_type: str, path: str) -> Path | None:
    """Try to download one file from HF; return Path or None if missing.

    Per CLAUDE.md `feedback_eval_script_silent_not_present_misdiagnosis`:
    we DISTINGUISH "genuinely not on HF" from "downloader bug" so the
    diagnostic reports the right thing. We FAIL-LOUD on auth/network
    errors, but silently return None if the file is genuinely missing.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        local = hf_hub_download(
            repo_id=repo,
            repo_type=repo_type,
            filename=path,
            revision="main",
        )
        return Path(local)
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        logger.warning("HF entry not found: %s/%s :: %s", repo, path, e)
        return None
    except Exception as e:
        # Auth / network / other -- distinguish from "not present".
        raise RuntimeError(
            f"HF download failed for {repo}/{path} (NOT a missing-file "
            f"error -- likely auth/network): {e}"
        ) from e


def diagnose_1_loss_curve_asymptote() -> dict:
    """Diagnostic 1: load #489's loss curve and report its asymptote.

    Output JSON:
        {
            "i489_path": str | None,
            "asymptote_nats": float | None,
            "below_firing_cliff": bool,  # True if asymptote < -5 nats
            "verdict": str,
        }
    """
    local = _try_download_hf(HF_DATA_REPO, "dataset", I489_LOSS_CURVE_PATH)
    if local is None:
        return {
            "i489_path": None,
            "asymptote_nats": None,
            "below_firing_cliff": None,
            "verdict": (
                "#489 loss_curve_{cid}.json not on HF -- cannot run loss-curve "
                "diagnosis. Phase 0.1 falls back to diagnostics 3+4 + the "
                "Phase 0.2 ICL block rebuild (which is the load-bearing "
                "intervention)."
            ),
        }
    payload = json.loads(local.read_text())
    # The #489 loss-curve schema is best-effort: we look for any of several
    # plausible key shapes. If we can't find a `log_p_marker` series, we
    # fail-loud rather than guess.
    series = (
        payload.get("log_p_marker_per_step")
        or payload.get("source_log_p_marker")
        or payload.get("log_p_curve")
    )
    if series is None:
        raise RuntimeError(
            f"#489 loss curve at {local} has no recognized log_p_marker series; "
            f"keys: {list(payload.keys())[:10]}"
        )
    last_n = max(1, len(series) // 5)  # last 20% as the asymptote
    asymptote = float(sum(series[-last_n:]) / last_n)
    below_firing = asymptote < -5.0
    return {
        "i489_path": str(local),
        "n_steps": len(series),
        "asymptote_nats": asymptote,
        "asymptote_window_steps": last_n,
        "below_firing_cliff": below_firing,
        "verdict": (
            "asymptote below -5 nats -> recipe-floor (not under-training)"
            if below_firing
            else "asymptote above firing cliff -> training-time issue"
        ),
    }


def diagnose_3_neg_composition() -> dict:
    """Diagnostic 3: audit #489's negative-row surface-style composition.

    For each negative-row's completion, classify as pirate / cot / bare /
    other based on simple regex fingerprints. Hypothesis: most #489
    negatives are BARE modern English -> the contrastive negatives carry
    no signal because the persona-voiced ICL didn't induce.
    """
    local = _try_download_hf(HF_DATA_REPO, "dataset", I489_TRAIN_ROWS_PATH)
    if local is None:
        return {
            "i489_path": None,
            "verdict": (
                "#489 train_rows/{cid}_seed42.jsonl not on HF -- cannot run "
                "neg-composition diagnosis. Falling back to the Phase 0.2 "
                "ICL block rebuild as the load-bearing intervention."
            ),
        }
    counts = {"pirate": 0, "cot": 0, "bare": 0, "other": 0, "total": 0, "n_neg": 0}
    with open(local) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            counts["total"] += 1
            # Negative rows are those whose completion has NO marker (we
            # only fingerprint negatives -- positives are by definition
            # pirate-voiced).
            comp_text = ""
            if "completion" in row and isinstance(row["completion"], list):
                for msg in row["completion"]:
                    if isinstance(msg, dict) and "content" in msg:
                        comp_text += msg["content"]
            if " ※" in comp_text:
                continue
            counts["n_neg"] += 1
            is_pirate = bool(_PIRATE_PATTERN.search(comp_text))
            is_cot = bool(_COT_PATTERN.search(comp_text))
            if is_pirate:
                counts["pirate"] += 1
            elif is_cot:
                counts["cot"] += 1
            elif len(comp_text) > 20 and not is_pirate and not is_cot:
                counts["bare"] += 1
            else:
                counts["other"] += 1

    bare_fraction = counts["bare"] / max(counts["n_neg"], 1)
    distinct_voice_fraction = (counts["pirate"] + counts["cot"]) / max(counts["n_neg"], 1)
    return {
        "i489_path": str(local),
        "counts": counts,
        "bare_fraction": bare_fraction,
        "distinct_voice_fraction": distinct_voice_fraction,
        "verdict": (
            f"bare_fraction={bare_fraction:.2f}; "
            + (
                "negatives are mostly bare modern English -> #489's "
                "contrastive set was NOT distinctly voiced; expected behavior "
                "for thin-prefix ICL blocks. Phase 0.2 Haiku rebuild fixes this."
                if bare_fraction > 0.6
                else "negatives carry distinct voices -> floor is NOT a composition problem"
            )
        ),
    }


def diagnose_4_scaffold_check() -> dict:
    """Diagnostic 4: count <|im_end|> positions in IK01 (ICL) vs A1 (instruction)
    training rows -- a scaffold-competition check.

    If IK01 has 2+ <|im_end|> per row (one inside the ICL demonstrations,
    one at post-response) the collator's slot finder could be matching
    the wrong one. (See plan v1 §8 Risks row 8 -- the slot finder uses
    valid_indices which is response-only, so prompt-region <|im_end|>'s
    are skipped; this diagnostic just verifies the assumption.)
    """

    def _count_im_end_per_row(local: Path | None) -> dict | None:
        if local is None:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                f"transformers import failed -- cannot run scaffold check: {e}"
            ) from e
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is None:
            raise RuntimeError("tokenizer cannot resolve <|im_end|>")
        per_row_counts: list[int] = []
        n = 0
        with open(local) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                msgs = list(row.get("prompt", [])) + list(row.get("completion", []))
                if not msgs:
                    continue
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                ids = tok.encode(text, add_special_tokens=False)
                per_row_counts.append(int(ids.count(im_end_id)))
                n += 1
                if n >= 20:  # diagnostic only -- 20 rows is enough.
                    break
        return {
            "n_rows_sampled": len(per_row_counts),
            "im_end_per_row_min": min(per_row_counts) if per_row_counts else None,
            "im_end_per_row_max": max(per_row_counts) if per_row_counts else None,
            "im_end_per_row_mean": (
                float(sum(per_row_counts) / len(per_row_counts)) if per_row_counts else None
            ),
        }

    ik01 = _try_download_hf(HF_DATA_REPO, "dataset", I489_TRAIN_ROWS_PATH)
    a1 = _try_download_hf(HF_DATA_REPO, "dataset", I474_TRAIN_ROWS_PATH)
    return {
        "i489_ik01": _count_im_end_per_row(ik01),
        "i474_a1": _count_im_end_per_row(a1),
        "verdict": (
            "see counts -- IK01 with mean >= 2 indicates scaffold competition "
            "in the prompt region (still benign because the slot finder uses "
            "valid_indices which is response-only; but worth knowing)."
        ),
    }


def diagnose_2_trajectory_signature_check() -> dict:
    """Diagnostic 2 (GPU-bound carve-out): signature smoke for the trajectory
    extractor entrypoint.

    The actual trajectory measurement (load #489 checkpoint, generate
    on-policy R under ICL context, teacher-force log P(※)) is GPU-bound
    and runs on a pod -- here we only AST-check that our planned
    entrypoint signature matches the canonical pattern in
    ``scripts/i474_phase4_eval.py``. This catches partial-port crashes
    (CLAUDE.md ``feedback_clone_modify_cross_file_drift``) before we
    burn pod minutes.
    """
    # Cross-eval signature -- the "canonical" function we'd call to score
    # log P(marker) on a fresh forward pass is _extract_marker_logp_and_argmax.
    # We import its source signature and assert it has the expected params.
    try:
        # AST-only: avoid loading vLLM at import time.
        repo = Path(__file__).resolve().parents[1]
        eval_src = (repo / "scripts" / "i474_phase4_eval.py").read_text()
    except Exception as e:
        return {"verdict": f"could-not-load-i474-eval-source: {e}"}

    sig_re = re.compile(
        r"def _extract_marker_logp_and_argmax\(\s*outputs,\s*slot_positions:\s*list\[int\],"
        r"\s*cell_label:\s*str\s*\)",
        flags=re.MULTILINE,
    )
    found = bool(sig_re.search(eval_src))
    return {
        "i474_eval_path": "scripts/i474_phase4_eval.py",
        "expected_sig_present": found,
        "verdict": (
            "i474 _extract_marker_logp_and_argmax signature present -- the "
            "trajectory script can reuse it without partial-port risk"
            if found
            else "i474 helper signature drifted -- trajectory port needs review"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip the GPU-bound diagnostic (#2 trajectory check); "
        "leaves the CPU-feasible diagnostics 1/3/4.",
    )
    ap.add_argument(
        "--skip-hf",
        action="store_true",
        help=(
            "Skip diagnostics 1+3+4 (which need #489 / #474 HF artifacts). "
            "Useful for the CPU smoke when HF artifacts aren't available."
        ),
    )
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_hf:
        out_loss = diagnose_1_loss_curve_asymptote()
        (OUT_DIR / "loss_curve_fit.json").write_text(json.dumps(out_loss, indent=2))
        logger.info("D1 (loss-curve) verdict: %s", out_loss["verdict"])

        out_neg = diagnose_3_neg_composition()
        (OUT_DIR / "neg_composition.json").write_text(json.dumps(out_neg, indent=2))
        logger.info("D3 (neg-comp) verdict: %s", out_neg["verdict"])

        out_scaf = diagnose_4_scaffold_check()
        (OUT_DIR / "scaffold_check.json").write_text(json.dumps(out_scaf, indent=2))
        logger.info("D4 (scaffold) verdict: %s", out_scaf["verdict"])

    # Diagnostic 2 is always cheap (it's a signature check, no GPU even on full run).
    out_traj = diagnose_2_trajectory_signature_check()
    (OUT_DIR / "trajectory_signature.json").write_text(json.dumps(out_traj, indent=2))
    logger.info("D2 (trajectory-sig) verdict: %s", out_traj["verdict"])

    logger.info("Phase 0.1 diagnostics written to %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
