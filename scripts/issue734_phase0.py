"""Issue #734 Phase 0 -- token-id split diagnostic (CPU-only, ~0 GPU-h).

Splits the #664 slot-bug magnitude into its two compounding defects (plan §4
Phase 0 / §12 assumption "decode->re-encode is a secondary contributor"):

  (a) the POST-TURN-END SLOT defect -- the #664 read appends the marker AFTER the
      response's assistant ``<|im_end|>\\n``, so it sits at a post-turn-end position
      (extra assistant ``<|im_end|>`` before the slot).
  (b) the DECODE->RE-ENCODE lossiness -- the #664 read decodes prompt+R to TEXT
      then ``tokenizer.encode``s it, which is NOT token-identical to the trained
      fused ``apply_chat_template(prompt + R + marker, tokenize=True)`` render.

For one representative cell (mk_librarian_contra_d1_seed42) over a small probe set,
this compares, per row:
  - the FUSED token ids (the corrected / in-loop render: prompt + R + marker),
  - the DECODE->RE-ENCODE ids of the same prompt+R text + appended marker (the #664
    mis-rooted construction),
and reports: the marker-slot position in each, the number of assistant turn-end
``<|im_end|>`` tokens before the marker slot in each (the (a) defect magnitude), and
the token-id edit distance between the fused completion-region ids and the
re-encoded ids (the (b) lossiness magnitude).

CPU-only: tokenizer + on-policy R only (no model forward). Writes
``eval_results/issue_734/phase0_token_id_split.json``.

Uses the model's OWN greedy R when a #664 raw-completion / marker_R cache row is
available (sha-pinned); otherwise a smoke R placeholder (``--smoke``). The split
analysis is invariant to the exact R text -- it measures the CONSTRUCTION
difference, not the response content -- so the smoke path is a faithful structural
check of the same code (CPU carve-out).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue734_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue734_phase0")


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Token-id edit distance between two id sequences (the (b) lossiness metric)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _fused_ids_and_slot(tokenizer, source_msgs, r_text):
    """Corrected fused render ids + marker slot (the in-loop construction)."""
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _find_subsequence

    row = RR.build_corrected_row(source_msgs, r_text, marker_text=C.MARKER_TEXT)
    full_ids = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False
    )
    if isinstance(full_ids, dict):
        full_ids = full_ids["input_ids"]
    full_ids = list(full_ids)
    marker_start = _find_subsequence(full_ids, [C.MARKER_ID])
    return full_ids, marker_start


def _misrooted_ids_and_slot(tokenizer, source_msgs, r_text):
    """Mis-rooted decode->re-encode ids + marker slot (the #664 construction).

    Mirrors ``issue664_extract_store._contexts_for_read`` + the
    ``compute_marker_slot_stats`` re-encode: take the decoded prompt+R+turn-end
    text, strip a trailing literal marker, re-encode, append the marker ids.
    """
    from explore_persona_space.train.sft import _find_subsequence

    prompt_text = tokenizer.apply_chat_template(
        source_msgs, tokenize=False, add_generation_prompt=True
    )
    # The model's OWN R ends with its assistant turn-end <|im_end|>\n in practice.
    r_with_turnend = r_text.rstrip() + "<|im_end|>\n"
    full_text = prompt_text + r_with_turnend
    stripped = full_text.rstrip()
    marker = C.MARKER_TEXT.strip()
    while stripped.endswith(marker):
        stripped = stripped[: -len(marker)].rstrip()
    ctx_ids = tokenizer.encode(stripped, add_special_tokens=False)
    marker_ids = tokenizer.encode(C.MARKER_TEXT, add_special_tokens=False)
    mis_ids = ctx_ids + marker_ids
    marker_start = _find_subsequence(mis_ids, [C.MARKER_ID])
    return mis_ids, marker_start


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #734 Phase 0 token-id split diagnostic.")
    ap.add_argument("--smoke", action="store_true", help="1-row smoke (no R cache needed)")
    ap.add_argument("--n-rows", type=int, default=3, help="rows to analyze (real run)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(C.INSTRUCT_ID, trust_remote_code=True)
    C.assert_marker_token(tokenizer)

    source = "librarian"
    questions = C.marker_question_pool(smoke=args.smoke)
    questions = questions[:1] if args.smoke else questions[: args.n_rows]
    # On-policy R: a representative natural response (the split is invariant to R
    # text -- it measures the CONSTRUCTION difference, see module docstring).
    r_text = "A consistent schedule and limiting screens before bed both help."

    rows_out = []
    n_imend_before_marker_fused = []
    n_imend_before_marker_mis = []
    edit_distances = []
    for q in questions:
        msgs = C.source_messages(source, q)
        fused_ids, fused_slot = _fused_ids_and_slot(tokenizer, msgs, r_text)
        mis_ids, mis_slot = _misrooted_ids_and_slot(tokenizer, msgs, r_text)
        assert fused_slot >= 0, "marker subsequence missing in fused render"
        assert mis_slot >= 0, "marker subsequence missing in mis-rooted render"
        # (a) post-turn-end magnitude: assistant <|im_end|> count before the marker.
        imend_fused = sum(1 for t in fused_ids[: fused_slot + 1] if t == C.IM_END_ID)
        imend_mis = sum(1 for t in mis_ids[: mis_slot + 1] if t == C.IM_END_ID)
        # (b) lossiness: edit distance between the two prefix id sequences up to the
        # marker (same prompt + R region, different tokenization path).
        ed = _levenshtein(fused_ids[: fused_slot + 1], mis_ids[: mis_slot + 1])
        n_imend_before_marker_fused.append(imend_fused)
        n_imend_before_marker_mis.append(imend_mis)
        edit_distances.append(ed)
        rows_out.append(
            {
                "question": q[:120],
                "fused_marker_slot": fused_slot,
                "misrooted_marker_slot": mis_slot,
                "imend_before_marker_fused": imend_fused,
                "imend_before_marker_misrooted": imend_mis,
                "prefix_edit_distance": ed,
            }
        )

    summary = {
        "experiment": "issue734_phase0_token_id_split",
        "source": source,
        "n_rows": len(rows_out),
        # (a) The post-turn-end slot defect: mis-rooted has STRICTLY MORE <|im_end|>
        # before the marker (the assistant turn-end), proving the slot is post-turn-end.
        "post_turn_end_defect": {
            "imend_before_marker_fused_mean": (
                sum(n_imend_before_marker_fused) / len(n_imend_before_marker_fused)
            ),
            "imend_before_marker_misrooted_mean": (
                sum(n_imend_before_marker_mis) / len(n_imend_before_marker_mis)
            ),
            "misrooted_has_extra_assistant_turn_end": all(
                m > f
                for m, f in zip(n_imend_before_marker_mis, n_imend_before_marker_fused, strict=True)
            ),
        },
        # (b) The decode->re-encode lossiness: non-zero edit distance means the two
        # paths produce DIFFERENT token ids for the same prompt+R region.
        "decode_reencode_lossiness": {
            "prefix_edit_distance_mean": sum(edit_distances) / len(edit_distances),
            "prefix_edit_distance_max": max(edit_distances),
            "any_lossy": any(ed > 0 for ed in edit_distances),
        },
        "rows": rows_out,
        "repro": C.repro_meta(),
        "smoke": args.smoke,
    }

    out_dir = C.EVAL_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase0_token_id_split.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "[phase0] wrote %s (post-turn-end extra=%s, lossy=%s, edit_dist_mean=%.2f)",
        out_path,
        summary["post_turn_end_defect"]["misrooted_has_extra_assistant_turn_end"],
        summary["decode_reencode_lossiness"]["any_lossy"],
        summary["decode_reencode_lossiness"]["prefix_edit_distance_mean"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
