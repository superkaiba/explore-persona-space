#!/usr/bin/env python3
"""Issue #761 — vectorized matched-probe v0(C,B) capture driver (the ONLY GPU code).

Plan §4.4-(1). For each behavior B in {sycophancy, refusal, harmful_compliance}
x each of 50 contexts C x each JUDGED probe in ``E0_expression.json`` ``per_probe``:
load the #658 on-policy completion, teacher-force ``(prompt + answer)``, capture the
residual-stream MEAN over the answer tokens at ALL 28 layers, batched at
``batch_probes`` probes per forward. Aggregate to the probe-mean matched
``v0(C,B)`` per layer per (context, behavior).

This is the single GPU step; the recompute (``issue761_recompute_mismatched_ridge.py``)
and the paired bootstrap (``issue761_paired_bootstrap.py``) are 0-GPU on the VM.

The capture machinery is REUSED, not rewritten:
- ``issue658_extract_base_store.load_hf_model`` / ``AnswerSpanCapture`` —
  loads ``Qwen/Qwen2.5-7B-Instruct`` bf16 on cuda:0 + the per-layer forward hooks.
- ``issue658_common.summarize_answer_span(span, "mean")`` — the exact ``mean``
  recipe that built the #658 store (``span.mean(dim=0)``), so the answer-span
  definition is identical by construction (the §6.0 byte-identical-recipe-at-capture
  premise, plan §8 risk (b)).
- ``issue594_common.messages_for_instance`` / ``load_battery`` — the 50-context
  battery; the e0_gen ``context_id`` field IS the battery instance ``id``.

``BatchedAnswerSpanCapture`` is the ONE vectorization of ``capture_v0_for_context``'s
batch-1 loop: same math, batched — the hook keeps the FULL ``(B, T, H)`` per layer,
left-padded so padding tokens never enter a row's answer span, and the per-row
position assert (captured span length == generated answer token count) is preserved
per row.

Matched-probe invariant (the whole point, plan §4.1): ``v0(C,B)`` is averaged over
EXACTLY the probes in ``E0[C][B].per_probe`` (the judged set), joined on probe text;
``judged ⊆ generated`` is asserted per (C, B) (a join miss fails loud).

Usage::

    # production (GPU)
    uv run python scripts/issue761_capture_matched_v0.py

    # smoke (1 context x 3 behaviors) — runs the IDENTICAL code path with
    # n_contexts=1; PASS_UNIFIED (smoke IS the sweep)
    uv run python scripts/issue761_capture_matched_v0.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

# Reused experiment code (see module docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue594_common import load_battery, messages_for_instance
from issue658_common import summarize_answer_span
from issue658_extract_base_store import AnswerSpanCapture, load_hf_model

# The recipe fingerprint is the SINGLE canonical copy in issue761_common; this
# driver imports it (do NOT re-declare a duplicate — a duplicated copy can drift
# undetected, defeating the cross-arm fingerprint equality assert; CONCERN
# fingerprint-assert-skips-matched-artifact, round-2). It is written into each
# .pt shard + v0_matched_by_behavior.json so the 0-GPU arms can assert equality.
from issue761_common import RECIPE_FINGERPRINT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue761_capture")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 28
HIDDEN = 3584
BEHAVIORS = ["sycophancy", "refusal", "harmful_compliance"]
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue761_matched_v0"
E0_PATH = REPO_ROOT / "eval_results" / "issue_658" / "E0_expression.json"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_761"
# Per-row (prompt+answer) token-length cap that bounds the (B, T, H) capture
# buffer. Raised from 1024 -> 4096 (round-2, CONCERN silent-left-truncation): the
# old 1024 silently LEFT-truncated long rows (dropping earliest prompt tokens,
# changing the teacher-forced conditioning); 4096 fits the observed matched-set
# distribution with a wide margin (#658 completion medians ~150 tok; the longest
# refusal/harmful probe PROMPTS run a few hundred tokens, so prompt+answer stays
# well under 4096). Overlength rows now FAIL LOUD by default (--strict-overlength)
# rather than being silently corrupted — a row over this cap is a real anomaly to
# investigate, not something to silently drop prompt tokens from. The longest row
# in the matched set is probed at startup + logged; if a future probe-set pushes
# past 4096, raise this cap rather than re-enabling truncation.
DEFAULT_MAX_T = 4096
DEFAULT_BATCH_PROBES = 16


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_store_context_ids() -> list[str]:
    """The canonical 50 context_ids in store order (plan §4.1 / store_manifest)."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO,
        "issue658_theory_assumptions/store/store_manifest.json",
        repo_type="dataset",
    )
    with open(p) as f:
        manifest = json.load(f)
    cids = manifest["context_ids"]
    assert len(cids) == 50, f"expected 50 context_ids, got {len(cids)}"
    assert manifest["n_layers"] == N_LAYERS, manifest["n_layers"]
    assert manifest["hidden"] == HIDDEN, manifest["hidden"]
    assert "mean" in manifest["v0_summary_recipes"], manifest["v0_summary_recipes"]
    return cids


def load_e0_gen(ctx_id: str, behavior: str) -> dict:
    """Load the #658 e0_gen completions JSON for one (context, behavior) from HF."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO,
        f"issue658_theory_assumptions/raw_completions/e0_gen/{ctx_id}__{behavior}.json",
        repo_type="dataset",
    )
    with open(p) as f:
        return json.load(f)


class BatchedAnswerSpanCapture(AnswerSpanCapture):
    """``AnswerSpanCapture`` whose hook keeps the FULL ``(B, T, H)`` per layer.

    One forward processes a left-padded batch of ``(prompt+answer)`` rows; for
    each row r the answer span ``[ans_start_r, ans_start_r + ans_len_r)`` is sliced
    (per-row lengths tracked by the caller), mean-reduced over its answer tokens at
    each layer to ``(n_layers, H)``, then probe-mean accumulated into ``v0(C,B)``.

    This is the ONE vectorization of ``capture_v0_for_context``'s batch-1 loop —
    same math (per-row ``summarize_answer_span(span, "mean")``), batched. Left-padding
    means a row's content occupies the TAIL positions, so an absolute position index
    is used (the caller passes ``ans_start_r`` already offset by the row's left pad).
    """

    def batched_answer_means(self, n_layers: int, spans: list[tuple[int, int]]) -> torch.Tensor:
        """(B, n_layers, H) fp32 CPU per-row answer-span MEAN per layer.

        ``spans[r] = (ans_start_r, ans_end_r)`` index the (left-padded) position
        axis — the answer tokens of row r only. Reduces the ``(B, T, H)`` buffer to
        per-row ``(n_layers, H)`` means immediately, then clears ``self.latest``.
        """
        b = len(spans)
        out = torch.zeros(b, n_layers, HIDDEN, dtype=torch.float32)
        for li in range(n_layers):
            hs = self.latest[li]  # (B, T, H) on device
            for r, (a0, a1) in enumerate(spans):
                assert 0 <= a0 < a1 <= hs.shape[1], (r, a0, a1, hs.shape[1])
                # (S, H) -> mean over the S answer tokens; exact `mean` recipe.
                span_rh = hs[r, a0:a1, :].to(torch.float32).cpu()
                out[r, li] = summarize_answer_span(span_rh, "mean")
        self.latest.clear()
        return out


class OverlengthRowError(RuntimeError):
    """A (prompt+answer) row exceeds ``max_t`` under the strict (default) policy.

    Raised instead of silently LEFT-truncating (which would drop earliest prompt
    tokens and change the teacher-forced conditioning invisibly — the per-row
    answer-span assert only checks answer length, so prompt-context loss was silent;
    CONCERN silent-left-truncation, round-2). Carries the offending cell + the row's
    true ``prompt_len + answer_len`` vs ``max_t``.
    """


def batched_capture_mean(
    model,
    tokenizer,
    instance: dict,
    tuples: list[tuple[str, str]],
    capture: BatchedAnswerSpanCapture,
    n_layers: int,
    *,
    batch_probes: int,
    max_t: int,
    behavior: str,
    strict_overlength: bool = True,
    skipped_overlength: list[dict] | None = None,
) -> tuple[torch.Tensor, int]:
    """Probe-mean matched ``v0(C,B)`` ``(n_layers, H)`` over ``tuples`` = [(probe, answer)].

    Left-pad-batches the ``(prompt+answer)`` rows ``batch_probes`` at a time; per row,
    captures the answer-span ``mean`` at every layer, then probe-means over all rows.
    Returns ``(v0, n_used)`` — ``n_used`` is the number of non-empty-answer probes
    actually summarized (the matched-N for this cell, modulo empty completions).

    Overlength handling (CONCERN silent-left-truncation, round-2): a row whose
    ``prompt_len + answer_len > max_t`` is NEVER silently truncated. Under
    ``strict_overlength=True`` (the default — the headline run) it raises
    :class:`OverlengthRowError` naming the (context, behavior, probe) + the row's true
    length vs ``max_t``. Under ``strict_overlength=False`` (an explicit
    ``--no-strict-overlength`` analysis opt-out) the row is SKIPPED (warn-loud) and
    recorded in ``skipped_overlength`` (a list the caller passes in); the headline run
    must keep that list empty. Either way the prompt context is never silently lost.

    The per-row answer-span position assert (captured length == generated answer
    token count) is preserved from ``capture_v0_for_context``.
    """
    pad_id = tokenizer.pad_token_id
    accum = torch.zeros(n_layers, HIDDEN, dtype=torch.float32)
    n_used = 0
    for start in range(0, len(tuples), batch_probes):
        batch = tuples[start : start + batch_probes]
        row_ids: list[torch.Tensor] = []
        row_ans_len: list[int] = []
        row_keep: list[bool] = []
        for probe, answer in batch:
            messages = messages_for_instance(instance, probe)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]
            ans_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"][
                0
            ]
            full = torch.cat([prompt_ids, ans_ids])
            total_len = int(full.shape[0])
            if max_t is not None and total_len > max_t:
                # NEVER silently left-truncate (drops earliest prompt tokens, changes
                # the teacher-forced conditioning invisibly). Fail loud by default;
                # warn-and-skip only under the explicit non-strict opt-out.
                if strict_overlength:
                    raise OverlengthRowError(
                        f"overlength row in ({instance['id']}, {behavior}): "
                        f"prompt_len={int(prompt_ids.shape[0])} + answer_len="
                        f"{int(ans_ids.shape[0])} = {total_len} > max_t={max_t}; "
                        f"probe={probe[:60]!r}. Raise --max-t rather than truncating "
                        "(silent left-truncation corrupts the answer-span conditioning)."
                    )
                logger.warning(
                    "overlength row SKIPPED (non-strict) in (%s, %s): "
                    "prompt+answer=%d > max_t=%d probe=%r",
                    instance["id"],
                    behavior,
                    total_len,
                    max_t,
                    probe[:40],
                )
                if skipped_overlength is not None:
                    skipped_overlength.append(
                        {
                            "context_id": instance["id"],
                            "behavior": behavior,
                            "probe": probe[:120],
                            "prompt_len": int(prompt_ids.shape[0]),
                            "answer_len": int(ans_ids.shape[0]),
                            "total_len": total_len,
                            "max_t": max_t,
                        }
                    )
                row_keep.append(False)
                row_ids.append(full)
                row_ans_len.append(int(ans_ids.shape[0]))
                continue
            row_keep.append(True)
            row_ids.append(full)
            row_ans_len.append(int(ans_ids.shape[0]))

        # drop empty-answer rows (no answer span) AND overlength-skipped rows
        keep = [i for i, n in enumerate(row_ans_len) if n > 0 and row_keep[i]]
        if not keep:
            for i in range(len(batch)):
                if row_ans_len[i] == 0:
                    logger.warning("empty completion for probe=%r — skipped", batch[i][0][:40])
            continue
        row_ids = [row_ids[i] for i in keep]
        row_ans_len = [row_ans_len[i] for i in keep]

        max_len = max(t.shape[0] for t in row_ids)
        b = len(row_ids)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros(b, max_len, dtype=torch.long)
        spans: list[tuple[int, int]] = []
        for r, ids in enumerate(row_ids):
            n = ids.shape[0]
            input_ids[r, max_len - n :] = ids  # LEFT-pad
            attn[r, max_len - n :] = 1
            ans_len = row_ans_len[r]
            a1 = max_len  # answer ends at the last real position
            a0 = max_len - ans_len  # answer starts ans_len before the end
            spans.append((a0, a1))

        with torch.no_grad():
            _ = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device))
        per_row = capture.batched_answer_means(n_layers, spans)  # (b, n_layers, H)
        # per-row position assert: captured span length == generated answer token count
        for r, (a0, a1) in enumerate(spans):
            assert (a1 - a0) == row_ans_len[r], (r, a1 - a0, row_ans_len[r])
        accum += per_row.sum(dim=0)
        n_used += b

    assert n_used > 0, f"instance {instance['id']}: every probe produced an empty answer"
    return accum / n_used, n_used


def probe_longest_row(
    tokenizer, e0: dict, inst_by_id: dict, context_ids: list[str]
) -> tuple[int, dict]:
    """Probe the longest ``prompt + answer`` token length over the WHOLE matched set.

    Runs the SAME tokenization the capture uses (chat-template prompt + answer, no
    forward) over every (context, behavior, judged-probe) row, returning the max
    ``prompt_len + answer_len`` and the offending (context, behavior, probe). Logged at
    startup so a future probe-set that would exceed ``max_t`` is visible BEFORE the GPU
    forwards run (round-2: pick max_t = max_observed + margin, never silently truncate).
    """
    longest = 0
    where: dict = {}
    for behavior in BEHAVIORS:
        for ctx_id in context_ids:
            gen = load_e0_gen(ctx_id, behavior)
            per_probe = e0["e0"][ctx_id][behavior]["per_probe"]
            judged = {x["probe"] for x in per_probe}
            inst = inst_by_id[ctx_id]
            for c in gen["cells"]:
                if c["probe"] not in judged or not c["completions"]:
                    continue
                messages = messages_for_instance(inst, c["probe"])
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                p_len = len(tokenizer(prompt_text, padding=False)["input_ids"])
                a_len = len(
                    tokenizer(c["completions"][0]["text"], add_special_tokens=False)["input_ids"]
                )
                if p_len + a_len > longest:
                    longest = p_len + a_len
                    where = {
                        "context_id": ctx_id,
                        "behavior": behavior,
                        "prompt_len": p_len,
                        "answer_len": a_len,
                    }
    return longest, where


def run_capture(
    *, smoke: bool, batch_probes: int, max_t: int, no_upload: bool, strict_overlength: bool = True
) -> Path:
    """The full matched-probe capture sweep. Returns the written JSON path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    context_ids = load_store_context_ids()
    if smoke:
        context_ids = context_ids[:1]
        logger.info("[smoke] capturing 1 context x %d behaviors", len(BEHAVIORS))

    with open(E0_PATH) as f:
        e0 = json.load(f)
    _, instances = load_battery()
    inst_by_id = {inst["id"]: inst for inst in instances}

    model, tokenizer = load_hf_model(MODEL_NAME, use_cuda=torch.cuda.is_available())
    capture = BatchedAnswerSpanCapture(model, N_LAYERS)

    # Probe the longest row up front so an over-cap distribution is visible BEFORE any
    # forward (CONCERN silent-left-truncation: pick max_t from the observed max, never
    # silently truncate). A longest row >= max_t under strict mode will raise in
    # batched_capture_mean; this just makes the anomaly loud at startup.
    longest, where = probe_longest_row(tokenizer, e0, inst_by_id, context_ids)
    logger.info(
        "longest matched row: prompt+answer=%d tokens (max_t=%d) at %s",
        longest,
        max_t,
        where,
    )
    if longest > max_t:
        logger.warning(
            "longest matched row (%d) EXCEEDS max_t (%d) — strict_overlength=%s will %s",
            longest,
            max_t,
            strict_overlength,
            "RAISE on those rows" if strict_overlength else "SKIP those rows",
        )

    # per-behavior shard dir for the .pt analysis tensors (HF upload, plan §4.4-1)
    shard_dir = OUT_DIR / "analysis_tensors"
    shard_dir.mkdir(parents=True, exist_ok=True)

    out_entries: list[dict] = []
    skipped_overlength: list[dict] = []
    # accumulate per-behavior matched v0 tensors + per-(ctx) matched_n
    per_behavior_v0: dict[str, dict[str, torch.Tensor]] = {b: {} for b in BEHAVIORS}
    per_behavior_matched_n: dict[str, dict[str, int]] = {b: {} for b in BEHAVIORS}
    try:
        for behavior in BEHAVIORS:
            for ctx_id in context_ids:
                gen = load_e0_gen(ctx_id, behavior)
                per_probe = e0["e0"][ctx_id][behavior]["per_probe"]
                judged = {x["probe"] for x in per_probe}
                # join on probe text; sycophancy uses the FIRST rollout's completion
                # (plan §4.4 the rollout-handling rule); refusal/harmful are n=1.
                tuples = [
                    (c["probe"], c["completions"][0]["text"])
                    for c in gen["cells"]
                    if c["probe"] in judged and c["completions"]
                ]
                # matched invariant: every judged probe must be in the generated set
                assert len(tuples) == len(judged), (
                    f"matched-probe join mismatch for ({ctx_id}, {behavior}): "
                    f"{len(tuples)} joined != {len(judged)} judged probes"
                )
                v0, n_used = batched_capture_mean(
                    model,
                    tokenizer,
                    inst_by_id[ctx_id],
                    tuples,
                    capture,
                    N_LAYERS,
                    batch_probes=batch_probes,
                    max_t=max_t,
                    behavior=behavior,
                    strict_overlength=strict_overlength,
                    skipped_overlength=skipped_overlength,
                )
                assert tuple(v0.shape) == (N_LAYERS, HIDDEN), tuple(v0.shape)
                per_behavior_v0[behavior][ctx_id] = v0
                # matched_n = the judged-probe count (the same-N control's per-(C,B) N).
                # Baked into the .pt shard below so the off-pod assembler reads it from
                # the UPLOADED shards (BLOCKER matched-metadata-not-durable, round-2 —
                # option (b): metadata travels with the data, no JSON dependency).
                per_behavior_matched_n[behavior][ctx_id] = len(tuples)
                out_entries.append(
                    {
                        "context_id": ctx_id,
                        "behavior": behavior,
                        "matched_n": len(tuples),
                        "n_used": n_used,
                        "v0_by_layer": v0.tolist(),  # (28, 3584)
                    }
                )
                logger.info(
                    "captured (%s, %s) matched_n=%d n_used=%d v0=%s",
                    ctx_id,
                    behavior,
                    len(tuples),
                    n_used,
                    tuple(v0.shape),
                )
    finally:
        capture.remove()

    # The headline run must skip NOTHING — a non-empty skip list under strict mode is
    # impossible (we'd have raised), but assert it for the non-strict path too so the
    # invariant ("skipped_overlength_rows empty in the headline run") is machine-checked.
    if strict_overlength:
        assert not skipped_overlength, (
            f"strict_overlength=True but {len(skipped_overlength)} rows skipped — "
            "this should be unreachable (overlength raises under strict mode)"
        )

    # write the per-behavior .pt shards (analysis tensors → HF). matched_n is BAKED IN
    # per context so the off-pod assembler recovers it from the uploaded shards alone
    # (BLOCKER matched-metadata-not-durable, round-2).
    for behavior, ctx_map in per_behavior_v0.items():
        shard_path = shard_dir / f"v0_matched_{behavior}.pt"
        torch.save(
            {
                "behavior": behavior,
                "context_ids": list(ctx_map.keys()),
                "v0": {c: t for c, t in ctx_map.items()},  # ctx -> (28, H) fp32
                "matched_n": dict(per_behavior_matched_n[behavior]),  # ctx -> int
                "recipe_fingerprint": RECIPE_FINGERPRINT,
                "smoke": smoke,
            },
            shard_path,
        )
        logger.info("wrote shard %s (%d contexts)", shard_path, len(ctx_map))

    out_path = OUT_DIR / "v0_matched_by_behavior.json"
    payload = {
        "task": 761,
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "hidden": HIDDEN,
        "summary_recipe": "mean",
        "behaviors": BEHAVIORS,
        "recipe_fingerprint": RECIPE_FINGERPRINT,
        "entries": out_entries,
        "skipped_overlength_rows": skipped_overlength,
        "longest_matched_row": {"total_len": longest, "max_t": max_t, "where": where},
        "metadata": {
            "git_commit": _git_commit(),
            "captured_at": _now_iso(),
            "smoke": smoke,
            "n_contexts": len(context_ids),
            "strict_overlength": strict_overlength,
            "torch_version": torch.__version__,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (%d entries)", out_path, len(out_entries))

    # upload the .pt analysis-tensor shards to HF before pod teardown (Upload Policy)
    if not no_upload and not smoke:
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        uploaded = upload_dataset_directory(
            shard_dir, f"{HF_PREFIX}/analysis_tensors", pattern="*.pt"
        )
        logger.info("uploaded %d analysis-tensor shards to HF", len(uploaded))

    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #761 matched-probe v0 capture")
    ap.add_argument("--smoke", action="store_true", help="1 context x 3 behaviors")
    ap.add_argument("--batch-probes", type=int, default=DEFAULT_BATCH_PROBES)
    ap.add_argument("--max-t", type=int, default=DEFAULT_MAX_T)
    ap.add_argument("--no-upload", action="store_true", help="skip the HF analysis-tensor upload")
    ap.add_argument(
        "--no-strict-overlength",
        dest="strict_overlength",
        action="store_false",
        help="warn-and-SKIP overlength rows instead of raising (default: strict raise). "
        "The headline run keeps strict on; an opt-out run records skipped rows in "
        "skipped_overlength_rows (which must be empty for the headline).",
    )
    ap.set_defaults(strict_overlength=True)
    args = ap.parse_args()

    out_path = run_capture(
        smoke=args.smoke,
        batch_probes=args.batch_probes,
        max_t=args.max_t,
        no_upload=args.no_upload,
        strict_overlength=args.strict_overlength,
    )

    if args.smoke:
        # smoke asserts: v0 shape (28, 3584) per cell, matched join exact, fingerprint serializes
        payload = json.loads(out_path.read_text())
        assert payload["recipe_fingerprint"] == RECIPE_FINGERPRINT, "fingerprint mismatch"
        json.dumps(payload["recipe_fingerprint"])  # serializes
        assert len(payload["entries"]) == len(BEHAVIORS), (
            f"smoke expected {len(BEHAVIORS)} entries (1 ctx x 3 beh), got "
            f"{len(payload['entries'])}"
        )
        for e in payload["entries"]:
            assert e["matched_n"] >= 50, (e["context_id"], e["behavior"], e["matched_n"])
            assert len(e["v0_by_layer"]) == N_LAYERS, len(e["v0_by_layer"])
            assert len(e["v0_by_layer"][0]) == HIDDEN, len(e["v0_by_layer"][0])
        # headline-run invariant: no row silently lost to overlength truncation
        assert payload["skipped_overlength_rows"] == [], payload["skipped_overlength_rows"]
        # matched_n is baked into the shards (BLOCKER matched-metadata-not-durable)
        for behavior in BEHAVIORS:
            blob = torch.load(
                OUT_DIR / "analysis_tensors" / f"v0_matched_{behavior}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert "matched_n" in blob and set(blob["matched_n"]) == set(blob["v0"]), (
                behavior,
                "shard matched_n missing or not keyed by every context",
            )
            assert blob["recipe_fingerprint"] == RECIPE_FINGERPRINT, behavior
        logger.info(
            "[smoke] PASS — %d cells, v0 (%d, %d), join exact, matched_n baked into shards, "
            "0 overlength-skipped, fingerprint serializes",
            len(payload["entries"]),
            N_LAYERS,
            HIDDEN,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
