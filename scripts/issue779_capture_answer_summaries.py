#!/usr/bin/env python3
"""Issue #779 follow-up: teacher-forced ANSWER-SUMMARY capture (GPU pod phase).

Re-captures, from the PERSISTED rollout TEXT (no regeneration ever), four
alternative per-rollout answer summaries at every layer, for the #779
training-source-ablation-hg corpora:

  - corpus (Arm B/C): ``{trait}_rollouts.json`` under
    ``issue779_monitoring/training-source-ablation-hg/behavior_corpus/`` —
    3 traits x 2400 contexts x 10 rollouts;
  - LMSYS (Arm A): ``lmsys_g_rollouts.json`` under
    ``.../lmsys_g_labels/`` — 5000 contexts x 1 rollout. NOTE these are the
    r7 g-label rollouts; the parent pass_b's ORIGINAL rollout text was never
    persisted (stripped at the sanitize step), so Arm-A summary targets come
    from THIS text, index-aligned to the cached pass_b ``cx_last`` (the g-label
    regen reloaded the same prompts in the same order, seed 42).

Per rollout x layer (fp16), with the SAME chat-template + span convention as
the original eval capture (``issue779_collect.capture_answer_vector``: span =
tokens [prompt_len, full_len) of ``apply_chat_template(messages + assistant
turn)``, which INCLUDES the ``<|im_end|>`` (id 151645) and trailing ``"\\n"``
(id 198) template-end tokens — verified against Qwen2.5-7B-Instruct):

  (a) ``v_last_turn``   — hidden state at the FINAL formatted-turn token
      (position full_len-1; the carry-forward position). Token id recorded per
      row (expected 198, the newline AFTER ``<|im_end|>``). This is the SAME
      position the eval rig's stored ``r2_last`` pooled projection reads.
  (b) ``v_last_content`` — hidden state at the last CONTENT token of the
      response proper (the position immediately BEFORE the turn-final
      ``<|im_end|>``). Invalid (NaN + valid=False) when the response text is
      empty. The eval rig stored NO content-only reference (span-last only).
  (c) ``v_max``          — element-wise MAX over the FULL response span
      [prompt_len, full_len), matching the span the original ``v_x`` mean and
      ``r2_max`` pooling used (i.e. INCLUDING the 2 template-end tokens).
  (d) ``v_first``        — hidden state at the FIRST response-content token
      (position prompt_len). Invalid when the response text is empty.

Batched (right-padded, sorted by length), bf16 model, forwards through
``model.model`` (the bare Qwen2Model) so lm_head logits are NEVER materialized
(the #779 r6 OOM class: an unread ``B x T x 152k`` allocation). Hook capture on
``model.model.layers[L]`` — same last-layer pre-final-norm convention as the
original ``analysis.extraction.extract_layer_activations`` hook path.

Checkpointed per trait per 500-context shard (``torch.save``); shards upload to
``issue779_monitoring/training-source-ablation-hg/final_token_capture/`` via
ONE bulk ``upload_folder`` commit per trait AS COMPLETED, verified with
``list_repo_files``. Fail loud everywhere; NaN never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib  # noqa: E402

import issue779_common as C  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import torch  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_capture_answer_summaries")

HF_ROUND_PREFIX = f"{C.HF_PREFIX}/training-source-ablation-hg"
CAPTURE_SUBDIR = "final_token_capture"
CORPUS_ROLLOUTS = "{prefix}/behavior_corpus/{trait}_rollouts.json"
CORPUS_PERSONAS = "{prefix}/corpus_specs/{trait}_personas.json"
LMSYS_ROLLOUTS = "{prefix}/lmsys_g_labels/lmsys_g_rollouts.json"

SUMMARIES = ("v_last_turn", "v_last_content", "v_max", "v_first")
IM_END_ID = 151645  # <|im_end|> — the token the turn-final template appends
SHARD_CTX = 500
CAPTURE_BATCH = int(os.environ.get("EPM_CAPTURE_BATCH", "16"))
N_LMSYS_EXPECTED = 5000
N_CORPUS_CTX_EXPECTED = 2400
N_ROLLOUTS_EXPECTED = 10
GPU_HOUR_BUDGET_WARN = 3.0


# ── input fetch + item building ───────────────────────────────────────────────


def _fetch_hf(hf_path: str, dest_dir: Path) -> Path:
    """Materialize one HF data-repo file into dest_dir (idempotent)."""
    dest = dest_dir / Path(hf_path).name
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[fetch] %s -> %s", hf_path, dest)
    got = hf_hub_download(
        repo_id=C.HF_DATA_REPO, filename=hf_path, repo_type="dataset", local_dir=dest_dir
    )
    got_p = Path(got)
    if got_p != dest:
        got_p.replace(dest)
    return dest


def build_corpus_items(trait: str, in_dir: Path) -> list[dict]:
    """Rebuild the corpus (messages, response) items from persisted rollout text.

    Structure (round-6+ schema, fail-loud validated): {"trait", "rollouts":
    {str(ci): {"persona_idx", "question_idx", "persona"?, "question",
    "responses": [n_rollouts strings]}}}. The persona TEXT comes from the row
    when present, else from the pinned corpus_specs personas by persona_idx —
    the SAME spec the rollouts were generated under. Messages mirror
    ``issue779_gen_behavior_corpus.build_corpus_contexts``: [{system persona},
    {user question}].
    """
    ro_path = _fetch_hf(CORPUS_ROLLOUTS.format(prefix=HF_ROUND_PREFIX, trait=trait), in_dir)
    with open(ro_path) as f:
        blob = json.load(f)
    if blob.get("trait") != trait:
        raise RuntimeError(f"rollouts trait {blob.get('trait')!r} != {trait!r} ({ro_path})")
    rollouts = blob["rollouts"]
    if len(rollouts) != N_CORPUS_CTX_EXPECTED:
        raise RuntimeError(f"{trait}: {len(rollouts)} rollout rows != {N_CORPUS_CTX_EXPECTED}")
    personas: list[str] | None = None
    if any("persona" not in rollouts[str(ci)] for ci in range(len(rollouts))):
        sp_path = _fetch_hf(CORPUS_PERSONAS.format(prefix=HF_ROUND_PREFIX, trait=trait), in_dir)
        with open(sp_path) as f:
            spec = json.load(f)
        personas = spec["personas"] if isinstance(spec, dict) else spec
        assert isinstance(personas, list) and personas, f"bad personas spec for {trait}"
    items: list[dict] = []
    for ci in range(len(rollouts)):
        row = rollouts[str(ci)]
        persona = row.get("persona")
        if persona is None:
            assert personas is not None
            persona = personas[row["persona_idx"]]
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": row["question"]},
        ]
        comps = row["responses"]
        if len(comps) != N_ROLLOUTS_EXPECTED:
            raise RuntimeError(f"{trait} ctx {ci}: {len(comps)} responses != 10")
        for ri, comp in enumerate(comps):
            items.append({"ci": ci, "ri": ri, "messages": messages, "response": comp})
    return items


def build_lmsys_items(in_dir: Path) -> list[dict]:
    """LMSYS g-label rollouts -> items. Bare user prompt, no system message."""
    ro_path = _fetch_hf(LMSYS_ROLLOUTS.format(prefix=HF_ROUND_PREFIX), in_dir)
    with open(ro_path) as f:
        blob = json.load(f)
    rollouts = blob["rollouts"]
    n = blob.get("n_contexts", len(rollouts))
    if len(rollouts) != n or n != N_LMSYS_EXPECTED:
        raise RuntimeError(f"lmsys rollouts: {len(rollouts)} rows, n_contexts={n} != 5000")
    items: list[dict] = []
    for ci in range(len(rollouts)):
        row = rollouts[str(ci)]
        comps = row["responses"]
        assert len(comps) == 1, (ci, len(comps))
        items.append(
            {
                "ci": ci,
                "ri": 0,
                "messages": [{"role": "user", "content": row["prompt"]}],
                "response": comps[0],
            }
        )
    return items


# ── batched capture core ──────────────────────────────────────────────────────


def _right_pad_batch(
    id_lists: list[list[int]], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad token-id lists into (input_ids, attention_mask, lens).

    Real tokens occupy 0..len-1 (pad at the end) so default position ids are
    correct per row. Mirrors the #779 worktree batched-capture convention.
    """
    lens = [len(x) for x in id_lists]
    max_len = max(lens)
    ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, : len(x)] = 1
    return ids.to(device), mask.to(device), lens


def _pad_id_for(tokenizer) -> int:
    pid = tokenizer.pad_token_id
    if pid is None:
        pid = tokenizer.eos_token_id
    if pid is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    return int(pid)


def _tokenize_item(tokenizer, item: dict) -> dict:
    """Tokenize one (messages, response) item; compute span + content boundary.

    Same convention as ``issue779_collect.capture_answer_vector``: prompt =
    ``apply_chat_template(messages, add_generation_prompt=True)``; full =
    ``apply_chat_template(messages + assistant turn)``. Span = [prompt_len,
    full_len). content_end = the position of the LAST ``<|im_end|>`` in the
    full ids (the template-appended turn end); content span = [prompt_len,
    content_end) — empty for an empty response string.
    """
    prompt_text = tokenizer.apply_chat_template(
        item["messages"], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, padding=False)["input_ids"]
    suffix = tokenizer.decode(prompt_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"position assert: {suffix!r} != {C.GENERATION_SUFFIX!r}"
    full_messages = [*item["messages"], {"role": "assistant", "content": item["response"]}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, padding=False)["input_ids"]
    prompt_len, full_len = len(prompt_ids), len(full_ids)
    assert full_len > prompt_len, (prompt_len, full_len)  # template appends >= 2 tokens
    im_end_positions = [p for p in range(prompt_len, full_len) if full_ids[p] == IM_END_ID]
    assert im_end_positions, f"no <|im_end|> in span (ci={item['ci']}, ri={item['ri']})"
    content_end = im_end_positions[-1]  # last <|im_end|> = the template turn end
    return {
        **item,
        "full_ids": full_ids,
        "prompt_len": prompt_len,
        "full_len": full_len,
        "content_end": content_end,  # content span = [prompt_len, content_end)
    }


@torch.no_grad()
def capture_summaries_batched(
    model, tokenizer, items: list[dict], layers: list[int], batch_size: int
) -> list[dict]:
    """Capture the 4 answer summaries per item x layer, right-padded batches.

    Forwards through ``model.model`` (bare decoder — NO lm_head logits) with
    hooks on ``model.model.layers[L]`` (raw pre-final-norm last layer, same as
    the original hook-path capture). Returns per item: {"summ": (4, L, H) fp16
    with NaN on invalid summaries, "valid": (4,) bool, "last_turn_token_id",
    "prompt_len", "span_len", "content_len"} in INPUT order.
    """
    blocks = model.model.layers
    pad_id = _pad_id_for(tokenizer)
    out: list[dict | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i]["full_len"])
    n_layers = len(layers)
    hidden = model.config.hidden_size

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(L: int):
        def _hook(_m, _i, output):
            captured[L] = output[0] if isinstance(output, tuple) else output

        return _hook

    for start in range(0, len(order), batch_size):
        sel = order[start : start + batch_size]
        batch = [items[i] for i in sel]
        ids_b, mask_b, _ = _right_pad_batch([b["full_ids"] for b in batch], pad_id, model.device)
        captured.clear()
        handles = [blocks[L].register_forward_hook(_make_hook(L)) for L in layers]
        try:
            model.model(input_ids=ids_b, attention_mask=mask_b)  # no lm_head, no logits
        finally:
            for h in handles:
                h.remove()
        for bi, gi in enumerate(sel):
            it = batch[bi]
            pl, fl, ce = it["prompt_len"], it["full_len"], it["content_end"]
            has_content = ce > pl
            summ = torch.full((4, n_layers, hidden), float("nan"), dtype=torch.float16)
            for li_pos, L in enumerate(layers):
                hs = captured[L][bi]  # (T, H) bf16, right-padded
                summ[0, li_pos] = hs[fl - 1].to(torch.float16).cpu()  # v_last_turn
                if has_content:
                    summ[1, li_pos] = hs[ce - 1].to(torch.float16).cpu()  # v_last_content
                    summ[3, li_pos] = hs[pl].to(torch.float16).cpu()  # v_first
                summ[2, li_pos] = (
                    hs[pl:fl].max(dim=0).values.to(torch.float16).cpu()
                )  # v_max, full span
            out[gi] = {
                "summ": summ,
                "valid": torch.tensor([True, has_content, True, has_content]),
                "last_turn_token_id": int(it["full_ids"][fl - 1]),
                "prompt_len": pl,
                "span_len": fl - pl,
                "content_len": ce - pl,
            }
        captured.clear()
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def equivalence_gate(model, tokenizer, layers: list[int]) -> dict:
    """Batched (padded) vs batch-1 equivalence on 3 variable-length items.

    Fail-fast gate before the sweep: a padding/indexing bug must crash here,
    not silently corrupt all captures. Mirrors the worktree's
    ``assert_batched_capture_equivalence`` recipe.
    """
    msgs = [
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi."}],
        [
            {"role": "system", "content": "You are a careful, verbose assistant."},
            {"role": "user", "content": "Explain in detail why the sky appears blue at noon."},
        ],
        [{"role": "user", "content": "Count to three."}],
    ]
    resps = [
        "Blue light scatters more.",
        "Because Rayleigh scattering favors short wavelengths across the whole sky.",
        "One two three.",
    ]
    items = [
        _tokenize_item(tokenizer, {"ci": i, "ri": 0, "messages": m, "response": r})
        for i, (m, r) in enumerate(zip(msgs, resps, strict=True))
    ]
    bat = capture_summaries_batched(model, tokenizer, items, layers, batch_size=3)
    ser = [
        capture_summaries_batched(model, tokenizer, [it], layers, batch_size=1)[0] for it in items
    ]
    max_abs, cos_min = 0.0, 1.0
    for s, b in zip(ser, bat, strict=True):
        assert torch.equal(s["valid"], b["valid"])
        for k in range(4):
            if not bool(s["valid"][k]):
                continue
            a = s["summ"][k].double().flatten()
            c = b["summ"][k].double().flatten()
            max_abs = max(max_abs, float((a - c).abs().max()))
            cos_min = min(cos_min, float(torch.dot(a, c) / (a.norm() * c.norm() + 1e-12)))
    assert cos_min >= 0.999 and max_abs < 0.1, (cos_min, max_abs)
    logger.info(
        "[gate] batched-vs-serial equivalence PASS (cos_min=%.6f max_abs=%.4f)", cos_min, max_abs
    )
    return {"cos_min": cos_min, "max_abs": max_abs}


# ── shard loop + upload ───────────────────────────────────────────────────────


def _shard_name(tag: str, k: int) -> str:
    return f"{tag}_summaries_shard{k:03d}.pt"


def _save_shard(
    path: Path,
    tag: str,
    layers: list[int],
    ctx_range: tuple[int, int],
    index: list[tuple[int, int]],
    rows: list[dict],
) -> None:
    tid = [r["last_turn_token_id"] for r in rows]
    torch.save(
        {
            "tag": tag,
            "summaries": list(SUMMARIES),
            "layers": layers,
            "context_range": list(ctx_range),
            "index": index,  # [(ci, ri)] aligned to rows
            "summ": torch.stack([r["summ"] for r in rows]),  # (n, 4, L, H) fp16, NaN=invalid
            "valid": torch.stack([r["valid"] for r in rows]),  # (n, 4) bool
            "last_turn_token_ids": torch.tensor(tid, dtype=torch.long),
            "prompt_lens": torch.tensor([r["prompt_len"] for r in rows], dtype=torch.long),
            "span_lens": torch.tensor([r["span_len"] for r in rows], dtype=torch.long),
            "content_lens": torch.tensor([r["content_len"] for r in rows], dtype=torch.long),
            "metadata": C.reproducibility_metadata(
                {
                    "script": "issue779_capture_answer_summaries",
                    "tag": tag,
                    "span_convention": (
                        "span=[prompt_len,full_len) incl <|im_end|>+\\n (matches "
                        "issue779_collect.capture_answer_vector); v_last_turn=full_len-1; "
                        "v_last_content=content_end-1 (pre-<|im_end|>); v_max=elementwise "
                        "max over full span; v_first=prompt_len"
                    ),
                }
            ),
        },
        path,
    )
    tmp_ids = set(tid)
    logger.info(
        "[shard] %s: %d rows, last-turn token ids %s", path.name, len(rows), sorted(tmp_ids)
    )


def _hf_capture_files() -> set[str]:
    from huggingface_hub import list_repo_files

    return {
        f
        for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_ROUND_PREFIX}/{CAPTURE_SUBDIR}/")
    }


def _upload_capture_dir(local_dir: Path, names: list[str]) -> None:
    """One bulk upload_folder commit for the given shard files + exact-set verify."""
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{HF_ROUND_PREFIX}/{CAPTURE_SUBDIR}"
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=prefix,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=names,
        commit_message=f"issue779 answer-summary capture: {len(names)} shard(s)",
    )
    repo = _hf_capture_files()
    missing = [n for n in names if f"{prefix}/{n}" not in repo]
    if missing:
        raise RuntimeError(f"capture upload verification FAILED: missing {missing}")
    logger.info("[upload] verified %d shard(s) under %s", len(names), prefix)


def run_tag(
    model,
    tokenizer,
    layers: list[int],
    tag: str,
    items: list[dict],
    n_ctx: int,
    out_dir: Path,
    hf_done: set[str],
    batch_size: int,
    t0: float,
    total_rollouts_all: int,
    done_rollouts_holder: list[int],
    smoke: bool = False,
) -> None:
    """Capture + checkpoint one corpus (trait or 'lmsys') in 500-context shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ci: dict[int, list[dict]] = {}
    for it in items:
        by_ci.setdefault(it["ci"], []).append(it)
    n_shards = (n_ctx + SHARD_CTX - 1) // SHARD_CTX
    new_names: list[str] = []
    prefix = f"{HF_ROUND_PREFIX}/{CAPTURE_SUBDIR}"
    for k in range(n_shards):
        name = _shard_name(tag, k)
        path = out_dir / name
        lo, hi = k * SHARD_CTX, min((k + 1) * SHARD_CTX, n_ctx)
        shard_items = [it for ci in range(lo, hi) for it in by_ci.get(ci, [])]
        if path.exists():
            logger.info("[%s] shard %d/%d already local; skip", tag, k + 1, n_shards)
            done_rollouts_holder[0] += len(shard_items)
            continue
        if f"{prefix}/{name}" in hf_done:
            logger.info("[%s] shard %d/%d already on HF; skip capture", tag, k + 1, n_shards)
            done_rollouts_holder[0] += len(shard_items)
            continue
        logger.info(
            "[%s] shard %d/%d: contexts [%d,%d) -> %d rollouts (tokenizing)",
            tag,
            k + 1,
            n_shards,
            lo,
            hi,
            len(shard_items),
        )
        tok_items = [_tokenize_item(tokenizer, it) for it in shard_items]
        rows = capture_summaries_batched(model, tokenizer, tok_items, layers, batch_size)
        index = [(it["ci"], it["ri"]) for it in shard_items]
        _save_shard(path, tag, layers, (lo, hi), index, rows)
        new_names.append(name)
        done_rollouts_holder[0] += len(shard_items)
        # Budget projection (guard: warn loudly past ~3 GPU-h projected).
        elapsed_h = (time.time() - t0) / 3600.0
        done = max(done_rollouts_holder[0], 1)
        proj_h = elapsed_h / done * total_rollouts_all
        logger.info(
            "[pace] %d/%d rollouts, %.2f h elapsed, %.2f h projected total",
            done,
            total_rollouts_all,
            elapsed_h,
            proj_h,
        )
        if proj_h > GPU_HOUR_BUDGET_WARN:
            logger.warning(
                "[budget] projected %.2f GPU-h exceeds %.1f h guard — continuing "
                "(corpus+LMSYS is the minimal set; nothing to drop)",
                proj_h,
                GPU_HOUR_BUDGET_WARN,
            )
    if new_names and not smoke:
        _upload_capture_dir(out_dir, new_names)
    elif new_names:
        logger.info("[%s] SMOKE: %d shard(s) kept local-only (no HF upload)", tag, len(new_names))
    else:
        logger.info("[%s] no new shards to upload", tag)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 answer-summary capture.")
    parser.add_argument("--model", default=C.DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/issue779_capture"))
    parser.add_argument("--in-dir", type=Path, default=Path("/workspace/issue779_capture/inputs"))
    parser.add_argument("--batch-size", type=int, default=CAPTURE_BATCH)
    parser.add_argument("--traits", nargs="+", default=[*C.TRAITS, "lmsys"])
    parser.add_argument("--smoke", action="store_true", help="2 contexts/trait, 20 rollouts total")
    parser.add_argument("--expected-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=C.EXPECTED_HIDDEN)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    n_layers = len(model.model.layers)
    assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
    assert model.config.hidden_size == args.expected_hidden
    layers = list(range(n_layers))

    gate = equivalence_gate(model, tokenizer, layers)
    out_dir = args.out_dir / ("shards_smoke" if args.smoke else "shards")
    hf_done = set() if args.smoke else _hf_capture_files()

    # Build all item sets up front (fail loud on any input mismatch first).
    tag_items: dict[str, tuple[list[dict], int]] = {}
    for tag in args.traits:
        if tag == "lmsys":
            items = build_lmsys_items(args.in_dir)
            n_ctx = N_LMSYS_EXPECTED
        else:
            items = build_corpus_items(tag, args.in_dir)
            n_ctx = N_CORPUS_CTX_EXPECTED
        if args.smoke:
            items = [it for it in items if it["ci"] < 2]
            n_ctx = 2
        tag_items[tag] = (items, n_ctx)
        logger.info("[inputs] %s: %d rollouts over %d contexts", tag, len(items), n_ctx)

    total = sum(len(v[0]) for v in tag_items.values())
    t0 = time.time()
    done_holder = [0]
    for tag, (items, n_ctx) in tag_items.items():
        run_tag(
            model,
            tokenizer,
            layers,
            tag,
            items,
            n_ctx,
            out_dir,
            hf_done,
            args.batch_size,
            t0,
            total,
            done_holder,
            smoke=args.smoke,
        )

    elapsed_h = (time.time() - t0) / 3600.0
    summary = {
        "tags": list(tag_items),
        "n_rollouts": total,
        "gpu_hours_wall": round(elapsed_h, 3),
        "equivalence_gate": gate,
        "smoke": args.smoke,
    }
    C.write_json_atomic(out_dir / "capture_summary.json", summary)
    logger.info("DONE: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
