#!/usr/bin/env python3
"""Issue #2224 P0c capture + 4b-1 predictor scoring (plan v3 §4).

Phase-dispatch driver (``--phase capture|score``; ``--import-check`` runs the
deferred-import + argparse-attribute completeness gate).

``--phase capture`` (P0c): BATCHED HF forwards over (prompt + dataset
response) AND (prompt + natural base response, from the P0b generation dir)
per pool sample, capturing at EVERY transformer block:

- ``resp_avg_dataset`` / ``resp_avg_natural`` — mean residual stream over the
  response tokens (the paper's response-avg pooling, #778 convention);
- ``last_prompt``  — context-end / last-prompt-token position (the paper's
  prompt-token predictor position);
- ``prefix_end``   — last token of the rendered prefix BEFORE the user query
  (the project prefix-arm read; glossary: prefix = everything before the
  user query, context = prefix + query).

Sharded fp16 ``.pt`` files under ``<out-root>/<corpus>/`` with
checkpoint-per-shard + regime-keyed resume (code-style.md checkpoint-per-
phase; a crash never loses more than one in-flight shard).

Tokenization is SEGMENT-CONCAT at the ids level — prefix | query |
template-suffix | response are encoded separately and concatenated — so
response/prefix span indices are exact BY CONSTRUCTION (no BPE seam merges;
for natural responses this matches generation-time conditioning). Deviation
vs issue778_lib's string-concat capture is recorded in the manifest; the
joint-vs-concat retokenization difference is counted per row as a
diagnostic. ``--verify-batched N`` runs the serial batch-1 oracle through
the same reductions and enforces cosine >= 0.999 per (row, kind, layer).

``--phase score`` (4b-1): the predictor arms per plan §4 arms table at the
CLI-passed read-out layer(s), computed from the capture summaries:

- ``raw``                — resp_avg_dataset · v̂
- ``exact_dp``           — (resp_avg_dataset − resp_avg_natural) · v̂  (paper ground truth)
- ``prompttoken_dp``     — (resp_avg_dataset − last_prompt) · v̂       (paper approximation)
- ``mapped_dp_context`` / ``mapped_dp_prefix``  — (resp_avg_dataset − M(v_C)) · v̂,
  v_C = context-end / prefix-end (BOTH arms — standing prefix+context rule)
- ``probe_diff_context`` / ``probe_diff_prefix`` — probe(resp_avg_dataset) − probe(M(v_C))

#2222-dependent artifacts are CLI paths so the P1 gate can inject pinned
revisions later; a requested arm whose artifact is absent FAILS LOUD (no
silent fallback):

- ``--persona-vectors-dir``  #778 ``rb/{trait}.pt`` — a BARE ``torch.Tensor``
  (28, 3584) fp32; shape/dtype asserted at load (plan §10).
- ``--map-context`` / ``--map-prefix``  frozen linear map ``.npz`` in the
  #1739 ``_save_map`` contract: ``w`` (Ly,d,d) fp16, ``x_mu``/``x_sd``/``y_mu``
  (Ly,1,d) fp32, ``layers`` (Ly,), ``meta`` json;
  apply = ``((x - x_mu)/x_sd) @ w + y_mu`` via the reviewed
  ``experiments.issue_1739.fits.apply_map``.
- ``--probe-dir``  Form-A ridge probe per trait: ``<trait>.npz`` with ``w``
  (d,), ``b`` (scalar), optional ``x_mu``/``x_sd`` (d,), optional ``layer``
  (asserted == the trait's read-out layer when present), optional ``meta``
  json. #2222 must emit this contract (or this loader is adapted at the P1
  gate).
- ``--layer`` / ``--layers-json``  read-out layer per trait (#2222's
  selection; 0-indexed block convention of the #778 rb rows — row 19 = the
  paper's 1-indexed layer 20).

Content hygiene: corpus/response text is never printed — sample_ids, counts
and score digests only.

Usage::

    uv run python scripts/issue2224_predictor_scores.py --phase capture \\
        --pool data/issue_2224/pools/lmsys.jsonl --corpus lmsys \\
        --natural-dir raw_completions/exact_dp_base_gen/lmsys \\
        --out-root analysis_tensors/predictor_summaries
    uv run python scripts/issue2224_predictor_scores.py --phase score \\
        --summaries-dir analysis_tensors/predictor_summaries/lmsys --corpus lmsys \\
        --persona-vectors-dir <rb-dir> --layers-json <path> \\
        --map-context <npz> --map-prefix <npz> --probe-dir <dir>
    uv run python scripts/issue2224_predictor_scores.py --import-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402
from issue2224_common import (  # noqa: E402
    CAPTURE_SCHEMA_VERSION,
    SCREENING_SCORES_DIR_DEFAULT,
    append_jsonl,
    atomic_write_json,
    load_jsonl,
    repro_meta,
    sha256_file,
)
from issue778_lib import HIDDEN_DIM, MODEL_NAME, N_LAYERS, TRAITS  # noqa: E402

logger = logging.getLogger("issue2224_predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARMS = ("raw", "exact_dp", "prompttoken_dp", "mapped_dp", "probe_diff")
KIND_NAMES = ("resp_avg_dataset", "resp_avg_natural", "last_prompt", "prefix_end")
USER_MARKER = "<|im_start|>user"  # Qwen chat-template user-turn opener
MIN_RESP_TOKENS = 1


# ── Tokenization (segment-concat, seam-safe spans) ───────────────────────────────


@dataclass
class SampleTok:
    """Per-sample tokenization: exact spans by construction (ids-level concat)."""

    sample_id: str
    prompt_ids: list[int]
    prefix_end: int  # index INTO prompt_ids of the last prefix token
    resp_ids: dict[str, list[int]] = field(default_factory=dict)  # kind -> ids
    truncated: dict[str, bool] = field(default_factory=dict)
    seam_differs: bool = False


def render_prompt_segments(tok, user_text: str) -> tuple[list[int], int, bool]:
    """Chat-templated prompt ids + prefix-end index (segment-concat).

    Renders ``apply_chat_template([{user}], add_generation_prompt=True)``,
    splits the rendered string at the user-query boundaries, and encodes the
    three segments separately so ``prefix_end`` / ``prompt_len`` are exact
    (no cross-boundary BPE merges). Returns (prompt_ids, prefix_end_idx,
    seam_differs_vs_joint_tokenization).
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
    m = rendered.rfind(USER_MARKER)
    q = rendered.find(user_text, m) if m >= 0 else rendered.rfind(user_text)
    if q < 0:
        raise RuntimeError(
            "user text not found verbatim in the rendered chat template — "
            "template convention changed; fix render_prompt_segments()"
        )
    prefix_str = rendered[:q]
    suffix_str = rendered[q + len(user_text) :]
    prefix_ids = tok.encode(prefix_str, add_special_tokens=False)
    query_ids = tok.encode(user_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_str, add_special_tokens=False)
    if not prefix_ids or not suffix_ids:
        raise RuntimeError("empty prefix/suffix segment — unexpected chat template shape")
    prompt_ids = prefix_ids + query_ids + suffix_ids
    seam_differs = prompt_ids != tok.encode(rendered, add_special_tokens=False)
    return prompt_ids, len(prefix_ids) - 1, seam_differs


def load_natural_map(natural_dir: Path, sample_ids: list[str]) -> dict[str, str]:
    """sample_id -> natural base response from the P0b generation dir.

    Contract (unit-2's P0b writer conforms): one or more ``*.jsonl`` files of
    rows ``{"sample_id": ..., "response": ...}``. FAIL LOUD on missing ids.
    """
    files = sorted(Path(natural_dir).glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"--natural-dir {natural_dir} contains no *.jsonl files")
    out: dict[str, str] = {}
    for fp in files:
        for r in load_jsonl(fp):
            out[str(r["sample_id"])] = str(r["response"])
    missing = [s for s in sample_ids if s not in out]
    if missing:
        raise RuntimeError(
            f"--natural-dir {natural_dir} is missing {len(missing)}/{len(sample_ids)} "
            f"sample_ids (first 5: {missing[:5]}) — P0b incomplete for this pool"
        )
    return out


def prepare_samples(
    tok, pool_rows: list[dict], natural_map: dict[str, str] | None, max_length: int
) -> tuple[list[SampleTok], dict]:
    """Tokenize every pool sample (prompt segments + response kinds)."""
    counters = {
        "n_samples": len(pool_rows),
        "seam_retokenization_differs": 0,
        "truncated_dataset": 0,
        "truncated_natural": 0,
    }
    out: list[SampleTok] = []
    for r in pool_rows:
        prompt_ids, prefix_end, seam = render_prompt_segments(tok, r["prompt"])
        st = SampleTok(
            sample_id=r["sample_id"],
            prompt_ids=prompt_ids,
            prefix_end=prefix_end,
            seam_differs=seam,
        )
        counters["seam_retokenization_differs"] += int(seam)
        budget = max_length - len(prompt_ids)
        if budget < MIN_RESP_TOKENS:
            raise RuntimeError(
                f"{r['sample_id']}: prompt is {len(prompt_ids)} tokens, no response budget "
                f"under --max-length {max_length} (pool 512-tok filter violated upstream?)"
            )
        kinds = {"dataset": r["response"]}
        if natural_map is not None:
            kinds["natural"] = natural_map[r["sample_id"]]
        for kind, text in kinds.items():
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"{r['sample_id']}: empty {kind} response after tokenization")
            if len(ids) > budget:
                ids = ids[:budget]
                st.truncated[kind] = True
                counters[f"truncated_{kind}"] += 1
            else:
                st.truncated[kind] = False
            st.resp_ids[kind] = ids
        out.append(st)
    return out, counters


# ── Batched capture ──────────────────────────────────────────────────────────────


def iter_batches(rows: list[dict], batch_tokens: int, max_batch: int):
    """Greedy length-sorted batches under a padded-token budget."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["n_total"])
    batch: list[int] = []
    max_t = 0
    for i in order:
        t = rows[i]["n_total"]
        new_max = max(max_t, t)
        if batch and (new_max * (len(batch) + 1) > batch_tokens or len(batch) >= max_batch):
            yield [rows[j] for j in batch]
            batch, max_t = [], 0
            new_max = t
        batch.append(i)
        max_t = new_max
    if batch:
        yield [rows[j] for j in batch]


def _reduce_hidden(hs, rows: list[dict], device) -> dict[str, "object"]:
    """Vectorized per-layer reductions over one padded batch.

    ``hs`` = ``outputs.hidden_states`` (tuple len L+1); rows carry
    prompt_len / resp_len / prefix_end / kind. Returns per-row reduction
    tensors on CPU fp16: resp_avg (n_rows, L, H) + last_prompt / prefix_end
    (dataset-kind rows only).
    """
    import torch

    n_layers = len(hs) - 1
    b = len(rows)
    t = hs[1].shape[1]
    plen = torch.tensor([r["n_prompt"] for r in rows], device=device)
    rlen = torch.tensor([r["n_resp"] for r in rows], device=device)
    pend = torch.tensor([r["prefix_end"] for r in rows], device=device)
    pos = torch.arange(t, device=device).unsqueeze(0)  # (1, T)
    resp_mask = ((pos >= plen.unsqueeze(1)) & (pos < (plen + rlen).unsqueeze(1))).to(torch.float32)
    arange_b = torch.arange(b, device=device)
    ds_rows = [i for i, r in enumerate(rows) if r["kind"] == "dataset"]
    ds_idx = torch.tensor(ds_rows, device=device, dtype=torch.long)

    resp_avg = torch.empty((b, n_layers, hs[1].shape[2]), dtype=torch.float16)
    last_prompt = torch.empty((len(ds_rows), n_layers, hs[1].shape[2]), dtype=torch.float16)
    prefix_end = torch.empty_like(last_prompt)
    for li in range(n_layers):
        h = hs[li + 1].float()  # (B, T, H) fp32 for the reduction
        ra = torch.einsum("bth,bt->bh", h, resp_mask) / rlen.to(torch.float32).unsqueeze(1)
        resp_avg[:, li] = ra.to("cpu", torch.float16)
        if ds_rows:
            last_prompt[:, li] = h[ds_idx, (plen - 1)[ds_idx]].to("cpu", torch.float16)
            prefix_end[:, li] = h[ds_idx, pend[ds_idx]].to("cpu", torch.float16)
        del h
    del arange_b
    return {
        "resp_avg": resp_avg,
        "last_prompt": last_prompt,
        "prefix_end": prefix_end,
        "ds_rows": ds_rows,
    }


def _forward_batch(model, rows: list[dict], device, pad_id: int):
    """Right-padded batched forward (explicit positions are the default
    arange under right padding — real tokens occupy 0..len-1, so no
    position_ids correction is needed; the #502 left-pad trap is avoided by
    construction)."""
    import torch

    t_max = max(r["n_total"] for r in rows)
    input_ids = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), t_max), dtype=torch.long)
    for i, r in enumerate(rows):
        ids = r["ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    return out


def _serial_reference(model, st: SampleTok, kind: str, device) -> dict[str, "object"]:
    """Batch-1, no-padding oracle for --verify-batched (plain indexing, fp32)."""
    import torch

    ids = st.prompt_ids + st.resp_ids[kind]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states
    n_layers = len(hs) - 1
    plen = len(st.prompt_ids)
    res = {
        "resp_avg": torch.stack(
            [hs[li + 1][0, plen:, :].float().mean(0) for li in range(n_layers)]
        ).cpu(),
        "last_prompt": torch.stack(
            [hs[li + 1][0, plen - 1, :].float() for li in range(n_layers)]
        ).cpu(),
        "prefix_end": torch.stack(
            [hs[li + 1][0, st.prefix_end, :].float() for li in range(n_layers)]
        ).cpu(),
    }
    del out
    return res


def _load_model(model_name: str, device: str, dtype_flag: str):
    """Explicit-device HF load (never device_map='auto' — silent CPU offload)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if dtype_flag == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_flag]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return tok, model, str(dtype)


def _capture_rows_for(samples: list[SampleTok], lo: int, hi: int) -> list[dict]:
    """Flatten samples[lo:hi] into per-(sample, kind) capture rows."""
    rows: list[dict] = []
    for pos in range(lo, hi):
        st = samples[pos]
        for kind, resp in st.resp_ids.items():
            rows.append(
                {
                    "pos": pos - lo,
                    "kind": kind,
                    "ids": st.prompt_ids + resp,
                    "n_prompt": len(st.prompt_ids),
                    "n_resp": len(resp),
                    "n_total": len(st.prompt_ids) + len(resp),
                    "prefix_end": st.prefix_end,
                }
            )
    return rows


def run_verify_batched(model, samples: list[SampleTok], args, device) -> None:
    """Batched-vs-serial equivalence gate (cosine >= 0.999 per row x kind x layer)."""
    import torch

    n = min(args.verify_batched, len(samples))
    if n < 2:
        raise RuntimeError("--verify-batched needs >= 2 samples so padding actually fires")
    sub = samples[:n]
    rows = _capture_rows_for(sub, 0, n)
    pad_id = model.config.eos_token_id or 0
    out = _forward_batch(model, rows, device, pad_id)
    red = _reduce_hidden(out.hidden_states, rows, device)
    del out
    worst = 1.0
    worst_tag = ""
    max_abs = 0.0
    for i, r in enumerate(rows):
        st = sub[r["pos"]]
        ref = _serial_reference(model, st, r["kind"], device)
        pairs = [("resp_avg", red["resp_avg"][i].float(), ref["resp_avg"])]
        if r["kind"] == "dataset":
            j = red["ds_rows"].index(i)
            pairs.append(("last_prompt", red["last_prompt"][j].float(), ref["last_prompt"]))
            pairs.append(("prefix_end", red["prefix_end"][j].float(), ref["prefix_end"]))
        for name, a, b in pairs:
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # per layer
            c = float(cos.min())
            max_abs = max(max_abs, float((a - b).abs().max()))
            if c < worst:
                worst, worst_tag = c, f"{st.sample_id}/{r['kind']}/{name}"
    logger.info(
        "[verify-batched] n=%d rows=%d min_cosine=%.6f (at %s) max_abs_diff=%.4g",
        n,
        len(rows),
        worst,
        worst_tag,
        max_abs,
    )
    if worst < 0.999:
        raise RuntimeError(
            f"batched-vs-serial equivalence FAILED: min cosine {worst:.6f} < 0.999 at {worst_tag}"
        )


def run_capture(args) -> int:
    """P0c: batched capture of all summary kinds, shard-checkpointed."""
    import torch

    if args.pool is None or args.corpus is None or args.out_root is None:
        raise RuntimeError("--phase capture requires --pool, --corpus and --out-root")
    pool_path = Path(args.pool)
    pool_rows = load_jsonl(pool_path)
    if args.limit:
        pool_rows = pool_rows[: args.limit]
    corpus_dir = Path(args.out_root) / args.corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model, dtype_str = _load_model(args.model, device, args.model_dtype)
    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)

    natural_map = None
    natural_files_sha = None
    if args.natural_dir:
        natural_map = load_natural_map(Path(args.natural_dir), [r["sample_id"] for r in pool_rows])
        natural_files_sha = {
            f.name: sha256_file(f) for f in sorted(Path(args.natural_dir).glob("*.jsonl"))
        }

    samples, tok_counters = prepare_samples(tok, pool_rows, natural_map, args.max_length)

    regime = {
        "schema": CAPTURE_SCHEMA_VERSION,
        "corpus": args.corpus,
        "model": args.model,
        "model_dtype": dtype_str,
        "max_length": args.max_length,
        "shard_size": args.shard_size,
        "limit": args.limit,
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "natural": natural_map is not None,
        "natural_files_sha256": natural_files_sha,
        "n_layers": n_layers,
        "hidden": hidden,
        "seam_convention": "segment-concat ids (prefix|query|template-suffix|response)",
        "pooling": {
            "answer": "response_avg",
            "context": "context_end (last prompt token)",
            "prefix": "prefix_end (last token before the user query)",
        },
    }

    manifest_path = corpus_dir / "manifest.json"
    shards_path = corpus_dir / "shards.jsonl"
    done_shards: set[int] = set()
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("regime") != regime:
            mismatched = sorted(
                k
                for k in set(regime) | set(prior.get("regime") or {})
                if (prior.get("regime") or {}).get(k) != regime.get(k)
            )
            if not args.force:
                raise RuntimeError(
                    f"capture regime MISMATCH vs existing manifest in {corpus_dir} "
                    f"(keys: {mismatched}); pass a fresh --out-root or --force to wipe"
                )
            import shutil

            logger.warning("[capture] --force: wiping %s (regime keys %s)", corpus_dir, mismatched)
            shutil.rmtree(corpus_dir)
            corpus_dir.mkdir(parents=True)
        elif shards_path.exists():
            for row in load_jsonl(shards_path):
                if (corpus_dir / row["file"]).exists():
                    done_shards.add(int(row["shard"]))
    atomic_write_json({"regime": regime, "status": "running"}, manifest_path)

    if args.verify_batched:
        run_verify_batched(model, samples, args, device)
        logger.info("[capture] --verify-batched PASSED; exiting (no shards written)")
        return 0

    n_shards = (len(samples) + args.shard_size - 1) // args.shard_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    for shard in range(n_shards):
        if shard in done_shards:
            logger.info("[capture] shard %d/%d RESUME-SKIP (already complete)", shard + 1, n_shards)
            continue
        lo = shard * args.shard_size
        hi = min(lo + args.shard_size, len(samples))
        n_sh = hi - lo
        rows = _capture_rows_for(samples, lo, hi)
        kinds_present = sorted({r["kind"] for r in rows})
        arrays = {
            f"resp_avg_{k}": np.zeros((n_sh, n_layers, hidden), dtype=np.float16)
            for k in kinds_present
        }
        arrays["last_prompt"] = np.zeros((n_sh, n_layers, hidden), dtype=np.float16)
        arrays["prefix_end"] = np.zeros((n_sh, n_layers, hidden), dtype=np.float16)
        for batch in iter_batches(rows, args.batch_tokens, args.max_batch):
            out = _forward_batch(model, batch, device, pad_id)
            red = _reduce_hidden(out.hidden_states, batch, device)
            del out
            ra = red["resp_avg"].numpy()
            for i, r in enumerate(batch):
                arrays[f"resp_avg_{r['kind']}"][r["pos"]] = ra[i]
            for j, i in enumerate(red["ds_rows"]):
                p = batch[i]["pos"]
                arrays["last_prompt"][p] = red["last_prompt"][j].numpy()
                arrays["prefix_end"][p] = red["prefix_end"][j].numpy()
        shard_samples = samples[lo:hi]
        payload = {
            "schema": CAPTURE_SCHEMA_VERSION,
            "shard": shard,
            "sample_ids": [s.sample_id for s in shard_samples],
            "kinds": {k: __import__("torch").from_numpy(v) for k, v in arrays.items()},
            "prompt_lens": [len(s.prompt_ids) for s in shard_samples],
            "prefix_end_idx": [s.prefix_end for s in shard_samples],
            "resp_lens": {k: [len(s.resp_ids[k]) for s in shard_samples] for k in kinds_present},
            "truncated": {
                k: [bool(s.truncated.get(k)) for s in shard_samples] for k in kinds_present
            },
        }
        import os as _os

        shard_file = corpus_dir / f"shard_{shard:05d}.pt"
        tmp = shard_file.with_name(f"{shard_file.name}.tmp.{_os.getpid()}")
        torch.save(payload, tmp)
        _os.replace(tmp, shard_file)
        append_jsonl(
            shards_path,
            {"shard": shard, "n": n_sh, "file": shard_file.name, "ts": int(time.time())},
        )
        print(
            f"[capture] shard {shard + 1}/{n_shards} corpus={args.corpus} n={n_sh} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    manifest = {
        "regime": regime,
        "status": "complete",
        "n_samples": len(samples),
        "n_shards": n_shards,
        "counters": tok_counters,
        "truncated_fraction_dataset": tok_counters["truncated_dataset"] / max(1, len(samples)),
        "meta": repro_meta("issue2224_predictor_scores.capture"),
    }
    atomic_write_json(manifest, manifest_path)
    logger.info(
        "[capture] DONE corpus=%s n=%d shards=%d (L=%d H=%d) counters=%s",
        args.corpus,
        len(samples),
        n_shards,
        n_layers,
        hidden,
        json.dumps(tok_counters),
    )
    return 0


# ── Scoring (4b-1) ───────────────────────────────────────────────────────────────


def load_rb(path: Path, expect_shape: tuple[int, int]) -> np.ndarray:
    """#778 persona vector: a BARE torch.Tensor, shape/dtype-asserted (plan §10)."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor):
        raise RuntimeError(f"{path}: expected a bare torch.Tensor, got {type(obj)}")
    if tuple(obj.shape) != tuple(expect_shape):
        raise RuntimeError(f"{path}: shape {tuple(obj.shape)} != expected {expect_shape}")
    if obj.dtype != torch.float32:
        raise RuntimeError(f"{path}: dtype {obj.dtype} != torch.float32")
    return obj.numpy().astype(np.float64)


def load_linear_map(path: Path):
    """Frozen linear map in the #1739 ``_save_map`` npz contract."""
    from explore_persona_space.experiments.issue_1739.fits import MapFit

    z = np.load(path, allow_pickle=False)
    required = {"w", "x_mu", "x_sd", "y_mu", "layers"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{path}: missing npz keys {sorted(missing)} — expected the #1739 _save_map "
            f"linear-map contract (w/x_mu/x_sd/y_mu/layers[/meta])"
        )
    meta = {}
    if "meta" in z.files:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {"_unparsed_meta": True}
    m = MapFit(
        w=z["w"].astype(np.float64),
        x_mu=z["x_mu"].astype(np.float64),
        x_sd=z["x_sd"].astype(np.float64),
        y_mu=z["y_mu"].astype(np.float64),
        diagnostics={},
        kind="linear",
    )
    return m, [int(x) for x in z["layers"]], meta


def apply_map_at_layer(m, map_layers: list[int], layer: int, x: np.ndarray) -> np.ndarray:
    """Apply the frozen map at ONE read-out layer via the reviewed apply_map."""
    from explore_persona_space.experiments.issue_1739.fits import MapFit, apply_map

    if layer not in map_layers:
        raise RuntimeError(
            f"read-out layer {layer} not in the map's layers {map_layers} — "
            f"#2222 must ship the map at the selected layer (0-indexed block convention)"
        )
    li = map_layers.index(layer)
    m1 = MapFit(
        w=m.w[li : li + 1],
        x_mu=m.x_mu[li : li + 1],
        x_sd=m.x_sd[li : li + 1],
        y_mu=m.y_mu[li : li + 1],
        diagnostics={},
        kind="linear",
    )
    return apply_map(x[None, :, :], m1)[0]


def load_probe(path: Path, hidden: int, layer: int) -> dict:
    """Form-A ridge probe npz: w (d,), b scalar, optional x_mu/x_sd/layer/meta."""
    z = np.load(path, allow_pickle=False)
    missing = {"w", "b"} - set(z.files)
    if missing:
        raise RuntimeError(
            f"{path}: missing npz keys {sorted(missing)} — expected the Form-A probe "
            f"contract (w (d,), b scalar, optional x_mu/x_sd (d,), layer, meta)"
        )
    w = np.asarray(z["w"], dtype=np.float64).ravel()
    if w.shape != (hidden,):
        raise RuntimeError(f"{path}: probe w shape {w.shape} != ({hidden},)")
    probe = {"w": w, "b": float(np.asarray(z["b"]).ravel()[0])}
    for k in ("x_mu", "x_sd"):
        if k in z.files:
            v = np.asarray(z[k], dtype=np.float64).ravel()
            if v.shape != (hidden,):
                raise RuntimeError(f"{path}: probe {k} shape {v.shape} != ({hidden},)")
            probe[k] = v
    if "layer" in z.files:
        probe_layer = int(np.asarray(z["layer"]).ravel()[0])
        if probe_layer != layer:
            raise RuntimeError(f"{path}: probe layer {probe_layer} != trait read-out layer {layer}")
    return probe


def probe_score(probe: dict, x: np.ndarray) -> np.ndarray:
    """Apply the probe to (n, d) answer-space vectors."""
    xx = x
    if "x_mu" in probe and "x_sd" in probe:
        xx = (x - probe["x_mu"]) / probe["x_sd"]
    return xx @ probe["w"] + probe["b"]


def load_layer_slices(corpus_dir: Path, layers: list[int]):
    """One pass over the capture shards: per-(kind, layer) fp16 slices."""
    import torch

    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"{corpus_dir}/manifest.json status={manifest.get('status')!r} — capture incomplete"
        )
    shard_rows = sorted(load_jsonl(corpus_dir / "shards.jsonl"), key=lambda r: r["shard"])
    ids: list[str] = []
    parts: dict[tuple[str, int], list[np.ndarray]] = {}
    kinds_seen: set[str] = set()
    for row in shard_rows:
        # Self-produced sha-pinned shard bundles (dict of tensors + python
        # lists) — weights_only=False is the sanctioned load for these.
        payload = torch.load(corpus_dir / row["file"], map_location="cpu", weights_only=False)
        ids.extend(payload["sample_ids"])
        for kind, tens in payload["kinds"].items():
            kinds_seen.add(kind)
            for layer in layers:
                parts.setdefault((kind, layer), []).append(tens[:, layer, :].numpy())
    slices = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    return ids, slices, kinds_seen, manifest


def run_score(args) -> int:
    """4b-1: predictor arms at the read-out layer(s) from the capture summaries."""
    if args.summaries_dir is None or args.corpus is None:
        raise RuntimeError("--phase score requires --summaries-dir and --corpus")
    corpus_dir = Path(args.summaries_dir)
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise RuntimeError(f"unknown arms {sorted(unknown)}; valid: {ARMS}")

    # Read-out layer per trait (#2222's selection) — REQUIRED, no default.
    if args.layers_json:
        layers_map = {k: int(v) for k, v in json.loads(Path(args.layers_json).read_text()).items()}
        missing = [t for t in traits if t not in layers_map]
        if missing:
            raise RuntimeError(f"--layers-json missing traits {missing}")
    elif args.layer is not None:
        layers_map = {t: int(args.layer) for t in traits}
    else:
        raise RuntimeError(
            "read-out layer required: pass --layer or --layers-json (#2222's read-out selection)"
        )

    if args.persona_vectors_dir is None:
        raise RuntimeError("--persona-vectors-dir required (the #778 rb/{trait}.pt tensors)")

    need_maps = bool({"mapped_dp", "probe_diff"} & set(arms))
    if need_maps:
        missing_flags = [
            f
            for f, v in (("--map-context", args.map_context), ("--map-prefix", args.map_prefix))
            if not v
        ]
        if missing_flags and not args.allow_single_mapping_arm:
            raise RuntimeError(
                f"arms {sorted({'mapped_dp', 'probe_diff'} & set(arms))} need BOTH mapping arms "
                f"(standing prefix+context rule); missing {missing_flags}. Pass them, or "
                f"--allow-single-mapping-arm to record an explicit one-arm deviation."
            )
    if "probe_diff" in arms and not args.probe_dir:
        raise RuntimeError("arm probe_diff requires --probe-dir (the #2222 Form-A probe npzs)")

    needed_layers = sorted({layers_map[t] for t in traits})
    ids, slices, kinds_seen, manifest = load_layer_slices(corpus_dir, needed_layers)
    regime = manifest["regime"]
    n_layers, hidden = int(regime["n_layers"]), int(regime["hidden"])
    for t in traits:
        if not (0 <= layers_map[t] < n_layers):
            raise RuntimeError(f"layer {layers_map[t]} for {t} out of range [0, {n_layers})")
    if "exact_dp" in arms and "resp_avg_natural" not in kinds_seen:
        raise RuntimeError(
            "arm exact_dp needs resp_avg_natural summaries, but the capture ran without "
            "--natural-dir (manifest regime natural=false) — run P0b + re-capture"
        )

    expect_rb = tuple(int(x) for x in args.expect_rb_shape.split(","))
    if expect_rb != (n_layers, hidden):
        raise RuntimeError(
            f"--expect-rb-shape {expect_rb} != capture dims ({n_layers}, {hidden}) — "
            f"persona vectors and summaries must share the layer/hidden convention"
        )

    maps = {}
    if need_maps:
        for arm_name, p in (("context", args.map_context), ("prefix", args.map_prefix)):
            if p:
                m, map_layers, map_meta = load_linear_map(Path(p))
                if m.w.shape[2] != hidden:
                    raise RuntimeError(f"{p}: map hidden dim {m.w.shape[2]} != {hidden}")
                pool_key = "input_pooling"
                expected_pool = {"context": "context_end", "prefix": "prefix_end"}[arm_name]
                if map_meta.get(pool_key) and map_meta[pool_key] != expected_pool:
                    raise RuntimeError(
                        f"{p}: map {pool_key}={map_meta[pool_key]!r} != {expected_pool!r} "
                        f"(plan §12 A11 pooling-convention mismatch)"
                    )
                maps[arm_name] = (m, map_layers, map_meta, str(p))

    out_dir = Path(args.out_dir) if args.out_dir else SCREENING_SCORES_DIR_DEFAULT / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    for trait in traits:
        layer = layers_map[trait]
        rb_path = Path(args.persona_vectors_dir) / f"{trait}.pt"
        if not rb_path.exists():
            raise RuntimeError(f"persona vector missing: {rb_path} (#778 rb/{trait}.pt)")
        rb = load_rb(rb_path, expect_rb)
        v = rb[layer]
        v_hat = v / np.linalg.norm(v)

        a_ds = slices[("resp_avg_dataset", layer)].astype(np.float64)
        c_last = slices[("last_prompt", layer)].astype(np.float64)
        p_end = slices[("prefix_end", layer)].astype(np.float64)

        preds: dict[str, np.ndarray] = {}
        if "context" in maps:
            m, ml, _, _ = maps["context"]
            preds["context"] = apply_map_at_layer(m, ml, layer, c_last)
        if "prefix" in maps:
            m, ml, _, _ = maps["prefix"]
            preds["prefix"] = apply_map_at_layer(m, ml, layer, p_end)

        scores: dict[str, np.ndarray] = {}
        if "raw" in arms:
            scores["raw"] = a_ds @ v_hat
        if "exact_dp" in arms:
            a_nat = slices[("resp_avg_natural", layer)].astype(np.float64)
            scores["exact_dp"] = (a_ds - a_nat) @ v_hat
        if "prompttoken_dp" in arms:
            scores["prompttoken_dp"] = (a_ds - c_last) @ v_hat
        if "mapped_dp" in arms:
            for side, pred in preds.items():
                scores[f"mapped_dp_{side}"] = (a_ds - pred) @ v_hat
        if "probe_diff" in arms:
            probe = load_probe(Path(args.probe_dir) / f"{trait}.npz", hidden, layer)
            for side, pred in preds.items():
                scores[f"probe_diff_{side}"] = probe_score(probe, a_ds) - probe_score(probe, pred)

        arm_stats = {
            k: {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "p5": float(np.percentile(x, 5)),
                "p50": float(np.percentile(x, 50)),
                "p95": float(np.percentile(x, 95)),
            }
            for k, x in scores.items()
        }
        payload = {
            "meta": {
                "issue": 2224,
                "corpus": args.corpus,
                "trait": trait,
                "readout_layer": layer,
                "layer_convention": "0-indexed block (#778 rb rows; row 19 = paper layer 20)",
                "arms": sorted(scores),
                "n_samples": len(ids),
                "artifacts": {
                    "rb": {"path": str(rb_path), "sha256": sha256_file(rb_path)},
                    "map_context": None
                    if "context" not in maps
                    else {"path": maps["context"][3], "sha256": sha256_file(maps["context"][3])},
                    "map_prefix": None
                    if "prefix" not in maps
                    else {"path": maps["prefix"][3], "sha256": sha256_file(maps["prefix"][3])},
                    "probe": None
                    if "probe_diff" not in arms
                    else {
                        "path": str(Path(args.probe_dir) / f"{trait}.npz"),
                        "sha256": sha256_file(Path(args.probe_dir) / f"{trait}.npz"),
                    },
                },
                "summaries_dir": str(corpus_dir),
                "capture_regime": regime,
                "pooling_ledger": regime["pooling"],
                "supervision_ledger": (
                    "map=trait-agnostic; persona vector=trait-description+judge-filter; "
                    "probe=judge-labels"
                ),
                "dv_note": "screening scores are predictors, never the construct (plan §6)",
                "arm_stats": arm_stats,
                "allow_single_mapping_arm": bool(args.allow_single_mapping_arm),
                "repro": repro_meta("issue2224_predictor_scores.score"),
            },
            "scores": {
                sid: {k: round(float(x[i]), 6) for k, x in scores.items()}
                for i, sid in enumerate(ids)
            },
        }
        out_path = out_dir / f"{trait}.json"
        atomic_write_json(payload, out_path)
        logger.info(
            "[score] corpus=%s trait=%s layer=%d arms=%s n=%d -> %s",
            args.corpus,
            trait,
            layer,
            sorted(scores),
            len(ids),
            out_path,
        )
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────

PHASES = {"capture": run_capture, "score": run_score}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #2224 P0c capture + 4b-1 predictor scoring (plan v3 §4)."
    )
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--list-phases", action="store_true", help="print the phase registry")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--corpus", default=None, help="pool corpus slug (lmsys | ultrachat | ...)")
    # capture args
    parser.add_argument(
        "--pool", type=Path, default=None, help="pool JSONL (issue2224_build_pools)"
    )
    parser.add_argument("--out-root", type=Path, default=None, help="capture out root (explicit)")
    parser.add_argument("--natural-dir", type=Path, default=None, help="P0b natural-response dir")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    parser.add_argument("--model-dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--batch-tokens", type=int, default=32768, help="padded-token batch budget")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap pool rows (smoke slices)")
    parser.add_argument(
        "--verify-batched",
        type=int,
        default=None,
        help="batched-vs-serial equivalence gate on N samples, then exit (no shards)",
    )
    parser.add_argument("--force", action="store_true", help="wipe a regime-mismatched capture dir")
    # score args
    parser.add_argument("--summaries-dir", type=Path, default=None, help="<out-root>/<corpus>")
    parser.add_argument("--traits", default=",".join(TRAITS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--persona-vectors-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=None, help="read-out layer for ALL traits")
    parser.add_argument("--layers-json", type=Path, default=None, help='{"evil": 19, ...}')
    parser.add_argument(
        "--map-context", type=Path, default=None, help="frozen map npz (context arm)"
    )
    parser.add_argument("--map-prefix", type=Path, default=None, help="frozen map npz (prefix arm)")
    parser.add_argument("--probe-dir", type=Path, default=None, help="dir of <trait>.npz probes")
    parser.add_argument(
        "--allow-single-mapping-arm",
        action="store_true",
        help="explicit one-mapping-arm deviation (standing rule wants BOTH; recorded in meta)",
    )
    parser.add_argument(
        "--expect-rb-shape",
        default=f"{N_LAYERS},{HIDDEN_DIM}",
        help="expected persona-vector shape (default the #778 (28,3584) contract)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="score output dir")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        return 0
    if args.import_check:
        import importlib

        for mod in ("numpy", "torch", "transformers"):
            importlib.import_module(mod)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.experiments.issue_1739.fits import (  # noqa: F401
            MapFit,
            apply_map,
        )
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_predictor_scores")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required (capture | score); see --list-phases")
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
