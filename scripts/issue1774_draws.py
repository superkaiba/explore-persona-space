"""#1774 P1 — multi-draw decode-noise generation (vLLM, K=5, temp 1.0) + teacher-forced
t1 capture at L14/18/19 (pod, GPU).

Contexts frozen at P0 into ``registry/draws_manifest.json``. Generation is chunked
(≤500 prompts/call, ``EPM_VLLM_GREEDY_CHUNK_SIZE`` convention) with hang-mitigation
knobs ON by default for this real-corpus long-prompt workload (#1324 checklist).
Raw completion TEXT uploads to HF the moment generation completes, BEFORE capture
(upload-policy: store-before-long-consumer). Capture reuses the #1092 rig
(`_capture_batch_loaded_model` — token-id concatenation + offset positions).

Stages: ``--stage pilot|gen|capture|upload`` with ``--shard i/n`` context sharding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE the heavy imports below (BLAS/torch
# pools freeze at import time; tests/test_shared_vm_thread_caps.py).
load_dotenv()

# vLLM v1 fork-EngineCore silent death (#628): set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# #1324 pre-launch checklist: hang-mitigation knobs ON for real-corpus prompts.
os.environ.setdefault("EPM_VLLM_ENFORCE_EAGER", "1")
os.environ.setdefault("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")

import numpy as np  # noqa: E402

import issue1774_common as c  # noqa: E402

GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 1024  # free-gen default (plan §4 P1)
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
CAPTURE_CHUNK = 500  # rows per persisted capture part (checkpoint-per-unit)
G2_MAX_ABS = float(os.environ.get("I1774_G2_MAX_ABS", "0.25"))
G2_COS_MIN = float(os.environ.get("I1774_G2_COS_MIN", "0.999"))


def _draw_contexts(out_root: str | None) -> tuple[list[int], int]:
    reg = json.loads((c.eval_out(out_root) / "registry/draws_manifest.json").read_text())
    return list(reg["manifest_indices"]), int(reg["k_draws"])


def _shard(items: list, shard: str) -> list:
    i, n = (int(x) for x in shard.split("/"))
    assert 0 <= i < n, shard
    return items[i::n]


def _render_rows(rows: list[dict], indices: list[int]) -> list[dict]:
    """Render (prefix, prompt) for each draw context under the instruct format."""
    from issue1092_gpu_phase import _prefix_turns, _query_text, _render_prompt_parts, load_store

    prefix_store = load_store(c.stage_dir() / "corpus", "prefix_store.jsonl")
    query_store = load_store(c.stage_dir() / "corpus", "query_store.jsonl")
    out = []
    for mi in indices:
        row = rows[mi]
        turns = _prefix_turns(prefix_store[str(row["prefix_id"])])
        query = _query_text(query_store[str(row["query_id"])])
        prefix_text, prompt = _render_prompt_parts(turns, query, "instruct")
        out.append(
            {
                "manifest_index": mi,
                "prefix_id": str(row["prefix_id"]),
                "query_id": str(row["query_id"]),
                "prefix_text": prefix_text,
                "prompt": prompt,
            }
        )
    return out


def _gen_path(out_root: str | None, shard: str) -> Path:
    tag = shard.replace("/", "of")
    return c.data_out(out_root) / "draws/raw_completions" / f"gen_shard{tag}.jsonl"


def stage_gen(args) -> int:
    rows = c.load_manifest()
    indices, k_draws = _draw_contexts(args.out_root)
    if args.limit:
        indices = indices[: args.limit]
    ctxs = _shard(_render_rows(rows, indices), args.shard)
    out_path = _gen_path(args.out_root, args.shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set[str] = set()
    if out_path.exists():
        done_keys = {f"{r['manifest_index']}" for r in c.jsonl_rows(out_path)}
        print(f"[p1-gen] resume: {len(done_keys)} contexts already generated")
    todo = [ctx for ctx in ctxs if str(ctx["manifest_index"]) not in done_keys]
    if not todo:
        print("[p1-gen] nothing to do")
        return 0

    from vllm import LLM, SamplingParams

    engine_kwargs: dict = {
        "model": c.INSTRUCT_MODEL,
        "revision": c.INSTRUCT_REVISION,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    }
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        engine_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        engine_kwargs["enable_prefix_caching"] = False
    llm = LLM(**engine_kwargs)
    sp = SamplingParams(
        n=k_draws,
        temperature=GEN_TEMPERATURE,
        top_p=1.0,
        max_tokens=GEN_MAX_TOKENS,
        seed=c.SEED_DRAWS,
    )
    n_chunks = (len(todo) + VLLM_CHUNK - 1) // VLLM_CHUNK
    t0 = time.time()
    for ci in range(0, len(todo), VLLM_CHUNK):
        chunk = todo[ci : ci + VLLM_CHUNK]
        print(
            f"[vllm-chunk] p1-gen chunk {ci // VLLM_CHUNK + 1}/{n_chunks} "
            f"({len(chunk)} prompts, K={k_draws}) elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        outs = llm.generate([ctx["prompt"] for ctx in chunk], sp, use_tqdm=False)
        with out_path.open("a", encoding="utf-8") as fh:
            for ctx, o in zip(chunk, outs, strict=True):
                rec = dict(ctx)
                rec["draws"] = [cand.text for cand in o.outputs]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[p1-gen] unit {min(ci + VLLM_CHUNK, len(todo))}/{len(todo)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    per_gen = (time.time() - t0) / max(1, len(todo) * k_draws)
    print(f"[p1-gen] done: {len(todo)} contexts x {k_draws} draws, {per_gen:.2f}s/gen")
    return 0


def _load_hf_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("I1774_CAPTURE_MODEL", c.INSTRUCT_MODEL)
    rev = c.INSTRUCT_REVISION if model_id == c.INSTRUCT_MODEL else None
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tok


def _capture_t1(gen_rows: list[dict], draw_k: int, model, tok, device: str) -> np.ndarray:
    """Teacher-forced t1 (answer-span mean) for one draw index; (n, n_layers_kept, D) fp16."""
    from issue1092_gpu_phase import _capture_batch_loaded_model

    n_layers = model.config.num_hidden_layers
    out = _capture_batch_loaded_model(
        prefix_texts=[r["prefix_text"] for r in gen_rows],
        prompts=[r["prompt"] for r in gen_rows],
        completions=[r["draws"][draw_k] if r["draws"][draw_k] else " " for r in gen_rows],
        prompt_format="instruct",
        model=model,
        tokenizer=tok,
        n_layers=n_layers,
        hidden_dim=model.config.hidden_size,
        device=device,
        log_label=f"p1-cap-draw{draw_k}",
    )
    # keep only the consumed layers (store convention: L -> hidden_states index L-1)
    kept = np.stack(
        [np.stack([s["t1"][layer - 1] for layer in c.LAYERS], axis=0) for s in out.summaries],
        axis=0,
    ).astype(np.float16)
    return kept  # (n, len(LAYERS), D)


def _capture_regime_guard(sdir, tag: str, regime: dict) -> None:
    """Fail-loud resume regime key (#722 r3): row_index + part files are resume
    state keyed on (limit, shard, k_draws, realized rows) — a re-run with a
    different ``--limit`` in the SAME out-root must REFUSE, never silently join
    fresh parts against a stale row_index."""
    rp = sdir / f"capture_regime_shard{tag}.json"
    if rp.exists():
        prior = json.loads(rp.read_text())
        if prior != regime:
            raise RuntimeError(
                f"stage_capture resume regime mismatch in {sdir}: prior={prior} "
                f"now={regime} — use a fresh --out-root (or clear the stale "
                "summaries) instead of resuming across regimes"
            )
    else:
        c.write_json_atomic(rp, regime)


def stage_capture(args) -> int:
    device = "cuda:0" if _cuda_ok() else "cpu"
    rows = c.load_manifest()
    indices, k_draws = _draw_contexts(args.out_root)
    if args.limit:
        indices = indices[: args.limit]
    gen_rows_all: list[dict] = []
    droot = c.data_out(args.out_root) / "draws/raw_completions"
    for p in sorted(droot.glob("gen_shard*.jsonl")):
        gen_rows_all.extend(c.jsonl_rows(p))
    by_mi = {int(r["manifest_index"]): r for r in gen_rows_all}
    missing = [mi for mi in indices if mi not in by_mi]
    assert not missing, f"{len(missing)} draw contexts have no generation rows: {missing[:5]}"
    ordered = [by_mi[mi] for mi in indices]
    ordered = _shard(ordered, args.shard)
    sdir = c.data_out(args.out_root) / "draws/summaries"
    sdir.mkdir(parents=True, exist_ok=True)
    tag = args.shard.replace("/", "of")
    # regime guard BEFORE model init (consumer contracts assert pre-GPU-load)
    _capture_regime_guard(
        sdir,
        tag,
        {
            "limit": int(args.limit or 0),
            "shard": str(args.shard),
            "k_draws": int(k_draws),
            "n_rows": len(ordered),
            "manifest_index_first_last": (
                [int(ordered[0]["manifest_index"]), int(ordered[-1]["manifest_index"])]
                if ordered
                else []
            ),
        },
    )
    model, tok = _load_hf_model(device)
    t0 = time.time()
    idx_path = sdir / f"row_index_shard{tag}.jsonl"
    if not idx_path.exists():
        with idx_path.open("w", encoding="utf-8") as fh:
            for r in ordered:
                fh.write(
                    json.dumps({k: r[k] for k in ("manifest_index", "prefix_id", "query_id")})
                    + "\n"
                )
    for k in range(k_draws):
        for part0 in range(0, len(ordered), CAPTURE_CHUNK):
            part_rows = ordered[part0 : part0 + CAPTURE_CHUNK]
            pj = part0 // CAPTURE_CHUNK
            done = all(
                (sdir / f"t1_L{layer}_draw{k}_shard{tag}_part{pj}.npy").exists()
                for layer in c.LAYERS
            )
            if done:
                continue
            kept = _capture_t1(part_rows, k, model, tok, device)
            for li, layer in enumerate(c.LAYERS):
                np.save(sdir / f"t1_L{layer}_draw{k}_shard{tag}_part{pj}.npy", kept[:, li])
            print(
                f"[p1-cap] unit draw{k} part{pj} rows={len(part_rows)} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    n_rows = len(ordered) * k_draws
    print(f"[p1-cap] done: {n_rows} row-draws, {(time.time() - t0) / max(1, n_rows):.2f}s/row")
    return 0


def _cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


def _drain_vllm_release(device: str, timeout_s: float = 60.0, frac_free: float = 0.55) -> None:
    """Bounded post-reap drain verify (gotchas #1090 pattern): wait until the
    reaped engine's HBM is actually released BEFORE the HF 7B load; engine
    teardown is async, so a single-shot check races the SIGKILL→driver-release
    window. Fail loud on timeout naming the residual (an orphaned EngineCore
    holds ~gpu_memory_utilization of HBM and would OOM the pilot's HF load)."""
    import torch

    if not device.startswith("cuda"):
        return
    idx = torch.device(device).index or 0
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        free, total = torch.cuda.mem_get_info(idx)
        if free / total >= frac_free:
            print(f"[p1-pilot] vLLM engine drained: free={free / 2**30:.1f} GiB", flush=True)
            return
        time.sleep(2.0)
    free, total = torch.cuda.mem_get_info(idx)
    raise RuntimeError(
        f"vLLM engine memory not released after {timeout_s:.0f}s: "
        f"free={free / 2**30:.1f}/{total / 2**30:.1f} GiB (< {frac_free:.0%} floor) — "
        "orphaned EngineCore would OOM the HF pilot load (gotchas: vLLM teardown)"
    )


def stage_pilot(args) -> int:
    """10-context K=2 pilot through the production entrypoints + identity gate.

    Measures per-gen + per-capture-row wall (the §9 pilot-gated basis) and runs
    the capture identity gate (teacher-forced full-forward vs KV-cached
    generate-hook states on spot rows). Refusal is artifact-routed: writes
    pilot_gate_report.json and exits rc=7 (never a bare rc=1 — gotchas #1415).
    """
    import torch

    rows = c.load_manifest()
    indices, _k = _draw_contexts(args.out_root)
    pilot_n = min(10, len(indices))
    ctxs = _render_rows(rows, indices[:pilot_n])
    report: dict = {"meta": c.repro_meta({"script": "issue1774_draws.py --stage pilot"})}
    device = "cuda:0" if _cuda_ok() else "cpu"

    # gen timing at production shape (vLLM, K=2), skipped off-GPU
    if device.startswith("cuda"):
        from vllm import LLM, SamplingParams

        t0 = time.time()
        llm = LLM(
            model=c.INSTRUCT_MODEL,
            revision=c.INSTRUCT_REVISION,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.5,
            enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1",
            enable_prefix_caching=not os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1",
        )
        sp = SamplingParams(
            n=2, temperature=GEN_TEMPERATURE, max_tokens=GEN_MAX_TOKENS, seed=c.SEED_DRAWS
        )
        t_gen0 = time.time()
        outs = llm.generate([x["prompt"] for x in ctxs], sp, use_tqdm=False)
        per_gen = (time.time() - t_gen0) / (pilot_n * 2)
        report["per_gen_s"] = per_gen
        report["engine_init_s"] = t_gen0 - t0
        completions = [o.outputs[0].text for o in outs]
        distinct = sum(1 for o in outs if o.outputs[0].text != o.outputs[1].text)
        report["n_distinct_draw_pairs"] = distinct  # asm 10: n=2 draws are independent
        # M1 (round 2): a bare `del llm` does NOT reap the v1 EngineCore
        # subprocess — its ~0.5-util KV cache stays pinned into the HF 7B load
        # (gotchas: vLLM in-process teardown). Reap, then bounded drain verify.
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(llm)
        del llm
        _drain_vllm_release(device)
    else:
        completions = ["pilot smoke completion." for _ in ctxs]
        report["per_gen_s"] = None

    model, tok = _load_hf_model(device)
    for r, comp in zip(ctxs, completions, strict=True):
        r["draws"] = [comp, comp]
    t_cap0 = time.time()
    kept = _capture_t1(ctxs, 0, model, tok, device)
    report["per_capture_row_s"] = (time.time() - t_cap0) / pilot_n
    assert kept.shape == (pilot_n, len(c.LAYERS), c.HIDDEN_DIM), kept.shape

    # identity gate on 8 spot rows: teacher-forced full forward == KV-cached
    # incremental forward at the SAME positions (the generate-time computation).
    from issue1092_gpu_phase import _boundary_suffix, _capture_row_ids_and_positions

    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

    # M1 (round 2): the gate never reads logits — skip the full-vocab
    # (B x T x 152k) lm_head materialization (introspection-guarded, #779).
    ltk = _logits_to_keep_kwargs(model, return_logits=False)
    spot = ctxs[: min(8, pilot_n)]
    max_abs, cos_min = 0.0, 1.0
    li = c.HEADLINE_LAYER - 1
    for r in spot:
        row_ids, pos = _capture_row_ids_and_positions(
            tok, r["prefix_text"], r["prompt"], r["draws"][0], _boundary_suffix("instruct")
        )
        ids = torch.tensor([row_ids], device=device)
        with torch.no_grad():
            full = model(input_ids=ids, output_hidden_states=True, **ltk).hidden_states[1:]
            n_prompt = pos["n_prompt"]
            pre = model(
                input_ids=ids[:, :n_prompt], use_cache=True, output_hidden_states=True, **ltk
            )
            inc = model(
                input_ids=ids[:, n_prompt:],
                past_key_values=pre.past_key_values,
                output_hidden_states=True,
                **ltk,
            ).hidden_states[1:]
        span = slice(pos["answer_start"], pos["answer_end"])
        tf = full[li][0, span, :].float().mean(0)
        inc_span = slice(span.start - n_prompt, span.stop - n_prompt)
        gh = inc[li][0, inc_span, :].float().mean(0)
        max_abs = max(max_abs, float((tf - gh).abs().max()))
        cos_min = min(cos_min, float(torch.nn.functional.cosine_similarity(tf, gh, dim=0)))
    report["identity_gate"] = {
        "n_spot_rows": len(spot),
        "max_abs": max_abs,
        "cos_min": cos_min,
        "bar_max_abs": G2_MAX_ABS,
        "bar_cos_min": G2_COS_MIN,
        "pass": bool(max_abs <= G2_MAX_ABS and cos_min >= G2_COS_MIN),
    }
    out = c.eval_out(args.out_root) / "draws"
    c.write_json_atomic(out / "pilot_gate_report.json", report)
    print(f"[p1-pilot] {json.dumps(report['identity_gate'])}")
    if not report["identity_gate"]["pass"]:
        print("[p1-pilot] IDENTITY GATE FAIL — halt (kill criterion; rc=7)")
        return 7
    return 0


def stage_upload(args) -> int:
    """Upload raw completion text (+ draw summaries when present) to HF, fail-loud."""
    from explore_persona_space.orchestrate import hub

    droot = c.data_out(args.out_root) / "draws"
    gen_files = sorted((droot / "raw_completions").glob("gen_shard*.jsonl"))
    assert gen_files, f"no generation shards under {droot / 'raw_completions'}"
    hub._upload(
        droot / "raw_completions",
        repo_id=c.DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{c.HF_UPLOAD_PREFIX}/raw_completions/draws",
    )
    expected = [f"{c.HF_UPLOAD_PREFIX}/raw_completions/draws/{p.name}" for p in gen_files]
    sdir = droot / "summaries"
    if sdir.exists() and any(sdir.glob("t1_*.npy")):
        hub._upload(
            sdir,
            repo_id=c.DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{c.HF_UPLOAD_PREFIX}/draws/summaries",
        )
        expected += [
            f"{c.HF_UPLOAD_PREFIX}/draws/summaries/{p.name}" for p in sorted(sdir.iterdir())
        ]
    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        c.DATA_REPO,
        expected,
        path_in_repo=c.HF_UPLOAD_PREFIX,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"p1 upload verify missing {len(missing)} paths: {sorted(missing)[:5]}")
    print(f"[p1-upload] verified {len(expected)} paths on {c.DATA_REPO}/{c.HF_UPLOAD_PREFIX}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["pilot", "gen", "capture", "upload"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="context cap (smoke)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real branch, then exit",
    )
    args = ap.parse_args(argv)
    if args.import_check:
        from issue1092_gpu_phase import (  # noqa: F401
            _boundary_suffix,
            _capture_batch_loaded_model,
            _capture_row_ids_and_positions,
            _prefix_turns,
            _query_text,
            _render_prompt_parts,
            load_store,
        )

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload,
            verify_repo_paths_uploaded,
        )

        print("[import-check] p1 deferred imports resolve")
        return 0
    print(f"[phase=p1_{args.stage}] shard={args.shard}")
    rc = {"pilot": stage_pilot, "gen": stage_gen, "capture": stage_capture, "upload": stage_upload}[
        args.stage
    ](args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit: heavy C-extension atexit race (gotchas #1689)


if __name__ == "__main__":
    main()
