"""Generation and capture workers; one launcher-pinned GPU per fresh child process."""

from __future__ import annotations

import json
import os
import time
import numpy as np
import issue1902_format_common as C


def tokenizer_for(model):
    """Load both encoding and template tokenizers at their explicit revisions."""
    from transformers import AutoTokenizer

    family = model.split("_")[0]
    mid, revision, _, _ = C.MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        mid, revision=revision, use_fast=True, padding_side="right"
    )
    template_id, template_revision, _, _ = C.MODELS[family + "_S"]
    template = AutoTokenizer.from_pretrained(template_id, revision=template_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if family == "olmo":
        rendered = C.render_prompt(family, "chat", "test", template)
        assert rendered == "<|endoftext|><|user|>\ntest\n<|assistant|>\n", repr(rendered)
    return tokenizer, template


def generate(root, model, first_chunk=False, shard=0, shards=1):
    """Generate only missing banks using the parent's within-family settings."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    manifest = json.loads((root / "manifest.json").read_text())
    family = model.split("_")[0]
    cohort = manifest[family]
    ids = cohort["ids"]
    fp = C.fingerprint(root)
    fresh = [b for b in C.banks(model) if b["fresh"]]
    if not fresh:
        return
    offsets = C.owned_offsets(model, len(ids), shard, shards, first_chunk)
    needed = [
        (b, off)
        for b in fresh
        for off in offsets
        if not C.complete(root / "banks" / model / b["name"] / f"chunk_{off:05d}.json", fp)
    ]
    if not needed:
        return
    tokenizer, template = tokenizer_for(model)
    mid, revision, _, _ = C.MODELS[model]
    max_model_len = 8192 if family == "qwen" else 4096
    llm = LLM(
        model=mid,
        revision=revision,
        tokenizer_revision=revision,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_num_seqs=128,
        tensor_parallel_size=1,
        seed=137,
    )
    pending = []
    for bank, offset in needed:
        started = time.monotonic()
        batch = ids[offset : offset + C.CHUNK]
        prompts, params, row_keys = [], [], []
        stop = (
            (["\nUser:"] if family == "qwen" else ["\nUser:", "User:"])
            if (bank["render"] == "plain")
            else (["<|im_end|>"] if family == "qwen" else None)
        )
        for cid in batch:
            prompt = C.render_prompt(family, bank["render"], cohort["questions"][cid], template)
            assert (
                len(tokenizer.encode(prompt, add_special_tokens=False)) + bank["cap"]
                <= max_model_len
            )
            for draw in range(5):
                if family == "olmo":
                    seed = [42, 45, 46, 47, 48][draw]
                else:
                    import hashlib

                    seed = int.from_bytes(
                        hashlib.sha256(f"1902-format|{model}|{cid}|{draw}".encode()).digest()[:4],
                        "little",
                    )
                prompts.append(prompt)
                params.append(
                    SamplingParams(
                        n=1,
                        temperature=1.0,
                        top_p=1.0 if family == "qwen" else 0.95,
                        max_tokens=bank["cap"],
                        stop=stop,
                        seed=seed,
                    )
                )
                row_keys.append((cid, draw, seed))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        assert len(outputs) == len(row_keys)
        records = []
        for output, (cid, draw, seed) in zip(outputs, row_keys, strict=True):
            assert len(output.outputs) == 1
            completion = output.outputs[0]
            records.append(
                dict(
                    id=cid,
                    query=cohort["questions"][cid],
                    draw=draw,
                    seed=seed,
                    answer=completion.text,
                    token_ids=list(completion.token_ids),
                    finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,
                    generated_token_count=len(completion.token_ids),
                    max_tokens=bank["cap"],
                    generation_checkpoint=model,
                    generation_render=bank["render"],
                    model_revision=revision,
                )
            )
        path = root / "banks" / model / bank["name"] / f"chunk_{offset:05d}.json"
        raw_files = C.write_raw(path, records)
        timing = path.with_suffix(".timing.json")
        seconds = time.monotonic() - started
        C.write_json(
            timing,
            dict(
                model=model,
                bank=bank,
                offset=offset,
                seconds=seconds,
                rows=len(records),
                generation_token_count=sum(r["generated_token_count"] for r in records),
                first_sampling_params=str(params[0]),
                stage="generation",
            ),
        )
        pending.extend([*raw_files, timing])
        if len(pending) >= 8 or first_chunk:
            C.upload_many(pending, root, fp)
            pending.clear()
        print(
            f"[phase=generation] {model}/{bank['name']} offset={offset} seconds={seconds:.1f}",
            flush=True,
        )
    if pending:
        C.upload_many(pending, root, fp)


def encode_entry(tokenizer, template, family, form, query, answer=None):
    """Keep answer spans explicit and never truncate a fixed answer text."""
    prompt = C.render_prompt(family, form, query, template)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert prompt_ids
    if answer is None:
        return dict(
            ids=prompt_ids,
            lo=len(prompt_ids) - 1,
            hi=len(prompt_ids),
            answer_tokens=0,
            stripped_chars=0,
            seam_straddles=False,
        )
    if not answer.strip():
        return None  # Undefined mean answer vector; counted, never a dummy target.
    if family == "olmo":
        normalized = answer.strip()  # Existing OLMo capture convention, same in both formats.
        segment = (" " if form == "plain" else "") + normalized
        answer_ids = tokenizer.encode(segment, add_special_tokens=False)
        assert answer_ids
        return dict(
            ids=prompt_ids + answer_ids,
            lo=len(prompt_ids),
            hi=len(prompt_ids) + len(answer_ids),
            answer_tokens=len(answer_ids),
            stripped_chars=len(answer) - len(normalized),
            seam_straddles=False,
        )
    # Existing Qwen offset-span policy; X itself is always captured prompt-only.
    text = prompt + answer + ("<|im_end|>" if form == "chat" else "")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prompt), len(prompt) + len(answer)
    selected = [
        i for i, (lo, hi) in enumerate(encoded["offset_mapping"]) if hi > start and lo < end
    ]
    assert selected and selected == list(range(selected[0], selected[-1] + 1))
    return dict(
        ids=encoded["input_ids"],
        lo=selected[0],
        hi=selected[-1] + 1,
        answer_tokens=len(selected),
        stripped_chars=0,
        seam_straddles=encoded["offset_mapping"][selected[0]][0] < start,
    )


def forward(model, tokenizer, entries, block_index, dim, *, fp32=True):
    """Length-batched block capture, preserving each parent's pooling convention."""
    import torch

    vectors = np.full((len(entries), dim), np.nan, dtype=np.float16)
    ordered = sorted(
        [i for i, e in enumerate(entries) if e is not None], key=lambda i: len(entries[i]["ids"])
    )
    assert all(
        len(entries[i]["ids"]) <= min(8192, model.config.max_position_embeddings) for i in ordered
    ), "Oversize full text; never truncate"
    hidden = {}

    def hook(_module, _inputs, output):
        hidden["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[block_index].register_forward_hook(hook)
    try:
        start = 0
        while start < len(ordered):
            end = start + 1
            while (
                end < len(ordered)
                and end - start < 8
                and (len(entries[ordered[end]]["ids"]) * (end - start + 1) <= 8192)
            ):
                end += 1
            indices = ordered[start:end]
            padded = tokenizer.pad(
                {"input_ids": [entries[i]["ids"] for i in indices]},
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                model.model(
                    **{k: v.to("cuda") for k, v in padded.items()},
                    use_cache=False,
                    output_hidden_states=False,
                )
            hs = hidden.pop("h")
            assert hs.shape[-1] == dim
            for row, i in enumerate(indices):
                entry = entries[i]
                values = hs[row, entry["lo"] : entry["hi"]]
                vectors[i] = (
                    (values.float() if fp32 else values).mean(0).to(torch.float16).cpu().numpy()
                )
            del hs, padded
            start = end
    finally:
        handle.remove()
    assert np.isfinite(vectors[ordered]).all()
    return vectors


def save_npz(path, **arrays):
    """Atomic array checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def capture_anchor(root, model_name, model, tokenizer, template, block_index):
    """Verify hook identity and rank the historical block against its neighbors."""
    import torch

    fp = C.fingerprint(root)
    audit_path = root / "audits" / f"{model_name}.json"
    if C.complete(audit_path, fp):
        return
    family = model_name.split("_")[0]
    reports = []
    for bank in C.banks(model_name):
        if bank["fresh"]:
            continue
        form = bank["render"] if family == "qwen" else "plain"
        raw = C.read_raw(root / "banks" / model_name / bank["name"] / "chunk_00000.json")
        with np.load(root / "references" / model_name / f"{form}.npz", allow_pickle=False) as p:
            by_id = {str(cid): i for i, cid in enumerate(p["ids"])}
            reference = p["y"].astype(np.float32)
        candidates = []
        # Match parent capture's old 1024 truncation only by selecting unaffected audit rows.
        for start in range(0, len(raw), 5):
            group = raw[start : start + 5]
            entries = [
                encode_entry(tokenizer, template, family, form, r["query"], r["answer"])
                for r in group
            ]
            if all(
                e is not None and (family != "olmo" or e["answer_tokens"] <= 1024) for e in entries
            ):
                candidates.append((group[0]["id"], entries))
            if len(candidates) == 8:  # Parent #2054 layer-adjudication audit size.
                break
        assert len(candidates) == 8, "Insufficient untruncated original rows for layer anchor"
        same, lower, upper, pooling_delta = [], [], [], []
        hook_error = 0.0
        for cid, entries in candidates:
            vectors = [[], [], []]
            for entry in entries:
                seen = {}

                def hook(_module, _inputs, output):
                    seen["h"] = output[0] if isinstance(output, tuple) else output

                handle = model.model.layers[block_index].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        result = model.model(
                            input_ids=torch.tensor([entry["ids"]], device="cuda"),
                            use_cache=False,
                            output_hidden_states=True,
                        )
                    difference = (
                        (seen.pop("h") - result.hidden_states[block_index + 1]).abs().max().item()
                    )
                    hook_error = max(hook_error, difference)
                    assert difference == 0.0, "Hook is not the declared hidden state"
                    for j, index in enumerate([block_index, block_index + 1, block_index + 2]):
                        values = result.hidden_states[index][0, entry["lo"] : entry["hi"]]
                        pooled = (values.float() if family == "olmo" else values).mean(0)
                        vectors[j].append(pooled.to(torch.float16).cpu().float().numpy())
                        if j == 1:
                            pooling_delta.append(
                                float(
                                    (values.float().mean(0) - values.mean(0).float()).norm().item()
                                )
                            )
                    del result
                finally:
                    handle.remove()
            target = reference[by_id[cid]]
            scale = np.linalg.norm(target)
            assert scale > 0
            lower.append(float(np.linalg.norm(np.mean(vectors[0], axis=0) - target) / scale))
            same.append(float(np.linalg.norm(np.mean(vectors[1], axis=0) - target) / scale))
            upper.append(float(np.linalg.norm(np.mean(vectors[2], axis=0) - target) / scale))
        assert np.all(np.array(same) < np.minimum(lower, upper)), "Historical layer rank failed"
        reports.append(
            dict(
                bank=bank["name"],
                capture_render=form,
                ids=[c[0] for c in candidates],
                historical_relative_error=same,
                lower_relative_error=lower,
                upper_relative_error=upper,
                hook_hidden_states_max_abs=hook_error,
                pooling_fp32_vs_bf16_l2=pooling_delta,
                historical_warning_cutoff=0.025,
                warnings=int((np.array(same) > 0.025).sum()),
                policy="#2054 adjudicated historical drift WARN; exact hook and adjacent-layer rank hard gates",
            )
        )
    C.write_json(
        audit_path,
        dict(
            model=model_name,
            block_index=block_index,
            hidden_states_index=block_index + 1,
            status="pass",
            reports=reports,
        ),
    )
    C.upload_many([audit_path], root, fp)
    print(
        f"[phase=layer_anchor] {model_name} exact hook identity and historical layer rank passed",
        flush=True,
    )


def capture(root, model_name, first_chunk=False, shard=0, shards=1):
    """Recapture every existing/new answer bank in both formats under its own checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM

    manifest = json.loads((root / "manifest.json").read_text())
    family = model_name.split("_")[0]
    cohort = manifest[family]
    offsets = C.owned_offsets(model_name, len(cohort["ids"]), shard, shards, first_chunk)
    if not offsets:
        return
    fp = C.fingerprint(root)
    tokenizer, template = tokenizer_for(model_name)
    mid, revision, layer, dim = C.MODELS[model_name]
    block_index = layer - 1 if family == "qwen" else layer
    model = (
        AutoModelForCausalLM.from_pretrained(
            mid, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    assert len(model.model.layers) >= layer
    capture_anchor(root, model_name, model, tokenizer, template, block_index)
    pending = []
    for offset in offsets:
        ids = cohort["ids"][offset : offset + C.CHUNK]
        for form in ("plain", "chat"):
            path = root / "contexts" / model_name / form / f"chunk_{offset:05d}.npz"
            if not C.complete(path, fp):
                entries = [
                    encode_entry(tokenizer, template, family, form, cohort["questions"][cid])
                    for cid in ids
                ]
                x = forward(model, tokenizer, entries, block_index, dim)
                save_npz(path, ids=np.array(ids), x=x)
                pending.append(path)
        for bank in C.banks(model_name):
            rawpath = root / "banks" / model_name / bank["name"] / f"chunk_{offset:05d}.json"
            assert C.complete(rawpath, fp), f"Unverified raw input: {rawpath}"
            rows = C.read_raw(rawpath)
            assert [(r["id"], r["draw"]) for r in rows] == [
                (cid, d) for cid in ids for d in range(5)
            ]
            for form in ("plain", "chat"):
                path = (
                    root / "captures" / model_name / bank["name"] / form / f"chunk_{offset:05d}.npz"
                )
                if C.complete(path, fp):
                    continue
                started = time.monotonic()
                entries = [
                    encode_entry(tokenizer, template, family, form, r["query"], r["answer"])
                    for r in rows
                ]
                w = forward(
                    model, tokenizer, entries, block_index, dim, fp32=family == "olmo"
                ).reshape(len(ids), 5, dim)
                valid = np.isfinite(w).all(2)
                counts = np.array([0 if e is None else e["answer_tokens"] for e in entries])
                save_npz(
                    path,
                    ids=np.array(ids),
                    w=w,
                    valid=valid,
                    answer_tokens=counts.reshape(len(ids), 5),
                    cap_mask=np.array([r["finish_reason"] == "length" for r in rows]).reshape(
                        len(ids), 5
                    ),
                )
                timing = path.with_suffix(".timing.json")
                seconds = time.monotonic() - started
                C.write_json(
                    timing,
                    dict(
                        model=model_name,
                        bank=bank["name"],
                        capture_render=form,
                        offset=offset,
                        stage="capture",
                        seconds=seconds,
                        answer_tokens=int(counts.sum()),
                        empty_draws=int((~valid).sum()),
                        answer_tokens_over_parent1024=int((counts > 1024).sum())
                        if family == "olmo"
                        else None,
                        stripped_chars=sum(e["stripped_chars"] for e in entries if e is not None),
                        seam_straddles=sum(e["seam_straddles"] for e in entries if e is not None),
                        layer=layer,
                        layer_index=block_index,
                        hidden_states_index=block_index + 1,
                        pool="float32 mean -> float16 store"
                        if family == "olmo"
                        else "BF16 mean -> float16 store",
                        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                    ),
                )
                pending.extend([path, timing])
                print(
                    f"[phase=capture] {model_name}/{bank['name']}/{form} offset={offset} seconds={seconds:.1f}",
                    flush=True,
                )
            if len(pending) >= 12 or first_chunk:
                C.upload_many(pending, root, fp)
                pending.clear()
    if pending:
        C.upload_many(pending, root, fp)
