"""Qwen generation and activation capture for issue #952's China follow-up.

Generation and teacher-forced capture are separate process phases so vLLM and
Transformers never coexist in one CUDA context.  Checkpoints are content-keyed,
raw generations upload before capture, and the terminal sentinel is written
only after revision-scoped Hub verification.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download

from explore_persona_space.analysis.extraction import extract_layer_activations
from explore_persona_space.orchestrate import hub

ISSUE = 952
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MAP_REV = "eef8eb1da43cd2212dfa73d8711fd64dc54c376f"
MAP_PREFIX = "issue779_monitoring/n1m_readout/weights"
LAYERS = (14, 19, 26)
HIDDEN = 3584
N_LAYERS = 28
N_DRAWS = 8
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 2048
SEED_BASE = 952_000
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue952_position_divergence/followups/china_refusal_topic_stratified_bilingual_v1"
EXPECTED_PROMPTS = 1080
EXPECTED_ROWS = EXPECTED_PROMPTS * N_DRAWS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_pt(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "vllm", "huggingface-hub")
    }


def _assert_package_versions() -> dict[str, str]:
    versions = _package_versions()
    expected = {
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "vllm": "0.11.0",
        "huggingface-hub": "0.36.2",
    }
    mismatched = {
        key: (versions[key], value)
        for key, value in expected.items()
        if versions[key].split("+")[0] != value
    }
    if mismatched:
        raise RuntimeError(f"incompatible package-version drift: {mismatched}")
    return versions


def _peak_host_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_bank(bank_path: Path, audit_path: Path, smoke: bool) -> tuple[list[dict], dict]:
    rows = _read_jsonl(bank_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or _sha256(bank_path) != audit["prompt_bank_sha256"]:
        raise RuntimeError("prompt bank is not the passed/hash-matched audited bank")
    if len(rows) != EXPECTED_PROMPTS or len({row["item_id"] for row in rows}) != len(rows):
        raise RuntimeError("prompt bank cardinality/uniqueness changed")
    if smoke:
        source_ids = sorted({row["source_prompt_id"] for row in rows})[:10]
        rows = [row for row in rows if row["source_prompt_id"] in set(source_ids)]
        if len(rows) != 120:
            raise RuntimeError(f"10-source-item smoke must have 120 prompts, got {len(rows)}")
    return rows, audit


def stage_inputs(
    out_root: Path, bank_path: Path | None, audit_path: Path | None
) -> tuple[Path, Path]:
    stage_path = out_root / "manifests" / "input_stage.json"
    prior_stage = json.loads(stage_path.read_text()) if stage_path.exists() else None
    if bank_path is not None and audit_path is not None:
        marker_path = bank_path.parent / "upload_verified.json"
        if not marker_path.exists():
            raise RuntimeError("explicit bank inputs require their uploaded verification marker")
        marker_revision = None
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        bank, audit = bank_path, audit_path
    else:
        local = out_root / "_hf_stage"
        api = HfApi()
        if prior_stage is not None and prior_stage.get("marker_source_revision") is None:
            raise RuntimeError("cannot switch from explicit local inputs to moving Hub inputs")
        marker_source_revision = prior_stage["marker_source_revision"] if prior_stage else None
        if marker_source_revision is None:
            marker_source_revision = hub.retry_transient(
                lambda: api.repo_info(HF_REPO, repo_type="dataset", revision="main").sha,
                what="issue952 input marker revision resolution",
            )
        marker_path = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    HF_REPO,
                    f"{HF_PREFIX}/inputs/upload_verified.json",
                    repo_type="dataset",
                    revision=marker_source_revision,
                    local_dir=local,
                ),
                what="issue952 input verification marker stage",
            )
        )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_revision = marker_source_revision
        data_revision = marker.get("data_revision")
        if not isinstance(data_revision, str) or not data_revision:
            raise RuntimeError("input marker lacks immutable data_revision")
        bank = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    HF_REPO,
                    f"{HF_PREFIX}/inputs/prompt_bank.jsonl",
                    repo_type="dataset",
                    revision=data_revision,
                    local_dir=local,
                ),
                what="issue952 prompt bank stage",
            )
        )
        audit = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    HF_REPO,
                    f"{HF_PREFIX}/inputs/bank_audit_report.json",
                    repo_type="dataset",
                    revision=data_revision,
                    local_dir=local,
                ),
                what="issue952 bank audit stage",
            )
        )
        canonical = out_root / "inputs"
        canonical.mkdir(parents=True, exist_ok=True)
        staged = {}
        for name, source in (
            ("upload_verified.json", marker_path),
            ("prompt_bank.jsonl", bank),
            ("bank_audit_report.json", audit),
        ):
            destination = canonical / name
            if destination.exists() and _sha256(destination) != _sha256(source):
                raise RuntimeError(f"canonical staged input drift: {name}")
            if not destination.exists():
                shutil.copyfile(source, destination)
            staged[name] = destination
        marker_path = staged["upload_verified.json"]
        bank = staged["prompt_bank.jsonl"]
        audit = staged["bank_audit_report.json"]
    if _sha256(bank) != marker.get("prompt_bank_sha256"):
        raise RuntimeError("staged prompt bank differs from immutable upload marker")
    if _sha256(audit) != marker.get("bank_audit_report_sha256"):
        raise RuntimeError("staged bank audit differs from immutable upload marker")
    stage = {
        "marker_source_revision": marker_revision,
        "data_revision": marker.get("data_revision"),
        "marker_sha256": _sha256(marker_path),
        "prompt_bank_sha256": _sha256(bank),
        "bank_audit_report_sha256": _sha256(audit),
    }
    if prior_stage is not None and stage != prior_stage:
        raise RuntimeError("GPU phase input stage differs from the frozen first-phase snapshot")
    _write_json(stage_path, stage)
    return bank, audit


def _expected_rollout_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{row['item_id']}-d{draw}" for row in rows for draw in range(N_DRAWS)]


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _context_ids(tok, prompt: str) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(ids, list) or not ids:
        raise RuntimeError("chat template produced no context tokens")
    return [int(x) for x in ids]


def _rendered(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _eot_ids(tok) -> list[int]:
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    if not isinstance(im_end, int) or im_end < 0 or not newline:
        raise RuntimeError("could not construct Qwen end-of-turn tail")
    return [im_end, *map(int, newline)]


def _regime(bank_sha: str, smoke: bool) -> dict[str, Any]:
    return {
        "issue": ISSUE,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "bank_sha256": bank_sha,
        "layers": list(LAYERS),
        "draws": N_DRAWS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_base": SEED_BASE,
        "smoke": smoke,
        "git_sha": _git_sha(),
    }


def _smoke_generation_compatibility(regime: dict[str, Any]) -> dict[str, Any]:
    excluded = {"smoke", "prompt_token_max", "max_model_len"}
    return {key: value for key, value in regime.items() if key not in excluded}


def _smoke_capture_compatibility(
    capture_regime: dict[str, Any], package_versions: dict[str, str]
) -> dict[str, Any]:
    return {
        "capture_regime": {
            key: value for key, value in capture_regime.items() if key != "rollouts_sha256"
        },
        "package_versions": package_versions,
    }


def phase_generate(
    out_root: Path,
    bank_path: Path,
    audit_path: Path,
    smoke: bool,
    shard_prompts: int,
    smoke_report: Path | None = None,
) -> dict[str, Any]:
    if not smoke:
        if smoke_report is None or not smoke_report.exists():
            raise RuntimeError("production generation requires a passed smoke timing report")
        smoke_gate = json.loads(smoke_report.read_text())
        if smoke_gate.get("passed") is not True:
            raise RuntimeError("production generation blocked by smoke timing gate")
    versions = _assert_package_versions()
    rows, _audit = _load_bank(bank_path, audit_path, smoke)
    tok = _tokenizer()
    contexts = [_context_ids(tok, row["prompt"]) for row in rows]
    prompt_max = max(map(len, contexts))
    max_model_len = max(4096, prompt_max + MAX_NEW_TOKENS + 32)
    if max_model_len > int(getattr(tok, "model_max_length", 32768)):
        raise RuntimeError("prompt+generation envelope exceeds tokenizer model maximum")
    bank_sha = _sha256(bank_path)
    if not smoke and smoke_gate["bank_sha256"] != bank_sha:
        raise RuntimeError("production bank differs from the smoke-tested bank")
    regime = _regime(bank_sha, smoke)
    regime["prompt_token_max"] = prompt_max
    regime["max_model_len"] = max_model_len
    regime["chat_template_sha256"] = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    regime["package_versions"] = versions
    regime["tokenizer_artifact_sha256"] = {
        name: _sha256(
            Path(
                hub.retry_transient(
                    lambda name=name: hf_hub_download(
                        MODEL, name, revision=MODEL_REV, repo_type="model"
                    ),
                    what=f"issue952 tokenizer artifact stage {name}",
                )
            )
        )
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    }
    if not smoke and smoke_gate.get("generation_compatibility") != (
        current_compatibility := _smoke_generation_compatibility(regime)
    ):
        raise RuntimeError(
            "production generation blocked: smoke code/model/tokenizer/regime is stale"
        )
    regime_fp = _sha_obj(regime)
    expected_ids = _expected_rollout_ids(rows)
    raw_dir = out_root / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        revision=MODEL_REV,
        tokenizer_revision=MODEL_REV,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.82,
        seed=SEED_BASE,
    )
    t0 = time.time()
    for start in range(0, len(rows), shard_prompts):
        chunk = rows[start : start + shard_prompts]
        chunk_expected_ids = _expected_rollout_ids(chunk)
        shard_path = raw_dir / f"rollouts_p{start:04d}_{start + len(chunk):04d}.jsonl"
        manifest_path = shard_path.with_suffix(".done.json")
        if shard_path.exists() and manifest_path.exists():
            done = json.loads(manifest_path.read_text())
            if (
                done.get("regime_fp") != regime_fp
                or done.get("n_rows") != len(chunk) * N_DRAWS
                or done.get("sha256") != _sha256(shard_path)
                or done.get("ordered_item_ids_sha256") != _sha_obj(chunk_expected_ids)
                or [row["item_id"] for row in _read_jsonl(shard_path)] != chunk_expected_ids
            ):
                raise RuntimeError(f"stale generation shard: {shard_path}")
            print(f"[gen] resume shard={start // shard_prompts + 1}")
            continue
        prompts: list[str] = []
        params = []
        specs = []
        for offset, row in enumerate(chunk):
            global_i = start + offset
            rendered = _rendered(tok, row["prompt"])
            for draw in range(N_DRAWS):
                prompts.append(rendered)
                params.append(
                    SamplingParams(
                        n=1,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_NEW_TOKENS,
                        seed=SEED_BASE + global_i * N_DRAWS + draw,
                    )
                )
                specs.append((row, global_i, draw, len(contexts[global_i])))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        realized = []
        for (row, global_i, draw, ctx_len), output in zip(specs, outputs, strict=True):
            if list(map(int, output.prompt_token_ids)) != contexts[global_i]:
                raise RuntimeError("vLLM/HF rendered prompt tokenization mismatch")
            sample = output.outputs[0]
            n_tokens = len(sample.token_ids)
            realized.append(
                {
                    "prompt_id": row["item_id"],
                    "item_id": f"{row['item_id']}-d{draw}",
                    "source_prompt_id": row["source_prompt_id"],
                    "topic": row["topic"],
                    "language": row["language"],
                    "content": row["content"],
                    "frame": row["frame"],
                    "draw": draw,
                    "seed": SEED_BASE + global_i * N_DRAWS + draw,
                    "question": row["prompt"],
                    "text": sample.text,
                    "completion_token_ids": list(map(int, sample.token_ids)),
                    "context_tokens": ctx_len,
                    "completion_tokens": n_tokens,
                    "finish_reason": sample.finish_reason,
                    "cap_hit": n_tokens >= MAX_NEW_TOKENS,
                    "audit_pass": bool(row["audit_pass"]),
                    "latency_s": (
                        float(output.metrics.finished_time - output.metrics.arrival_time)
                        if output.metrics is not None and output.metrics.finished_time is not None
                        else None
                    ),
                }
            )
        _write_jsonl(shard_path, realized)
        _write_json(
            manifest_path,
            {
                "regime_fp": regime_fp,
                "n_prompts": len(chunk),
                "n_rows": len(realized),
                "sha256": _sha256(shard_path),
                "ordered_item_ids_sha256": _sha_obj(chunk_expected_ids),
            },
        )
        print(
            f"[gen] shard={start // shard_prompts + 1} prompts={len(chunk)} "
            f"rows={len(realized)} elapsed={time.time() - t0:.1f}s"
        )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shard_paths = sorted(raw_dir.glob("rollouts_p*.jsonl"))
    all_rows = [row for path in shard_paths for row in _read_jsonl(path)]
    expected = len(rows) * N_DRAWS
    realized_ids = [row["item_id"] for row in all_rows]
    if (
        len(all_rows) != expected
        or len(set(realized_ids)) != expected
        or realized_ids != expected_ids
    ):
        raise RuntimeError(f"generation coverage mismatch: {len(all_rows)} != {expected}")
    cap_frac = sum(row["cap_hit"] for row in all_rows) / len(all_rows)
    final_path = raw_dir / "rollouts.jsonl"
    _write_jsonl(final_path, all_rows)
    report = {
        "regime": regime,
        "regime_fp": regime_fp,
        "n_prompts": len(rows),
        "n_rows": len(all_rows),
        "ordered_item_ids_sha256": _sha_obj(expected_ids),
        "n_empty": sum(not row["text"] for row in all_rows),
        "n_cap_hit": sum(row["cap_hit"] for row in all_rows),
        "cap_hit_fraction": cap_frac,
        "rollouts_sha256": _sha256(final_path),
        "elapsed_s": time.time() - t0,
        "tokens_generated": sum(row["completion_tokens"] for row in all_rows),
        "tokens_per_second": sum(row["completion_tokens"] for row in all_rows)
        / max(time.time() - t0, 1e-9),
        "p90_request_latency_s": float(
            torch.quantile(
                torch.tensor(
                    [row["latency_s"] for row in all_rows if row["latency_s"] is not None]
                ),
                0.9,
            )
        )
        if any(row["latency_s"] is not None for row in all_rows)
        else None,
        "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": _peak_host_rss_bytes(),
        "package_versions": versions,
    }
    _write_json(out_root / "manifests" / "generation.json", report)
    print(
        f"[gen] complete prompts={len(rows)} rows={len(all_rows)} cap_frac={cap_frac:.6f} "
        f"sha={report['rollouts_sha256'][:12]}"
    )
    if cap_frac > 0.005:
        raise RuntimeError(f"cap-hit fraction {cap_frac:.4f} exceeds 0.005 gate")
    if report["n_empty"]:
        raise RuntimeError(f"generation produced {report['n_empty']} empty completions")
    return report


def _upload_folder(folder: Path, path_in_repo: str, message: str) -> dict[str, Any]:
    info = hub.retry_transient(
        lambda: HfApi().upload_folder(
            repo_id=HF_REPO,
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=message,
        ),
        what=message,
    )
    return {"commit_url": str(info), "revision": getattr(info, "oid", None)}


def phase_upload_raw(out_root: Path) -> dict[str, Any]:
    report = json.loads((out_root / "manifests" / "generation.json").read_text())
    target_prefix = f"{HF_PREFIX}/smoke" if report["regime"]["smoke"] else HF_PREFIX
    rollouts = out_root / "raw_completions" / "rollouts.jsonl"
    if _sha256(rollouts) != report["rollouts_sha256"] or report["cap_hit_fraction"] > 0.005:
        raise RuntimeError("raw upload blocked by generation identity/cap gate")
    result = _upload_folder(
        out_root / "raw_completions",
        f"{target_prefix}/raw_completions",
        "Issue 952: bilingual China Qwen rollouts",
    )
    result["rollouts_sha256"] = _sha256(rollouts)
    result["generation_manifest_sha256"] = _sha256(out_root / "manifests" / "generation.json")
    check_rev = result["revision"] or "main"
    rollout_exists = hub.retry_transient(
        lambda: HfApi().file_exists(
            HF_REPO,
            f"{target_prefix}/raw_completions/rollouts.jsonl",
            repo_type="dataset",
            revision=check_rev,
        ),
        what="issue952 raw rollout upload verification",
    )
    if not rollout_exists:
        raise RuntimeError("revision-scoped raw rollout verification failed")
    _write_json(out_root / "manifests" / "raw_upload.json", result)
    manifest_result = _upload_folder(
        out_root / "manifests",
        f"{target_prefix}/manifests",
        "Issue 952: bilingual China generation manifests",
    )
    final_rev = manifest_result["revision"] or "main"
    for path in (
        f"{target_prefix}/raw_completions/rollouts.jsonl",
        f"{target_prefix}/manifests/generation.json",
        f"{target_prefix}/manifests/raw_upload.json",
    ):
        exists = hub.retry_transient(
            lambda path=path: HfApi().file_exists(
                HF_REPO, path, repo_type="dataset", revision=final_rev
            ),
            what=f"issue952 raw manifest upload verification {path}",
        )
        if not exists:
            raise RuntimeError(f"revision-scoped raw/manifests verification failed: {path}")
    print(f"[upload-raw] verified revision={final_rev}")
    return result


def _load_hf_model():
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("production activation capture requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.bfloat16)
    model = model.to("cuda:0")
    model.eval()
    if model.config.hidden_size != HIDDEN or model.config.num_hidden_layers != N_LAYERS:
        raise RuntimeError("model shape mismatch")
    return model


def _right_pad(rows: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    tmax = max(map(len, rows))
    ids = torch.full((len(rows), tmax), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), tmax), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    return ids.cuda(), mask.cuda()


def _with_eot_tail(comp: list[int], eot: list[int]) -> list[int]:
    """Preserve exact generated ids and append only missing turn-end ids."""

    if len(comp) >= len(eot) and comp[-len(eot) :] == eot:
        return comp
    if comp[-1:] == eot[:1]:
        return comp + eot[1:]
    return comp + eot


@torch.no_grad()
def _capture_contexts(model, tok, rows: list[dict], batch_size: int) -> torch.Tensor:
    token_rows = [_context_ids(tok, row["prompt"]) for row in rows]
    out = torch.zeros((len(rows), len(LAYERS), HIDDEN), dtype=torch.float32)
    pad = tok.pad_token_id
    for start in range(0, len(rows), batch_size):
        chunk = token_rows[start : start + batch_size]
        ids, mask = _right_pad(chunk, pad)
        acts = extract_layer_activations(model, ids, LAYERS, attention_mask=mask)
        for i, token_ids in enumerate(chunk):
            out[start + i] = torch.stack(
                [acts[layer][i, len(token_ids) - 1].float().cpu() for layer in LAYERS]
            )
        del acts, ids, mask
    return out


@torch.no_grad()
def _capture_answers(
    model, tok, prompts: dict[str, str], rows: list[dict], batch_size: int
) -> tuple[torch.Tensor, list[int], list[dict]]:
    eot = _eot_ids(tok)
    ctx_cache = {pid: _context_ids(tok, prompt) for pid, prompt in prompts.items()}
    specs = []
    empty = []
    for i, row in enumerate(rows):
        comp = row.get("completion_token_ids")
        if not isinstance(comp, list) or any(not isinstance(token_id, int) for token_id in comp):
            raise RuntimeError("rollout lacks exact vLLM completion token ids")
        if not comp:
            empty.append(i)
        specs.append((ctx_cache[row["prompt_id"]], comp))
    out = torch.zeros((len(rows), len(LAYERS), HIDDEN), dtype=torch.float32)
    boundaries = []
    pad = tok.pad_token_id
    for start in range(0, len(rows), batch_size):
        chunk_specs = specs[start : start + batch_size]
        nonempty = [(j, ctx, comp) for j, (ctx, comp) in enumerate(chunk_specs) if comp]
        if nonempty:
            completion_with_tail = [_with_eot_tail(comp, eot) for _, _ctx, comp in nonempty]
            token_rows = [
                ctx + tailed
                for (_, ctx, _comp), tailed in zip(nonempty, completion_with_tail, strict=True)
            ]
            ids, mask = _right_pad(token_rows, pad)
            acts = extract_layer_activations(model, ids, LAYERS, attention_mask=mask)
            for b, ((local_j, ctx, _comp), tailed) in enumerate(
                zip(nonempty, completion_with_tail, strict=True)
            ):
                span = slice(len(ctx), len(ctx) + len(tailed))
                out[start + local_j] = torch.stack(
                    [acts[layer][b, span].float().mean(0).cpu() for layer in LAYERS]
                )
            del acts, ids, mask
        for ctx, comp in chunk_specs:
            tailed_len = len(_with_eot_tail(comp, eot))
            boundaries.append(
                {
                    "ctx_len": len(ctx),
                    "completion_len": len(comp),
                    "span_start": len(ctx),
                    "span_end": len(ctx) + len(comp),
                    "tail_end": len(ctx) + tailed_len,
                }
            )
    return out, empty, boundaries


def phase_capture(
    out_root: Path,
    bank_path: Path,
    audit_path: Path,
    smoke: bool,
    batch_size: int,
    answer_shard_rows: int,
    smoke_report: Path | None = None,
) -> dict[str, Any]:
    if not (out_root / "manifests" / "raw_upload.json").exists():
        raise RuntimeError("capture blocked: raw-rollout upload has not been verified")
    bank_rows, _audit = _load_bank(bank_path, audit_path, smoke)
    rollouts_path = out_root / "raw_completions" / "rollouts.jsonl"
    gen = json.loads((out_root / "manifests" / "generation.json").read_text())
    if (
        _sha256(rollouts_path) != gen["rollouts_sha256"]
        or _sha256(bank_path) != gen["regime"]["bank_sha256"]
    ):
        raise RuntimeError("rollout hash drift before capture")
    rollout_rows = _read_jsonl(rollouts_path)
    expected = len(bank_rows) * N_DRAWS
    if (
        len(rollout_rows) != expected
        or [row["item_id"] for row in rollout_rows] != _expected_rollout_ids(bank_rows)
        or gen.get("ordered_item_ids_sha256") != _sha_obj(_expected_rollout_ids(bank_rows))
    ):
        raise RuntimeError("rollout coverage mismatch before capture")
    prompts = {row["item_id"]: row["prompt"] for row in bank_rows}
    if any(row["question"] != prompts.get(row["prompt_id"]) for row in rollout_rows):
        raise RuntimeError("rollout question text differs from the frozen prompt bank")
    tok = _tokenizer()
    capture_regime = {
        "model_revision": MODEL_REV,
        "layers": list(LAYERS),
        "bank_sha256": _sha256(bank_path),
        "rollouts_sha256": gen["rollouts_sha256"],
        "context_position": "context_last_generation_prompt_token",
        "answer_pooling": "completion_plus_im_end_newline_mean",
        "serialized_dtype": "fp32",
        "git_sha": _git_sha(),
    }
    versions = _package_versions()
    if not smoke:
        if smoke_report is None or not smoke_report.exists():
            raise RuntimeError("production capture requires the passed smoke timing report")
        smoke_gate = json.loads(smoke_report.read_text())
        if smoke_gate.get("passed") is not True or smoke_gate.get(
            "capture_compatibility"
        ) != _smoke_capture_compatibility(capture_regime, versions):
            raise RuntimeError("production capture blocked: smoke capture regime is stale")
    model = _load_hf_model()
    t0 = time.time()
    capture_regime_fp = _sha_obj(capture_regime)
    vc_path = out_root / "analysis_tensors" / "vc.pt"
    if vc_path.exists():
        prior_vc = torch.load(vc_path, map_location="cpu", weights_only=False)
        if (
            prior_vc.get("capture_regime_fp") != capture_regime_fp
            or prior_vc.get("item_ids") != [row["item_id"] for row in bank_rows]
            or prior_vc["vc"].shape != (len(bank_rows), len(LAYERS), HIDDEN)
        ):
            raise RuntimeError("stale context capture checkpoint")
    else:
        vc = _capture_contexts(model, tok, bank_rows, batch_size)
        _save_pt(
            vc_path,
            {
                "issue": ISSUE,
                "layers": list(LAYERS),
                "item_ids": [row["item_id"] for row in bank_rows],
                "vc": vc,
                "dtype": "fp32",
                "position": "context_last_generation_prompt_token",
                "model_revision": MODEL_REV,
                "bank_sha256": _sha256(bank_path),
                "capture_regime": capture_regime,
                "capture_regime_fp": capture_regime_fp,
            },
        )
        print(f"[capture-vc] rows={len(bank_rows)} sha={_sha256(vc_path)[:12]}")
    answer_paths = []
    all_empty = []
    for start in range(0, len(rollout_rows), answer_shard_rows):
        chunk = rollout_rows[start : start + answer_shard_rows]
        path = out_root / "analysis_tensors" / f"va_{start:05d}_{start + len(chunk):05d}.pt"
        if path.exists():
            store = torch.load(path, map_location="cpu", weights_only=False)
            if (
                store.get("capture_regime_fp") != capture_regime_fp
                or len(store["index"]) != len(chunk)
                or [row["item_id"] for row in store["index"]] != [row["item_id"] for row in chunk]
                or store["va_tail_incl"].shape != (len(chunk), len(LAYERS), HIDDEN)
            ):
                raise RuntimeError(f"stale answer capture shard: {path}")
            answer_paths.append(path)
            all_empty.extend(start + int(i) for i in store["empty_rows"])
            print(f"[capture-va] resume rows={start}:{start + len(chunk)}")
            continue
        va, empty, bounds = _capture_answers(model, tok, prompts, chunk, batch_size)
        for row, bound in zip(chunk, bounds, strict=True):
            if row["context_tokens"] != bound["ctx_len"]:
                raise RuntimeError("generation/capture context-token boundary mismatch")
            if row["completion_tokens"] != bound["completion_len"]:
                raise RuntimeError("generation/capture completion-token boundary mismatch")
        _save_pt(
            path,
            {
                "issue": ISSUE,
                "layers": list(LAYERS),
                "index": [
                    {
                        "item_id": row["item_id"],
                        "prompt_id": row["prompt_id"],
                        "draw": row["draw"],
                        **bound,
                    }
                    for row, bound in zip(chunk, bounds, strict=True)
                ],
                "va_tail_incl": va,
                "empty_rows": empty,
                "dtype": "fp32",
                "pooling": "completion_plus_im_end_newline_mean",
                "model_revision": MODEL_REV,
                "rollouts_sha256": gen["rollouts_sha256"],
                "capture_regime": capture_regime,
                "capture_regime_fp": capture_regime_fp,
            },
        )
        answer_paths.append(path)
        all_empty.extend(start + i for i in empty)
        print(
            f"[capture-va] rows={start}:{start + len(chunk)} empty={len(empty)} "
            f"sha={_sha256(path)[:12]} elapsed={time.time() - t0:.1f}s"
        )
    report = {
        "issue": ISSUE,
        "model_revision": MODEL_REV,
        "layers": list(LAYERS),
        "n_contexts": len(bank_rows),
        "n_answer_rows": len(rollout_rows),
        "n_empty_answer_rows": len(all_empty),
        "empty_answer_rows": all_empty,
        "vc_sha256": _sha256(vc_path),
        "va_files": {path.name: _sha256(path) for path in answer_paths},
        "rollouts_sha256": gen["rollouts_sha256"],
        "capture_regime_fp": capture_regime_fp,
        "capture_regime": capture_regime,
        "elapsed_s": time.time() - t0,
        "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": _peak_host_rss_bytes(),
        "examples_per_second": len(rollout_rows) / max(time.time() - t0, 1e-9),
        "serialized_bytes": vc_path.stat().st_size
        + sum(path.stat().st_size for path in answer_paths),
        "package_versions": versions,
    }
    _write_json(out_root / "manifests" / "capture.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if all_empty:
        raise RuntimeError(f"activation capture has {len(all_empty)} empty answer rows")
    return report


def phase_upload_capture(out_root: Path) -> dict[str, Any]:
    report = json.loads((out_root / "manifests" / "capture.json").read_text())
    generation = json.loads((out_root / "manifests" / "generation.json").read_text())
    target_prefix = f"{HF_PREFIX}/smoke" if generation["regime"]["smoke"] else HF_PREFIX
    tensor_dir = out_root / "analysis_tensors"
    if _sha256(tensor_dir / "vc.pt") != report["vc_sha256"]:
        raise RuntimeError("context capture hash drift")
    for name, sha in report["va_files"].items():
        if _sha256(tensor_dir / name) != sha:
            raise RuntimeError(f"answer capture hash drift: {name}")
    result = _upload_folder(
        tensor_dir,
        f"{target_prefix}/analysis_tensors",
        "Issue 952: bilingual China Qwen activation captures",
    )
    check_rev = result["revision"] or "main"
    required = [f"{target_prefix}/analysis_tensors/vc.pt"] + [
        f"{target_prefix}/analysis_tensors/{name}" for name in report["va_files"]
    ]
    api = HfApi()
    missing = []
    for path in required:
        exists = hub.retry_transient(
            lambda path=path: api.file_exists(
                HF_REPO, path, repo_type="dataset", revision=check_rev
            ),
            what=f"issue952 tensor upload verification {path}",
        )
        if not exists:
            missing.append(path)
    if missing:
        raise RuntimeError(f"revision-scoped tensor upload verification missing {len(missing)}")
    # Real consumer-open probe from the uploaded revision.
    probe = hub.retry_transient(
        lambda: hf_hub_download(
            HF_REPO,
            required[0],
            repo_type="dataset",
            revision=check_rev,
            local_dir=out_root / "consumer_probe",
            force_download=True,
        ),
        what="issue952 uploaded context capture consumer probe",
    )
    opened = torch.load(probe, map_location="cpu", weights_only=False)
    if opened["vc"].shape != (report["n_contexts"], len(LAYERS), HIDDEN):
        raise RuntimeError("uploaded context tensor consumer-open shape mismatch")
    result["verified_files"] = len(required)
    result["consumer_open_shape"] = list(opened["vc"].shape)
    result["capture_manifest_sha256"] = _sha256(out_root / "manifests" / "capture.json")
    result["vc_sha256"] = report["vc_sha256"]
    result["va_files"] = report["va_files"]
    _write_json(out_root / "manifests" / "capture_upload.json", result)
    manifest_result = _upload_folder(
        out_root / "manifests",
        f"{target_prefix}/manifests",
        "Issue 952: bilingual China capture manifests",
    )
    final_rev = manifest_result["revision"] or "main"
    for path in (
        *required,
        f"{target_prefix}/manifests/generation.json",
        f"{target_prefix}/manifests/raw_upload.json",
        f"{target_prefix}/manifests/capture.json",
        f"{target_prefix}/manifests/capture_upload.json",
    ):
        exists = hub.retry_transient(
            lambda path=path: api.file_exists(
                HF_REPO, path, repo_type="dataset", revision=final_rev
            ),
            what=f"issue952 capture manifest upload verification {path}",
        )
        if not exists:
            raise RuntimeError(f"revision-scoped capture/manifests verification failed: {path}")
    print(f"[upload-capture] verified={len(required) + 4} revision={final_rev}")
    return result


def _smoke_map_consumer_probe(out_root: Path) -> dict[str, Any]:
    """Open every frozen map and check its registered/raw affine application."""

    store = torch.load(
        out_root / "analysis_tensors" / "vc.pt", map_location="cpu", weights_only=False
    )
    checks = {}
    for layer_pos, layer in enumerate(LAYERS):
        path = Path(
            hub.retry_transient(
                lambda layer=layer: hf_hub_download(
                    HF_REPO,
                    f"{MAP_PREFIX}/L{layer}/ridge.pt",
                    repo_type="dataset",
                    revision=MAP_REV,
                ),
                what=f"issue952 frozen map consumer probe L{layer}",
            )
        )
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if (
            bundle.get("kind") != "ridge"
            or int(bundle.get("layer", -1)) != layer
            or bundle["W"].shape != (HIDDEN, HIDDEN)
        ):
            raise RuntimeError(f"frozen map schema mismatch at layer {layer}")
        x = store["vc"][0, layer_pos].double()
        w = bundle["W"].double()
        xmu = bundle["xmu"].double()
        xsd = bundle["xsd"].double()
        ymu = bundle["ymu"].double()
        registered = ((x - xmu) / xsd) @ w + ymu
        a_map = w / xsd[:, None]
        intercept = ymu - (xmu / xsd) @ w
        affine = x @ a_map + intercept
        max_abs = float(torch.max(torch.abs(registered - affine)))
        if not torch.allclose(registered, affine, rtol=1e-10, atol=1e-8):
            raise RuntimeError(f"map consumer parity failed at layer {layer}: {max_abs}")
        checks[str(layer)] = {
            "map_sha256": _sha256(path),
            "max_abs_parity_error": max_abs,
            "output_norm": float(torch.linalg.vector_norm(registered)),
        }
    return {"map_revision": MAP_REV, "layers": checks}


def phase_finalize(out_root: Path) -> dict[str, Any]:
    generation = json.loads((out_root / "manifests" / "generation.json").read_text())
    capture = json.loads((out_root / "manifests" / "capture.json").read_text())
    raw_upload = json.loads((out_root / "manifests" / "raw_upload.json").read_text())
    capture_upload = json.loads((out_root / "manifests" / "capture_upload.json").read_text())
    if generation["n_rows"] != capture["n_answer_rows"]:
        raise RuntimeError("final generation/capture row mismatch")
    result = {
        "issue": ISSUE,
        "status": "done",
        "generation": generation,
        "capture": capture,
        "raw_upload": raw_upload,
        "capture_upload": capture_upload,
        "hf_prefix": HF_PREFIX,
        "timestamp_unix": time.time(),
    }
    if generation["regime"]["smoke"]:
        consumer_probe = _smoke_map_consumer_probe(out_root)
        scale = EXPECTED_PROMPTS / generation["n_prompts"]
        projected_gpu_s = scale * (generation["elapsed_s"] + capture["elapsed_s"])
        projected_upper_s = 1.25 * projected_gpu_s + 0.3 * 3600
        smoke_timing = {
            "passed": projected_upper_s <= 6 * 3600
            and generation["p90_request_latency_s"] is not None,
            "bank_sha256": generation["regime"]["bank_sha256"],
            "smoke_prompts": generation["n_prompts"],
            "smoke_draws": generation["n_rows"],
            "projected_gpu_hours": projected_gpu_s / 3600,
            "projected_upper_gpu_hours": projected_upper_s / 3600,
            "gpu_hour_fence": 6,
            "wall_hour_fence": 8,
            "generation_tokens_per_second": generation["tokens_per_second"],
            "capture_examples_per_second": capture["examples_per_second"],
            "serialized_mb_per_second": capture["serialized_bytes"]
            / max(capture["elapsed_s"], 1e-9)
            / 1e6,
            "peak_hbm_bytes": max(generation["peak_hbm_bytes"], capture["peak_hbm_bytes"]),
            "peak_host_rss_bytes": max(
                generation["peak_host_rss_bytes"], capture["peak_host_rss_bytes"]
            ),
            "p90_request_latency_s": generation["p90_request_latency_s"],
            "map_consumer_probe": consumer_probe,
            "generation_compatibility": _smoke_generation_compatibility(generation["regime"]),
            "capture_compatibility": _smoke_capture_compatibility(
                capture["capture_regime"], capture["package_versions"]
            ),
        }
        _write_json(out_root / "manifests" / "smoke_timing.json", smoke_timing)
        result["smoke_timing"] = smoke_timing
        if not smoke_timing["passed"]:
            raise RuntimeError("smoke timing/telemetry gate failed; production remains blocked")
    _write_json(out_root / "issue952_china_definitive_done.json", result)
    target_prefix = f"{HF_PREFIX}/smoke" if generation["regime"]["smoke"] else HF_PREFIX
    if generation["regime"]["smoke"]:
        _upload_folder(
            out_root / "manifests",
            f"{target_prefix}/manifests",
            "Issue 952: bilingual China smoke timing manifest",
        )
    info = hub.retry_transient(
        lambda: HfApi().upload_file(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_or_fileobj=str(out_root / "issue952_china_definitive_done.json"),
            path_in_repo=f"{target_prefix}/issue952_china_definitive_done.json",
            commit_message="Issue 952: bilingual China GPU terminal sentinel",
        ),
        what="issue952 GPU terminal sentinel upload",
    )
    revision = getattr(info, "oid", None) or "main"
    sentinel_exists = hub.retry_transient(
        lambda: HfApi().file_exists(
            HF_REPO,
            f"{target_prefix}/issue952_china_definitive_done.json",
            repo_type="dataset",
            revision=revision,
        ),
        what="issue952 GPU terminal sentinel verification",
    )
    if not sentinel_exists:
        raise RuntimeError("revision-scoped GPU terminal sentinel verification failed")
    print("[finalize] terminal sentinel written")
    return result


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=("gen", "upload-raw", "capture", "upload-capture", "finalize"),
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard-prompts", type=int, default=90)
    ap.add_argument("--capture-batch", type=int, default=4)
    ap.add_argument("--answer-shard-rows", type=int, default=720)
    ap.add_argument("--smoke-report", type=Path)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    bank, audit = stage_inputs(args.out_root, args.bank, args.audit)
    if args.phase == "gen":
        phase_generate(
            args.out_root,
            bank,
            audit,
            args.smoke,
            args.shard_prompts,
            args.smoke_report,
        )
    elif args.phase == "upload-raw":
        phase_upload_raw(args.out_root)
    elif args.phase == "capture":
        phase_capture(
            args.out_root,
            bank,
            audit,
            args.smoke,
            args.capture_batch,
            args.answer_shard_rows,
            args.smoke_report,
        )
    elif args.phase == "upload-capture":
        phase_upload_capture(args.out_root)
    else:
        phase_finalize(args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
