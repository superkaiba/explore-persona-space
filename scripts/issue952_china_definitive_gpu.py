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
REGISTERED_SOURCE_ITEMS = 90
PROMPTS_PER_SOURCE = 12
REGISTERED_PROMPTS = REGISTERED_SOURCE_ITEMS * PROMPTS_PER_SOURCE
SMOKE_SEED_BASE = SEED_BASE + REGISTERED_PROMPTS * N_DRAWS
REPAIRED_HF_PREFIX = "issue952_position_divergence/followups/china_refusal_wording_withholding_v2"
REPAIRED_CONTRACT = "china-country-cue-v2"
REPAIRED_SEED_BASE = 9_520_000
REPAIRED_EXTENSION_POLICY = {
    "contract": "china-cap-extension-v2",
    "trigger": "initial finish_reason == length fraction > threshold within language/content/frame",
    "threshold": 0.02,
    "max_passes": 1,
    "max_new_tokens": 4096,
    "shard_rows": 90,
    "question_and_seed": "unchanged from initial rollout",
    "remaining_truncation": "retain and report after the single extension pass",
}
REPAIRED_CONTENT_CELLS = {
    "sensitive_full": ("sensitive", "present"),
    "sensitive_country_neutral": ("sensitive", "absent"),
    "matched_non_china": ("control", "absent"),
    "matched_non_china_country": ("control", "present"),
}


def _study_config(study: str) -> dict[str, Any]:
    """Resolve a study explicitly without changing the legacy module constants."""
    if study == "v1":
        return {
            "hf_prefix": HF_PREFIX,
            "source_items": REGISTERED_SOURCE_ITEMS,
            "prompts_per_source": PROMPTS_PER_SOURCE,
            "seed_base": SEED_BASE,
            "smoke_seed_base": SMOKE_SEED_BASE,
        }
    if study == "repaired-v2":
        return {
            "hf_prefix": REPAIRED_HF_PREFIX,
            "source_items": 85,
            "prompts_per_source": 16,
            "seed_base": REPAIRED_SEED_BASE,
            "smoke_seed_base": REPAIRED_SEED_BASE + 85 * 16 * N_DRAWS,
        }
    raise ValueError(f"unknown China study: {study}")


def _validate_study(regime: dict[str, Any], study: str) -> None:
    """Refuse cross-study consumers before any capture or external write."""
    config = _study_config(study)
    if regime.get("study", "v1") != study:
        raise RuntimeError("selected study differs from the generation regime")
    if study == "repaired-v2" and (
        regime.get("hf_prefix") != config["hf_prefix"]
        or regime.get("bank_contract") != REPAIRED_CONTRACT
    ):
        raise RuntimeError("repaired generation study namespace/contract drift")


def _output_prefix(smoke: bool, attempt: int, study: str = "v1") -> str:
    """Return the immutable per-attempt output namespace; inputs stay canonical."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    base = f"{_study_config(study)['hf_prefix']}/attempt{attempt}"
    return f"{base}/smoke" if smoke else base


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


def _accepted_bank_identity(rows: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    """Return the hash-bound accepted-source identity and its derived widths."""
    source_ids = sorted(audit["passing_item_ids"])
    accepted_rows = [row for row in rows if row["source_prompt_id"] in set(source_ids)]
    return {
        "accepted_source_ids": source_ids,
        "accepted_source_ids_sha256": _sha_obj(source_ids),
        "n_accepted_source_items": len(source_ids),
        "n_accepted_prompts": len(accepted_rows),
        "n_expected_rollouts": len(accepted_rows) * N_DRAWS,
    }


def _validate_repaired_bank(rows: list[dict], audit: dict[str, Any]) -> None:
    """Require the complete audited factorial lattice and exact cue additions."""
    from scripts.issue952_china_repair_bank import validate_pairs

    if audit.get("contract") != REPAIRED_CONTRACT:
        raise RuntimeError("repaired bank audit contract is missing or incompatible")
    source_ids = set(audit["passing_item_ids"])
    if len(source_ids) != 85 or source_ids != {row["source_prompt_id"] for row in rows}:
        raise RuntimeError("repaired bank must retain all 85 accepted source identities")
    expected = {
        (source, language, frame, content)
        for source in source_ids
        for language in ("en", "zh")
        for frame in ("direct", "academic")
        for content in REPAIRED_CONTENT_CELLS
    }
    realized = {
        (row["source_prompt_id"], row["language"], row["frame"], row["content"]) for row in rows
    }
    if realized != expected or len(rows) != len(expected):
        raise RuntimeError("repaired bank factorial cells are incomplete or duplicated")
    for row in rows:
        subject, cue_level = REPAIRED_CONTENT_CELLS[row["content"]]
        if (
            row.get("bank_contract") != REPAIRED_CONTRACT
            or row.get("subject_arm") != subject
            or row.get("country_cue") != cue_level
            or (cue_level == "absent" and row.get("cue_text") != "")
        ):
            raise RuntimeError("repaired bank row contract/country-cue metadata drift")
    for field in ("source_bank_sha256", "metadata_sha256", "independent_audit_sha256"):
        digest = audit.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeError(f"repaired bank lacks hash-bound provenance: {field}")
    validate_pairs(rows)


def _load_bank(
    bank_path: Path, audit_path: Path, smoke: bool, study: str = "v1"
) -> tuple[list[dict], dict]:
    """Validate the full registered bank and select only audit-accepted rows."""
    rows = _read_jsonl(bank_path)
    config = _study_config(study)
    prompts_per_source = config["prompts_per_source"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or _sha256(bank_path) != audit["prompt_bank_sha256"]:
        raise RuntimeError("prompt bank is not the passed/hash-matched audited bank")
    if len(rows) != config["source_items"] * prompts_per_source or len(
        {row["item_id"] for row in rows}
    ) != len(rows):
        raise RuntimeError("prompt bank cardinality/uniqueness changed")
    all_source_ids = {row["source_prompt_id"] for row in rows}
    passing_ids = audit.get("passing_item_ids")
    if (
        not isinstance(passing_ids, list)
        or len(passing_ids) != len(set(passing_ids))
        or not set(passing_ids) <= all_source_ids
        or len(all_source_ids) != config["source_items"]
        or any(
            sum(row["source_prompt_id"] == source_id for row in rows) != prompts_per_source
            for source_id in all_source_ids
        )
        or audit.get("n_audit_passing_items") != len(passing_ids)
    ):
        raise RuntimeError("audit passing-item identity/cardinality changed")
    passing = set(passing_ids)
    if any(
        not isinstance(row.get("audit_pass"), bool)
        or row["audit_pass"] != (row["source_prompt_id"] in passing)
        for row in rows
    ):
        raise RuntimeError("row audit_pass flags disagree with passing_item_ids")
    if study == "repaired-v2":
        _validate_repaired_bank(rows, audit)
    identity = _accepted_bank_identity(rows, audit)
    if identity["n_accepted_prompts"] != len(passing_ids) * prompts_per_source:
        raise RuntimeError("accepted prompt width disagrees with accepted source identities")
    rows = [row for row in rows if row["source_prompt_id"] in passing]
    if smoke:
        source_ids = sorted({row["source_prompt_id"] for row in rows})[:10]
        rows = [row for row in rows if row["source_prompt_id"] in set(source_ids)]
        if len(source_ids) != 10 or len(rows) != 10 * prompts_per_source:
            raise RuntimeError(
                f"10-source-item smoke must have {10 * prompts_per_source} accepted prompts, "
                f"got {len(rows)}"
            )
    return rows, audit


def stage_inputs(
    out_root: Path, bank_path: Path | None, audit_path: Path | None, study: str = "v1"
) -> tuple[Path, Path]:
    """Stage the selected study's immutable audited inputs and freeze their identity."""
    prefix = _study_config(study)["hf_prefix"]
    if (bank_path is None) != (audit_path is None):
        raise ValueError("explicit --bank and --audit must be supplied together")
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
                    f"{prefix}/inputs/upload_verified.json",
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
                    f"{prefix}/inputs/prompt_bank.jsonl",
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
                    f"{prefix}/inputs/bank_audit_report.json",
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
    full_rows = _read_jsonl(bank)
    _, audit_payload = _load_bank(bank, audit, smoke=False, study=study)
    accepted = _accepted_bank_identity(full_rows, audit_payload)
    stage = {
        "marker_source_revision": marker_revision,
        "data_revision": marker.get("data_revision"),
        "marker_sha256": _sha256(marker_path),
        "prompt_bank_sha256": _sha256(bank),
        "bank_audit_report_sha256": _sha256(audit),
        **{key: value for key, value in accepted.items() if key != "accepted_source_ids"},
    }
    if study == "repaired-v2":
        stage.update(study=study, hf_prefix=prefix, bank_contract=REPAIRED_CONTRACT)
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


def _regime(
    bank_sha: str,
    smoke: bool,
    *,
    attempt: int = 1,
    accepted: dict[str, Any] | None = None,
    selected_rows: list[dict[str, Any]] | None = None,
    study: str = "v1",
) -> dict[str, Any]:
    """Describe all output-affecting generation settings and bank selection."""
    config = _study_config(study)
    seed_base = config["smoke_seed_base"] if smoke else config["seed_base"]
    regime = {
        "issue": ISSUE,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "bank_sha256": bank_sha,
        "layers": list(LAYERS),
        "draws": N_DRAWS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_base": seed_base,
        "seed_namespace": "smoke-disjoint-v1" if smoke else "registered-production-v1",
        "seed_formula": (
            f"{SMOKE_SEED_BASE} + selected_prompt_index * {N_DRAWS} + draw"
            if smoke
            else f"{SEED_BASE} + prompt_index * {N_DRAWS} + draw"
        ),
        "seed_policy": "disjoint-smoke-production-v1",
        "smoke": smoke,
        "attempt": attempt,
        "git_sha": _git_sha(),
    }
    if study == "repaired-v2":
        regime.update(
            study=study,
            hf_prefix=config["hf_prefix"],
            bank_contract=REPAIRED_CONTRACT,
            registered_source_items=config["source_items"],
            prompts_per_source=config["prompts_per_source"],
            seed_namespace="repaired-smoke-v2" if smoke else "repaired-production-v2",
            seed_formula=f"{seed_base} + selected_prompt_index * {N_DRAWS} + draw",
            seed_policy="disjoint-v1-v2-smoke-production-v2",
            cap_extension_policy=dict(REPAIRED_EXTENSION_POLICY),
        )
    if accepted is not None and selected_rows is not None:
        selected_sources = sorted({row["source_prompt_id"] for row in selected_rows})
        regime.update(
            {
                **{key: value for key, value in accepted.items() if key != "accepted_source_ids"},
                "n_selected_prompts": len(selected_rows),
                "selected_source_ids_sha256": _sha_obj(selected_sources),
            }
        )
    return regime


def _repaired_token_pair_checks(rows: list[dict], tok: Any) -> dict[str, Any]:
    """Validate every source's cue pair through the actual rendered tokenizer inputs."""
    from scripts.issue952_china_repair_bank import validate_pairs

    result = validate_pairs(
        rows,
        encode=lambda prompt: tok.encode(_rendered(tok, prompt), add_special_tokens=False),
    )
    if result.get("token_checks_run") is not True or result.get("n_pairs") != 85 * 2 * 2 * 2:
        raise RuntimeError("repaired tokenizer gate did not cover every registered cue pair")
    return result


def _repaired_cap_diagnostics(rows: list[dict], *, allow_extension: bool = True) -> dict[str, Any]:
    """Identify only truncated rows in cells above the registered two-percent threshold."""
    cells: dict[str, list[dict]] = {}
    for row in rows:
        key = ":".join(row[field] for field in ("language", "content", "frame"))
        cells.setdefault(key, []).append(row)
    per_cell = {}
    extension_ids = []
    for key, cell_rows in sorted(cells.items()):
        truncated = [row["item_id"] for row in cell_rows if row["finish_reason"] == "length"]
        fraction = len(truncated) / len(cell_rows)
        above_threshold = fraction > REPAIRED_EXTENSION_POLICY["threshold"]
        needs_extension = above_threshold and allow_extension
        per_cell[key] = {
            "n_rows": len(cell_rows),
            "n_cap_hit": len(truncated),
            "cap_hit_fraction": fraction,
            "extension_required": needs_extension,
            "above_cap_threshold": above_threshold,
        }
        if needs_extension:
            extension_ids.extend(truncated)
    return {
        "cap_hit_definition": "finish_reason == length",
        "cap_extension_threshold_per_cell": REPAIRED_EXTENSION_POLICY["threshold"],
        "cap_by_cell": per_cell,
        "cap_extension_required_item_ids": sorted(extension_ids),
        "cap_gate_passed": not any(cell["above_cap_threshold"] for cell in per_cell.values()),
    }


def _smoke_generation_compatibility(regime: dict[str, Any]) -> dict[str, Any]:
    """Compare every shared setting while allowing the registered smoke slice/seeds."""
    excluded = {
        "smoke",
        "prompt_token_max",
        "max_model_len",
        "n_selected_prompts",
        "selected_source_ids_sha256",
        "seed_base",
        "seed_namespace",
        "seed_formula",
    }
    return {key: value for key, value in regime.items() if key not in excluded}


def _smoke_capture_compatibility(
    capture_regime: dict[str, Any], package_versions: dict[str, str]
) -> dict[str, Any]:
    """Bind capture implementations and settings across distinct smoke responses."""
    return {
        "capture_regime": {
            key: value
            for key, value in capture_regime.items()
            if key
            not in {
                "rollouts_sha256",
                "n_selected_prompts",
                "selected_source_ids_sha256",
                "generation_fingerprint",
            }
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
    attempt: int = 1,
    study: str = "v1",
) -> dict[str, Any]:
    """Generate complete, exact-token rollouts only after all selected study guards pass."""
    _study_config(study)
    if shard_prompts < 1:
        raise ValueError("shard_prompts must be >= 1")
    if not smoke:
        if smoke_report is None or not smoke_report.exists():
            raise RuntimeError("production generation requires a completed smoke timing report")
        smoke_gate = json.loads(smoke_report.read_text())
        if smoke_gate.get("passed") is not True:
            raise RuntimeError("production generation blocked by incomplete smoke evidence")
        if smoke_gate.get("attempt") != attempt:
            raise RuntimeError("production generation smoke gate belongs to another attempt")
    versions = _assert_package_versions()
    rows, audit = _load_bank(bank_path, audit_path, smoke, study=study)
    full_rows = _read_jsonl(bank_path)
    accepted = _accepted_bank_identity(full_rows, audit)
    tok = _tokenizer()
    pair_checks = _repaired_token_pair_checks(full_rows, tok) if study == "repaired-v2" else None
    contexts = [_context_ids(tok, row["prompt"]) for row in rows]
    prompt_max = max(map(len, contexts))
    max_model_len = max(4096, prompt_max + MAX_NEW_TOKENS + 32)
    if max_model_len > int(getattr(tok, "model_max_length", 32768)):
        raise RuntimeError("prompt+generation envelope exceeds tokenizer model maximum")
    bank_sha = _sha256(bank_path)
    if not smoke and smoke_gate["bank_sha256"] != bank_sha:
        raise RuntimeError("production bank differs from the smoke-tested bank")
    regime = _regime(
        bank_sha, smoke, attempt=attempt, accepted=accepted, selected_rows=rows, study=study
    )
    if study == "repaired-v2":
        regime.update(
            country_cue_validation=pair_checks,
            shard_prompts=shard_prompts,
            bank_audit_report_sha256=_sha256(audit_path),
            source_bank_sha256=audit["source_bank_sha256"],
            metadata_sha256=audit["metadata_sha256"],
            independent_audit_sha256=audit["independent_audit_sha256"],
        )
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
        seed=regime["seed_base"],
    )
    t0 = time.time()
    resumed_shards = 0
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
                or not isinstance(done.get("elapsed_s"), (int, float))
                or done["elapsed_s"] < 0
                or [row["item_id"] for row in _read_jsonl(shard_path)] != chunk_expected_ids
            ):
                raise RuntimeError(f"stale generation shard: {shard_path}")
            resumed_shards += 1
            print(f"[gen] resume shard={start // shard_prompts + 1}")
            continue
        shard_t0 = time.time()
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
                        seed=regime["seed_base"] + global_i * N_DRAWS + draw,
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
                    **(
                        {
                            key: row[key]
                            for key in (
                                "subject_arm",
                                "country_cue",
                                "cue_text",
                                "base_prompt_sha256",
                                "bank_contract",
                            )
                        }
                        if study == "repaired-v2"
                        else {}
                    ),
                    "draw": draw,
                    "seed": regime["seed_base"] + global_i * N_DRAWS + draw,
                    "question": row["prompt"],
                    "text": sample.text,
                    "completion_token_ids": list(map(int, sample.token_ids)),
                    "context_tokens": ctx_len,
                    "completion_tokens": n_tokens,
                    "finish_reason": sample.finish_reason,
                    "cap_hit": (
                        sample.finish_reason == "length"
                        if study == "repaired-v2"
                        else n_tokens >= MAX_NEW_TOKENS
                    ),
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
                "elapsed_s": time.time() - shard_t0,
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
    realized_manifests = set(raw_dir.glob("rollouts_p*.done.json"))
    if realized_manifests != {path.with_suffix(".done.json") for path in shard_paths}:
        raise RuntimeError("stale generation shard-manifest residue is present")
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
    final_path = raw_dir / (
        "rollouts.initial.jsonl" if study == "repaired-v2" else "rollouts.jsonl"
    )
    if study == "repaired-v2" and final_path.exists():
        if _read_jsonl(final_path) != all_rows:
            raise RuntimeError("immutable initial rollouts differ from generation checkpoints")
    else:
        _write_jsonl(final_path, all_rows)
    shard_manifests = [json.loads(path.read_text()) for path in sorted(realized_manifests)]
    cumulative_elapsed_s = sum(float(row["elapsed_s"]) for row in shard_manifests)
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
        "elapsed_s": cumulative_elapsed_s,
        "fresh_invocation_elapsed_s": time.time() - t0,
        "timing_evidence_complete": len(shard_manifests)
        == (len(rows) + shard_prompts - 1) // shard_prompts,
        "n_resumed_shards": resumed_shards,
        "tokens_generated": sum(row["completion_tokens"] for row in all_rows),
        "tokens_per_second": sum(row["completion_tokens"] for row in all_rows)
        / max(cumulative_elapsed_s, 1e-9),
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
        "accepted_bank": accepted,
        "cap_gate_passed": cap_frac <= 0.005,
    }
    if study == "repaired-v2":
        report.update(_repaired_cap_diagnostics(all_rows))
        report["cap_extension_policy_checked"] = False
        initial_manifest = out_root / "manifests" / "generation.initial.json"
        if initial_manifest.exists():
            initial = json.loads(initial_manifest.read_text())
            if _generation_fingerprint(initial) != _generation_fingerprint(report):
                raise RuntimeError("immutable initial generation manifest differs from checkpoints")
            # A gen resume must not overwrite either the original timing or the merged final view.
            report = {**initial, "n_resumed_shards": resumed_shards}
        else:
            _write_json(initial_manifest, report)
            _write_jsonl(raw_dir / "rollouts.jsonl", all_rows)
            _write_json(out_root / "manifests" / "generation.json", report)
    else:
        _write_json(out_root / "manifests" / "generation.json", report)
    print(
        f"[gen] complete prompts={len(rows)} rows={len(all_rows)} cap_frac={cap_frac:.6f} "
        f"sha={report['rollouts_sha256'][:12]}"
    )
    if report["n_empty"]:
        raise RuntimeError(f"generation produced {report['n_empty']} empty completions")
    return report


def _extension_baseline(out_root: Path) -> tuple[dict, list[dict], str]:
    """Read the immutable initial draw set and reject any changed provenance or coverage."""
    manifest = out_root / "manifests" / "generation.initial.json"
    initial = json.loads(manifest.read_text())
    _validate_study(initial["regime"], "repaired-v2")
    raw = out_root / "raw_completions" / "rollouts.initial.jsonl"
    rows = _read_jsonl(raw)
    ids = [row["item_id"] for row in rows]
    if (
        initial["regime"].get("cap_extension_policy") != REPAIRED_EXTENSION_POLICY
        or _sha_obj(initial["regime"]) != initial["regime_fp"]
        or _sha256(raw) != initial["rollouts_sha256"]
        or len(rows) != initial["n_rows"]
        or len(set(ids)) != len(ids)
        or _sha_obj(ids) != initial["ordered_item_ids_sha256"]
        or initial.get("timing_evidence_complete") is not True
    ):
        raise RuntimeError("initial generation evidence is stale or extension-incompatible")
    return initial, rows, _sha256(manifest)


def _extension_lineage(original: dict, initial_sha: str) -> dict:
    """Bind one extended response to its unchanged original draw and doubled cap."""
    return {
        "contract": REPAIRED_EXTENSION_POLICY["contract"],
        "initial_generation_manifest_sha256": initial_sha,
        "initial_row_sha256": _sha_obj(original),
        "original_finish_reason": original["finish_reason"],
        "original_completion_tokens": original["completion_tokens"],
        "max_new_tokens": REPAIRED_EXTENSION_POLICY["max_new_tokens"],
        "pass": 1,
    }


def _validate_extended_rows(originals: list[dict], extended: list[dict], initial_sha: str) -> None:
    """Reject wrong IDs, changed questions/seeds, missing tokens, or fabricated merge lineage."""
    if [row["item_id"] for row in extended] != [row["item_id"] for row in originals]:
        raise RuntimeError(
            "extension rollout identity/order differs from the selected original IDs"
        )
    output_fields = {
        "text",
        "completion_token_ids",
        "completion_tokens",
        "finish_reason",
        "cap_hit",
        "latency_s",
    }
    for original, row in zip(originals, extended, strict=True):
        retained = {key: value for key, value in original.items() if key not in output_fields}
        observed = {
            key: value
            for key, value in row.items()
            if key not in output_fields and key != "cap_extension"
        }
        tokens = row.get("completion_token_ids")
        if (
            original["finish_reason"] != "length"
            or retained != observed
            or row.get("cap_extension") != _extension_lineage(original, initial_sha)
            or not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int for token in tokens)
            or row.get("completion_tokens") != len(tokens)
            or len(tokens) > REPAIRED_EXTENSION_POLICY["max_new_tokens"]
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or row.get("finish_reason") not in ("stop", "length")
            or row.get("cap_hit") is not (row["finish_reason"] == "length")
        ):
            raise RuntimeError("extension response changed original metadata or has invalid output")


def _validate_cap_extension(out_root: Path, generation: dict[str, Any]) -> None:
    """Require exactly one complete, hash-bound policy pass before consuming repaired outputs."""
    if generation["regime"].get("study", "v1") != "repaired-v2":
        return
    manifest_path = out_root / "manifests" / "cap_extension.json"
    if generation.get("cap_extension_policy_checked") is not True or not manifest_path.exists():
        raise RuntimeError("repaired generation requires the completed cap-extension policy phase")
    extension = json.loads(manifest_path.read_text())
    initial, original, initial_sha = _extension_baseline(out_root)
    required = _repaired_cap_diagnostics(original)["cap_extension_required_item_ids"]
    required_set = set(required)
    selected = [row for row in original if row["item_id"] in required_set]
    extended_path = out_root / "raw_completions" / "rollouts.extensions.jsonl"
    extended = _read_jsonl(extended_path)
    _validate_extended_rows(selected, extended, initial_sha)
    replacements = {row["item_id"]: row for row in extended}
    merged = [replacements.get(row["item_id"], row) for row in original]
    final_path = out_root / "raw_completions" / "rollouts.jsonl"
    if (
        extension.get("policy") != REPAIRED_EXTENSION_POLICY
        or extension.get("policy_checked") is not True
        or extension.get("initial_generation_manifest_sha256") != initial_sha
        or extension.get("initial_rollouts_sha256") != initial["rollouts_sha256"]
        or extension.get("required_item_ids") != required
        or extension.get("n_extended_rows") != len(selected)
        or extension.get("extended_rollouts_sha256") != _sha256(extended_path)
        or extension.get("merged_rollouts_sha256") != _sha256(final_path)
        or generation.get("cap_extension_manifest_sha256") != _sha256(manifest_path)
        or generation.get("rollouts_sha256") != _sha256(final_path)
        or generation.get("regime") != initial["regime"]
        or generation.get("regime_fp") != initial["regime_fp"]
        or generation.get("n_rows") != initial["n_rows"]
        or generation.get("n_prompts") != initial["n_prompts"]
        or generation.get("ordered_item_ids_sha256") != initial["ordered_item_ids_sha256"]
        or _read_jsonl(final_path) != merged
    ):
        raise RuntimeError("cap-extension policy evidence or exact-ID merge is stale")
    shard_files = extension.get("shard_files")
    if not isinstance(shard_files, dict):
        raise RuntimeError("cap-extension shard evidence is missing")
    shard_dir = out_root / "raw_completions" / "extensions"
    if {path.name for path in shard_dir.glob("*")} != set(shard_files):
        raise RuntimeError("cap-extension shard census is stale")
    if any(_sha256(shard_dir / name) != digest for name, digest in shard_files.items()):
        raise RuntimeError("cap-extension shard bytes changed after completion")


def _generate_extension_shards(
    out_root: Path, initial: dict, selected: list[dict], initial_sha: str, tok: Any
) -> tuple[list[dict], float, dict[str, str]]:
    """Generate only missing extension shards; completed shards never load another engine."""
    from vllm import LLM, SamplingParams

    max_tokens = REPAIRED_EXTENSION_POLICY["max_new_tokens"]
    max_model_len = max(4096, initial["regime"]["prompt_token_max"] + max_tokens + 32)
    if max_model_len > int(getattr(tok, "model_max_length", 32768)):
        raise RuntimeError("extension prompt+generation exceeds tokenizer model maximum")
    extension_fp = _sha_obj(
        {
            "initial_generation_manifest_sha256": initial_sha,
            "policy": REPAIRED_EXTENSION_POLICY,
            "max_model_len": max_model_len,
        }
    )
    shard_dir = out_root / "raw_completions" / "extensions"
    shard_dir.mkdir(parents=True, exist_ok=True)
    llm = None
    extended, shard_files = [], {}
    elapsed_s = 0.0
    width = REPAIRED_EXTENSION_POLICY["shard_rows"]
    for start in range(0, len(selected), width):
        chunk = selected[start : start + width]
        path = shard_dir / f"rollouts_r{start:05d}_{start + len(chunk):05d}.jsonl"
        done_path = path.with_suffix(".done.json")
        expected_ids = [row["item_id"] for row in chunk]
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if (
                done.get("extension_fp") != extension_fp
                or done.get("sha256") != _sha256(path)
                or done.get("ordered_item_ids_sha256") != _sha_obj(expected_ids)
                or not isinstance(done.get("elapsed_s"), (int, float))
                or done["elapsed_s"] < 0
            ):
                raise RuntimeError("stale completed cap-extension shard")
            realized = _read_jsonl(path)
            _validate_extended_rows(chunk, realized, initial_sha)
        else:
            if llm is None:
                llm = LLM(
                    model=MODEL,
                    revision=MODEL_REV,
                    tokenizer_revision=MODEL_REV,
                    dtype="bfloat16",
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.82,
                    seed=initial["regime"]["seed_base"],
                )
            t0 = time.time()
            contexts = [_context_ids(tok, row["question"]) for row in chunk]
            outputs = llm.generate(
                [_rendered(tok, row["question"]) for row in chunk],
                [
                    SamplingParams(
                        n=1,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=max_tokens,
                        seed=row["seed"],
                    )
                    for row in chunk
                ],
                use_tqdm=False,
            )
            realized = []
            for original, context, output in zip(chunk, contexts, outputs, strict=True):
                if (
                    list(map(int, output.prompt_token_ids)) != context
                    or len(context) != original["context_tokens"]
                ):
                    raise RuntimeError("extension generation prompt-token identity mismatch")
                sample = output.outputs[0]
                realized.append(
                    {
                        **original,
                        "text": sample.text,
                        "completion_token_ids": list(map(int, sample.token_ids)),
                        "completion_tokens": len(sample.token_ids),
                        "finish_reason": sample.finish_reason,
                        "cap_hit": sample.finish_reason == "length",
                        "latency_s": (
                            float(output.metrics.finished_time - output.metrics.arrival_time)
                            if output.metrics is not None
                            and output.metrics.finished_time is not None
                            else None
                        ),
                        "cap_extension": _extension_lineage(original, initial_sha),
                    }
                )
            _write_jsonl(path, realized)
            _validate_extended_rows(chunk, realized, initial_sha)
            done = {
                "extension_fp": extension_fp,
                "sha256": _sha256(path),
                "ordered_item_ids_sha256": _sha_obj(expected_ids),
                "elapsed_s": time.time() - t0,
            }
            _write_json(done_path, done)
        extended.extend(realized)
        elapsed_s += done["elapsed_s"]
        shard_files.update({path.name: _sha256(path), done_path.name: _sha256(done_path)})
        print(
            f"[extend] rows={start}:{start + len(chunk)}/{len(selected)} elapsed={elapsed_s:.1f}s"
        )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if {path.name for path in shard_dir.glob("*")} != set(shard_files):
        raise RuntimeError("unexpected cap-extension shard residue")
    return extended, elapsed_s, shard_files


def phase_extend(
    out_root: Path, bank_path: Path, audit_path: Path, attempt: int = 1, study: str = "v1"
) -> dict[str, Any]:
    """Apply the registered single doubled-cap pass and publish its exact-ID merged view."""
    phase_start = time.time()
    if study != "repaired-v2":
        raise ValueError("extend is only available with --study repaired-v2")
    initial, original, initial_sha = _extension_baseline(out_root)
    if initial["regime"]["attempt"] != attempt:
        raise RuntimeError("extension attempt differs from original generation")
    bank_rows, _ = _load_bank(bank_path, audit_path, initial["regime"]["smoke"], study=study)
    if (
        _sha256(bank_path) != initial["regime"]["bank_sha256"]
        or _sha256(audit_path) != initial["regime"]["bank_audit_report_sha256"]
        or _expected_rollout_ids(bank_rows) != [row["item_id"] for row in original]
        or initial["regime"]["git_sha"] != _git_sha()
    ):
        raise RuntimeError("extension bank/audit/code differs from original generation")
    manifest_path = out_root / "manifests" / "cap_extension.json"
    generation_path = out_root / "manifests" / "generation.json"
    if manifest_path.exists() and generation_path.exists():
        current = json.loads(generation_path.read_text())
        if current.get("cap_extension_policy_checked") is True:
            _validate_cap_extension(out_root, current)
            print("[extend] resumed verified completed extension policy")
            return current
    initial_caps = _repaired_cap_diagnostics(original)
    required = initial_caps["cap_extension_required_item_ids"]
    required_set = set(required)
    selected = [row for row in original if row["item_id"] in required_set]
    if selected:
        versions = _assert_package_versions()
        tok = _tokenizer()
        if (
            versions != initial["package_versions"]
            or _repaired_token_pair_checks(_read_jsonl(bank_path), tok)
            != initial["regime"]["country_cue_validation"]
            or hashlib.sha256(tok.chat_template.encode()).hexdigest()
            != initial["regime"]["chat_template_sha256"]
        ):
            raise RuntimeError("extension tokenizer/runtime differs from initial generation")
        extended, elapsed_s, shard_files = _generate_extension_shards(
            out_root, initial, selected, initial_sha, tok
        )
    else:
        extended, elapsed_s, shard_files = [], 0.0, {}
    replacements = {row["item_id"]: row for row in extended}
    merged = [replacements.get(row["item_id"], row) for row in original]
    raw_dir = out_root / "raw_completions"
    _write_jsonl(raw_dir / "rollouts.extensions.jsonl", extended)
    _write_jsonl(raw_dir / "rollouts.jsonl", merged)
    final_caps = _repaired_cap_diagnostics(merged, allow_extension=False)
    remaining = [row["item_id"] for row in merged if row["finish_reason"] == "length"]
    extension = {
        "policy": dict(REPAIRED_EXTENSION_POLICY),
        "policy_checked": True,
        "status": "applied_once" if selected else "no_extension_required",
        "initial_generation_manifest_sha256": initial_sha,
        "initial_rollouts_sha256": initial["rollouts_sha256"],
        "required_item_ids": required,
        "n_extended_rows": len(extended),
        "shard_files": shard_files,
        "elapsed_s": elapsed_s,
        "extended_rollouts_sha256": _sha256(raw_dir / "rollouts.extensions.jsonl"),
        "merged_rollouts_sha256": _sha256(raw_dir / "rollouts.jsonl"),
        "initial_cap_by_cell": initial_caps["cap_by_cell"],
        "remaining_cap_hit_item_ids": remaining,
        "remaining_cap_by_cell": final_caps["cap_by_cell"],
    }
    _write_json(manifest_path, extension)
    total_elapsed = initial["elapsed_s"] + elapsed_s
    compute_tokens = initial["tokens_generated"] + sum(row["completion_tokens"] for row in extended)
    latencies = [row["latency_s"] for row in [*original, *extended] if row["latency_s"] is not None]
    report = {
        **initial,
        **final_caps,
        "rollouts_sha256": extension["merged_rollouts_sha256"],
        "initial_generation_manifest_sha256": initial_sha,
        "initial_rollouts_sha256": initial["rollouts_sha256"],
        "cap_extension_manifest_sha256": _sha256(manifest_path),
        "cap_extension_policy_checked": True,
        "cap_extension_status": extension["status"],
        "cap_extension_initial_required_item_ids": required,
        "cap_extension_required_item_ids": [],
        "remaining_cap_hit_item_ids": remaining,
        "n_empty": sum(not row["text"] for row in merged),
        "n_cap_hit": len(remaining),
        "cap_hit_fraction": len(remaining) / len(merged),
        "elapsed_s": total_elapsed,
        "extension_elapsed_s": elapsed_s,
        "initial_fresh_invocation_elapsed_s": initial["fresh_invocation_elapsed_s"],
        "fresh_invocation_elapsed_s": time.time() - phase_start,
        "p90_request_latency_s": float(torch.quantile(torch.tensor(latencies), 0.9))
        if latencies
        else None,
        "tokens_generated": sum(row["completion_tokens"] for row in merged),
        "tokens_generated_including_extensions": compute_tokens,
        "tokens_per_second": compute_tokens / max(total_elapsed, 1e-9),
        "peak_hbm_bytes": max(initial["peak_hbm_bytes"], torch.cuda.max_memory_allocated())
        if selected
        else initial["peak_hbm_bytes"],
        "peak_host_rss_bytes": max(initial["peak_host_rss_bytes"], _peak_host_rss_bytes()),
    }
    _write_json(generation_path, report)
    _validate_cap_extension(out_root, report)
    print(f"[extend] checked selected={len(selected)} remaining_truncated={len(remaining)}")
    return report


def _upload_folder(folder: Path, path_in_repo: str, message: str) -> dict[str, Any]:
    """Upload one folder and require an immutable returned commit revision."""
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
    revision = getattr(info, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"{message} did not return an immutable revision")
    return {"commit_url": str(info), "revision": revision}


def _verify_remote_file(
    out_root: Path,
    *,
    revision: str,
    remote_path: str,
    expected_sha256: str,
) -> Path:
    """Download a file from an exact revision and verify its local byte hash."""
    if not revision or revision == "main":
        raise RuntimeError("remote byte verification requires an immutable revision")
    fetched = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                remote_path,
                repo_type="dataset",
                revision=revision,
                local_dir=out_root / "_upload_verify" / revision[:12],
                force_download=True,
            ),
            what=f"issue952 exact-revision byte verification {remote_path}",
        )
    )
    if _sha256(fetched) != expected_sha256:
        raise RuntimeError(f"revision-scoped uploaded byte hash mismatch: {remote_path}")
    return fetched


def _generation_fingerprint(report: dict[str, Any]) -> str:
    """Bind generation identity fields used by every downstream phase."""
    return _sha_obj(
        {
            "regime_fp": report["regime_fp"],
            "rollouts_sha256": report["rollouts_sha256"],
            "n_prompts": report["n_prompts"],
            "n_rows": report["n_rows"],
            "ordered_item_ids_sha256": report["ordered_item_ids_sha256"],
            **(
                {"cap_extension_manifest_sha256": report.get("cap_extension_manifest_sha256")}
                if report["regime"].get("study") == "repaired-v2"
                else {}
            ),
        }
    )


def _validate_raw_upload(
    out_root: Path, generation: dict[str, Any], raw_upload: dict[str, Any]
) -> None:
    """Reject stale upload evidence before capture or finalization."""
    _validate_cap_extension(out_root, generation)
    rollouts = out_root / "raw_completions" / "rollouts.jsonl"
    generation_path = out_root / "manifests" / "generation.json"
    if (
        _sha256(rollouts) != generation.get("rollouts_sha256")
        or raw_upload.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or raw_upload.get("generation_manifest_sha256") != _sha256(generation_path)
        or raw_upload.get("generation_fingerprint") != _generation_fingerprint(generation)
        or raw_upload.get("raw_payload_revision") != raw_upload.get("revision")
        or not raw_upload.get("raw_payload_revision")
    ):
        raise RuntimeError("raw upload evidence is stale or generation-incompatible")
    if generation["regime"].get("study") == "repaired-v2":
        prefix = _output_prefix(
            generation["regime"]["smoke"], generation["regime"]["attempt"], "repaired-v2"
        )
        expected = {
            f"{prefix}/raw_completions/{name}": digest
            for name, digest in _raw_payload_hashes(out_root).items()
        }
        if raw_upload.get("byte_verified_sha256") != expected:
            raise RuntimeError("raw upload does not verify the original and extension payloads")


def _raw_payload_hashes(out_root: Path) -> dict[str, str]:
    """Census every persisted raw rollout and checkpoint for exact-revision verification."""
    raw_dir = out_root / "raw_completions"
    return {
        path.relative_to(raw_dir).as_posix(): _sha256(path)
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file()
    }


def phase_upload_raw(out_root: Path, attempt: int = 1, study: str = "v1") -> dict[str, Any]:
    """Upload and byte-verify the current generation artifacts."""
    report = json.loads((out_root / "manifests" / "generation.json").read_text())
    _validate_study(report["regime"], study)
    _validate_cap_extension(out_root, report)
    if report["regime"].get("attempt") != attempt:
        raise RuntimeError("raw upload attempt differs from generation regime")
    target_prefix = _output_prefix(report["regime"]["smoke"], attempt, study=study)
    rollouts = out_root / "raw_completions" / "rollouts.jsonl"
    if (
        _sha256(rollouts) != report["rollouts_sha256"]
        or report.get("timing_evidence_complete") is not True
    ):
        raise RuntimeError("raw upload blocked by generation identity/timing evidence gate")
    generation_path = out_root / "manifests" / "generation.json"
    generation_fingerprint = _generation_fingerprint(report)
    result = _upload_folder(
        out_root / "raw_completions",
        f"{target_prefix}/raw_completions",
        "Issue 952: bilingual China Qwen rollouts",
    )
    result["raw_payload_revision"] = result["revision"]
    result["rollouts_sha256"] = _sha256(rollouts)
    result["generation_manifest_sha256"] = _sha256(generation_path)
    result["generation_fingerprint"] = generation_fingerprint
    _verify_remote_file(
        out_root,
        revision=result["raw_payload_revision"],
        remote_path=f"{target_prefix}/raw_completions/rollouts.jsonl",
        expected_sha256=result["rollouts_sha256"],
    )
    raw_expected = {}
    if study == "repaired-v2":
        raw_expected = {
            f"{target_prefix}/raw_completions/{name}": digest
            for name, digest in _raw_payload_hashes(out_root).items()
        }
        for path, expected_sha in raw_expected.items():
            _verify_remote_file(
                out_root,
                revision=result["raw_payload_revision"],
                remote_path=path,
                expected_sha256=expected_sha,
            )
        result["byte_verified_sha256"] = raw_expected
    _write_json(out_root / "manifests" / "raw_upload.json", result)
    manifest_result = _upload_folder(
        out_root / "manifests",
        f"{target_prefix}/manifests",
        "Issue 952: bilingual China generation manifests",
    )
    final_rev = manifest_result["revision"]
    expected = {
        **raw_expected,
        f"{target_prefix}/raw_completions/rollouts.jsonl": _sha256(rollouts),
        f"{target_prefix}/manifests/generation.json": _sha256(generation_path),
        f"{target_prefix}/manifests/raw_upload.json": _sha256(
            out_root / "manifests" / "raw_upload.json"
        ),
    }
    if study == "repaired-v2":
        for name in ("generation.initial.json", "cap_extension.json"):
            expected[f"{target_prefix}/manifests/{name}"] = _sha256(out_root / "manifests" / name)
    for path, expected_sha in expected.items():
        _verify_remote_file(
            out_root, revision=final_rev, remote_path=path, expected_sha256=expected_sha
        )
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
    attempt: int = 1,
    study: str = "v1",
) -> dict[str, Any]:
    """Capture the selected study's exact generation tokens after verified raw upload."""
    if batch_size < 1 or answer_shard_rows < 1:
        raise ValueError("capture batch size and answer shard rows must be >= 1")
    raw_upload_path = out_root / "manifests" / "raw_upload.json"
    if not raw_upload_path.exists():
        raise RuntimeError("capture blocked: raw-rollout upload has not been verified")
    bank_rows, audit = _load_bank(bank_path, audit_path, smoke, study=study)
    accepted = _accepted_bank_identity(_read_jsonl(bank_path), audit)
    rollouts_path = out_root / "raw_completions" / "rollouts.jsonl"
    gen = json.loads((out_root / "manifests" / "generation.json").read_text())
    _validate_study(gen["regime"], study)
    _validate_cap_extension(out_root, gen)
    if study == "repaired-v2" and (
        gen["regime"].get("smoke") != smoke
        or gen["regime"].get("bank_audit_report_sha256") != _sha256(audit_path)
    ):
        raise RuntimeError("repaired capture smoke/audit identity differs from generation")
    if gen.get("regime", {}).get("attempt") != attempt:
        raise RuntimeError("capture attempt differs from generation regime")
    raw_upload = json.loads(raw_upload_path.read_text())
    _validate_raw_upload(out_root, gen, raw_upload)
    if (
        _sha256(rollouts_path) != gen["rollouts_sha256"]
        or _sha256(bank_path) != gen["regime"]["bank_sha256"]
        or gen["regime"].get("accepted_source_ids_sha256") != accepted["accepted_source_ids_sha256"]
        or gen["regime"].get("n_accepted_prompts") != accepted["n_accepted_prompts"]
        or gen["regime"].get("n_selected_prompts") != len(bank_rows)
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
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": accepted["n_accepted_source_items"],
        "n_accepted_prompts": accepted["n_accepted_prompts"],
        "n_selected_prompts": len(bank_rows),
        "selected_source_ids_sha256": gen["regime"]["selected_source_ids_sha256"],
        "generation_fingerprint": _generation_fingerprint(gen),
    }
    if study == "repaired-v2":
        capture_regime.update(
            study=study,
            hf_prefix=_study_config(study)["hf_prefix"],
            bank_contract=REPAIRED_CONTRACT,
            bank_audit_report_sha256=_sha256(audit_path),
            source_bank_sha256=audit["source_bank_sha256"],
            metadata_sha256=audit["metadata_sha256"],
            independent_audit_sha256=audit["independent_audit_sha256"],
            capture_batch=batch_size,
            answer_shard_rows=answer_shard_rows,
            cap_extension_policy=gen["regime"]["cap_extension_policy"],
        )
    versions = _package_versions()
    if not smoke:
        if smoke_report is None or not smoke_report.exists():
            raise RuntimeError("production capture requires a completed smoke timing report")
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
            or not isinstance(prior_vc.get("elapsed_s"), (int, float))
            or prior_vc["elapsed_s"] < 0
        ):
            raise RuntimeError("stale context capture checkpoint")
    else:
        vc_t0 = time.time()
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
                "elapsed_s": time.time() - vc_t0,
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
                or not isinstance(store.get("elapsed_s"), (int, float))
                or store["elapsed_s"] < 0
            ):
                raise RuntimeError(f"stale answer capture shard: {path}")
            answer_paths.append(path)
            all_empty.extend(start + int(i) for i in store["empty_rows"])
            print(f"[capture-va] resume rows={start}:{start + len(chunk)}")
            continue
        shard_t0 = time.time()
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
                "elapsed_s": time.time() - shard_t0,
            },
        )
        answer_paths.append(path)
        all_empty.extend(start + i for i in empty)
        print(
            f"[capture-va] rows={start}:{start + len(chunk)} empty={len(empty)} "
            f"sha={_sha256(path)[:12]} elapsed={time.time() - t0:.1f}s"
        )
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False)
    realized_answer_paths = set((out_root / "analysis_tensors").glob("va_*.pt"))
    if realized_answer_paths != set(answer_paths):
        raise RuntimeError("stale answer capture shard residue is present")
    va_elapsed_s = 0.0
    for path in answer_paths:
        va_elapsed_s += float(torch.load(path, map_location="cpu", weights_only=False)["elapsed_s"])
    cumulative_elapsed_s = float(vc_store["elapsed_s"]) + va_elapsed_s
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
        "elapsed_s": cumulative_elapsed_s,
        "fresh_invocation_elapsed_s": time.time() - t0,
        "timing_evidence_complete": len(answer_paths)
        == (len(rollout_rows) + answer_shard_rows - 1) // answer_shard_rows,
        "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": _peak_host_rss_bytes(),
        "examples_per_second": len(rollout_rows) / max(time.time() - t0, 1e-9),
        "serialized_bytes": vc_path.stat().st_size
        + sum(path.stat().st_size for path in answer_paths),
        "package_versions": versions,
        "accepted_bank": accepted,
        "generation_fingerprint": _generation_fingerprint(gen),
    }
    if study == "repaired-v2":
        report["country_cue_capture"] = _repaired_capture_diagnostics(bank_rows, vc_store["vc"])
    _write_json(out_root / "manifests" / "capture.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if all_empty:
        raise RuntimeError(f"activation capture has {len(all_empty)} empty answer rows")
    return report


def _repaired_capture_diagnostics(rows: list[dict], vc: torch.Tensor) -> dict[str, Any]:
    """Persist actual cue-delta norms; numerical degeneracy remains explicit."""
    if vc.shape != (len(rows), len(LAYERS), HIDDEN) or not torch.isfinite(vc).all():
        raise RuntimeError("repaired context capture has invalid shape or nonfinite values")
    index = {
        (row["source_prompt_id"], row["language"], row["frame"], row["content"]): i
        for i, row in enumerate(rows)
    }
    absent_indices, present_indices, pair_ids = [], [], []
    for absent, present in (
        ("sensitive_country_neutral", "sensitive_full"),
        ("matched_non_china", "matched_non_china_country"),
    ):
        for i, row in enumerate(rows):
            if row["content"] != absent:
                continue
            j = index[(row["source_prompt_id"], row["language"], row["frame"], present)]
            absent_indices.append(i)
            present_indices.append(j)
            pair_ids.append([row["item_id"], rows[j]["item_id"]])
    norms = torch.linalg.vector_norm(vc[present_indices] - vc[absent_indices], dim=-1)
    return {
        "layers": list(LAYERS),
        "n_pairs": len(pair_ids),
        "pair_ids_absent_present": pair_ids,
        "cue_delta_norms": norms.tolist(),
        "n_zero_delta_pairs_by_layer": (norms == 0).sum(dim=0).tolist(),
        "minimum_delta_norm_by_layer": norms.min(dim=0).values.tolist(),
    }


def _capture_fingerprint(report: dict[str, Any]) -> str:
    """Bind capture identity fields used by upload and finalization."""
    return _sha_obj(
        {
            "capture_regime_fp": report["capture_regime_fp"],
            "generation_fingerprint": report["generation_fingerprint"],
            "vc_sha256": report["vc_sha256"],
            "va_files": report["va_files"],
            "n_contexts": report["n_contexts"],
            "n_answer_rows": report["n_answer_rows"],
        }
    )


def _validate_capture_artifacts(
    out_root: Path, generation: dict[str, Any], capture: dict[str, Any]
) -> None:
    """Reject stale or incomplete local activation-capture evidence."""
    tensor_dir = out_root / "analysis_tensors"
    va_files = capture.get("va_files")
    if (
        not isinstance(va_files, dict)
        or capture.get("generation_fingerprint") != _generation_fingerprint(generation)
        or capture.get("n_answer_rows") != generation.get("n_rows")
        or capture.get("timing_evidence_complete") is not True
        or _sha256(tensor_dir / "vc.pt") != capture.get("vc_sha256")
    ):
        raise RuntimeError("capture evidence is stale or generation-incompatible")
    for name, sha in va_files.items():
        if _sha256(tensor_dir / name) != sha:
            raise RuntimeError(f"answer capture hash drift: {name}")


def _validate_capture_upload(
    out_root: Path, capture: dict[str, Any], capture_upload: dict[str, Any]
) -> None:
    """Reject stale capture-upload markers before terminal finalization."""
    capture_path = out_root / "manifests" / "capture.json"
    raw_upload_path = out_root / "manifests" / "raw_upload.json"
    byte_verified = capture_upload.get("byte_verified_sha256")
    expected_verified = {"vc.pt": capture.get("vc_sha256"), **capture.get("va_files", {})}
    realized_verified = (
        {Path(path).name: sha for path, sha in byte_verified.items()}
        if isinstance(byte_verified, dict)
        else None
    )
    if (
        capture_upload.get("capture_manifest_sha256") != _sha256(capture_path)
        or capture_upload.get("raw_upload_manifest_sha256") != _sha256(raw_upload_path)
        or capture_upload.get("capture_fingerprint") != _capture_fingerprint(capture)
        or capture_upload.get("vc_sha256") != capture.get("vc_sha256")
        or capture_upload.get("va_files") != capture.get("va_files")
        or realized_verified != expected_verified
        or set(capture_upload.get("byte_verified_files", [])) != set(byte_verified or {})
        or capture_upload.get("tensor_payload_revision") != capture_upload.get("revision")
        or not capture_upload.get("tensor_payload_revision")
    ):
        raise RuntimeError("capture upload evidence is stale or capture-incompatible")


def phase_upload_capture(out_root: Path, attempt: int = 1, study: str = "v1") -> dict[str, Any]:
    """Upload and byte-verify current capture artifacts and manifests."""
    report = json.loads((out_root / "manifests" / "capture.json").read_text())
    generation = json.loads((out_root / "manifests" / "generation.json").read_text())
    _validate_study(generation["regime"], study)
    raw_upload = json.loads((out_root / "manifests" / "raw_upload.json").read_text())
    if generation["regime"].get("attempt") != attempt:
        raise RuntimeError("capture upload attempt differs from generation regime")
    target_prefix = _output_prefix(generation["regime"]["smoke"], attempt, study=study)
    tensor_dir = out_root / "analysis_tensors"
    _validate_raw_upload(out_root, generation, raw_upload)
    _validate_capture_artifacts(out_root, generation, report)
    result = _upload_folder(
        tensor_dir,
        f"{target_prefix}/analysis_tensors",
        "Issue 952: bilingual China Qwen activation captures",
    )
    check_rev = result["revision"]
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
    # Every tensor is load-bearing; verify every uploaded byte at both immutable snapshots.
    verify_paths = required
    expected_shas = {
        f"{target_prefix}/analysis_tensors/vc.pt": report["vc_sha256"],
        **{
            f"{target_prefix}/analysis_tensors/{name}": sha
            for name, sha in report["va_files"].items()
        },
    }
    downloaded = {
        path: _verify_remote_file(
            out_root, revision=check_rev, remote_path=path, expected_sha256=expected_shas[path]
        )
        for path in verify_paths
    }
    probe = downloaded[required[0]]
    opened = torch.load(probe, map_location="cpu", weights_only=False)
    if opened["vc"].shape != (report["n_contexts"], len(LAYERS), HIDDEN):
        raise RuntimeError("uploaded context tensor consumer-open shape mismatch")
    result["verified_files"] = len(required)
    result["consumer_open_shape"] = list(opened["vc"].shape)
    result["capture_manifest_sha256"] = _sha256(out_root / "manifests" / "capture.json")
    result["raw_upload_manifest_sha256"] = _sha256(out_root / "manifests" / "raw_upload.json")
    result["capture_fingerprint"] = _capture_fingerprint(report)
    result["vc_sha256"] = report["vc_sha256"]
    result["va_files"] = report["va_files"]
    result["tensor_payload_revision"] = check_rev
    result["byte_verified_files"] = sorted(verify_paths)
    result["byte_verified_sha256"] = {path: expected_shas[path] for path in sorted(verify_paths)}
    _write_json(out_root / "manifests" / "capture_upload.json", result)
    manifest_result = _upload_folder(
        out_root / "manifests",
        f"{target_prefix}/manifests",
        "Issue 952: bilingual China capture manifests",
    )
    final_rev = manifest_result["revision"]
    manifest_expected = {
        f"{target_prefix}/manifests/generation.json": _sha256(
            out_root / "manifests" / "generation.json"
        ),
        f"{target_prefix}/manifests/raw_upload.json": _sha256(
            out_root / "manifests" / "raw_upload.json"
        ),
        f"{target_prefix}/manifests/capture.json": _sha256(out_root / "manifests" / "capture.json"),
        f"{target_prefix}/manifests/capture_upload.json": _sha256(
            out_root / "manifests" / "capture_upload.json"
        ),
    }
    for path, expected_sha in manifest_expected.items():
        _verify_remote_file(
            out_root, revision=final_rev, remote_path=path, expected_sha256=expected_sha
        )
    for path in verify_paths:
        _verify_remote_file(
            out_root, revision=final_rev, remote_path=path, expected_sha256=expected_shas[path]
        )
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


def phase_finalize(out_root: Path, attempt: int = 1, study: str = "v1") -> dict[str, Any]:
    """Validate every phase and publish the terminal sentinel last."""
    sentinel_path = out_root / "issue952_china_definitive_done.json"
    generation = json.loads((out_root / "manifests" / "generation.json").read_text())
    _validate_study(generation["regime"], study)
    sentinel_path.unlink(missing_ok=True)
    _validate_cap_extension(out_root, generation)
    capture = json.loads((out_root / "manifests" / "capture.json").read_text())
    raw_upload = json.loads((out_root / "manifests" / "raw_upload.json").read_text())
    capture_upload = json.loads((out_root / "manifests" / "capture_upload.json").read_text())
    _validate_raw_upload(out_root, generation, raw_upload)
    _validate_capture_artifacts(out_root, generation, capture)
    _validate_capture_upload(out_root, capture, capture_upload)
    if generation["n_rows"] != capture["n_answer_rows"]:
        raise RuntimeError("final generation/capture row mismatch")
    if generation["regime"].get("attempt") != attempt:
        raise RuntimeError("finalize attempt differs from generation regime")
    target_prefix = _output_prefix(generation["regime"]["smoke"], attempt, study=study)
    _verify_remote_file(
        out_root,
        revision=raw_upload["raw_payload_revision"],
        remote_path=f"{target_prefix}/raw_completions/rollouts.jsonl",
        expected_sha256=generation["rollouts_sha256"],
    )
    _verify_remote_file(
        out_root,
        revision=capture_upload["tensor_payload_revision"],
        remote_path=f"{target_prefix}/analysis_tensors/vc.pt",
        expected_sha256=capture["vc_sha256"],
    )
    result = {
        "schema_version": 1,
        "kind": "issue952_china_definitive_gpu",
        "version": 1,
        "issue": ISSUE,
        "status": "done",
        "note": "GPU generation, capture, and exact-revision uploads verified",
        "generation": generation,
        "capture": capture,
        "raw_upload": raw_upload,
        "capture_upload": capture_upload,
        "hf_prefix": _study_config(study)["hf_prefix"],
        "timestamp_unix": time.time(),
    }
    if study == "repaired-v2":
        result.update(study=study, bank_contract=REPAIRED_CONTRACT)
    if generation["regime"]["smoke"]:
        consumer_probe = _smoke_map_consumer_probe(out_root)
        if (
            generation.get("timing_evidence_complete") is not True
            or capture.get("timing_evidence_complete") is not True
        ):
            raise RuntimeError("smoke finalization requires complete cumulative timing evidence")
        scale = generation["accepted_bank"]["n_accepted_prompts"] / generation["n_prompts"]
        projected_gpu_s = scale * (generation["elapsed_s"] + capture["elapsed_s"])
        projected_upper_s = 1.25 * projected_gpu_s + 0.3 * 3600
        smoke_timing = {
            "passed": True,
            "attempt": attempt,
            "projected_timing_passed": projected_upper_s <= 6 * 3600
            and generation["p90_request_latency_s"] is not None,
            "continuation_allowed": True,
            "bank_sha256": generation["regime"]["bank_sha256"],
            "accepted_source_ids_sha256": generation["regime"]["accepted_source_ids_sha256"],
            "n_accepted_prompts": generation["regime"]["n_accepted_prompts"],
            "n_expected_production_rollouts": generation["regime"]["n_expected_rollouts"],
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
    pending_path = out_root / "issue952_china_definitive_done.pending.json"
    pending_path.unlink(missing_ok=True)
    _write_json(pending_path, result)
    if generation["regime"]["smoke"]:
        smoke_manifest_upload = _upload_folder(
            out_root / "manifests",
            f"{target_prefix}/manifests",
            "Issue 952: bilingual China smoke timing manifest",
        )
        _verify_remote_file(
            out_root,
            revision=smoke_manifest_upload["revision"],
            remote_path=f"{target_prefix}/manifests/smoke_timing.json",
            expected_sha256=_sha256(out_root / "manifests" / "smoke_timing.json"),
        )
    info = hub.retry_transient(
        lambda: HfApi().upload_file(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_or_fileobj=str(pending_path),
            path_in_repo=f"{target_prefix}/issue952_china_definitive_done.json",
            commit_message="Issue 952: bilingual China GPU terminal sentinel",
        ),
        what="issue952 GPU terminal sentinel upload",
    )
    revision = getattr(info, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("GPU terminal sentinel upload did not return an immutable revision")
    _verify_remote_file(
        out_root,
        revision=revision,
        remote_path=f"{target_prefix}/issue952_china_definitive_done.json",
        expected_sha256=_sha256(pending_path),
    )
    os.replace(pending_path, sentinel_path)
    print("[finalize] terminal sentinel written")
    return result


def build_argparser() -> argparse.ArgumentParser:
    """Expose all legacy phases with an explicit opt-in for the repaired study."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=("gen", "extend", "upload-raw", "capture", "upload-capture", "finalize"),
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--study", choices=("v1", "repaired-v2"), default="v1")
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard-prompts", type=int, default=90)
    ap.add_argument("--capture-batch", type=int, default=4)
    ap.add_argument("--answer-shard-rows", type=int, default=720)
    ap.add_argument("--smoke-report", type=Path)
    ap.add_argument("--attempt", type=int, default=1)
    return ap


def main() -> int:
    """Dispatch each isolated process phase under the same explicit study identity."""
    args = build_argparser().parse_args()
    if args.attempt < 1:
        raise SystemExit("--attempt must be >= 1")
    args.out_root.mkdir(parents=True, exist_ok=True)
    bank, audit = stage_inputs(args.out_root, args.bank, args.audit, study=args.study)
    if args.phase == "gen":
        phase_generate(
            args.out_root,
            bank,
            audit,
            args.smoke,
            args.shard_prompts,
            args.smoke_report,
            args.attempt,
            study=args.study,
        )
    elif args.phase == "extend":
        phase_extend(args.out_root, bank, audit, args.attempt, study=args.study)
    elif args.phase == "upload-raw":
        phase_upload_raw(args.out_root, args.attempt, study=args.study)
    elif args.phase == "capture":
        phase_capture(
            args.out_root,
            bank,
            audit,
            args.smoke,
            args.capture_batch,
            args.answer_shard_rows,
            args.smoke_report,
            args.attempt,
            study=args.study,
        )
    elif args.phase == "upload-capture":
        phase_upload_capture(args.out_root, args.attempt, study=args.study)
    else:
        phase_finalize(args.out_root, args.attempt, study=args.study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
