"""Pinned native-model execution for the exploratory J/R experiment.

The official Jacobian estimator is reused without its permissive prompt-skipping
or weak resume predicates. Each checkpoint binds exact token IDs and run inputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import time
import unicodedata
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch

from explore_persona_space.analysis.workspace_lenses import dense_r_rules
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance


def file_sha256(path: Path) -> str:
    """Hash file bytes with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value) -> str:
    """Hash a canonical JSON value; float-derived cache keys are not used."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def save_json(path: Path, value) -> None:
    """Atomically replace a JSON checkpoint; reject nonfinite JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensors(path: Path, value) -> None:
    """Atomically replace a tensor checkpoint; the caller verifies its contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_identity(config_path: Path, selection_path: Path, role: str) -> dict:
    """Persist code, exact input hashes and installed runtime versions."""
    return {
        "config_sha256": file_sha256(config_path),
        "selection_sha256": file_sha256(selection_path),
        "model_role": role,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface-hub")
        },
        "code": as_metadata_dict(git_provenance(), phase="workspace-jr"),
    }


def load_workspace_jr_config(config_path: Path) -> dict:
    """Load the registered YAML as a plain resolved dictionary."""
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict) or config.get("schema_version") != "workspace-jr-v1":
        raise ValueError(f"Unsupported workspace J/R config: {config_path}")
    return config


def selected_prompts(selection_path: Path, audit_path: Path, subset: str) -> list[dict]:
    """Resolve frozen IDs from byte-verified source files, downloading when absent."""
    from huggingface_hub import hf_hub_download

    selection = json.loads(selection_path.read_text())
    audit = json.loads(audit_path.read_text())
    selected = selection["subsets"][subset]
    if not selected:
        raise ValueError(f"Empty frozen subset: {subset}")
    sources = {}
    for split in {row["source_split"] for row in selected}:
        record = selection["source_records"][split]
        source = audit["manifest_sources"]["splits"][record["filename"]]
        path = Path(source["path"])
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    repo_id="superkaiba1/explore-persona-space-data",
                    repo_type="dataset",
                    revision="815ff6d976c686af8672b27cfdfb1ce6b419c02c",
                    filename=f"issue1491_scale_ladder/manifest/{record['filename']}",
                )
            )
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Source bytes differ: {path}")
        sources[split] = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    result = []
    for row in selected:
        source = sources[row["source_split"]][row["source_row_index"]]
        prompt = source["prompt"]
        digest = hashlib.sha256(unicodedata.normalize("NFC", prompt).encode()).hexdigest()
        if digest != row["prompt_sha256"] or source["ladder_local_id"] != row["ladder_local_id"]:
            raise ValueError("Frozen selection identity mismatch")
        result.append({**row, "prompt": prompt})
    return result


def _safe_distribution_versions(names: Sequence[str]) -> dict[str, str]:
    """Return installed distribution versions without making import failures fatal."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _cuda_report() -> dict:
    """Report CUDA availability and memory without requiring nvidia-smi."""
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free, total = torch.cuda.mem_get_info()
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_gib": total / 2**30,
                    "free_gib": free / 2**30,
                }
            )
    return {"available": available, "device_count": len(devices), "devices": devices}


def _disk_report(paths: Sequence[Path]) -> dict[str, dict[str, float]]:
    """Report free space for the unique existing parents of relevant paths."""
    report = {}
    for path in paths:
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        report[str(probe)] = {
            "total_gib": usage.total / 2**30,
            "used_gib": usage.used / 2**30,
            "free_gib": usage.free / 2**30,
        }
    return report


def native_loader_status() -> dict:
    """Inspect whether this Python stack can name/load the native Qwen3.5 wrapper."""
    status = {
        "transformers_qwen3_5_module": importlib.util.find_spec("transformers.models.qwen3_5")
        is not None,
        "qwen3_5_class_exported": False,
        "auto_causal_lm_available": False,
        "auto_image_text_to_text_available": False,
        "selected_loader": None,
        "warnings": [],
    }
    try:
        from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

        status["qwen3_5_class_exported"] = True
        status["selected_loader"] = "Qwen3_5ForConditionalGeneration"
    except ImportError:
        status["warnings"].append("native_qwen3_5_class_not_exported_by_installed_transformers")
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401

            status["auto_causal_lm_available"] = True
        except ImportError:
            pass
        try:
            from transformers import AutoModelForImageTextToText  # noqa: F401

            status["auto_image_text_to_text_available"] = True
        except ImportError:
            pass
        status["warnings"].append("native_qwen3_5_unavailable_upgrade_runtime_before_loading")
    return status


def preflight_status(
    config_path: Path,
    selection_path: Path,
    audit_path: Path,
    *,
    role: str,
    output_path: Path,
    minimum_local_free_gib: float = 80.0,
) -> dict:
    """Record outcome-independent local readiness and explicit launch blockers."""
    config = load_workspace_jr_config(config_path)
    if role not in config["models"]:
        raise ValueError(f"Unknown model role {role!r}; available={sorted(config['models'])}")
    prompts = selected_prompts(selection_path, audit_path, "calibration")
    cuda = _cuda_report()
    disk = _disk_report([Path.cwd(), output_path, config_path, selection_path, audit_path])
    loader = native_loader_status()
    blockers = []
    if not cuda["available"]:
        blockers.append("no_cuda_device_visible_to_torch")
    if min(record["free_gib"] for record in disk.values()) < minimum_local_free_gib:
        blockers.append(f"local_free_disk_below_{minimum_local_free_gib:g}GiB")
    if loader["selected_loader"] is None:
        blockers.append("no_importable_huggingface_loader_for_selected_checkpoint")
    status = {
        "schema_version": "workspace-jr-runtime-preflight-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": run_identity(config_path, selection_path, role),
        "workflow_status": config.get("status"),
        "workflow_exception": config.get("workflow_exception"),
        "role": role,
        "model_id": config["selection"][role],
        "model_spec": config["models"][role],
        "calibration_contexts_resolved": len(prompts),
        "runtime_versions": _safe_distribution_versions(
            ("torch", "transformers", "huggingface-hub", "omegaconf", "numpy")
        ),
        "native_loader": loader,
        "cuda": cuda,
        "disk": disk,
        "minimum_local_free_gib": minimum_local_free_gib,
        "local_real_model_execution_ready": not blockers,
        "blockers": blockers,
        "warnings": loader["warnings"],
        "does_not_read_experimental_outcomes": True,
        "experimental_component_results": None,
    }
    save_json(output_path, status)
    return status


def _resolve_text_decoder(model):
    """Find the native text decoder under Qwen3.5 multimodal/text wrappers."""
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
    ]
    for candidate in candidates:
        if (
            candidate is not None
            and hasattr(candidate, "layers")
            and hasattr(candidate, "embed_tokens")
            and hasattr(candidate, "norm")
        ):
            return candidate
    raise ValueError("Could not resolve Qwen text decoder with layers/embed_tokens/norm")


def load_native(config: dict, role: str, *, device: str, dtype: torch.dtype):
    """Load the exact dense Qwen3.5 checkpoint with its native text architecture."""
    from transformers import AutoTokenizer
    from transformers import Qwen3_5ForConditionalGeneration as model_cls

    name, spec = config["selection"][role], config["models"][role]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=spec["revision"])
    model = (
        model_cls.from_pretrained(
            name,
            revision=spec["revision"],
            dtype=dtype,
            device_map={"": device},
            attn_implementation="eager",
            trust_remote_code=True,
        )
        .eval()
        .requires_grad_(False)
    )
    text = _resolve_text_decoder(model)
    if len(text.layers) != spec["n_layers"] or text.config.hidden_size != spec["d_model"]:
        raise ValueError("Loaded checkpoint geometry differs from frozen selection")
    if text.config.model_type != "qwen3_5_text":
        raise ValueError(f"Unexpected architecture: {text.config.model_type}")
    return model, tokenizer, text


class ExactTokenLensModel:
    """Official estimator adapter whose encode operation consumes frozen token IDs."""

    def __init__(self, text_decoder, tokenizer, token_rows: dict[str, list[int]]):
        self._text = text_decoder
        self.layers = self._text.layers
        self.n_layers = len(self.layers)
        self.d_model = self._text.config.hidden_size
        self.tokenizer = tokenizer
        self.token_rows = token_rows
        self.device = self._text.embed_tokens.weight.device

    def encode(self, prompt: str, *, max_length: int):
        ids = self.token_rows[prompt]
        if len(ids) > max_length:
            raise ValueError("Calibration tokens exceed frozen maximum")
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def forward(self, input_ids):
        return self._text(input_ids=input_ids, use_cache=False)


def calibration_tokens(tokenizer, prompts: list[dict], config: dict) -> dict:
    """Freeze raw-corpus tokens identically for both lenses, with visible short rows."""
    lens = config["lenses"]
    rows, excluded = [], []
    for row in prompts:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=True)[
            : lens["calibration_max_tokens"]
        ]
        record = {"prompt_sha256": row["prompt_sha256"], "token_ids": ids}
        if len(ids) <= lens["skip_first"] + 1:
            excluded.append({**record, "reason": "no_valid_calibration_positions"})
        else:
            rows.append(record)
    if not rows:
        raise ValueError("No valid calibration contexts")
    return {"rows": rows, "excluded": excluded, "special_tokens": "native_raw_encode_true"}


def validate_token_manifest(
    manifest: dict,
    *,
    config_path: Path,
    selection_path: Path,
    config: dict,
    role: str,
) -> dict:
    """Reject token manifests that do not match the requested model role/run inputs."""
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Token manifest is missing its run identity block")
    expected = {
        "identity.model_role": role,
        "identity.config_sha256": file_sha256(config_path),
        "identity.selection_sha256": file_sha256(selection_path),
        "model_id": config["selection"][role],
        "model_revision": config["models"][role]["revision"],
    }
    observed = {
        "identity.model_role": identity.get("model_role"),
        "identity.config_sha256": identity.get("config_sha256"),
        "identity.selection_sha256": identity.get("selection_sha256"),
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
    }
    mismatches = [
        {"field": field, "expected": value, "observed": observed[field]}
        for field, value in expected.items()
        if observed[field] != value
    ]
    if mismatches:
        raise ValueError(f"Token manifest identity mismatch: {mismatches}")
    if not isinstance(manifest.get("rows"), list) or not isinstance(manifest.get("excluded"), list):
        raise ValueError("Token manifest must contain rows and excluded lists")
    if manifest.get("subset") != "calibration":
        raise ValueError("Lenses may only use the frozen calibration subset")
    frozen = json.loads(selection_path.read_text())["subsets"]["calibration"]
    expected_hashes = {row["prompt_sha256"] for row in frozen}
    observed_hashes = [row["prompt_sha256"] for row in manifest["rows"] + manifest["excluded"]]
    if len(set(observed_hashes)) != len(observed_hashes) or set(observed_hashes) != expected_hashes:
        raise ValueError(
            "Calibration token manifest has missing, duplicate or non-calibration rows"
        )
    return {
        "status": "ok",
        "role": role,
        "model_id": expected["model_id"],
        "model_revision": expected["model_revision"],
        "n_rows": len(manifest["rows"]),
        "n_excluded": len(manifest["excluded"]),
    }


def native_forward_validation(model, text, token_ids: list[int], spec: dict) -> dict:
    """Check native forward identity and finite ordinary/modified activation VJPs."""
    ids = torch.tensor([token_ids], device=text.embed_tokens.weight.device)
    source, target = spec["source_layer"], spec["target_layer"]

    def execute(modified):
        activations, handles = {}, []

        def capture_source(_module, _inputs, output):
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            tensor.requires_grad_(True)
            activations["source"] = tensor

        def capture_target(_module, _inputs, output):
            activations["target"] = output if isinstance(output, torch.Tensor) else output[0]

        scope = dense_r_rules(text.layers, final_norm=text.norm) if modified else nullcontext([])
        try:
            handles.append(text.layers[source].register_forward_hook(capture_source))
            handles.append(text.layers[target].register_forward_hook(capture_target))
            with scope as patched:
                output = text(input_ids=ids, use_cache=False).last_hidden_state
                generator = torch.Generator(device=ids.device).manual_seed(20260912)
                cotangent = torch.randn(
                    activations["target"].shape,
                    generator=generator,
                    device=ids.device,
                    dtype=torch.float32,
                ).to(output.dtype)
                gradient = torch.autograd.grad(
                    activations["target"], activations["source"], cotangent
                )[0]
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Native activation backward returned nonfinite values")
                return output.detach(), activations["target"].detach(), gradient.detach(), patched
        finally:
            for handle in handles:
                handle.remove()

    ordinary = execute(False)
    modified = execute(True)
    torch.testing.assert_close(ordinary[0], modified[0], rtol=0, atol=0)
    torch.testing.assert_close(ordinary[1], modified[1], rtol=0, atol=0)
    numerical = native_j_numerical_validation(text, ids, spec)
    return {
        "forward_bit_identical": True,
        "hook_bit_identical": True,
        "ordinary_vjp_finite": True,
        "modified_vjp_finite": True,
        "ordinary_gradient_norm": float(ordinary[2].float().norm()),
        "modified_gradient_norm": float(modified[2].float().norm()),
        "patched_paths": modified[3],
        "ordinary_numerical_validation": numerical,
        "model_class": type(model).__name__,
    }


def native_j_numerical_validation(text, ids: torch.Tensor, spec: dict) -> dict:
    """Check native ordinary VJPs against source-hook finite differences.

    Only the suffix after the measured source is cast to fp32, so the 27B
    checkpoint fits one 80GB GPU. Prefix states and positional constants stay
    native. Three independent cotangents select their gradient directions to
    avoid a near-zero scalar derivative. Two successive finite-difference
    scales must pass. R coefficients are never compared with derivatives.
    """
    source, target = spec["source_layer"], spec["target_layer"]
    suffix = [*text.layers[source + 1 :], text.norm]
    old_dtypes = [next(module.parameters()).dtype for module in suffix]
    state, handles = {}, []

    def inject(_module, _inputs, output):
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        value = tensor.detach().float()
        if "delta" in state:
            value = value + state["delta"]
        value.requires_grad_(True)
        state["source"] = value
        return value if isinstance(output, torch.Tensor) else (value, *output[1:])

    def capture(_module, _inputs, output):
        state["target"] = output if isinstance(output, torch.Tensor) else output[0]

    records = []
    try:
        for module in suffix:
            module.float()
        handles.append(text.layers[source].register_forward_hook(inject))
        handles.append(text.layers[target].register_forward_hook(capture))
        for seed in (20260912, 20260913, 20260914):
            state.clear()
            text(input_ids=ids, use_cache=False)
            generator = torch.Generator(device=ids.device).manual_seed(seed)
            cotangent = torch.randn(state["target"].shape, generator=generator, device=ids.device)
            cotangent /= cotangent.norm()
            gradient = torch.autograd.grad(state["target"], state["source"], cotangent)[0].detach()
            if not torch.isfinite(gradient).all() or gradient.norm() <= 1e-10:
                raise ValueError("Degenerate or nonfinite native numerical J test")
            direction = gradient / gradient.norm() * state["source"].detach().norm()
            analytic = float((gradient.double() * direction.double()).sum())
            checks = []
            for epsilon in (0.01, 0.003, 0.001):
                values = []
                for sign in (1, -1):
                    state["delta"] = sign * epsilon * direction
                    with torch.no_grad():
                        text(input_ids=ids, use_cache=False)
                        values.append(float((state["target"].double() * cotangent.double()).sum()))
                finite_difference = (values[0] - values[1]) / (2 * epsilon)
                relative_error = abs(finite_difference - analytic) / abs(analytic)
                checks.append(
                    {
                        "epsilon": epsilon,
                        "analytic": analytic,
                        "finite_difference": finite_difference,
                        "relative_error": relative_error,
                    }
                )
            if not any(
                checks[i]["relative_error"] < 0.02 and checks[i + 1]["relative_error"] < 0.02
                for i in range(len(checks) - 1)
            ):
                raise ValueError(f"Native ordinary-J finite differences failed: {checks}")
            records.append({"seed": seed, "checks": checks})
    finally:
        for handle in handles:
            handle.remove()
        for module, dtype in zip(suffix, old_dtypes, strict=True):
            module.to(dtype=dtype)
    return {
        "status": "passed",
        "precision": "fp32_suffix_native_prefix_and_positions",
        "relative_tolerance": 0.02,
        "records": records,
        "limitation": "controls numerical derivatives in fp32; not bf16 coefficient equality",
    }


def fit_lens_prompts(
    model,
    tokenizer,
    config: dict,
    role: str,
    token_manifest: dict,
    output: Path,
    identity: dict,
    *,
    dim_batch: int,
    start: int = 0,
    stop: int | None = None,
) -> None:
    """Checkpoint one matched J/R pair per prompt using the official estimator."""
    vendor = Path(__file__).resolve().parents[3] / "external/jacobian-lens"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from jlens.fitting import jacobian_for_prompt

    rows = token_manifest["rows"]
    end = len(rows) if stop is None else stop
    if not 0 <= start < end <= len(rows):
        raise ValueError("Invalid calibration shard interval")
    token_rows = {row["prompt_sha256"]: row["token_ids"] for row in rows}
    spec, lens = config["models"][role], config["lenses"]
    adapter = ExactTokenLensModel(_resolve_text_decoder(model), tokenizer, token_rows)
    contract = {
        **identity,
        "tokens_sha256": content_sha256(token_manifest),
        "dim_batch": dim_batch,
        "actual_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": str(adapter._text.config._attn_implementation),
        "model_class": type(model).__name__,
    }
    contract_digest = content_sha256(contract)
    for index in range(start, end):
        row = rows[index]
        path = output / f"prompt-{index:04d}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved["contract_sha256"] != contract_digest
                or saved["prompt_sha256"] != row["prompt_sha256"]
            ):
                raise ValueError(f"Stale lens checkpoint: {path}")
            for arm in ("J", "R"):
                matrix = saved[arm]
                if (
                    matrix.shape != (adapter.d_model, adapter.d_model)
                    or not torch.isfinite(matrix).all()
                ):
                    raise ValueError(f"Invalid lens checkpoint: {path}, {arm}")
            print(f"lens prompt={index + 1}/{len(rows)} resume=verified", flush=True)
            continue
        saved = {
            "contract_sha256": contract_digest,
            "contract": contract,
            "prompt_sha256": row["prompt_sha256"],
            "token_ids": row["token_ids"],
        }
        for arm in ("J", "R"):
            began = time.monotonic()
            scope = dense_r_rules(adapter.layers) if arm == "R" else nullcontext()
            with scope:
                matrices, seq_len, n_valid = jacobian_for_prompt(
                    adapter,
                    row["prompt_sha256"],
                    [spec["source_layer"]],
                    target_layer=spec["target_layer"],
                    dim_batch=dim_batch,
                    max_seq_len=lens["calibration_max_tokens"],
                    skip_first=lens["skip_first"],
                )
            matrix = matrices[spec["source_layer"]]
            if not torch.isfinite(matrix).all():
                raise FloatingPointError(f"Nonfinite {arm} lens at calibration index {index}")
            saved[arm] = matrix
            saved[f"{arm}_seconds"] = time.monotonic() - began
            saved["sequence_length"], saved["valid_positions"] = seq_len, n_valid
            print(
                f"lens prompt={index + 1}/{len(rows)} arm={arm} "
                f"seconds={saved[f'{arm}_seconds']:.2f}",
                flush=True,
            )
        save_tensors(path, saved)
        print(f"lens prompt={index + 1}/{len(rows)} saved={path}", flush=True)
