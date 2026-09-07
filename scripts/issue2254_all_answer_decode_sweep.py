#!/usr/bin/env python3
"""#2254 matched context-only versus pure decode-only answer steering.

GPU phases generate a lower-dose layer-14 ``r_B`` sweep with
``DeltaHook(decode_only=True)``, same-runtime reverse-map context comparators,
and no-hook bridges for the zero-dose cells.  Every retained generation is
persisted; no 4096-token regeneration is allowed because the 2048-token cap is
part of the matched response-integrity outcome (plan v15).

This module performs generation and raw-artifact verification only.  The
companion analysis driver owns blinded Codex judging and statistical reduction.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import resource
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

import scripts.issue2254_first_k_steering as fk  # noqa: E402
import scripts.issue2254_preimage as i2254  # noqa: E402
from explore_persona_space.experiments.issue1415 import steering  # noqa: E402
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    HF_DATA_REPO,
    HIDDEN_DIM,
    MODEL_NAME,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2254.all_answer_decode")

FOLLOWUP_LABEL = "all_answer_decode_sweep"
OUT_REL = Path("eval_results/issue_2254") / FOLLOWUP_LABEL
HF_PREFIX = f"{i2254.HF_PREFIX}/{FOLLOWUP_LABEL}"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BEHAVIORS = ("evil", "sycophancy")
LAYER = 14
RHO_FILE_REL = Path("eval_results/issue_2254/norm_probe/rho_by_layer.json")
RHO_FILE_SHA256 = "ac7a31f19016c7272654cef1fc8a700fa23678ff10c9828218a126a85d103d56"
RHO_L14 = 63.24640552926044
DOSES = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0)
CONTEXT_DOSE = {"evil": 4.0, "sycophancy": 2.0}
N_QUESTIONS = 20
N_DRAWS = 6
SEED_BASE = 42
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
EXPECTED_EOS_TOKEN_IDS = (151645, 151643)
EXPECTED_PAD_TOKEN_ID = 151643
CJK_PATTERN = r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]"
PILOT_CELL_ID = "evil__rb__decodeonly__L14__c4"
PLANNED_GPU_COUNT = 4
MAX_FLEET_WALL_SECONDS = 90 * 60
PILOT_WALL_SAFETY_FACTOR = 2.0
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_MEMORY_FRACTION = 0.90
EXPECTED_DIRECTION_SHA256 = {
    "revmap": {
        "evil": "fe9fb192bca8b908025c3e080088e679cebfa1505c0573da7b6e99cb1c5c8173",
        "sycophancy": "f1ac96523a73e7013ab809858f45bb11723bef85575d2568728b573566856a71",
    },
    "rb_bf16": {
        "evil": "4848093616721216f7cbf47f51ae11d28a623b42745387408ea546232da3d4e4",
        "sycophancy": "7856cdf61a448b3d1037cd0c44fcb9137a43b82cf21c424ddb9c3a98898db8e0",
    },
}
SENTINEL_DIR = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))


class DecodeSweepError(RuntimeError):
    """Fail-loud generation, provenance, or artifact error."""


@dataclass(frozen=True)
class CellSpec:
    behavior: str
    route: str
    dose: float

    @property
    def direction(self) -> str:
        if self.route == "decodeonly":
            return "rb"
        if self.route == "context":
            return "revmap"
        if self.route == "nohook":
            return "none"
        raise DecodeSweepError(f"unknown route {self.route!r}")

    @property
    def cell_id(self) -> str:
        return f"{self.behavior}__{self.direction}__{self.route}__L{LAYER}__{dose_token(self.dose)}"


def dose_token(c: float) -> str:
    """Stable exact dose token, including fractional lower-dose rungs."""
    lookup = {
        0.0: "c0",
        1 / 64: "c1over64",
        1 / 32: "c1over32",
        1 / 16: "c1over16",
        1 / 8: "c1over8",
        1 / 4: "c1over4",
        1 / 2: "c1over2",
        1.0: "c1",
        2.0: "c2",
        4.0: "c4",
    }
    try:
        return lookup[float(c)]
    except KeyError as exc:
        raise DecodeSweepError(f"unregistered dose {c!r}") from exc


def registered_cells() -> list[CellSpec]:
    cells = [CellSpec(b, "decodeonly", c) for b in BEHAVIORS for c in DOSES]
    cells += [CellSpec(b, "context", CONTEXT_DOSE[b]) for b in BEHAVIORS]
    cells += [CellSpec(b, "nohook", 0.0) for b in BEHAVIORS]
    if len(cells) != 24 or len({cell.cell_id for cell in cells}) != 24:
        raise DecodeSweepError("registered generation family is not 24 unique cells")
    return cells


def analysis_cells() -> list[CellSpec]:
    cells = [cell for cell in registered_cells() if cell.route != "nohook"]
    if len(cells) != 22:
        raise DecodeSweepError("analysis family is not 22 cells")
    return cells


def round_root(out_root: Path | str) -> Path:
    return Path(out_root) / OUT_REL


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(raw.encode("utf-8"))


def _token_id_tuple(value: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else [value]
    return tuple(int(item) for item in values)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return _sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = proc.stdout.strip()
    if proc.returncode or len(commit) != 40:
        raise DecodeSweepError(f"could not resolve Git commit: {proc.stderr.strip()}")
    return commit


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _write_sentinel(name: str, payload: dict) -> None:
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            SENTINEL_DIR / f"issue-2254-{name}.json",
            {"issue": 2254, "phase": name, "status": "done", "ts": time.time(), **payload},
        )
    except OSError as exc:
        logger.info("[sentinel] unavailable: %s", exc)


def _stage_inputs(out_root: Path | str) -> dict:
    """Load and hash every generation input without loading the base model."""
    from transformers import GenerationConfig

    root = round_root(out_root)
    i2254._stage_e1_assets()
    rho_path = _REPO_ROOT / RHO_FILE_REL
    if not rho_path.is_file():
        i2254._ensure_git_input(str(RHO_FILE_REL), str(RHO_FILE_REL.parent))
    if _sha256_file(rho_path) != RHO_FILE_SHA256:
        raise DecodeSweepError("rho_by_layer.json differs from the registered SHA-256")
    rho_payload = json.loads(rho_path.read_text(encoding="utf-8"))
    rho = float(rho_payload["rho_pooled_median"][f"L{LAYER}"])
    if rho != RHO_L14:
        raise DecodeSweepError(f"rho L14 changed: {rho} != {RHO_L14}")

    rb_all = i2254._load_rb_all()
    directions: dict[str, dict[str, torch.Tensor]] = {"rb": {}, "revmap": {}}
    direction_rows = []
    for behavior in BEHAVIORS:
        rb = torch.as_tensor(rb_all[behavior][LAYER], dtype=torch.float32)
        rb = (rb / rb.norm()).to(torch.bfloat16)
        if rb.shape != (HIDDEN_DIM,) or not torch.isfinite(rb).all():
            raise DecodeSweepError(f"invalid r_B direction for {behavior}")
        rb_sha = _tensor_sha256(rb)
        if rb_sha != EXPECTED_DIRECTION_SHA256["rb_bf16"][behavior]:
            raise DecodeSweepError(
                f"{behavior} r_B BF16 bytes changed: {rb_sha} != "
                f"{EXPECTED_DIRECTION_SHA256['rb_bf16'][behavior]}"
            )
        directions["rb"][behavior] = rb

        rev_path = _REPO_ROOT / OUT_REL.parent / "directions" / f"{behavior}_revmap_L14.pt"
        if not rev_path.is_file():
            i2254._ensure_direction_vec(_REPO_ROOT / OUT_REL.parent, behavior, "revmap", LAYER)
        rev_file_sha = _sha256_file(rev_path)
        if rev_file_sha != EXPECTED_DIRECTION_SHA256["revmap"][behavior]:
            raise DecodeSweepError(f"{behavior} reverse-map direction file changed")
        payload = torch.load(rev_path, map_location="cpu", weights_only=True)
        rev = payload["direction"].float()
        rev = (rev / rev.norm()).to(torch.bfloat16)
        if rev.shape != (HIDDEN_DIM,) or not torch.isfinite(rev).all():
            raise DecodeSweepError(f"invalid reverse-map direction for {behavior}")
        directions["revmap"][behavior] = rev
        direction_rows.append(
            {
                "behavior": behavior,
                "rb_bf16_sha256": rb_sha,
                "revmap_file": str(rev_path.relative_to(_REPO_ROOT)),
                "revmap_file_sha256": rev_file_sha,
                "revmap_bf16_sha256": _tensor_sha256(rev),
                "rb_norm_fp32_after_bf16": float(rb.float().norm()),
                "revmap_norm_fp32_after_bf16": float(rev.float().norm()),
            }
        )

    question_rows = []
    for behavior in BEHAVIORS:
        questions = i2254._eval_questions(behavior)[:N_QUESTIONS]
        if len(questions) != N_QUESTIONS or len(set(questions)) != N_QUESTIONS:
            raise DecodeSweepError(f"{behavior}: eval bank is not 20 unique questions")
        question_rows.append(
            {
                "behavior": behavior,
                "n_questions": len(questions),
                "compact_json_sha256": _canonical_sha256(questions),
            }
        )

    generation_config = GenerationConfig.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    eos_token_ids = _token_id_tuple(generation_config.eos_token_id)
    if eos_token_ids != EXPECTED_EOS_TOKEN_IDS:
        raise DecodeSweepError(
            f"pinned generation EOS IDs changed: {eos_token_ids} != {EXPECTED_EOS_TOKEN_IDS}"
        )
    if int(generation_config.pad_token_id) != EXPECTED_PAD_TOKEN_ID:
        raise DecodeSweepError(
            f"pinned generation pad ID changed: {generation_config.pad_token_id} != "
            f"{EXPECTED_PAD_TOKEN_ID}"
        )

    manifest = {
        "experiment": FOLLOWUP_LABEL,
        "plan_version": 15,
        "git_commit": _git_commit(),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "pass_b_revision": i2254.HF_REV,
        "rho_path": str(RHO_FILE_REL),
        "rho_sha256": RHO_FILE_SHA256,
        "rho_pooled_median_L14": rho,
        "directions": direction_rows,
        "questions": question_rows,
        "generation": {
            "n_questions": N_QUESTIONS,
            "n_draws": N_DRAWS,
            "seed_base": SEED_BASE,
            "effective_seeds": list(range(SEED_BASE, SEED_BASE + N_DRAWS)),
            "temperature": TEMPERATURE,
            "top_p": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "cap_regeneration": False,
            "eos_token_ids": list(eos_token_ids),
            "pad_token_id": int(generation_config.pad_token_id),
        },
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in registered_cells()],
    }
    _atomic_json(root / "inputs_manifest.json", manifest)
    logger.info("[phase=stage_inputs] staged %d registered cells", len(registered_cells()))
    _write_sentinel("decode-stage-inputs", {"cells": len(registered_cells())})
    return manifest


def _load_directions(manifest: dict) -> dict[str, dict[str, torch.Tensor]]:
    """Reload directions only when bytes match the frozen staged manifest."""
    rb_all = i2254._load_rb_all()
    output: dict[str, dict[str, torch.Tensor]] = {"rb": {}, "revmap": {}}
    rows = {row["behavior"]: row for row in manifest.get("directions", [])}
    if set(rows) != set(BEHAVIORS):
        raise DecodeSweepError("inputs manifest does not contain both direction rows")
    for behavior in BEHAVIORS:
        rb = torch.as_tensor(rb_all[behavior][LAYER], dtype=torch.float32)
        rb = (rb / rb.norm()).to(torch.bfloat16)
        rb_sha = _tensor_sha256(rb)
        if rb.shape != (HIDDEN_DIM,) or not torch.isfinite(rb).all():
            raise DecodeSweepError(f"invalid r_B direction for {behavior}")
        if rb_sha != EXPECTED_DIRECTION_SHA256["rb_bf16"][behavior] or rb_sha != rows[behavior].get(
            "rb_bf16_sha256"
        ):
            raise DecodeSweepError(f"{behavior} r_B differs from the frozen manifest")

        expected_rel = str(OUT_REL.parent / "directions" / f"{behavior}_revmap_L14.pt")
        if rows[behavior].get("revmap_file") != expected_rel:
            raise DecodeSweepError(f"{behavior} reverse-map manifest path changed")
        rev_path = _REPO_ROOT / expected_rel
        if not rev_path.is_file():
            raise DecodeSweepError(f"staged reverse-map direction is missing: {rev_path}")
        rev_file_sha = _sha256_file(rev_path)
        if rev_file_sha != EXPECTED_DIRECTION_SHA256["revmap"][behavior] or rev_file_sha != rows[
            behavior
        ].get("revmap_file_sha256"):
            raise DecodeSweepError(f"{behavior} reverse-map file differs from frozen manifest")
        payload = torch.load(rev_path, map_location="cpu", weights_only=True)
        rev = payload["direction"].float()
        rev = (rev / rev.norm()).to(torch.bfloat16)
        rev_sha = _tensor_sha256(rev)
        if rev.shape != (HIDDEN_DIM,) or not torch.isfinite(rev).all():
            raise DecodeSweepError(f"invalid reverse-map direction for {behavior}")
        if rev_sha != rows[behavior].get("revmap_bf16_sha256"):
            raise DecodeSweepError(f"{behavior} reverse-map tensor differs from frozen manifest")
        output["rb"][behavior] = rb
        output["revmap"][behavior] = rev
    return output


def _load_model_and_tokenizer(manifest: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise DecodeSweepError("generation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    if not bool(getattr(model.config, "use_cache", False)):
        raise DecodeSweepError("pinned model config does not enable KV caching")
    eos_ids = _token_id_tuple(model.generation_config.eos_token_id)
    expected_eos = tuple(manifest["generation"]["eos_token_ids"])
    if eos_ids != expected_eos or eos_ids != EXPECTED_EOS_TOKEN_IDS:
        raise DecodeSweepError(
            f"loaded generation EOS IDs differ from frozen inputs: {eos_ids} != {expected_eos}"
        )
    pad_id = int(model.generation_config.pad_token_id)
    if pad_id != manifest["generation"]["pad_token_id"] or pad_id != EXPECTED_PAD_TOKEN_ID:
        raise DecodeSweepError("loaded generation pad ID differs from frozen inputs")
    if tokenizer.pad_token_id != pad_id:
        raise DecodeSweepError("tokenizer and generation-config pad IDs differ")
    return model, tokenizer


def _trim_generated_ids(
    raw_ids: list[int], eos_token_ids: int | list[int] | tuple[int, ...]
) -> tuple[list[int], str]:
    eos_ids = set(_token_id_tuple(eos_token_ids))
    if not eos_ids:
        raise DecodeSweepError("generation EOS set is empty")
    for index, token_id in enumerate(raw_ids):
        if token_id in eos_ids:
            return raw_ids[: index + 1], "eos_token"
    if len(raw_ids) != MAX_NEW_TOKENS:
        raise DecodeSweepError(f"non-EOS generation has {len(raw_ids)} != cap")
    return raw_ids, "length"


def _repetition_diagnostic(token_ids: list[int]) -> dict:
    """Flag gross loops when an exact eight-token n-gram occurs at least five times."""
    if len(token_ids) < 8:
        maximum = 0
    else:
        counts = collections.Counter(
            tuple(token_ids[index : index + 8]) for index in range(len(token_ids) - 7)
        )
        maximum = max(counts.values(), default=0)
    return {"max_exact_8gram_count": int(maximum), "flag": bool(maximum >= 5)}


def _resource_snapshot(path: Path) -> dict:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    host_total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    # Linux reports ru_maxrss in KiB.
    host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    disk = shutil.disk_usage(path)
    return {
        "cuda_device": int(device),
        "cuda_name": properties.name,
        "hbm_total_bytes": int(properties.total_memory),
        "hbm_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "hbm_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host_total_bytes": host_total,
        "host_max_rss_bytes": host_rss,
        "disk_free_bytes": int(disk.free),
    }


def _evaluate_pilot_gate(record: dict) -> dict:
    """Evaluate the production-shape high-dose timing and resource gate."""
    if record.get("cell_id") != PILOT_CELL_ID:
        raise DecodeSweepError(f"pilot gate requires {PILOT_CELL_ID}")
    resources = record.get("resources")
    if not isinstance(resources, dict):
        raise DecodeSweepError("pilot record has no resource measurements")
    completed = sum(len(row) for row in record["seeds"][str(SEED_BASE)]["completions"])
    wall = float(record["wall_seconds"])
    pilot_completed_at = float(record["completed_at_epoch"])
    projected_wall = wall * len(registered_cells()) / PLANNED_GPU_COUNT
    projected_wall_upper = projected_wall * PILOT_WALL_SAFETY_FACTOR
    projected_gpu_hours_upper = projected_wall_upper * PLANNED_GPU_COUNT / 3600
    checks = {
        "production_shape_120_completions": completed == N_QUESTIONS * N_DRAWS,
        "projected_wall_within_1p5h": projected_wall_upper <= MAX_FLEET_WALL_SECONDS,
        "projected_gpu_hours_within_6h": projected_gpu_hours_upper <= 6.0,
        "hbm_headroom": resources["hbm_max_reserved_bytes"]
        <= MAX_MEMORY_FRACTION * resources["hbm_total_bytes"],
        "host_memory_headroom": resources["host_max_rss_bytes"]
        <= MAX_MEMORY_FRACTION * resources["host_total_bytes"],
        "disk_headroom": resources["disk_free_bytes"] >= MIN_DISK_FREE_BYTES,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pilot_cell_id": PILOT_CELL_ID,
        "pilot_record_sha256": _canonical_sha256(record),
        "pilot_wall_seconds": wall,
        "pilot_completed_at_epoch": pilot_completed_at,
        "deadline_epoch": pilot_completed_at + MAX_FLEET_WALL_SECONDS,
        "pilot_completions": completed,
        "planned_cells": len(registered_cells()),
        "planned_gpu_count": PLANNED_GPU_COUNT,
        "wall_safety_factor": PILOT_WALL_SAFETY_FACTOR,
        "projected_fleet_wall_seconds": projected_wall,
        "projected_fleet_wall_upper_seconds": projected_wall_upper,
        "projected_gpu_hours_upper": projected_gpu_hours_upper,
        "hard_fleet_wall_seconds": MAX_FLEET_WALL_SECONDS,
        "resources": resources,
        "checks": checks,
    }


def _require_production_gate(root: Path, deadline_epoch: float | None) -> dict:
    gate_path = root / "generation" / "pilot_gate_report.json"
    if not gate_path.is_file():
        raise DecodeSweepError("full generation requires the production-shape pilot gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not all(gate.get("checks", {}).values()):
        raise DecodeSweepError("production-shape pilot gate is absent or not passing")
    pilot_path = root / "generation" / "raw_completions" / f"{PILOT_CELL_ID}.json"
    if not pilot_path.is_file():
        raise DecodeSweepError("pilot gate record is missing")
    pilot_record = json.loads(pilot_path.read_text(encoding="utf-8"))
    if gate.get("pilot_record_sha256") != _canonical_sha256(pilot_record):
        raise DecodeSweepError("pilot gate is not bound to the current pilot record")
    now = time.time()
    if deadline_epoch is None:
        raise DecodeSweepError("full generation requires --deadline-epoch")
    frozen_deadline = float(gate.get("deadline_epoch", 0))
    if deadline_epoch != frozen_deadline:
        raise DecodeSweepError(
            f"shard deadline {deadline_epoch} differs from pilot-bound {frozen_deadline}"
        )
    if frozen_deadline != float(pilot_record["completed_at_epoch"]) + MAX_FLEET_WALL_SECONDS:
        raise DecodeSweepError("pilot-bound deadline is malformed")
    if frozen_deadline <= now:
        raise DecodeSweepError("generation deadline has already expired")
    return gate


def _check_deadline(deadline_epoch: float | None) -> None:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise DecodeSweepError("generation exceeded the shared 1.5-hour hard fence")


def _arm_deadline(deadline_epoch: float | None) -> None:
    if deadline_epoch is None:
        return
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        raise DecodeSweepError("generation deadline has already expired")

    def _deadline_handler(_signum, _frame):
        raise DecodeSweepError("generation hit the shared 1.5-hour hard fence")

    signal.signal(signal.SIGALRM, _deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def _disarm_deadline() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)


def _generate_batch_record(
    model,
    tokenizer,
    contexts: list[dict],
    *,
    hook: fk.RecordedHook | None,
    deadline_epoch: float | None,
) -> dict:
    """Production batched generator with raw token IDs and finish-reason audit."""
    ids = [steering.context_token_ids(tokenizer, context) for context in contexts]
    texts = [steering.render_context(tokenizer, context) for context in contexts]
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size, prompt_len = input_ids.shape
    if batch_size != N_QUESTIONS or max(map(len, ids)) != prompt_len:
        raise DecodeSweepError("batched prompt shape differs from the registered design")
    for row, item_ids in enumerate(ids):
        row_len = int(attention_mask[row].sum().item())
        if row_len != len(item_ids) or input_ids[row, prompt_len - row_len :].tolist() != item_ids:
            raise DecodeSweepError(f"context-token parity failed at row {row}")

    completions: list[list[str]] = [[] for _ in contexts]
    token_ids: list[list[list[int]]] = [[] for _ in contexts]
    finish_reasons: list[list[str]] = [[] for _ in contexts]
    token_counts: list[list[int]] = [[] for _ in contexts]
    do_sample = TEMPERATURE > 0
    eos_token_ids = _token_id_tuple(model.generation_config.eos_token_id)
    if eos_token_ids != EXPECTED_EOS_TOKEN_IDS:
        raise DecodeSweepError("generation EOS set drifted after model load")
    for draw in range(N_DRAWS):
        _check_deadline(deadline_epoch)
        torch.manual_seed(SEED_BASE + draw)
        if hook is not None:
            hook.arm(expected_prompt_len=prompt_len)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=TEMPERATURE,
            top_p=None,
            top_k=None,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=list(eos_token_ids),
            use_cache=True,
        )
        _check_deadline(deadline_epoch)
        if output.shape[0] != batch_size or output.shape[1] <= prompt_len:
            raise DecodeSweepError(f"unexpected generate output shape {tuple(output.shape)}")
        for row in range(batch_size):
            trimmed, finish_reason = _trim_generated_ids(
                [int(v) for v in output[row, prompt_len:].tolist()], eos_token_ids
            )
            text = tokenizer.decode(trimmed, skip_special_tokens=True)
            if not text.strip():
                raise DecodeSweepError(f"empty completion at row={row} draw={draw}")
            completions[row].append(text)
            token_ids[row].append(trimmed)
            finish_reasons[row].append(finish_reason)
            token_counts[row].append(len(trimmed))
    return {
        "completions": completions,
        "token_ids": token_ids,
        "finish_reasons": finish_reasons,
        "token_counts": token_counts,
        "prompt_len": prompt_len,
    }


def _hook_for_cell(model, cell: CellSpec, direction: torch.Tensor):
    alpha = float(cell.dose * RHO_L14)
    if cell.route == "decodeonly":
        return fk.build_recorded_hook(model, fk.PURE_DECODE_POSITION, [LAYER], [direction], [alpha])
    if cell.route == "context":
        return fk.build_recorded_hook(model, "lastctx", [LAYER], [direction], [alpha])
    if cell.route == "nohook":
        return None
    raise DecodeSweepError(f"unknown cell route {cell.route}")


def _regime_payload(cell: CellSpec, manifest: dict, direction_sha: str | None) -> dict:
    question_row = next(row for row in manifest["questions"] if row["behavior"] == cell.behavior)
    return {
        "cell": asdict(cell),
        "cell_id": cell.cell_id,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "direction_sha256": direction_sha,
        "rho_sha256": RHO_FILE_SHA256,
        "rho_L14": RHO_L14,
        "alpha": cell.dose * RHO_L14,
        "question_sha256": question_row["compact_json_sha256"],
        "generation": manifest["generation"],
        "git_commit": manifest["git_commit"],
        "hook_mode": cell.route,
    }


def _generate_cell(
    model,
    tokenizer,
    cell: CellSpec,
    directions: dict,
    manifest: dict,
    resource_path: Path,
    deadline_epoch: float | None,
) -> dict:
    questions = i2254._eval_questions(cell.behavior)[:N_QUESTIONS]
    contexts = i2254._contexts_for_questions(questions)
    direction = None if cell.route == "nohook" else directions[cell.direction][cell.behavior]
    direction_sha = None if direction is None else _tensor_sha256(direction)
    regime = _regime_payload(cell, manifest, direction_sha)
    hook = None if direction is None else _hook_for_cell(model, cell, direction)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if hook is None:
        generated = _generate_batch_record(
            model,
            tokenizer,
            contexts,
            hook=None,
            deadline_epoch=deadline_epoch,
        )
        traces: list[dict] = []
    else:
        with hook:
            generated = _generate_batch_record(
                model,
                tokenizer,
                contexts,
                hook=hook,
                deadline_epoch=deadline_epoch,
            )
        traces = hook.draw_traces
        if len(traces) != N_DRAWS:
            raise DecodeSweepError(f"{cell.cell_id}: {len(traces)} traces != {N_DRAWS}")

    n_length = sum(reason == "length" for rows in generated["finish_reasons"] for reason in rows)
    all_texts = [text for rows in generated["completions"] for text in rows]
    all_token_ids = [ids for rows in generated["token_ids"] for ids in rows]
    repetition = [_repetition_diagnostic(ids) for ids in all_token_ids]
    cjk_flags = [bool(re.search(CJK_PATTERN, text)) for text in all_texts]
    torch.cuda.synchronize()
    wall_seconds = time.time() - started
    record = {
        "experiment": FOLLOWUP_LABEL,
        "plan_version": 15,
        "cell_id": cell.cell_id,
        "cell": {
            "behavior": cell.behavior,
            "route": cell.route,
            "direction": cell.direction,
            "position": cell.route,
            "layer": LAYER,
            "c": cell.dose,
        },
        "base_model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "pass_b_revision": i2254.HF_REV,
        "q_of_context": list(range(N_QUESTIONS)),
        "effective_seeds": list(range(SEED_BASE, SEED_BASE + N_DRAWS)),
        "seeds": {
            str(SEED_BASE): {
                "completions": generated["completions"],
                "token_ids": generated["token_ids"],
                "finish_reasons": generated["finish_reasons"],
                "token_counts": generated["token_counts"],
                "edit_traces": traces,
            }
        },
        "alpha": float(cell.dose * RHO_L14),
        "rho_pooled_median_L14": RHO_L14,
        "direction_bf16_sha256": direction_sha,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": None,
        "use_cache": True,
        "cap_regeneration": False,
        "cap_hit_fraction": n_length / (N_QUESTIONS * N_DRAWS),
        "n_length_finished": n_length,
        "text_diagnostics": {
            "cjk_pattern": CJK_PATTERN,
            "cjk_fraction": sum(cjk_flags) / len(cjk_flags),
            "repetition_definition": "exact 8-token n-gram occurs at least five times",
            "repetition_fraction": sum(row["flag"] for row in repetition) / len(repetition),
            "max_exact_8gram_count": max(row["max_exact_8gram_count"] for row in repetition),
        },
        "wall_seconds": wall_seconds,
        "resources": _resource_snapshot(resource_path),
        "completed_at_epoch": time.time(),
        "regime": regime,
        "regime_fp": _canonical_sha256(regime),
    }
    if hook is not None:
        pos = fk.PURE_DECODE_POSITION if cell.route == "decodeonly" else "lastctx"
        record["hook_impl"] = fk._hook_impl_record({"position": pos}, 1)
        record["expected_edit_profile"] = fk.expected_edit_profile(pos, 1)
        record["edit_trace_check"] = fk.assert_cell_edit_traces(record)
    else:
        record["hook_impl"] = {"steer_hook_class": None, "mode": "nohook", "n_layers": 0}
        record["expected_edit_profile"] = None
        record["edit_trace_check"] = {"checked_draws": 0, "per_draw": []}
    return i2254._run_metadata(record)


def _validate_record(record: dict, cell: CellSpec, expected_fp: str) -> None:
    if record.get("cell_id") != cell.cell_id or record.get("regime_fp") != expected_fp:
        raise DecodeSweepError(f"{cell.cell_id}: cell id/regime fingerprint mismatch")
    if record.get("base_model") != MODEL_NAME or record.get("model_revision") != MODEL_REVISION:
        raise DecodeSweepError(f"{cell.cell_id}: model provenance mismatch")
    if (
        record.get("max_new_tokens") != MAX_NEW_TOKENS
        or record.get("cap_regeneration") is not False
    ):
        raise DecodeSweepError(f"{cell.cell_id}: common-horizon generation mismatch")
    seed = record.get("seeds", {}).get(str(SEED_BASE), {})
    for field in ("completions", "token_ids", "finish_reasons", "token_counts"):
        rows = seed.get(field)
        if not isinstance(rows, list) or len(rows) != N_QUESTIONS:
            raise DecodeSweepError(f"{cell.cell_id}: {field} does not have 20 rows")
        if any(not isinstance(row, list) or len(row) != N_DRAWS for row in rows):
            raise DecodeSweepError(f"{cell.cell_id}: {field} does not have six draws per row")
    diagnostics = record.get("text_diagnostics", {})
    if diagnostics.get("cjk_pattern") != CJK_PATTERN or not all(
        isinstance(diagnostics.get(field), (int, float))
        for field in ("cjk_fraction", "repetition_fraction", "max_exact_8gram_count")
    ):
        raise DecodeSweepError(f"{cell.cell_id}: text diagnostics are absent or malformed")
    resources = record.get("resources", {})
    required_resources = {
        "hbm_total_bytes",
        "hbm_max_allocated_bytes",
        "hbm_max_reserved_bytes",
        "host_total_bytes",
        "host_max_rss_bytes",
        "disk_free_bytes",
    }
    if not required_resources.issubset(resources) or any(
        not isinstance(resources[field], int) or resources[field] <= 0
        for field in required_resources
    ):
        raise DecodeSweepError(f"{cell.cell_id}: resource diagnostics are absent or malformed")
    if (
        not isinstance(record.get("completed_at_epoch"), (int, float))
        or record["completed_at_epoch"] <= 0
    ):
        raise DecodeSweepError(f"{cell.cell_id}: completion timestamp is absent or malformed")
    if cell.route == "decodeonly":
        profile = record.get("expected_edit_profile", {})
        if profile.get("prefill") is not False or profile.get("all_decode") is not True:
            raise DecodeSweepError(f"{cell.cell_id}: pure-decode profile mismatch")
        if record.get("edit_trace_check", {}).get("checked_draws") != N_DRAWS:
            raise DecodeSweepError(f"{cell.cell_id}: decode trace coverage mismatch")
    elif cell.route == "context":
        profile = record.get("expected_edit_profile", {})
        if profile.get("prefill") is not True or profile.get("all_decode") is not False:
            raise DecodeSweepError(f"{cell.cell_id}: context profile mismatch")
    elif record.get("edit_trace_check", {}).get("checked_draws") != 0:
        raise DecodeSweepError(f"{cell.cell_id}: no-hook bridge has edit traces")


def _pack_and_upload(source: Path, destination: Path, hf_path: str, group: str) -> int:
    import scripts.issue2220_readwrite as rw2220

    if destination.exists():
        shutil.rmtree(destination)
    shards = rw2220._pack_tree_to_jsonl_shards(source, destination, group=group, pattern="*.json")
    if shards < 1:
        raise DecodeSweepError(f"packing {source} produced no JSONL shards")
    i2254._upload_folder_to_hf(destination, hf_path, allow=["*.jsonl", "*.json"])
    return shards


def phase_generate(args) -> None:
    """Generate a selected full-shape pilot cell or one across-cell shard."""
    root = round_root(args.out_root)
    manifest_path = root / "inputs_manifest.json"
    if not manifest_path.is_file():
        _stage_inputs(args.out_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("git_commit") != _git_commit():
        raise DecodeSweepError("inputs manifest was staged from a different Git commit")
    cells = registered_cells()
    if args.cell_id:
        cells = [cell for cell in cells if cell.cell_id == args.cell_id]
        if len(cells) != 1:
            raise DecodeSweepError(f"--cell-id did not select exactly one cell: {args.cell_id}")
    else:
        _require_production_gate(root, args.deadline_epoch)
        if args.force:
            raise DecodeSweepError(
                "--force is forbidden for full shards; rerun the pilot to create a new gate"
            )
        if args.num_shards != PLANNED_GPU_COUNT:
            raise DecodeSweepError(f"full generation requires exactly {PLANNED_GPU_COUNT} shards")
        if not 0 <= args.shard_id < args.num_shards:
            raise DecodeSweepError("shard-id must be in [0, num-shards)")
        cells = cells[args.shard_id :: args.num_shards]
    if not cells:
        raise DecodeSweepError("selected generation shard is empty")
    _arm_deadline(args.deadline_epoch)

    raw_root = root / "generation" / "raw_completions"
    raw_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < MIN_DISK_FREE_BYTES:
        raise DecodeSweepError("less than 10 GiB disk headroom before model load")
    directions = _load_directions(manifest)
    model, tokenizer = _load_model_and_tokenizer(manifest)
    started = time.time()
    generated_names: list[str] = []
    for index, cell in enumerate(cells, 1):
        _check_deadline(args.deadline_epoch)
        direction = None if cell.route == "nohook" else directions[cell.direction][cell.behavior]
        direction_sha = None if direction is None else _tensor_sha256(direction)
        expected_fp = _canonical_sha256(_regime_payload(cell, manifest, direction_sha))
        path = raw_root / f"{cell.cell_id}.json"
        if path.is_file() and not args.force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            _validate_record(cached, cell, expected_fp)
            logger.info("[phase=generate] unit %d/%d %s cached", index, len(cells), cell.cell_id)
        else:
            record = _generate_cell(
                model,
                tokenizer,
                cell,
                directions,
                manifest,
                root,
                args.deadline_epoch,
            )
            _validate_record(record, cell, expected_fp)
            _atomic_json(path, record)
            logger.info(
                "[phase=generate] unit %d/%d %s wall=%.1fs cap_hit=%.3f",
                index,
                len(cells),
                cell.cell_id,
                record["wall_seconds"],
                record["cap_hit_fraction"],
            )
        generated_names.append(path.name)
        _check_deadline(args.deadline_epoch)

    shard_stage = root / "generation" / f"stage_shard{args.shard_id}"
    if shard_stage.exists():
        shutil.rmtree(shard_stage)
    shard_stage.mkdir(parents=True)
    for name in generated_names:
        shutil.copy2(raw_root / name, shard_stage / name)
    n_pack = _pack_and_upload(
        shard_stage,
        root / "generation" / f"pack_shard{args.shard_id}",
        f"{HF_PREFIX}/raw_completions/shard{args.shard_id}",
        f"decode_sweep_shard{args.shard_id}",
    )
    shutil.rmtree(shard_stage)
    gate = None
    if args.cell_id == PILOT_CELL_ID:
        pilot_record = json.loads((raw_root / f"{PILOT_CELL_ID}.json").read_text(encoding="utf-8"))
        gate = _evaluate_pilot_gate(pilot_record)
        _atomic_json(root / "generation" / "pilot_gate_report.json", gate)
        i2254._upload_folder_to_hf(
            root / "generation",
            f"{HF_PREFIX}/generation_manifests",
            allow=["pilot_gate_report.json"],
        )
    _write_sentinel(
        f"decode-generate-shard{args.shard_id}",
        {
            "cells": len(cells),
            "pack_shards": n_pack,
            "wall_seconds": time.time() - started,
            "git_commit": _git_commit(),
            "pilot_gate_status": None if gate is None else gate["status"],
        },
    )
    if gate is not None and gate["status"] != "PASS":
        raise DecodeSweepError(f"production-shape pilot gate failed: {gate['checks']}")
    _disarm_deadline()


def _bridge_parity(records: dict[str, dict], behavior: str) -> dict:
    decode = records[CellSpec(behavior, "decodeonly", 0.0).cell_id]
    bridge = records[CellSpec(behavior, "nohook", 0.0).cell_id]
    ds = decode["seeds"][str(SEED_BASE)]
    bs = bridge["seeds"][str(SEED_BASE)]
    fields = ("completions", "token_ids", "finish_reasons", "token_counts")
    equal = {field: ds[field] == bs[field] for field in fields}
    if not all(equal.values()):
        raise DecodeSweepError(f"{behavior}: c0 decode hook differs from no-hook bridge: {equal}")
    return {"behavior": behavior, "fields_equal": equal, "n_responses": 120}


def _validate_downloaded_documents(
    docs: dict[str, object], expected_raw: set[str], verification: dict
) -> None:
    expected = expected_raw | {"inputs_manifest.json", "generation_manifest.json"}
    if set(docs) != expected:
        raise DecodeSweepError(
            f"verified pack document set differs: missing={sorted(expected - set(docs))} "
            f"extras={sorted(set(docs) - expected)}"
        )
    expected_doc_hashes = verification.get("document_canonical_sha256", {})
    if set(expected_doc_hashes) != expected:
        raise DecodeSweepError("verification does not hash-bind every packed document")
    for name, doc in docs.items():
        if _canonical_sha256(doc) != expected_doc_hashes[name]:
            raise DecodeSweepError(f"verified packed document differs for {name}")
    generation_manifest = docs["generation_manifest.json"]
    manifest_rows = {row["path"]: row for row in generation_manifest.get("files", [])}
    if set(manifest_rows) != expected_raw:
        raise DecodeSweepError("generation manifest raw-document set differs")
    for name in expected_raw:
        if _canonical_sha256(docs[name]) != manifest_rows[name].get("sha256"):
            raise DecodeSweepError(f"raw document differs from generation manifest for {name}")


def phase_verify(args) -> None:
    """Validate all local records, c0 bridges, pack, upload, and remote exact set."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    root = round_root(args.out_root)
    manifest = json.loads((root / "inputs_manifest.json").read_text(encoding="utf-8"))
    directions = _load_directions(manifest)
    raw_root = root / "generation" / "raw_completions"
    records: dict[str, dict] = {}
    file_rows = []
    for cell in registered_cells():
        path = raw_root / f"{cell.cell_id}.json"
        if not path.is_file():
            raise DecodeSweepError(f"missing registered generation {path.name}")
        record = json.loads(path.read_text(encoding="utf-8"))
        direction = None if cell.route == "nohook" else directions[cell.direction][cell.behavior]
        direction_sha = None if direction is None else _tensor_sha256(direction)
        fp = _canonical_sha256(_regime_payload(cell, manifest, direction_sha))
        _validate_record(record, cell, fp)
        records[cell.cell_id] = record
        file_rows.append(
            {
                "cell_id": cell.cell_id,
                "path": path.name,
                # `sha256` is deliberately over the parsed JSON document so it
                # remains verifiable after JSONL pack/unpack. Preserve the
                # original pretty-printed file hash separately.
                "sha256": _canonical_sha256(record),
                "source_file_sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "cap_hit_fraction": record["cap_hit_fraction"],
            }
        )
    bridges = [_bridge_parity(records, behavior) for behavior in BEHAVIORS]
    generation_manifest = {
        "experiment": FOLLOWUP_LABEL,
        "git_commit": _git_commit(),
        "n_generation_cells": len(file_rows),
        "n_analysis_cells": len(analysis_cells()),
        "n_retained_responses": len(file_rows) * N_QUESTIONS * N_DRAWS,
        "n_analysis_responses": len(analysis_cells()) * N_QUESTIONS * N_DRAWS,
        "bridge_parity": bridges,
        "files": file_rows,
    }
    _atomic_json(root / "generation" / "generation_manifest.json", generation_manifest)
    all_stage = root / "generation" / "stage_all"
    if all_stage.exists():
        shutil.rmtree(all_stage)
    all_stage.mkdir(parents=True)
    for row in file_rows:
        shutil.copy2(raw_root / row["path"], all_stage / row["path"])
    shutil.copy2(root / "inputs_manifest.json", all_stage / "inputs_manifest.json")
    shutil.copy2(
        root / "generation" / "generation_manifest.json",
        all_stage / "generation_manifest.json",
    )
    pack_root = root / "generation" / "pack_all"
    n_pack = _pack_and_upload(
        all_stage,
        pack_root,
        f"{HF_PREFIX}/raw_completions/all",
        "decode_sweep_all",
    )
    shutil.rmtree(all_stage)

    api = HfApi()
    revision = hub.retry_transient(
        lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset").sha,
        what="resolve decode-sweep upload revision",
    )
    remote_entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                HF_DATA_REPO,
                path_in_repo=f"{HF_PREFIX}/raw_completions/all",
                repo_type="dataset",
                revision=revision,
            )
        ),
        what="verify decode-sweep packed raw exact set",
    )
    remote_names = sorted(Path(entry.path).name for entry in remote_entries)
    local_names = sorted(path.name for path in pack_root.iterdir() if path.is_file())
    if remote_names != local_names:
        raise DecodeSweepError(
            f"remote packed set differs: remote={remote_names} local={local_names}"
        )
    docs: dict[str, object] = {}
    pack_hashes = {}
    for name in local_names:
        remote_path = hub.retry_transient(
            lambda n=name: hf_hub_download(
                HF_DATA_REPO,
                f"{HF_PREFIX}/raw_completions/all/{n}",
                repo_type="dataset",
                revision=revision,
            ),
            what=f"download-verify {name}",
        )
        pack_sha = _sha256_file(pack_root / name)
        if _sha256_file(Path(remote_path)) != pack_sha:
            raise DecodeSweepError(f"remote packed bytes differ for {name}")
        pack_hashes[name] = pack_sha
        if name.endswith(".jsonl"):
            for line in Path(remote_path).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["path"] in docs:
                    raise DecodeSweepError(f"duplicate packed document {row['path']}")
                docs[row["path"]] = row["doc"]
    expected_doc_names = {row["path"] for row in file_rows} | {
        "inputs_manifest.json",
        "generation_manifest.json",
    }
    if set(docs) != expected_doc_names:
        raise DecodeSweepError("downloaded packed document set is incomplete or has extras")
    for row in file_rows:
        if docs[row["path"]] != records[row["cell_id"]]:
            raise DecodeSweepError(f"downloaded packed JSON differs for {row['cell_id']}")
        if _canonical_sha256(docs[row["path"]]) != row["sha256"]:
            raise DecodeSweepError(f"downloaded packed JSON hash differs for {row['cell_id']}")

    document_hashes = {name: _canonical_sha256(doc) for name, doc in docs.items()}

    verification = {
        "status": "PASS",
        "hf_repo": HF_DATA_REPO,
        "hf_revision": revision,
        "hf_prefix": f"{HF_PREFIX}/raw_completions/all",
        "remote_files": remote_names,
        "remote_file_sha256": pack_hashes,
        "document_canonical_sha256": document_hashes,
        "n_pack_shards": n_pack,
        "n_documents": len(docs),
        "expected_documents": sorted(expected_doc_names),
        "download_hash_verified": True,
        "document_equality_verified": True,
        "bridge_parity": bridges,
    }
    _atomic_json(root / "generation" / "upload_verification.json", verification)
    i2254._upload_folder_to_hf(
        root / "generation",
        f"{HF_PREFIX}/generation_manifests",
        allow=["generation_manifest.json", "upload_verification.json"],
    )
    _write_sentinel(
        "decode-generation-verified",
        {
            "cells": len(file_rows),
            "hf_revision": revision,
            "pack_shards": n_pack,
            "git_commit": _git_commit(),
        },
    )
    logger.info(
        "[phase=verify] PASS cells=%d responses=%d hf_revision=%s",
        len(file_rows),
        generation_manifest["n_retained_responses"],
        revision,
    )


def phase_download(args) -> None:
    """Stage the verified canonical all-pack from Hugging Face onto this host."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    root = round_root(args.out_root)
    api = HfApi()
    pointer_revision = hub.retry_transient(
        lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset").sha,
        what="resolve decode generation verification pointer",
    )
    verification_path = hub.retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO,
            f"{HF_PREFIX}/generation_manifests/upload_verification.json",
            repo_type="dataset",
            revision=pointer_revision,
        ),
        what="download decode generation verification pointer",
    )
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or not verification.get("download_hash_verified"):
        raise DecodeSweepError("remote generation verification is absent or not passing")
    raw_revision = verification["hf_revision"]
    prefix = verification["hf_prefix"]
    entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=raw_revision,
            )
        ),
        what="list verified decode generation pack",
    )
    names = sorted(Path(entry.path).name for entry in entries)
    if names != sorted(verification["remote_files"]):
        raise DecodeSweepError("verified generation pack exact set changed at pinned revision")
    docs: dict[str, object] = {}
    expected_pack_hashes = verification.get("remote_file_sha256", {})
    if set(expected_pack_hashes) != set(names):
        raise DecodeSweepError("verification does not hash-bind every packed file")
    for name in names:
        downloaded = hub.retry_transient(
            lambda n=name: hf_hub_download(
                HF_DATA_REPO,
                f"{prefix}/{n}",
                repo_type="dataset",
                revision=raw_revision,
            ),
            what=f"stage verified generation pack {name}",
        )
        if _sha256_file(Path(downloaded)) != expected_pack_hashes[name]:
            raise DecodeSweepError(f"verified packed bytes differ for {name}")
        if name.endswith(".jsonl"):
            for line in Path(downloaded).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["path"] in docs:
                    raise DecodeSweepError(f"duplicate packed document {row['path']}")
                docs[row["path"]] = row["doc"]
    expected_raw = {f"{cell.cell_id}.json" for cell in registered_cells()}
    _validate_downloaded_documents(docs, expected_raw, verification)
    raw_root = root / "generation" / "raw_completions"
    raw_root.mkdir(parents=True, exist_ok=True)
    for name in expected_raw:
        _atomic_json(raw_root / name, docs[name])
    _atomic_json(root / "inputs_manifest.json", docs["inputs_manifest.json"])
    _atomic_json(root / "generation" / "generation_manifest.json", docs["generation_manifest.json"])
    _atomic_json(root / "generation" / "upload_verification.json", verification)
    logger.info(
        "[phase=download] staged %d raw cells from revision %s", len(expected_raw), raw_revision
    )


def phase_stage_inputs(args) -> None:
    _stage_inputs(args.out_root)


PHASES = {
    "stage_inputs": phase_stage_inputs,
    "generate": phase_generate,
    "verify": phase_verify,
    "download": phase_download,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--out-root", default=str(_REPO_ROOT))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--cell-id")
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    unknown = sorted(set(phases) - set(PHASES))
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices={sorted(PHASES)}")
    for phase in phases:
        print(f"[phase={phase}]", flush=True)
        PHASES[phase](args)


if __name__ == "__main__":
    main()
