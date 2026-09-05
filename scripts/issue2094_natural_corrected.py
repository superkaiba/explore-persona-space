#!/usr/bin/env python3
"""Corrected natural task/subject context-end patching follow-up for #2094.

The primary intervention replaces the final-context-token block output at
layer 19.  The all-28-layer replacement is a separately labelled secondary
arm.  All generations share an identical forced ``Response:\n`` opening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: E402

from explore_persona_space.experiments.issue2094.hooks import joint_hooks  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS = 28
PRIMARY_LAYER = 19
ALL_LAYERS = tuple(range(N_LAYERS))
FORCED_OPENING = "Response:\n"
INITIAL_MAX_NEW_TOKENS = 768
RERUN_MAX_NEW_TOKENS = 1536
GREEDY_SEED = 0
SCHEMA_VERSION = 1

SUBJECTS: tuple[tuple[str, str], ...] = (
    ("vancouver", "Chinese food and culture in Vancouver"),
    ("japan", "Tokyo and Mount Fuji for a family visiting Japan"),
    ("rome", "Ancient Roman landmarks and history in Rome"),
)

TASKS: tuple[tuple[str, str], ...] = (
    (
        "itinerary",
        "Hello. Create a three-day itinerary, organized day by day, for exploring "
        "{subject}. Include a morning, afternoon, and evening plan for each day. "
        "Keep the entire answer under 220 words.",
    ),
    (
        "quiz",
        "Hello. Create a five-question multiple-choice quiz about {subject}. Give "
        "four options per question and a compact answer key. Keep the entire answer "
        "under 220 words.",
    ),
    (
        "briefing",
        "Hello. Write a practical visitor briefing about {subject}. Organize it into "
        "exactly four short titled sections covering context, highlights, etiquette "
        "or cautions, and planning. Keep the entire answer under 220 words.",
    ),
)

FORMAT_INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("bullets", "Use exactly five bullet points, with no prose before or after the list."),
    (
        "paragraph",
        "Use one continuous paragraph, with no bullets, numbering, or headings.",
    ),
)


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    family: str
    text: str
    task: str
    subject: str
    output_format: str | None


def prompt_bank() -> list[PromptRecord]:
    """Return the fixed 3x3 bank plus six exact-format minimal pairs."""
    records: list[PromptRecord] = []
    for task, template in TASKS:
        for subject, subject_text in SUBJECTS:
            records.append(
                PromptRecord(
                    prompt_id=f"main__{task}__{subject}",
                    family="main",
                    text=template.format(subject=subject_text),
                    task=task,
                    subject=subject,
                    output_format=None,
                )
            )
    for output_format, instruction in FORMAT_INSTRUCTIONS:
        for subject, subject_text in SUBJECTS:
            records.append(
                PromptRecord(
                    prompt_id=f"format__{output_format}__{subject}",
                    family="format",
                    text=(
                        "Hello. Explain the most useful things a first-time visitor "
                        f"should know about {subject_text} in 140 to 180 words. "
                        f"{instruction}"
                    ),
                    task="explanation",
                    subject=subject,
                    output_format=output_format,
                )
            )
    assert len(records) == 15
    assert len({r.prompt_id for r in records}) == len(records)
    assert all(r.text.startswith("Hello. ") for r in records)
    return records


def _directed_pairs(values: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(recipient, donor) for recipient in values for donor in values if donor != recipient]


def planned_rows() -> list[dict[str, Any]]:
    """Build the preregistered 129-row generation census."""
    bank = prompt_bank()
    by_id = {p.prompt_id: p for p in bank}
    rows: list[dict[str, Any]] = []

    def add(
        arm: str,
        axis: str,
        layer_setting: str,
        recipient_id: str,
        donor_id: str | None,
    ) -> None:
        rec = by_id[recipient_id]
        donor = by_id[donor_id] if donor_id else None
        gen_id = "__".join((arm, axis, layer_setting, recipient_id, f"from_{donor_id or 'none'}"))
        rows.append(
            {
                "gen_id": gen_id,
                "arm": arm,
                "axis": axis,
                "layer_setting": layer_setting,
                "recipient_prompt_id": recipient_id,
                "donor_prompt_id": donor_id,
                "recipient_task": rec.task,
                "recipient_subject": rec.subject,
                "recipient_format": rec.output_format,
                "donor_task": donor.task if donor else None,
                "donor_subject": donor.subject if donor else None,
                "donor_format": donor.output_format if donor else None,
            }
        )

    for p in bank:
        add("unpatched", "anchor", "none", p.prompt_id, None)
    for setting in ("L19", "all28"):
        for p in bank:
            add("self_patch", "self", setting, p.prompt_id, p.prompt_id)

        tasks = tuple(t[0] for t in TASKS)
        subjects = tuple(s[0] for s in SUBJECTS)
        for subject in subjects:
            for recipient_task, donor_task in _directed_pairs(tasks):
                add(
                    "donor_patch",
                    "same_subject_different_task",
                    setting,
                    f"main__{recipient_task}__{subject}",
                    f"main__{donor_task}__{subject}",
                )
        for task in tasks:
            for recipient_subject, donor_subject in _directed_pairs(subjects):
                add(
                    "donor_patch",
                    "same_task_different_subject",
                    setting,
                    f"main__{task}__{recipient_subject}",
                    f"main__{task}__{donor_subject}",
                )
        for subject in subjects:
            for recipient_format, donor_format in _directed_pairs(("bullets", "paragraph")):
                add(
                    "donor_patch",
                    "positive_format_control",
                    setting,
                    f"format__{recipient_format}__{subject}",
                    f"format__{donor_format}__{subject}",
                )

    assert len(rows) == 129, len(rows)
    assert len({r["gen_id"] for r in rows}) == len(rows)
    return rows


def rows_for_smoke(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small census exercising every production intervention path."""
    wanted = {
        ("unpatched", "anchor", "none", "main__itinerary__vancouver", None),
        (
            "self_patch",
            "self",
            "L19",
            "main__itinerary__vancouver",
            "main__itinerary__vancouver",
        ),
        (
            "self_patch",
            "self",
            "all28",
            "main__itinerary__vancouver",
            "main__itinerary__vancouver",
        ),
        (
            "donor_patch",
            "same_subject_different_task",
            "L19",
            "main__itinerary__vancouver",
            "main__quiz__vancouver",
        ),
        (
            "donor_patch",
            "same_subject_different_task",
            "all28",
            "main__itinerary__vancouver",
            "main__quiz__vancouver",
        ),
        (
            "donor_patch",
            "same_task_different_subject",
            "L19",
            "main__itinerary__vancouver",
            "main__itinerary__japan",
        ),
        (
            "donor_patch",
            "same_task_different_subject",
            "all28",
            "main__itinerary__vancouver",
            "main__itinerary__japan",
        ),
        (
            "donor_patch",
            "positive_format_control",
            "L19",
            "format__bullets__vancouver",
            "format__paragraph__vancouver",
        ),
        (
            "donor_patch",
            "positive_format_control",
            "all28",
            "format__bullets__vancouver",
            "format__paragraph__vancouver",
        ),
    }
    selected = [
        r
        for r in rows
        if (
            r["arm"],
            r["axis"],
            r["layer_setting"],
            r["recipient_prompt_id"],
            r["donor_prompt_id"],
        )
        in wanted
    ]
    assert len(selected) == len(wanted), (len(selected), len(wanted))
    return selected


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def context_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer(render_prompt(tokenizer, prompt), add_special_tokens=False)["input_ids"]


def assert_assistant_header(tokenizer, ids: list[int]) -> None:
    tail = tokenizer.decode(ids[-3:])
    if tail != "<|im_start|>assistant\n":
        raise AssertionError(f"unexpected assistant header suffix: {tail!r}")


class LastTokenCapture:
    """Capture raw decoder-block outputs at each row's last real token."""

    def __init__(self, model, layers: tuple[int, ...]):
        self.layers = layers
        self.blocks = model.model.layers
        self.handles: list[Any] = []
        self.captured: dict[int, torch.Tensor] = {}

    def __enter__(self):
        for layer in self.layers:

            def make_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.captured[layer_idx] = hidden[:, -1, :].detach().cpu()

                return hook

            self.handles.append(self.blocks[layer].register_forward_hook(make_hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def encode_left_padded(tokenizer, prompts: list[str]) -> tuple[dict[str, torch.Tensor], list[int]]:
    rendered = [render_prompt(tokenizer, p) for p in prompts]
    per_row = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in rendered]
    for ids in per_row:
        assert_assistant_header(tokenizer, ids)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_side
    width = int(enc["input_ids"].shape[1])
    for row, ids in enumerate(per_row):
        observed = int(enc["attention_mask"][row].sum())
        assert observed == len(ids)
        assert enc["input_ids"][row, width - len(ids) :].tolist() == ids
    return enc, [len(ids) for ids in per_row]


@torch.no_grad()
def capture_source_states(
    model, tokenizer, bank: list[PromptRecord], batch_size: int
) -> torch.Tensor:
    """Return ``(n_prompts, 28, hidden)`` raw block states on CPU."""
    chunks: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for start in range(0, len(bank), batch_size):
        chunk = bank[start : start + batch_size]
        enc, _ = encode_left_padded(tokenizer, [p.text for p in chunk])
        with LastTokenCapture(model, ALL_LAYERS) as capture:
            model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                use_cache=False,
            )
        if set(capture.captured) != set(ALL_LAYERS):
            raise RuntimeError("source capture missed decoder layers")
        chunks.append(torch.stack([capture.captured[layer] for layer in ALL_LAYERS], dim=1))
    states = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(states.float()).all()):
        raise RuntimeError("non-finite source state")
    return states


class ForcedOpeningProcessor:
    """Force an exact token prefix while leaving later logits untouched."""

    def __init__(self, prompt_width: int, prefix_ids: list[int]):
        self.prompt_width = int(prompt_width)
        self.prefix_ids = list(prefix_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = int(input_ids.shape[1]) - self.prompt_width
        if 0 <= step < len(self.prefix_ids):
            token_id = self.prefix_ids[step]
            forced = torch.full_like(scores, -torch.inf)
            forced[:, token_id] = scores[:, token_id]
            return forced
        return scores


def classify_termination(token_ids: list[int], eos_ids: set[int]) -> tuple[str, list[int]]:
    """Trim decode padding at the first EOS and classify EOS versus length stop."""
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return "eos", token_ids[: index + 1]
    return "length", token_ids


def _layers(setting: str) -> tuple[int, ...]:
    if setting == "none":
        return ()
    if setting == "L19":
        return (PRIMARY_LAYER,)
    if setting == "all28":
        return ALL_LAYERS
    raise ValueError(setting)


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


@torch.no_grad()
def generate_chunk(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    bank_by_id: dict[str, PromptRecord],
    bank_index: dict[str, int],
    source_states: torch.Tensor,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate one homogeneous layer-setting chunk and return complete row records."""
    settings = {r["layer_setting"] for r in rows}
    assert len(settings) == 1
    setting = next(iter(settings))
    layers = _layers(setting)
    prompts = [bank_by_id[r["recipient_prompt_id"]].text for r in rows]
    enc, row_lengths = encode_left_padded(tokenizer, prompts)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    width = int(input_ids.shape[1])
    prefix_ids = tokenizer(FORCED_OPENING, add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise RuntimeError("forced opening tokenized to zero tokens")
    processor = LogitsProcessorList([ForcedOpeningProcessor(width, prefix_ids)])
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [int(eos)])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = min(eos_ids)

    torch.manual_seed(GREEDY_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GREEDY_SEED)

    def run_generate():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=sorted(eos_ids),
            pad_token_id=pad_id,
            logits_processor=processor,
            return_dict_in_generate=True,
        ).sequences

    stack = None
    if layers:
        positions = [[length - 1] for length in row_lengths]
        deltas_per_layer: list[list[torch.Tensor]] = []
        for layer in layers:
            deltas_per_layer.append(
                [
                    source_states[bank_index[str(r["donor_prompt_id"])], layer].unsqueeze(0)
                    for r in rows
                ]
            )
        stack = joint_hooks(model, layers)
        with stack:
            stack.arm_batch_per_layer(
                row_lengths,
                positions,
                deltas_per_layer,
                mode="replace",
                alpha=1.0,
            )
            stack.arm(width)
            sequences = run_generate()
        if stack.n_edits != len(layers):
            raise RuntimeError(f"wrong hook edit count: {stack.n_edits} != {len(layers)}")
    else:
        sequences = run_generate()

    realized = stack.realized_edits if stack is not None else []
    outputs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        raw_new = sequences[row_index, width:].tolist()
        termination, actual_ids = classify_termination(raw_new, eos_ids)
        if actual_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(f"forced opening mismatch for {row['gen_id']}")
        text = tokenizer.decode(actual_ids, skip_special_tokens=True)
        if not text.startswith(FORCED_OPENING):
            raise RuntimeError(f"decoded opening mismatch for {row['gen_id']}: {text[:30]!r}")

        telemetry: dict[str, Any] | None = None
        if layers:
            row_edits = [item for item in realized or [] if item["row"] == row_index]
            if len(row_edits) != len(layers):
                raise RuntimeError(f"missing edit telemetry for {row['gen_id']}")
            errors: list[float] = []
            norms: dict[str, float] = {}
            cosines: dict[str, float] = {}
            recipient_index = bank_index[row["recipient_prompt_id"]]
            donor_index = bank_index[str(row["donor_prompt_id"])]
            applied_bytes = bytearray()
            for item in row_edits:
                layer = int(item["layer"])
                expected = source_states[donor_index, layer].float()
                applied = item["applied"].squeeze(0)
                errors.append(float((applied - expected).abs().max()))
                norms[str(layer)] = float(expected.norm())
                cosines[str(layer)] = float(
                    torch.nn.functional.cosine_similarity(
                        expected,
                        source_states[recipient_index, layer].float(),
                        dim=0,
                    )
                )
                applied_bytes.extend(applied.contiguous().numpy().tobytes())
                if item["positions_unpadded"] != [row_lengths[row_index] - 1]:
                    raise RuntimeError(f"wrong unpadded edit position for {row['gen_id']}")
                if item["positions_padded"] != [width - 1]:
                    raise RuntimeError(f"wrong padded edit position for {row['gen_id']}")
            max_error = max(errors)
            if max_error != 0.0:
                raise RuntimeError(
                    f"non-exact applied source state for {row['gen_id']}: {max_error}"
                )
            telemetry = {
                "n_layers": len(layers),
                "layers": list(layers),
                "n_prefill_edits": len(row_edits),
                "positions_unpadded": [row_lengths[row_index] - 1],
                "position_padded": width - 1,
                "max_abs_source_error": max_error,
                "source_norm_by_layer": norms,
                "source_recipient_cosine_by_layer": cosines,
                "applied_sha256": hashlib.sha256(applied_bytes).hexdigest(),
            }

        outputs.append(
            {
                **row,
                "schema_version": SCHEMA_VERSION,
                "recipient_prompt": bank_by_id[row["recipient_prompt_id"]].text,
                "donor_prompt": (
                    bank_by_id[str(row["donor_prompt_id"])].text if row["donor_prompt_id"] else None
                ),
                "forced_opening": FORCED_OPENING,
                "forced_opening_token_ids": prefix_ids,
                "output_text": text,
                "generated_token_ids": actual_ids,
                "n_new_tokens": len(actual_ids),
                "termination_reason": termination,
                "max_new_tokens_used": max_new_tokens,
                "injection_telemetry": telemetry,
            }
        )
    return outputs


def _validate_cached_row(cached: dict[str, Any], planned: dict[str, Any]) -> bool:
    for key in (
        "gen_id",
        "arm",
        "axis",
        "layer_setting",
        "recipient_prompt_id",
        "donor_prompt_id",
    ):
        if cached.get(key) != planned.get(key):
            raise RuntimeError(f"cached row metadata mismatch for {planned['gen_id']}: {key}")
    return cached.get("termination_reason") == "eos"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    assert_out_root_headroom(out, 2.0, phase="natural-task-subject", canary_gb=1.0)
    rows = planned_rows()
    if args.smoke:
        rows = rows_for_smoke(rows)
    bank = prompt_bank()
    bank_by_id = {p.prompt_id: p for p in bank}
    bank_index = {p.prompt_id: i for i, p in enumerate(bank)}
    bank_payload = [p.__dict__ for p in bank]
    bank_hash = stable_json_hash(bank_payload)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "prompt_bank.json", {"sha256": bank_hash, "prompts": bank_payload})

    print("[phase=model_load]", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "revision": MODEL_REVISION,
        "device_map": "cuda",
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, **load_kwargs
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, **load_kwargs
        ).eval()
    if len(model.model.layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} layers, got {len(model.model.layers)}")
    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if model_revision != MODEL_REVISION or tokenizer_revision != MODEL_REVISION:
        raise RuntimeError(
            f"resolved revision mismatch: model={model_revision}, tokenizer={tokenizer_revision}"
        )

    print("[phase=capture]", flush=True)
    source_path = out / "source_states.pt"
    if source_path.exists():
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        expected = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_bank_sha256": bank_hash,
            "layers": list(ALL_LAYERS),
            "convention": "raw decoder-block output at final context token",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"stale source state metadata: {key}")
        source_states = payload["states"]
    else:
        source_states = capture_source_states(model, tokenizer, bank, args.batch_size)
        torch.save(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_bank_sha256": bank_hash,
                "layers": list(ALL_LAYERS),
                "convention": "raw decoder-block output at final context token",
                "states": source_states,
            },
            source_path,
        )
    expected_shape = (len(bank), N_LAYERS, int(model.config.hidden_size))
    if tuple(source_states.shape) != expected_shape:
        raise RuntimeError(f"source state shape {tuple(source_states.shape)} != {expected_shape}")

    metadata = {
        "experiment": "issue2094_natural_task_subject_corrected",
        "schema_version": SCHEMA_VERSION,
        "started_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": tokenizer_revision,
        "model_dtype": str(next(model.parameters()).dtype),
        "n_layers": N_LAYERS,
        "primary_layer": PRIMARY_LAYER,
        "all_layer_treatment": list(ALL_LAYERS),
        "context_position": "last token of add_generation_prompt=True chat render",
        "patch_mode": "replace once during prefill; decode unedited",
        "generation": {
            "do_sample": False,
            "seed": GREEDY_SEED,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "initial_max_new_tokens": INITIAL_MAX_NEW_TOKENS,
            "cap_rerun_max_new_tokens": RERUN_MAX_NEW_TOKENS,
            "forced_opening": FORCED_OPENING,
            "forced_opening_token_ids": tokenizer(FORCED_OPENING, add_special_tokens=False)[
                "input_ids"
            ],
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "prompt_bank_sha256": bank_hash,
        "planned_n_rows": len(rows),
        "full_planned_n_rows": 129,
        "smoke": bool(args.smoke),
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "provenance": as_metadata_dict(
            git_provenance(cwd=Path.cwd()), phase="natural-task-subject-generation"
        ),
    }
    _atomic_json(out / "metadata.json", metadata)

    row_dir = out / "rows"
    row_dir.mkdir(exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = row_dir / f"{hashlib.sha256(row['gen_id'].encode()).hexdigest()}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if _validate_cached_row(cached, row):
                cache[row["gen_id"]] = cached

    print("[phase=generation]", flush=True)
    started = time.time()
    for max_tokens in (INITIAL_MAX_NEW_TOKENS, RERUN_MAX_NEW_TOKENS):
        pending = [r for r in rows if r["gen_id"] not in cache]
        if not pending:
            break
        if max_tokens == RERUN_MAX_NEW_TOKENS:
            pending = [
                r
                for r in pending
                if (row_dir / f"{hashlib.sha256(r['gen_id'].encode()).hexdigest()}.json").exists()
            ]
        for setting in ("none", "L19", "all28"):
            group = [r for r in pending if r["layer_setting"] == setting]
            for start in range(0, len(group), args.batch_size):
                chunk = group[start : start + args.batch_size]
                if not chunk:
                    continue
                records = generate_chunk(
                    model,
                    tokenizer,
                    chunk,
                    bank_by_id,
                    bank_index,
                    source_states,
                    max_tokens,
                )
                for record in records:
                    path = row_dir / f"{hashlib.sha256(record['gen_id'].encode()).hexdigest()}.json"
                    _atomic_json(path, record)
                    if record["termination_reason"] == "eos":
                        cache[record["gen_id"]] = record
                print(
                    f"generated setting={setting} batch={start // args.batch_size + 1} "
                    f"max_tokens={max_tokens} complete={len(cache)}/{len(rows)}",
                    flush=True,
                )
                if (time.time() - started) / 3600 > args.hard_stop_hours:
                    raise RuntimeError(
                        "hard wall-time stop exceeded before generation census completed"
                    )

    missing = [r["gen_id"] for r in rows if r["gen_id"] not in cache]
    if missing:
        raise RuntimeError(
            f"{len(missing)} rows still capped/missing after cap rerun: {missing[:5]}"
        )
    ordered = [cache[r["gen_id"]] for r in rows]
    if any(r["termination_reason"] != "eos" for r in ordered):
        raise RuntimeError("non-EOS row survived final census")
    if len({r["gen_id"] for r in ordered}) != len(rows):
        raise RuntimeError("duplicate generation IDs")
    _write_jsonl(out / "generations.jsonl", ordered)

    metadata["finished_utc"] = datetime.now(UTC).isoformat()
    metadata["generation_wall_seconds"] = time.time() - started
    metadata["realized_n_rows"] = len(ordered)
    metadata["n_cap_reruns"] = sum(
        r["max_new_tokens_used"] == RERUN_MAX_NEW_TOKENS for r in ordered
    )
    metadata["peak_cuda_allocated_gb"] = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    )
    _atomic_json(out / "metadata.json", metadata)
    done = {
        "schema_version": SCHEMA_VERSION,
        "finished_utc": metadata["finished_utc"],
        "prompt_bank_sha256": bank_hash,
        "realized_n_rows": len(ordered),
        "all_rows_eos": True,
        "generations_sha256": hashlib.sha256((out / "generations.jsonl").read_bytes()).hexdigest(),
    }
    _atomic_json(out / "DONE.json", done)

    if args.write_sentinel:
        sentinel_dir = Path("/workspace/logs")
        sentinel = sentinel_dir / f"issue-2094-epm_results-{int(time.time())}.json"
        _atomic_json(
            sentinel,
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 2094,
                "gate": "natural-task-subject-corrected-generation",
                "blocks_pipeline": False,
                "by": "issue2094_natural_corrected.py",
                "ts": datetime.now(UTC).isoformat(),
                "note": {
                    "followup_label": "natural-task-subject-corrected",
                    "out_root": str(out),
                    "realized_n_rows": len(ordered),
                    "all_rows_eos": True,
                    "generation_wall_seconds": metadata["generation_wall_seconds"],
                },
            },
        )
    print("[phase=done]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/issue2094-natural-task-subject-corrected"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--hard-stop-hours", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-sentinel", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.hard_stop_hours <= 0:
        parser.error("--hard-stop-hours must be positive")
    run(args)


if __name__ == "__main__":
    main()
