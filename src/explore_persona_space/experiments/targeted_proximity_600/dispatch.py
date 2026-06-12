# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ΔG + Qwen marker " ※" + × intentional
"""Task #600 §4.7 — unified smoke=sweep dispatcher (fork of leave_one_out_505/dispatch.py).

One dispatcher, one per-(cell, seed) subprocess shape, one logging surface.
Smoke = ``--smoke`` = the SAME ``main()`` with ``spec_iter`` reduced to ONE
(cell, seed) — the first NEAR cell at seed 42 — same subprocess shape
(``scripts/i600_run_cell.py``), same GPU-pin env injection, same eval path,
same sentinel JSON. The sweep runs the same code path over the full
``--cells/--seeds`` subset. Every phase's cell list derives from
``spec_iter``: train/eval per subprocess, smoke gates from ``spec_iter[0]``'s
trajectory, uploads from the artifacts this run produced. The analysis phase
is NOT executed on the pod at all (VM, post-teardown — plan §9).

Constants are RE-PINNED to the #600 plan recipe (r16/α32 attn-only, lr 5e-6,
1 epoch smoke-laddered) — NOT #505's rescued r32/lr1e-5/3ep values.

The #600 path NEVER reads the #472 ``R_eval.json`` artifact (plan §10: R_eval
is UNFIT — missing 15 bank personas, generated at a different commit). The
training mixes consume only ``R_train.json``; eval is on-policy generation.
``tests/test_issue600_panel_disjointness.py`` pins this structurally.

Pod-side contract (poll_pipeline.py): ``[phase=...]`` log lines terminating
in a single ``[phase=done]`` on graceful completion, plus an end-of-run
sentinel JSON carrying ``sentinel_schema_version`` / ``kind`` / ``version``.
Per-cell completion echoes never carry the ``[phase=done]`` token (#545).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.targeted_proximity_600 import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS_DEFAULT,
    EXPECTED_MARKER_TOKEN_ID,
    EXPECTED_SHA256,
    EXPECTED_STEPS_PER_EPOCH,
    GRAD_ACCUM,
    HF_ADAPTER_PATH_PREFIX,
    HF_DATA_PREFIX,
    HF_DATA_PREFIX_INPUTS,
    HF_DATA_REPO,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    LORA_TARGETS_ATTN_ONLY,
    MARKER_BAND_LOG_ONLY,
    MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
    MARKER_TEXT,
    MAX_LORA_RANK_EVAL,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
    N_NEG_PERSONAS,
    NEG_ROWS_PER_PERSONA,
    POS_ROWS,
    QWEN_IM_END_TOKEN_ID,
    SEEDS,
    SOURCE_DG_BAND_NATS,
    SOURCE_LOGP_CEILING_EPS_NATS,
    SOURCE_MANIFEST_TOL_NATS,
    SOURCE_PERSONA,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
    WANDB_PROJECT,
)
from explore_persona_space.experiments.targeted_proximity_600 import (
    BYSTANDER_ARGMAX_CEILING as ARGMAX_CEILING,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    CellSpec600,
    cell_specs_from_manifest,
    first_near_slug,
    load_manifest,
)

log = logging.getLogger("issue_600.dispatch")

TERMINAL_FRAC = 1.0


# ── Path resolvers (env-overridable for local smokes / tests). ──────────────


def _output_root() -> Path:
    return Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_600"))


def _data_root() -> Path:
    return Path(os.environ.get("EPM_DATA_ROOT", "data/issue_600"))


def _i472_data_root() -> Path:
    return Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472"))


def _manifest_path() -> Path:
    return Path(os.environ.get("EPM_I600_MANIFEST", str(_output_root() / "panel_selection.json")))


def _repo_root() -> Path:
    """Repo root: env ``REPO_ROOT`` when set (GCP lane runs from ``$WORKLOAD_ROOT``),
    else resolved from this module's location (src layout, editable install)."""
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not (root / "scripts").is_dir():
            raise FileNotFoundError(f"REPO_ROOT={env_root} has no scripts/ dir — wrong root.")
        return root
    root = Path(__file__).resolve().parents[4]
    if not (root / "scripts").is_dir():
        raise FileNotFoundError(
            f"Cannot resolve repo root from {__file__} (no scripts/ under {root})."
        )
    return root


# ── Train kwargs (single source of truth; pinned + unit-tested). ────────────


def train_overrides_600(epochs: int) -> dict:
    """The pinned #600 train_one_cell override kwargs (plan §4.5 / §11).

    Single source of truth for the recipe re-pin so
    ``tests/test_issue600_collator_post_response_slot.py`` can assert the
    load-bearing conjunction + the attn-only targets + log-only band mode
    without spinning up a trainer.
    """
    return {
        "lora_r_override": LORA_R,
        "lora_alpha_override": LORA_ALPHA,
        "lr_override": LEARNING_RATE,
        "epochs_override": int(epochs),
        # THE load-bearing conjunction (defaults on main are OFF/None).
        "marker_suppress_at_post_response_slot": MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        "marker_im_end_token_id": QWEN_IM_END_TOKEN_ID,
        # Band callback attaches (marker_band_stop default True) in LOG-ONLY
        # mode: telemetry without stopping → all cells run matched steps.
        "marker_band_log_only_override": MARKER_BAND_LOG_ONLY,
        # Attn-only quad — without this the rig silently trains the 7-module
        # default (#505's demonstrated floor regime at r16/lr5e-6).
        "lora_targets_override": list(LORA_TARGETS_ATTN_ONLY),
    }


# ── Marker invariant. ────────────────────────────────────────────────────────


def assert_marker_tokenization(tokenizer) -> None:
    """In-process marker assert (incident #537: convention-only checks are not enough)."""
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"[invariant] marker {MARKER_TEXT!r} tokenizes to {encoded}, expected "
            f"[{EXPECTED_MARKER_TOKEN_ID}]. Tokenizer drift — aborting."
        )


# ── Phase 0 helpers: inherited #472 artifacts (NO R_eval on the #600 path). ──


def _sha256_file(path: Path) -> str:
    """Streaming sha256 of ``path`` (the EXPECTED_SHA256 pin check)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_pinned_sha256(path: Path, local_rel: str) -> None:
    """Fail-loud content pin: ``path`` must hash to ``EXPECTED_SHA256[local_rel]``.

    The 2026-06-11 incident class: an HF mirror of a reused artifact silently
    diverged from the verified local generation (issue472_neg_geometry/
    R_train.json + centroids_L10.pt were a different git generation, dac5749
    vs b68e560) and the divergence surfaced only as a KeyError ten frames deep
    in build_cell. Every prefetch/autofetch of a pinned input asserts the hash
    at the trust boundary instead.
    """
    expected = EXPECTED_SHA256[local_rel]
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"[invariant] pinned input {local_rel} at {path} has sha256 {actual} but the "
            f"verified #600 snapshot pins {expected}. The file is a DIFFERENT generation "
            f"(stale mirror / corrupted download) — refusing to proceed. Re-upload the "
            f"verified copy to {HF_DATA_REPO}/{HF_DATA_PREFIX_INPUTS}/{local_rel} or fix "
            f"the on-disk copy."
        )


def _prefetch_inherited_artifacts(i472_root: Path) -> None:
    """Idempotent, sha256-pinned HF prefetch of the inherited #472 inputs.

    Fetches from the issue-600-OWNED snapshot ``HF_DATA_PREFIX_INPUTS`` (NOT
    the shared ``issue472_neg_geometry/`` mirrors, two of which are a stale
    generation — the 2026-06-11 smoke crash). Every file — downloaded OR
    already on disk — is hash-asserted against ``EXPECTED_SHA256`` so a stale
    copy from ANY source fails loud at phase=prefetch.

    Deliberately does NOT fetch ``R_eval.json`` — plan §10 marks it UNFIT
    (missing 15 bank personas) and the #600 path must never read it.
    """
    targets = sorted(EXPECTED_SHA256)  # local_rel == path under HF_DATA_PREFIX_INPUTS
    missing = [rel for rel in targets if not (i472_root / rel).exists()]
    if missing:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                f"inherited #472 artifacts missing locally and HF_TOKEN unset → cannot "
                f"prefetch: {missing}."
            )
        from huggingface_hub import hf_hub_download

        i472_root.mkdir(parents=True, exist_ok=True)
        for local_rel in missing:
            remote = f"{HF_DATA_PREFIX_INPUTS}/{local_rel}"
            log.info("[phase=prefetch] %s ← %s/%s", local_rel, HF_DATA_REPO, remote)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO, filename=remote, repo_type="dataset", token=token
            )
            target = i472_root / local_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != Path(downloaded).resolve():
                import shutil

                shutil.copyfile(downloaded, target)
    # Pin check covers EVERY target — fresh downloads AND pre-existing files
    # (a stale on-disk copy is the same crash class as a stale mirror).
    for local_rel in targets:
        assert_pinned_sha256(i472_root / local_rel, local_rel)
        log.info(
            "[phase=prefetch] OK %s (%d bytes, sha256 pin verified)",
            i472_root / local_rel,
            (i472_root / local_rel).stat().st_size,
        )


def _load_bank_and_r_train() -> tuple[dict[str, str], dict, list[str]]:
    """Load the #472 persona bank + R_train (ONLY — never R_eval) via the canonical helpers."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        NO_PERSONA_KEY,
        load_r_artifact,
    )

    i472 = _i472_data_root()
    persona_bank = load_persona_bank(i472 / "persona_bank.json")
    r_train = load_r_artifact(i472 / "on_policy_R" / "R_train.json")
    # Consume-time coverage assert (the r_generate EXIT assertion, enforced
    # where the artifact is CONSUMED): every bank persona + no_persona must
    # have completions in R_train. A stale/foreign R_train generation fails
    # HERE with the missing personas named, not as a KeyError in build_cell
    # (the 2026-06-11 GCE smoke crash: r_train lacked 'bartender').
    missing_personas = sorted((set(persona_bank) | {NO_PERSONA_KEY}) - set(r_train))
    if missing_personas:
        raise RuntimeError(
            f"[invariant] R_train at {i472 / 'on_policy_R' / 'R_train.json'} is missing "
            f"{len(missing_personas)} personas required by the bank ∪ {{no_persona}}: "
            f"{missing_personas}. The artifact is a different generation than the bank — "
            f"refusing to build training mixes."
        )
    any_p = next(iter(r_train))
    q_train = sorted(r_train[any_p])
    return persona_bank, r_train, q_train


# ── Realized-panel verifier (plan §4.4 — against builder OUTPUT, not prose). ─


def verify_realized_panel(
    jsonl_path: Path,
    *,
    persona_bank: dict[str, str],
    expected_panel: list[str],
    source: str,
    targets: list[str],
    pos_rows: int = POS_ROWS,
    neg_rows_per_persona: int = NEG_ROWS_PER_PERSONA,
    marker_text: str = MARKER_TEXT,
) -> dict:
    """Re-read the REALIZED training JSONL and assert the panel invariants.

    The #527/#538 incident class: the realized panel must be recovered from
    the rows' system prompts (mapped back to persona names via the bank),
    never assumed from the spec. Raises AssertionError on any violation;
    returns the verdict payload on success.
    """
    prompt_to_name = {prompt: name for name, prompt in persona_bank.items()}
    if len(prompt_to_name) != len(persona_bank):
        raise AssertionError("persona bank has duplicate prompts — cannot invert.")

    pos_count = 0
    neg_counts: dict[str, int] = {}
    n_rows = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            row = json.loads(line)
            sys_msgs = [m for m in row["prompt"] if m["role"] == "system"]
            if len(sys_msgs) != 1:
                raise AssertionError(f"row {n_rows}: expected exactly 1 system msg.")
            persona = prompt_to_name.get(sys_msgs[0]["content"])
            if persona is None:
                raise AssertionError(f"row {n_rows}: system prompt not in the bank.")
            completion = row["completion"][0]["content"]
            if marker_text in completion:
                if persona != source:
                    raise AssertionError(
                        f"row {n_rows}: marker-bearing row under {persona!r} != source."
                    )
                pos_count += 1
            else:
                if persona == source:
                    raise AssertionError(f"row {n_rows}: marker-less row under the source.")
                neg_counts[persona] = neg_counts.get(persona, 0) + 1

    realized_panel = sorted(neg_counts)
    expected_sorted = sorted(expected_panel)
    if realized_panel != expected_sorted:
        raise AssertionError(
            f"realized negative panel {realized_panel} != intended {expected_sorted}"
        )
    if source in realized_panel:
        raise AssertionError(f"source {source!r} appears as a negative.")
    overlap = set(realized_panel) & set(targets)
    if overlap:
        raise AssertionError(f"panel ∩ TARGETS != ∅: {sorted(overlap)}")
    if pos_count != pos_rows:
        raise AssertionError(f"positives {pos_count} != {pos_rows}")
    bad_counts = {p: c for p, c in neg_counts.items() if c != neg_rows_per_persona}
    if bad_counts:
        raise AssertionError(f"per-negative row counts off: {bad_counts}")
    expected_total = pos_rows + neg_rows_per_persona * len(expected_panel)
    if n_rows != expected_total:
        raise AssertionError(f"total rows {n_rows} != {expected_total}")
    payload = {
        "jsonl": str(jsonl_path),
        "realized_panel": realized_panel,
        "n_positive": pos_count,
        "neg_counts": neg_counts,
        "n_rows": n_rows,
        "verdict": "pass",
    }
    log.info("[panel-verify] PASS %s: panel=%s rows=%d", jsonl_path.name, realized_panel, n_rows)
    return payload


# ── Collator label-mask gate (plan §4.7 gate e). ─────────────────────────────


def collator_mask_gate(jsonl_path: Path, tokenizer) -> dict:
    """Run the REAL MarkerOnlyDataCollator over one positive + one negative row.

    Asserts (the #505 gate-(h) contract, on REAL rows + the REAL chat
    template): the positive row's loss-bearing token ids are exactly
    {marker} ∪ {the trailing valid token} (the trailing token under the Qwen
    template is the "\\n" after ``<|im_end|>`` — the recipe doc's "{marker,
    EOS}" shorthand), and the negative row's ONLY loss-bearing token is the
    first ``<|im_end|>`` after R. Raises on any violation.
    """
    import torch

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    pos_row = neg_row = None
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if MARKER_TEXT in row["completion"][0]["content"]:
                pos_row = pos_row or row
            else:
                neg_row = neg_row or row
            if pos_row and neg_row:
                break
    if pos_row is None or neg_row is None:
        raise AssertionError(f"{jsonl_path} lacks a positive or negative row.")

    def _ids_and_labels(row: dict) -> tuple[list[int], list[int]]:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise AssertionError("chat-template prompt is not a prefix of the full conversation.")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return full_ids, labels

    rows = [_ids_and_labels(pos_row), _ids_and_labels(neg_row)]
    t_max = max(len(ids) for ids, _ in rows)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((2, t_max), pad_id, dtype=torch.long)
    labels = torch.full((2, t_max), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(labs)] = torch.tensor(labs)

    collator = MarkerOnlyDataCollator(
        inner_collator=lambda b: b,
        marker_token_ids=[EXPECTED_MARKER_TOKEN_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        im_end_token_id=QWEN_IM_END_TOKEN_ID,
    )
    out = collator({"input_ids": input_ids, "labels": labels})

    pos_kept = [int(input_ids[0, j]) for j in (out["labels"][0] != -100).nonzero().flatten()]
    neg_kept = [int(input_ids[1, j]) for j in (out["labels"][1] != -100).nonzero().flatten()]
    if EXPECTED_MARKER_TOKEN_ID not in pos_kept:
        raise AssertionError(f"positive row loss mask lacks the marker id: kept={pos_kept}")
    allowed_pos = {EXPECTED_MARKER_TOKEN_ID, QWEN_IM_END_TOKEN_ID, 198}  # 198 = "\n"
    if not set(pos_kept) <= allowed_pos:
        raise AssertionError(f"positive row keeps unexpected loss tokens: {pos_kept}")
    if neg_kept != [QWEN_IM_END_TOKEN_ID]:
        raise AssertionError(
            f"negative row's loss tokens must be exactly [<|im_end|>]; got {neg_kept}"
        )
    payload = {
        "jsonl": str(jsonl_path),
        "positive_kept_token_ids": pos_kept,
        "negative_kept_token_ids": neg_kept,
        "suppress_at_post_response_slot": MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        "im_end_token_id": QWEN_IM_END_TOKEN_ID,
        "verdict": "pass",
    }
    log.info("[collator-gate] PASS: pos kept=%s, neg kept=%s", pos_kept, neg_kept)
    return payload


# ── Adapter-config parity assert (plan §4.5; #480 template). ─────────────────


def assert_adapter_config_parity(adapter_dir: Path) -> dict:
    """Diff the REALIZED peft adapter_config.json against the pinned #600 geometry.

    Verifies target_modules == attn-only quad, r=16, α=32, rsLoRA on,
    modules_to_save empty — the silent-7-module-degrade guard (plan §4.5) AND
    the gauge assert precondition for the logit readout.
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json missing at {cfg_path}")
    realized = json.loads(cfg_path.read_text())
    checks: dict[str, tuple[object, object]] = {
        "r": (realized.get("r"), LORA_R),
        "lora_alpha": (realized.get("lora_alpha"), LORA_ALPHA),
        "use_rslora": (realized.get("use_rslora"), True),
        "target_modules": (
            sorted(realized.get("target_modules") or []),
            sorted(LORA_TARGETS_ATTN_ONLY),
        ),
        "modules_to_save": (realized.get("modules_to_save") or None, None),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise RuntimeError(
            f"adapter-config parity FAILED at {cfg_path}: {mismatches} — the declared "
            "attn-only r16/α32 recipe did not reach the realized adapter (the silent "
            "7-module degrade class, plan §4.5); refusing to eval."
        )
    log.info("[adapter-parity] PASS at %s (%d keys)", cfg_path, len(checks))
    return {k: v[1] for k, v in checks.items()}


def eval_names_for_cell(
    held_out_panel: list[str],
    panel: tuple[str, ...] | list[str],
    extra_eval_personas: tuple[str, ...] | None = None,
) -> list[str]:
    """The eval persona list: held_out ∪ {source} ∪ panel ∪ extras (plan §4.6).

    ``extra_eval_personas=None`` reproduces the #600 behavior exactly; #610
    threads ``("qwen_default", "assistant")`` because neither is in any
    default eval set of the no-default arm (the primary DV would silently
    not exist — unit-tested in tests/test_issue610_overrides_backcompat.py).
    """
    return sorted(
        set(held_out_panel) | {SOURCE_PERSONA} | set(panel) | set(extra_eval_personas or ())
    )


# ── Per-(cell, seed) body — the subprocess target (scripts/i600_run_cell.py). ─


def run_one_cell(
    *,
    cell_slug: str,
    seed: int,
    gpu_id: int,
    epochs: int,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    data_root: Path | None = None,
    spec_override: CellSpec600 | None = None,
    extra_eval_personas: tuple[str, ...] | None = None,
    hf_adapter_prefix: str | None = None,
    run_name_prefix: str | None = None,
) -> dict:
    """Build → verify → train → eval ONE (cell, seed). Runs inside the subprocess.

    Same body for smoke and sweep (smoke IS this, once). Persists, per cell:
    the training JSONL (+ build manifest), the realized-panel verify JSON,
    the collator-gate JSON, the band-callback trajectory JSON, the
    adapter-parity JSON, trajectory.json (four-float leaves; compute_kl=True
    PINNED), raw_completions.json, and a done sentinel.

    #610 extensions (ALL default to None → byte-equivalent #600 behavior):
      spec_override        — bypass the manifest cell registry (whose
                             ``cell_specs_from_manifest`` hard-asserts
                             ``qwen_default`` ∈ panel — structurally
                             incompatible with the #610 no-default arm).
      extra_eval_personas  — appended to the eval persona set; required for
                             #610 because ``qwen_default``/``assistant`` are
                             in NO default eval set of the no-default arm
                             (the primary DV would silently not exist).
      hf_adapter_prefix    — HF path prefix for the inline adapter upload
                             (default: the #600 ``adapters/issue_600``).
      run_name_prefix      — WandB run-name prefix (default ``issue600_``).
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
        build_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )
    from explore_persona_space.experiments.leave_one_out_505.eval_trajectory_505 import (
        run_trajectory_eval_with_guard,
    )

    out_root = output_root or _output_root()
    data_root_ = data_root or _data_root()
    manifest = load_manifest(manifest_path or _manifest_path())
    if spec_override is not None:
        if spec_override.slug != cell_slug:
            raise ValueError(
                f"spec_override.slug {spec_override.slug!r} != cell_slug {cell_slug!r}."
            )
        spec = spec_override
    else:
        specs = cell_specs_from_manifest(manifest)
        spec = next((s for s in specs if s.slug == cell_slug), None)
        if spec is None:
            raise KeyError(
                f"Unknown #600 cell slug {cell_slug!r}; registry has {[s.slug for s in specs]}"
            )

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_tokenization(tokenizer)

    persona_bank, r_train, q_train = _load_bank_and_r_train()
    target_names = [t["name"] for t in manifest["targets"]]

    # ── Build the training JSONL (explicit panel; no auto-prepend path). ─────
    train_jsonl = data_root_ / f"{cell_slug}_seed{seed}.jsonl"
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    build_cell(
        cell_slug,
        train_jsonl,
        r_train=r_train,
        q_train=q_train,
        persona_bank=persona_bank,
        seed=seed,
        cell_specs=(
            (spec.slug, spec.plain_name, "explicit", N_NEG_PERSONAS, NEG_ROWS_PER_PERSONA, True),
        ),
        negative_personas_override=list(spec.panel),
    )

    cell_out = out_root / "sweep" / cell_slug / f"seed_{seed}"
    cell_out.mkdir(parents=True, exist_ok=True)

    # ── Gate (f): realized-panel verifier on the BUILT JSONL. ────────────────
    verify_payload = verify_realized_panel(
        train_jsonl,
        persona_bank=persona_bank,
        expected_panel=list(spec.panel),
        source=SOURCE_PERSONA,
        targets=target_names,
    )
    (cell_out / "panel_verify.json").write_text(json.dumps(verify_payload, indent=2))

    # ── Gate (e): collator label-mask probe on the BUILT JSONL (CPU). ────────
    collator_payload = collator_mask_gate(train_jsonl, tokenizer)
    (cell_out / "collator_gate.json").write_text(json.dumps(collator_payload, indent=2))

    # ── Train (63 matched steps × epochs; band callback in log-only mode). ───
    adapter_out = cell_out / "adapter"
    ckpt_root = cell_out / "checkpoints"
    band_traj_path = cell_out / "band_trajectory.json"
    train_result = train_one_cell(
        cell_slug=cell_slug,
        seed=seed,
        train_jsonl=train_jsonl,
        output_dir=adapter_out,
        ckpt_root=ckpt_root,
        fractions=TRAJECTORY_CHECKPOINT_FRACTIONS,
        base_model=BASE_MODEL,
        report_to="wandb",
        gpu_id=gpu_id,
        hf_path_in_repo_override=(
            f"{hf_adapter_prefix or HF_ADAPTER_PATH_PREFIX}/{cell_slug}_seed{seed}"
        ),
        run_name_override=f"{run_name_prefix or 'issue600_'}{cell_slug}_seed{seed}",
        marker_band_trajectory_path_override=str(band_traj_path),
        **train_overrides_600(epochs),
    )

    # ── Matched-step assert (load-bearing for the paired design, §4.8). ──────
    expected_steps = EXPECTED_STEPS_PER_EPOCH * int(epochs)
    terminal_key = f"{TERMINAL_FRAC:.2f}"
    ckpt_index = train_result["checkpoint_index"]
    realized_steps = ckpt_index.get(terminal_key, {}).get("step")
    if realized_steps != expected_steps:
        raise RuntimeError(
            f"[{cell_slug}_seed{seed}] realized terminal step {realized_steps} != expected "
            f"{expected_steps} (epochs={epochs}, "
            f"rows={POS_ROWS + N_NEG_PERSONAS * NEG_ROWS_PER_PERSONA}, "
            f"eff batch {BATCH_SIZE}×{GRAD_ACCUM}) — matched training amounts are load-bearing; "
            "refusing to eval an unmatched cell."
        )

    # ── Gate (g) precondition: band-callback telemetry must EXIST. ───────────
    if not band_traj_path.exists():
        raise RuntimeError(
            f"[{cell_slug}_seed{seed}] band-callback trajectory missing at {band_traj_path} — "
            "the log-only band callback never attached/logged (the #480 paper-mitigation "
            "class). Refusing to continue."
        )
    band_payload = json.loads(band_traj_path.read_text())
    if not band_payload.get("records"):
        raise RuntimeError(
            f"[{cell_slug}_seed{seed}] band-callback trajectory at {band_traj_path} has zero "
            "records — telemetry never fired."
        )
    final_band_delta = float(band_payload["delta_nats"][-1])

    # ── Adapter-config parity (terminal + one mid-run checkpoint). ───────────
    parity = assert_adapter_config_parity(adapter_out)
    first_frac_key = f"{TRAJECTORY_CHECKPOINT_FRACTIONS[0]:.2f}"
    first_ckpt_dir = ckpt_index.get(first_frac_key, {}).get("path")
    if first_ckpt_dir:
        assert_adapter_config_parity(Path(first_ckpt_dir))
    (cell_out / "adapter_parity.json").write_text(json.dumps(parity, indent=2))

    # ── Eval: on-policy trajectory + four-float capture + guards. ────────────
    checkpoint_specs = [
        {"frac": float(k), "step": v.get("step"), "adapter_path": v.get("path")}
        for k, v in ckpt_index.items()
    ]
    held_out_panel = manifest["held_out_panel"]
    q_eval = manifest["q_eval"]
    eval_names = eval_names_for_cell(held_out_panel, spec.panel, extra_eval_personas)
    eval_personas = {p: persona_bank[p] for p in eval_names}
    out_path = cell_out / "trajectory.json"
    run_trajectory_eval_with_guard(
        cell_slug=cell_slug,
        seed=seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=SOURCE_PERSONA,
        source_prompt=persona_bank[SOURCE_PERSONA],
        out_path=out_path,
        base_model=BASE_MODEL,
        max_new_tokens=MAX_NEW_TOKENS_GEN,
        headline_frac=TERMINAL_FRAC,
        # PINNED (plan §4.6): the four-float leaves are written only under
        # compute_kl=True; skipping KL silently drops them and voids gates
        # (g)/(h). The #505 fork's --no-kl smoke speed-up is REMOVED.
        compute_kl=True,
        max_lora_rank=MAX_LORA_RANK_EVAL,
        max_model_len=MAX_MODEL_LEN,
        # #534 adapter-application cross-check: in-loop (teacher-forced band
        # callback) vs off-line (on-policy eval) source read at the terminal
        # fraction; fails loud above tol.
        source_guard_meta={
            "expected_by_frac": {TERMINAL_FRAC: final_band_delta},
            "band_stop_fired": False,
            "tol_nats": SOURCE_MANIFEST_TOL_NATS,
        },
        raw_completions_out_path=cell_out / "raw_completions.json",
    )

    result = {
        "cell_slug": cell_slug,
        "seed": seed,
        "target": spec.target,
        "condition": spec.condition,
        "panel": list(spec.panel),
        "epochs": int(epochs),
        "realized_terminal_step": realized_steps,
        "trajectory_path": str(out_path),
        "band_trajectory_path": str(band_traj_path),
        "train_jsonl": str(train_jsonl),
        "adapter_dir": str(adapter_out),
        "checkpoint_index": ckpt_index,
        "final_band_delta_nats": final_band_delta,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (cell_out / "done.json").write_text(json.dumps(result, indent=2))
    return result


# ── Smoke gates (a)-(h). ─────────────────────────────────────────────────────


def _terminal_checkpoint(payload: dict) -> dict:
    for ck in payload["checkpoints"]:
        if abs(float(ck["frac"]) - TERMINAL_FRAC) < 1e-6:
            return ck
    fracs = [c["frac"] for c in payload["checkpoints"]]
    raise KeyError(f"trajectory has no terminal checkpoint; fracs={fracs}")


def check_smoke_gates_600(
    *,
    trajectory_path: Path,
    band_trajectory_path: Path,
    verify_payload: dict,
    collator_payload: dict,
    checkpoint_index: dict,
    expected_steps: int,
    panel_personas: list[str],
    smoke_out_path: Path,
) -> dict:
    """Run the §4.7 (a)-(h) gates against the written smoke artifacts."""
    payload = json.loads(trajectory_path.read_text())
    ck = _terminal_checkpoint(payload)
    src = ck["source_self"]
    dg = float(src["delta_g_mean"])
    band_low, band_high = SOURCE_DG_BAND_NATS

    held_out = ck["held_out"]
    source = payload["source"]
    bystanders = [p for p in held_out if p != source]
    argmax_flags = [
        bool(leaf["argmax_marker"]) for p in bystanders for leaf in held_out[p].values()
    ]
    bystander_argmax_rate = sum(argmax_flags) / len(argmax_flags) if argmax_flags else 0.0

    # (d) the source's own per-q records ride held_out (villain is in the
    # eval persona list per plan §4.6), so n_marker_in_R is readable.
    if source not in held_out:
        raise AssertionError(
            "smoke trajectory lacks the source's per-q held_out records — the eval "
            "persona list must include the source (plan §4.6)."
        )
    n_marker_in_source_r = sum(
        int(leaf.get("n_marker_in_R", 0)) for leaf in held_out[source].values()
    )

    guard_verdict = ck.get("eval_guard_diagnostic", {}).get("guard_verdict")
    manifest_check = ck.get("source_manifest_check") or {}
    band_records = json.loads(band_trajectory_path.read_text()).get("records", [])
    realized_steps = checkpoint_index.get(f"{TERMINAL_FRAC:.2f}", {}).get("step")

    gates = {
        "gate_a_band": band_low <= dg <= band_high,
        "gate_b_sub_saturation": (
            bystander_argmax_rate < ARGMAX_CEILING
            and float(src["g_logp_mean"]) <= -SOURCE_LOGP_CEILING_EPS_NATS
        ),
        "gate_c_eval_guard_positive_control": guard_verdict == "pass_b_norm_ok",
        "gate_d_no_marker_in_source_R": n_marker_in_source_r == 0,
        "gate_e_collator_mask": collator_payload.get("verdict") == "pass",
        "gate_f_panel_disjointness": verify_payload.get("verdict") == "pass",
        "gate_g_telemetry": (
            len(band_records) >= 1
            and len(checkpoint_index) == len(TRAJECTORY_CHECKPOINT_FRACTIONS)
            and realized_steps == expected_steps
            and bool(payload.get("logit_fields"))
        ),
        "gate_h_offline_vs_inloop_source": manifest_check.get("guard_verdict") == "pass",
    }
    out = {
        "frac": TERMINAL_FRAC,
        "source_dg_mean_nats": dg,
        "source_trained_logp_mean": float(src["g_logp_mean"]),
        "source_emission_p": float(src.get("emission_p", 0.0)),
        "bystander_argmax_rate": bystander_argmax_rate,
        "n_bystander_probes": len(argmax_flags),
        "n_marker_in_source_R": n_marker_in_source_r,
        "expected_band_nats": list(SOURCE_DG_BAND_NATS),
        "bystander_argmax_ceiling": ARGMAX_CEILING,
        "eval_guard_verdict": guard_verdict,
        "source_manifest_check": manifest_check,
        "n_band_trajectory_records": len(band_records),
        "realized_terminal_step": realized_steps,
        "expected_terminal_step": expected_steps,
        "n_checkpoints": len(checkpoint_index),
        "panel_personas": panel_personas,
        "trajectory_path": str(trajectory_path),
        **gates,
        "all_gates_passed": all(gates.values()),
        "floor_failed": dg < band_low,
        "saturation_failed": dg > band_high or not gates["gate_b_sub_saturation"],
    }
    smoke_out_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_out_path.write_text(json.dumps(out, indent=2))
    log.info(
        "[smoke-gate] dg=%.2f band=[%s,%s]; bystander_argmax=%.3f; gates=%s; all=%s",
        dg,
        band_low,
        band_high,
        bystander_argmax_rate,
        {k: v for k, v in gates.items()},
        out["all_gates_passed"],
    )
    return out


# ── Subprocess scheduler (sweep parallelism; CVD pinned in the LAUNCHER env). ─


def _run_cells_subprocess(
    spec_iter: list[tuple[CellSpec600, int]],
    *,
    n_gpus: int,
    max_parallel: int,
    epochs: int,
    manifest_path: Path,
    out_root: Path,
    data_root: Path,
    script_name: str = "i600_run_cell.py",
) -> tuple[list[dict], list[dict]]:
    """Run each (cell, seed) as scripts/i600_run_cell.py pinned to one GPU.

    The CVD pin is exported in the LAUNCHER environment (gotcha #545: any
    import-time cuInit freezes the device list before the in-process clobber)
    AND threaded as --gpu-id so sft.py's in-process clobber rewrites the same
    value. Per-cell logs under ``out_root/logs/``; per-cell completion lines
    deliberately do NOT carry the ``[phase=done]`` token (reserved for the
    dispatcher's single terminal line — incident #545).

    Resume contract (``EPM_SKIP_EXISTING=1``): a (cell, seed) whose
    ``done.json`` AND ``trajectory.json`` both exist is not re-run; its
    persisted ``done.json`` is appended to ``results`` (with
    ``skipped_existing: true`` and path fields re-pointed at the current
    ``out_root``) so downstream phases treat it exactly like a fresh rc=0
    completion. A ``trajectory.json`` without ``done.json`` is an INCOMPLETE
    prior run and is re-run.
    """
    script = _repo_root() / "scripts" / script_name
    if not script.exists():
        raise FileNotFoundError(f"run-cell entrypoint missing at {script}")
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = os.environ.get("EPM_SKIP_EXISTING", "").lower() in {"1", "true", "yes"}
    pending: list[tuple[CellSpec600, int]] = []
    results: list[dict] = []
    for spec, seed in spec_iter:
        cell_dir = out_root / "sweep" / spec.slug / f"seed_{seed}"
        traj = cell_dir / "trajectory.json"
        done_path = cell_dir / "done.json"
        if skip_existing and traj.exists() and done_path.exists():
            # Synthesize a result entry EQUIVALENT to a successful run (rc=0).
            # done.json is the per-cell completion sentinel run_one_cell writes
            # LAST, so its presence == the full build→train→eval chain finished.
            # Without this entry the smoke phase's `if failures or not results`
            # gate misreads a skipped-but-complete cell as a crash (2026-06-11
            # EPM_SKIP_EXISTING relaunch incident) and the sweep undercounts
            # completed cells. Path fields are re-pointed at THIS run's
            # out_root (done.json may store pod-cwd-relative paths from the
            # producing run) so gates (a)-(h) re-evaluate from the persisted
            # artifacts as located NOW — gate (g) reads the persisted
            # band-trajectory records, never a live WandB run.
            entry = json.loads(done_path.read_text())
            entry["trajectory_path"] = str(traj)
            entry["band_trajectory_path"] = str(cell_dir / "band_trajectory.json")
            entry["skipped_existing"] = True
            results.append(entry)
            log.info(
                "[skip-existing] %s_seed%d: done.json + trajectory.json present — "
                "synthesized completed result; gates re-evaluate from disk",
                spec.slug,
                seed,
            )
            continue
        if skip_existing and traj.exists():
            log.warning(
                "[skip-existing] %s_seed%d: trajectory.json present but done.json MISSING — "
                "prior run incomplete; re-running the cell",
                spec.slug,
                seed,
            )
        pending.append((spec, seed))

    max_parallel = max(1, min(max_parallel, n_gpus))
    running: dict[int, tuple[subprocess.Popen, CellSpec600, int, Path]] = {}
    failures: list[dict] = []
    next_idx = 0
    while next_idx < len(pending) or running:
        # Launch onto free GPUs.
        free_gpus = [g for g in range(n_gpus) if g not in running]
        while next_idx < len(pending) and free_gpus and len(running) < max_parallel:
            gpu = free_gpus.pop(0)
            spec, seed = pending[next_idx]
            next_idx += 1
            cell_log = log_dir / f"{spec.slug}_seed{seed}.log"
            cmd = [
                sys.executable,
                str(script),
                "--cell",
                spec.slug,
                "--seed",
                str(seed),
                "--gpu-id",
                str(gpu),
                "--epochs",
                str(epochs),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(out_root),
                "--data-root",
                str(data_root),
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            log.info(
                "[launch] %s_seed%d on GPU %d → %s (cmd: %s)",
                spec.slug,
                seed,
                gpu,
                cell_log,
                shlex.join(cmd),
            )
            with open(cell_log, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, spec, seed, cell_log)
        # Reap finished.
        for gpu in list(running):
            proc, spec, seed, cell_log = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            del running[gpu]
            done_path = out_root / "sweep" / spec.slug / f"seed_{seed}" / "done.json"
            if rc == 0 and done_path.exists():
                results.append(json.loads(done_path.read_text()))
                log.info("[cell-complete] %s_seed%d rc=0", spec.slug, seed)
            else:
                failures.append(
                    {"cell_slug": spec.slug, "seed": seed, "rc": rc, "log": str(cell_log)}
                )
                log.error(
                    "[cell-FAILED] %s_seed%d rc=%s — see %s (tail):", spec.slug, seed, rc, cell_log
                )
                try:
                    tail = cell_log.read_text().splitlines()[-25:]
                    for line in tail:
                        log.error("    %s", line)
                except OSError:
                    pass
        if running:
            time.sleep(10)
    return results, failures


# ── Uploads (plan §4.9; Upload Policy). ──────────────────────────────────────


def _upload_phase(out_root: Path, data_root: Path, manifest_path: Path) -> None:
    """Training JSONLs + manifests + panel_selection → HF data repo; raw completions too.

    Adapters were already uploaded inline by train_lora (cfg.hf_upload=True,
    per-cell hf_path_in_repo). Fail-loud throughout (Upload Policy).
    """
    from explore_persona_space.orchestrate.hub import (
        _upload,
        upload_dataset_directory,
        upload_raw_completions_to_data_repo,
    )

    log.info("[phase=upload] training JSONLs from %s", data_root)
    upload_dataset_directory(data_root, f"{HF_DATA_PREFIX}/training_data", pattern="*.jsonl")
    upload_dataset_directory(
        data_root, f"{HF_DATA_PREFIX}/training_data", pattern="*.manifest.json"
    )
    log.info("[phase=upload] design manifest %s", manifest_path)
    url = _upload(
        local_path=manifest_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_DATA_PREFIX}/panel_selection.json",
        # upload_as_file is load-bearing: without it a FILE path falls into the
        # upload_folder branch, verification looks for a folder prefix, finds 0
        # files, and returns "" (smoke crash 2026-06-11 on pod-600).
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"panel_selection.json upload returned empty URL ({manifest_path}).")
    log.info("[phase=upload] raw completions under %s", out_root)
    upload_raw_completions_to_data_repo(experiment_name=HF_DATA_PREFIX, eval_results_dir=out_root)


# ── Pod sentinel (poll_pipeline.py contract). ────────────────────────────────


def _write_results_sentinel(note_payload: dict, *, out_root: Path) -> Path:
    """Write the end-of-run sentinel with poll_pipeline's required keys."""
    sentinel_dir = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    if not sentinel_dir.is_dir():
        fallback = out_root / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        log.warning(
            "[sentinel] %s missing (not on a pod?) — writing sentinel to %s",
            sentinel_dir,
            fallback,
        )
        sentinel_dir = fallback
    path = sentinel_dir / f"issue-600-epm_results-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 600,
        "by": "i600_dispatch",
        "ts": datetime.now(UTC).isoformat(),
        "note": json.dumps(note_payload, indent=2),
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("[sentinel] wrote %s", path)
    return path


# ── Main. ────────────────────────────────────────────────────────────────────


def _resolve_cells(cells_arg: str, specs: tuple[CellSpec600, ...]) -> list[CellSpec600]:
    """Resolve --cells: 'all', an integer cap, or comma-separated slugs."""
    if cells_arg == "all":
        return list(specs)
    tokens = [t.strip() for t in cells_arg.split(",") if t.strip()]
    if len(tokens) == 1 and tokens[0].isdigit():
        return list(specs[: int(tokens[0])])
    by_slug = {s.slug: s for s in specs}
    unknown = [t for t in tokens if t not in by_slug]
    if unknown:
        raise KeyError(f"--cells contains unknown slugs {unknown}; registry: {sorted(by_slug)}")
    return [by_slug[t] for t in tokens]


def _resolve_seeds(seeds_arg: str) -> list[int]:
    """Resolve --seeds: comma-separated seed VALUES (must be ⊆ the pinned SEEDS)."""
    seeds = [int(t) for t in seeds_arg.split(",") if t.strip()]
    unknown = [s for s in seeds if s not in SEEDS]
    if unknown:
        raise ValueError(f"--seeds contains non-registered seeds {unknown}; pinned: {SEEDS}")
    return seeds


def main(
    *,
    smoke: bool,
    cells: str,
    seeds: str,
    n_gpus: int,
    max_parallel: int,
    epochs: int,
    plan_only: bool = False,
    no_upload: bool = False,
) -> int:
    """Run the unified §4.7 smoke=sweep pipeline. Returns the shell exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    # uv run python does NOT auto-load .env; subprocesses inherit THIS env
    # (the #397 round-10' dispatcher-env incident class).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    log.info(
        "[phase=start] smoke=%s cells=%r seeds=%r n_gpus=%d max_parallel=%d epochs=%d "
        "plan_only=%s host=%s",
        smoke,
        cells,
        seeds,
        n_gpus,
        max_parallel,
        epochs,
        plan_only,
        socket.gethostname(),
    )
    out_root = _output_root()
    data_root = _data_root()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # ── Phase 0a: marker tokenizer invariant (in-process, before anything). ──
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_tokenization(tokenizer)
    log.info("[phase=marker_check] %r → id=%d (OK)", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    # ── Phase 0b: inherited artifacts + design manifest. ─────────────────────
    _prefetch_inherited_artifacts(_i472_data_root())
    persona_bank, _r_train, _q_train = _load_bank_and_r_train()
    log.info("[phase=load_bank] %d personas", len(persona_bank))

    manifest_path = _manifest_path()
    manifest = load_manifest(manifest_path)
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        _content_hash,
    )

    bank_hash = _content_hash(persona_bank)
    if bank_hash != manifest["bank_content_hash"]:
        raise RuntimeError(
            f"persona-bank content hash {bank_hash[:12]} != manifest's "
            f"{manifest['bank_content_hash'][:12]} — the committed design manifest was "
            "selected against a DIFFERENT bank; refusing to train."
        )
    specs = cell_specs_from_manifest(manifest)
    log.info("[phase=manifest] %d cells registered; bank hash OK", len(specs))

    # ── Phase 1: spec_iter — THE single cell-list every later phase derives from.
    if smoke:
        smoke_slug = first_near_slug(specs)
        spec_iter = [(s, SEEDS[0]) for s in specs if s.slug == smoke_slug]
    else:
        cell_list = _resolve_cells(cells, specs)
        seed_list = _resolve_seeds(seeds)
        spec_iter = [(spec, seed) for spec in cell_list for seed in seed_list]
    log.info(
        "[phase=plan] %d (cell, seed) pairs: %s",
        len(spec_iter),
        [(s.slug, seed) for s, seed in spec_iter],
    )
    if plan_only:
        _write_results_sentinel(
            {
                "mode": "plan_only",
                "smoke": smoke,
                "n_pairs": len(spec_iter),
                "pairs": [(s.slug, seed) for s, seed in spec_iter],
            },
            out_root=out_root,
        )
        log.info("[phase=done] plan-only: %d pairs validated, nothing launched", len(spec_iter))
        return 0

    # ── Phase 2: per-(cell, seed) subprocesses (same shape, smoke and sweep). ─
    log.info("[phase=train_eval_start] %d pairs", len(spec_iter))
    results, failures = _run_cells_subprocess(
        spec_iter,
        n_gpus=n_gpus,
        max_parallel=max_parallel,
        epochs=epochs,
        manifest_path=manifest_path,
        out_root=out_root,
        data_root=data_root,
    )

    # ── Phase 3: smoke gates (smoke only). ───────────────────────────────────
    gate_payload: dict | None = None
    if smoke:
        if failures or not results:
            log.error("[phase=smoke_gate_fail] smoke cell crashed: %s", failures)
            _write_results_sentinel(
                {"mode": "smoke", "verdict": "CRASH", "failures": failures},
                out_root=out_root,
            )
            return 2
        r = results[0]
        cell_dir = out_root / "sweep" / r["cell_slug"] / f"seed_{r['seed']}"
        gate_payload = check_smoke_gates_600(
            trajectory_path=Path(r["trajectory_path"]),
            band_trajectory_path=Path(r["band_trajectory_path"]),
            verify_payload=json.loads((cell_dir / "panel_verify.json").read_text()),
            collator_payload=json.loads((cell_dir / "collator_gate.json").read_text()),
            checkpoint_index=r["checkpoint_index"],
            expected_steps=EXPECTED_STEPS_PER_EPOCH * int(epochs),
            panel_personas=r["panel"],
            smoke_out_path=out_root / "smoke" / "smoke_gate.json",
        )

    # ── Phase 4: uploads (the artifacts THIS run produced). ──────────────────
    if no_upload:
        log.warning("[phase=upload] SKIPPED (--no-upload; local smoke only)")
    else:
        _upload_phase(out_root, data_root, manifest_path)

    # ── Phase 5: sentinel + terminal phase line. ─────────────────────────────
    note = {
        "mode": "smoke" if smoke else "sweep",
        "epochs": epochs,
        "n_pairs": len(spec_iter),
        "n_completed": len(results),
        "n_skipped_existing": sum(1 for r in results if r.get("skipped_existing")),
        "failures": failures,
        "smoke_gate": gate_payload,
        "output_root": str(out_root),
    }
    _write_results_sentinel(note, out_root=out_root)

    if smoke and gate_payload is not None and not gate_payload["all_gates_passed"]:
        log.error(
            "[phase=smoke_gate_fail] %s",
            {k: v for k, v in gate_payload.items() if k.startswith("gate_")},
        )
        return 2
    if failures:
        log.error(
            "[phase=cells_failed] %d of %d pairs failed: %s",
            len(failures),
            len(spec_iter),
            failures,
        )
        return 4
    log.info("[phase=done] all %d (cell, seed) pairs finished successfully", len(spec_iter))
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """argparse entrypoint (used by ``scripts/i600_dispatch.py``)."""
    p = argparse.ArgumentParser(description="Task #600 unified smoke=sweep dispatcher")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="§4.7 smoke: ONE (cell, seed) — the first NEAR cell at seed 42 — through the "
        "identical subprocess path, then gates (a)-(h).",
    )
    p.add_argument(
        "--cells",
        type=str,
        default="all",
        help="'all', an integer cap, or comma-separated c600_* slugs.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seed VALUES (⊆ pinned {42,137,219}).",
    )
    p.add_argument("--n-gpus", type=int, default=8, help="Physical GPUs available.")
    p.add_argument("--max-parallel", type=int, default=8, help="Concurrent cells cap.")
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT,
        help="§4.7 smoke-ladder re-pin (1→2→3); ONE value for ALL cells (matched steps).",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate marker/bank/manifest + print the launch plan; spawn nothing "
        "(local dry-run; exercises the cell-iteration plumbing + sentinel writer).",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the HF upload phase (LOCAL runs only — pods must upload).",
    )
    args = p.parse_args(argv)
    return main(
        smoke=args.smoke,
        cells=args.cells,
        seeds=args.seeds,
        n_gpus=args.n_gpus,
        max_parallel=args.max_parallel,
        epochs=args.epochs,
        plan_only=args.plan_only,
        no_upload=args.no_upload,
    )


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
