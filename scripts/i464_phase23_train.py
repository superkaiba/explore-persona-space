"""Phase 2 (smoke) + Phase 3 (sweep) — train ONE LoRA per (arm, seed) with
marker-only loss over the 2-persona MIX (issue #464 plan v2 §4.1 + §4.4).

Per (arm, seed) LoRA:
  * Read R_canon[persona, q] from data/issue_464/R_canon_train.json
    (Phase 1 output; MF-B(1) — SAME R across all arms).
  * Build positive rows for BOTH personas mixed (30 Q_train x 2 personas
    x N_DUPES_POS dupes = 600 default rows). Each row's prompt+completion
    is constructed by BUILD_TRAIN_PROMPT_AND_COMPLETION(arm, persona, q,
    R_canon, tok) — see i464_encodings.py.
  * Train with marker_only_loss=True + tail_tokens=0 + multi-marker
    collator (issue #464 patch: list[str] of marker texts), so loss
    lands ONLY on each persona's own marker token (+ EOS).
  * Hyperparameters inherited from #460 (lr=1e-5, 5 epochs, bs=4 x
    grad_accum=4, r=32, alpha=64, dropout=0.05).
  * Optional MF-C trajectory callback (every 10% of steps) — wired when
    --traj-probe-file is passed.

Phase 2 smoke uses the SAME script with --conds system_plain_seed42 and
no other flags (REAL recipe — same epochs, dupes, hyperparams). The
dispatcher then invokes scripts/i464_phase2_smoke_check.py as a separate
process for the implant gate (vLLM-after-HF GPU conflict mitigation —
CLAUDE.md task #399).

CLI:
    # Phase 2 smoke (real recipe; smoke gate runs separately afterward):
    uv run python scripts/i464_phase23_train.py --cell system_plain_seed42

    # Single sweep cell:
    uv run python scripts/i464_phase23_train.py --cell role_seed137 --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_train_answers,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

load_dotenv()

logger = logging.getLogger("i464.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"

# Plan §4.1 Phase 3: 30 q x 2 personas x N_DUPES_POS dupes = 600 rows / LoRA.
N_DUPES_POS = 10
LOCAL_DATA_DIR = Path("data/issue_464")
TRAIN_ROW_DIR = Path("data/issue_464/train_rows")
# Synthetic key for the default-assistant negative encoding (must match
# scripts/i464_cn_generate_R_default.py::DEFAULT_KEY).
DEFAULT_NEG_KEY = "default"

SEEDS_BY_ISSUE: dict[int, tuple[int, ...]] = {
    # #464 parent (3 seeds, recipe inherited).
    464: (42, 137, 1337),
    # #529: 5 seeds (statistical-power bump per plan §12 Assumption 8;
    # 7/21 added on top of #464's 42/137/1337). Seed list is the only
    # accepted set when --issue 529 is passed.
    529: (42, 137, 1337, 7, 21),
    # #533: lr=5e-6 corrective re-run of #529's grid — single-variable
    # change (lr), so the seed set / epoch suffix / HF-prefix shape all
    # mirror #529's.
    533: (42, 137, 1337, 7, 21),
}
# Legacy alias preserved for any external importer that referenced
# ``SEEDS`` directly (none in-repo, but a thin-wrapper path could rely
# on it).
SEEDS = SEEDS_BY_ISSUE[464]


def _parse_cell(cell: str, issue: int = 464) -> tuple[enc.Arm, int]:
    """Parse 'arm_seedSEED' → (arm, seed). Raises on malformed input.

    ``issue`` selects which seed set is accepted (see
    ``SEEDS_BY_ISSUE``). The seed must be in that set; otherwise raises
    so an off-by-one seed never silently lands at the wrong HF subpath.
    """
    if "_seed" not in cell:
        raise ValueError(f"--cell {cell!r} must look like 'arm_seed42'")
    arm, seed_str = cell.rsplit("_seed", 1)
    if arm not in enc.ARMS:
        raise ValueError(f"unknown arm {arm!r} in --cell {cell!r}; valid: {enc.ARMS}")
    try:
        seed = int(seed_str)
    except ValueError as e:
        raise ValueError(f"--cell {cell!r}: seed part {seed_str!r} is not int") from e
    seed_set = SEEDS_BY_ISSUE.get(issue)
    if seed_set is None:
        raise ValueError(
            f"--issue {issue} has no registered seed set; want one of {list(SEEDS_BY_ISSUE)}"
        )
    if seed not in seed_set:
        raise ValueError(f"--cell {cell!r}: seed {seed} not in --issue {issue} seed set {seed_set}")
    return arm, seed  # type: ignore[return-value]


def _load_R_canon(split: str) -> dict[str, dict[str, dict]]:
    """Load R_canon for ``split`` in {'train', 'test'}; HF fallback or local override.

    Override via ``EPM_LOCAL_R_CANON_DIR``: when set, read
    ``<override>/R_canon_<split>.json`` directly. RAISE if env is
    set but file missing — never silently fall through to HF (the
    override is for `--no-upload` smoke isolation; silent HF
    fallback would defeat it). Production behavior (env unset)
    unchanged.
    """
    override_dir = os.environ.get("EPM_LOCAL_R_CANON_DIR")
    if override_dir:
        override_path = Path(override_dir) / f"R_canon_{split}.json"
        if not override_path.exists():
            raise RuntimeError(
                f"EPM_LOCAL_R_CANON_DIR={override_dir!r} set but R_canon_{split}.json "
                f"missing at {override_path}."
            )
        logger.info("Using local R_canon override (split=%s): %s", split, override_path)
        local = override_path
    else:
        local = LOCAL_DATA_DIR / f"R_canon_{split}.json"
        if not local.exists():
            logger.info("R_canon_%s.json missing locally; pulling from HF data repo.", split)
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_R_PATH_PREFIX}/R_canon_{split}.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(
                    f"HF download claimed success but {local} is missing/empty (src {downloaded})."
                )

    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(
            f"R_canon_{split}.json schema_version={payload.get('schema_version')!r}, "
            f"expected 'i464_v2_matched_R' — refuse to mix R versions."
        )
    return payload["completions"]


def _load_R_canon_default_train() -> dict[str, dict[str, dict]]:
    """Load R_canon for the default-assistant TRAIN encoding (cn-only).

    Reads ``data/issue_464/R_canon_default_train.json`` (the artifact
    produced by ``scripts/i464_cn_generate_R_default.py``). HF fallback
    points at the same data-repo prefix so a pod that hasn't run the
    generator can still pull the artifact.

    Returns a shape-matched ``{"default": {q: {response_text, ...}}}``
    dict the cn negative-row builder can merge with the per-persona
    R_canon_train map.

    Raises:
        RuntimeError if neither the local file nor the HF copy is present.
        AssertionError if the schema_version drifts.
    """
    override_dir = os.environ.get("EPM_LOCAL_R_CANON_DIR")
    if override_dir:
        override_path = Path(override_dir) / "R_canon_default_train.json"
        if not override_path.exists():
            raise RuntimeError(
                f"EPM_LOCAL_R_CANON_DIR={override_dir!r} set but "
                f"R_canon_default_train.json missing at {override_path}."
            )
        logger.info("Using local R_canon_default_train override: %s", override_path)
        local = override_path
    else:
        local = LOCAL_DATA_DIR / "R_canon_default_train.json"
        if not local.exists():
            logger.info("R_canon_default_train.json missing locally; pulling from HF data repo.")
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_R_PATH_PREFIX}/R_canon_default_train.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(
                    f"HF download claimed success but {local} is missing/empty (src {downloaded})."
                )

    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_cn_default_R_v1":
        raise AssertionError(
            f"R_canon_default_train.json schema_version={payload.get('schema_version')!r}, "
            f"expected 'i464_cn_default_R_v1' — refuse to mix R versions."
        )
    completions = payload["completions"]
    if DEFAULT_NEG_KEY not in completions:
        raise AssertionError(
            f"R_canon_default_train.json missing top-level key {DEFAULT_NEG_KEY!r}; "
            f"keys={list(completions.keys())}"
        )
    return completions


def _build_negative_row(
    arm: enc.Arm,
    neg_encoding: str,
    q: str,
    R_neg: str,
    tokenizer,
) -> tuple[str, str]:
    """Build (prompt, completion) for ONE marker-less contrastive-negative row.

    ``neg_encoding`` is either a persona name (``pirate``/``villain`` — the
    OTHER persona vs the cell's target) OR ``DEFAULT_NEG_KEY`` (``"default"``
    — the bare default-assistant encoding).

    - Persona negative: uses ``BUILD_TRAIN_PROMPT_AND_COMPLETION(arm,
      neg_persona, q, R_neg, tok)`` for the prompt (the helper would
      normally append the neg persona's marker; we discard the
      helper-built completion and rebuild it with NO marker so the only
      loss-bearing token under ``MarkerOnlyDataCollator(tail_tokens=0)``
      becomes EOS at the post-response slot).
    - Default negative: uses ``BUILD_EVAL_PROMPT('default_assistant', q,
      tok)`` for the prompt — the bare neutral-system + plain
      ``<|im_start|>assistant\\n`` chat-template prefix, regardless of
      the positive's arm. This matches the brief's "negatives use [...]
      the default ``<|im_start|>assistant`` header" clause: the default
      negative is the canonical safety target (leakage-to-default).

    The completion has NO marker appended in either case. Under the cn
    train path's collator (``MarkerOnlyDataCollator`` constructed with
    ``marker_text=[' ※']`` and ``tail_tokens=0``), a row whose
    ``input_ids`` does NOT contain ``MARKER_PIRATE_ID`` is treated as a
    negative and the only loss-bearing token is EOS at the post-response
    slot — i.e. the row explicitly trains "after a response under THIS
    encoding, emit EOS, NOT  ※."

    Returns ``(prompt_text, completion_text)`` where completion is
    exactly ``R_neg`` (no marker).
    """
    if neg_encoding == DEFAULT_NEG_KEY:
        prompt_text = enc.BUILD_EVAL_PROMPT("default_assistant", q, tokenizer)
    else:
        # Treat neg_encoding as a Persona name; build via the same helper
        # the positives use, then discard the helper's marker.
        if neg_encoding not in enc.PERSONAS:
            raise ValueError(
                f"neg_encoding={neg_encoding!r} not in PERSONAS={enc.PERSONAS} and "
                f"!= {DEFAULT_NEG_KEY!r}"
            )
        prompt_text, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
            arm,
            neg_encoding,
            q,
            R_neg,
            tokenizer,  # type: ignore[arg-type]
        )
    # Negative completion: response only, NO marker. Whitespace contract:
    # the helper-built positive completion is "{R}{marker}" with no
    # boundary char between R and marker; the negative is just "{R}".
    completion_text = R_neg
    return prompt_text, completion_text


def _build_training_rows(  # noqa: C901 - cn follow-up added contrastive-negatives branch pushes complexity to ~22
    arm: enc.Arm,
    seed: int,
    q_train_answers: dict[str, str],
    R_canon_train: dict[str, dict[str, dict]],
    tokenizer,
    n_dupes: int,
    single_persona: enc.Persona | None = None,
    shared_marker: bool = False,
    contrastive_negatives: bool = False,
    R_canon_default_train: dict[str, dict[str, dict]] | None = None,
    issue_prefix: str = "i464",
    epoch_suffix: str = "",
) -> Path:
    """Build the 30 x P x n_dupes rows for ONE cell and write JSONL.

    Row shape (prompt-completion STRING format):
        {"prompt": "<chat-template prefix ending at role-open>",
         "completion": "<R_canon[persona, q]><marker_text>"}

    Default behavior (single_persona=None, shared_marker=False) — UNCHANGED
    from the original #464 sweep: 30 q x 2 personas x n_dupes rows where
    each persona's row carries that persona's OWN marker (pirate→ ※,
    villain→ ¶).

    Positive-only single-persona follow-up
    (single_persona ∈ {pirate, villain}, shared_marker=True):
        - Only ``single_persona``'s 30 questions are emitted (one persona
          per LoRA — co-residence removed).
        - EVERY row's completion is suffixed with the SHARED pirate marker
          ` ※` (id 83399), even villain rows. The collator is passed
          ``[" ※"]`` so the loss-bearing slot is uniquely ` ※`+EOS.

    Contrastive-negatives follow-up (cn — single_persona + shared_marker +
    contrastive_negatives=True; requires R_canon_default_train):
        - POSITIVES: 30 q x n_dupes rows of (target persona prompt + R +
          ` ※`) — IDENTICAL to the positive-only path above.
        - NEGATIVES (marker-LESS): 30 q x (n_dupes // 2) rows for EACH of
          two negative encodings (must split cleanly, so n_dupes is required
          to be even when contrastive_negatives is set):
            (a) OTHER persona, SAME arm: ``BUILD_TRAIN_PROMPT_AND_COMPLETION
                (arm, other_persona, q, R_canon[other], tok)`` prompt, then
                completion replaced by ``R_canon[other, q]`` with NO marker
                appended.
            (b) DEFAULT assistant: ``BUILD_EVAL_PROMPT('default_assistant',
                q, tok)`` prompt, completion = ``R_canon[default, q]`` with
                NO marker appended.
          With ``MarkerOnlyDataCollator(marker_text=[' ※'], tail_tokens=0)``
          a row whose input_ids do NOT contain MARKER_PIRATE_ID is treated
          as a negative and the only loss-bearing token is EOS at the
          post-response slot — i.e. each negative row explicitly trains
          "after a response under THIS encoding, emit EOS, NOT ※."
        - Total: 30 q x n_dupes positives + 30 q x (n_dupes//2) other-neg +
          30 q x (n_dupes//2) default-neg = 600 rows / cell at n_dupes=10
          (300 + 150 + 150), mirroring the parent #464 sweep's 600 rows.

    Marker count == 1 per POSITIVE row (asserted on the first row); under
    ``shared_marker``, the positive's loss-bearing token id is asserted to
    be ``MARKER_PIRATE_ID`` (83399). Under ``contrastive_negatives``, each
    negative row's completion is additionally asserted to contain ZERO
    copies of MARKER_PIRATE_ID (would otherwise collapse the contrast).
    """
    if shared_marker and single_persona is None:
        raise ValueError("--shared-marker requires --single-persona; use one persona per LoRA")
    if contrastive_negatives:
        if not (shared_marker and single_persona is not None):
            raise ValueError(
                "--contrastive-negatives requires --shared-marker AND --single-persona"
            )
        if R_canon_default_train is None:
            raise ValueError(
                "--contrastive-negatives requires R_canon_default_train (loaded via "
                "_load_R_canon_default_train())"
            )
        if DEFAULT_NEG_KEY not in R_canon_default_train:
            raise AssertionError(
                f"R_canon_default_train missing key {DEFAULT_NEG_KEY!r}; "
                f"keys={list(R_canon_default_train.keys())}"
            )
        if n_dupes % 2 != 0:
            raise ValueError(
                f"--contrastive-negatives requires an EVEN n_dupes (so the "
                f"per-negative-encoding dupe count is an integer); got n_dupes={n_dupes}"
            )
    questions = sorted(q_train_answers.keys())
    if len(questions) == 0:
        raise AssertionError("q_train_answers is empty — cannot build training rows.")
    if len(questions) != 30:
        # Real-recipe path uses exactly 30 (Q_train); the smoke/CPU path
        # truncates intentionally. Warn so a misconfigured pod run is
        # visible in the log, but do not abort.
        logger.warning(
            "Expected 30 Q_train questions, got %d (smoke or CPU-stub mode?).",
            len(questions),
        )

    if single_persona is not None:
        if single_persona not in enc.PERSONAS:
            raise ValueError(f"--single-persona {single_persona!r} not in {enc.PERSONAS}")
        active_personas: tuple[enc.Persona, ...] = (single_persona,)
    else:
        active_personas = enc.PERSONAS

    rows: list[dict] = []
    sanity_count = {p: 0 for p in active_personas}
    for persona in active_personas:
        if persona not in R_canon_train:
            raise AssertionError(f"R_canon_train missing persona={persona!r}")
        # Under shared_marker, the row's marker is ALWAYS the pirate
        # marker ( ※ id 83399) regardless of the row's training persona.
        # Without shared_marker, each persona keeps its own marker.
        row_marker_text = enc.MARKER_PIRATE_TEXT if shared_marker else enc.marker_text_for(persona)
        row_marker_id = enc.MARKER_PIRATE_ID if shared_marker else enc.marker_id_for(persona)
        for q in questions:
            if q not in R_canon_train[persona]:
                raise AssertionError(f"R_canon_train[{persona}] missing q={q!r}")
            R = R_canon_train[persona][q]["response_text"]
            # BUILD_TRAIN_PROMPT_AND_COMPLETION normally appends the persona's
            # OWN marker. For shared_marker we ignore the helper's marker
            # suffix and rebuild the completion with the shared marker.
            prompt_text, completion_text = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
                arm, persona, q, R, tokenizer
            )
            if shared_marker:
                completion_text = f"{R}{row_marker_text}"
            # Tokenization sanity (first row per persona): the chosen marker
            # is present exactly once AND is the row's loss-bearing token id.
            if sanity_count[persona] < 1:
                full_ids = tokenizer.encode(
                    prompt_text + completion_text + "<|im_end|>\n",
                    add_special_tokens=False,
                )
                cnt = full_ids.count(row_marker_id)
                if cnt != 1:
                    raise AssertionError(
                        f"arm={arm} persona={persona} shared_marker={shared_marker}: "
                        f"tokenized row has {cnt} copies of marker id {row_marker_id}, "
                        f"expected 1. First 80 ids: {full_ids[:80]}"
                    )
                if shared_marker:
                    # Loss-bearing token under MarkerOnlyDataCollator with
                    # tail_tokens=0 is the marker token (and EOS); the
                    # marker id MUST equal MARKER_PIRATE_ID, never the
                    # per-persona marker that would be ` ¶` for villain.
                    completion_only_ids = tokenizer.encode(
                        completion_text, add_special_tokens=False
                    )
                    if enc.MARKER_PIRATE_ID not in completion_only_ids:
                        raise AssertionError(
                            f"shared-marker villain row's completion does NOT "
                            f"contain MARKER_PIRATE_ID={enc.MARKER_PIRATE_ID}; "
                            f"completion ids: {completion_only_ids[-10:]}"
                        )
                sanity_count[persona] += 1
            row = {"prompt": prompt_text, "completion": completion_text}
            for _ in range(n_dupes):
                rows.append(row)

    # ── Contrastive-negatives rows (cn only) ────────────────────────────
    # Built AFTER all positives so the JSONL has positives-then-negatives;
    # ordering doesn't affect training (SFTTrainer shuffles), it just
    # makes the row file easier to eyeball.
    n_pos_rows = len(rows)
    n_neg_other_rows = 0
    n_neg_default_rows = 0
    if contrastive_negatives:
        # The 2-of-2 type-narrowing for the type checker / readability;
        # the guards above already raise if either is wrong.
        assert single_persona is not None
        assert R_canon_default_train is not None
        target_persona = single_persona
        other_persona: enc.Persona = "villain" if target_persona == "pirate" else "pirate"
        n_dupes_neg = n_dupes // 2  # 5 by default (n_dupes=10 → 5+5)
        # Fail-loud if either negative encoding is missing per-q R_canon.
        if other_persona not in R_canon_train:
            raise AssertionError(
                f"contrastive-negatives: R_canon_train missing other persona "
                f"{other_persona!r}; keys={list(R_canon_train.keys())}"
            )
        # Sanity-check the first row per negative encoding: marker id MUST
        # NOT appear in completion (otherwise the row would be mis-typed as
        # a positive by the collator's marker-substring search).
        for neg_encoding, R_source in (
            (other_persona, R_canon_train[other_persona]),
            (DEFAULT_NEG_KEY, R_canon_default_train[DEFAULT_NEG_KEY]),
        ):
            sanity_emitted = False
            for q in questions:
                if q not in R_source:
                    raise AssertionError(
                        f"contrastive-negatives: R_canon for neg_encoding={neg_encoding!r} "
                        f"missing q={q!r}"
                    )
                R_neg = R_source[q]["response_text"]
                prompt_text, completion_text = _build_negative_row(
                    arm, neg_encoding, q, R_neg, tokenizer
                )
                if not sanity_emitted:
                    # Completion MUST be free of MARKER_PIRATE_ID — the
                    # row's only loss-bearing token under tail_tokens=0
                    # is EOS, and ANY ※ id in the completion would
                    # cause the collator to misclassify the row.
                    completion_only_ids = tokenizer.encode(
                        completion_text, add_special_tokens=False
                    )
                    cnt = completion_only_ids.count(enc.MARKER_PIRATE_ID)
                    if cnt != 0:
                        raise AssertionError(
                            f"contrastive-negatives: neg_encoding={neg_encoding!r} "
                            f"row's completion contains MARKER_PIRATE_ID "
                            f"({enc.MARKER_PIRATE_ID}) {cnt} time(s); negatives must "
                            f"be marker-free. completion tail: {completion_only_ids[-10:]}"
                        )
                    # And the FULL tokenization MUST contain zero ※ as
                    # well (defense-in-depth: the prompt builder also
                    # never adds the marker).
                    full_ids = tokenizer.encode(
                        prompt_text + completion_text + "<|im_end|>\n",
                        add_special_tokens=False,
                    )
                    if full_ids.count(enc.MARKER_PIRATE_ID) != 0:
                        raise AssertionError(
                            f"contrastive-negatives: neg_encoding={neg_encoding!r} "
                            f"FULL row contains MARKER_PIRATE_ID — refusing to write."
                        )
                    sanity_emitted = True
                row = {"prompt": prompt_text, "completion": completion_text}
                for _ in range(n_dupes_neg):
                    rows.append(row)
            if neg_encoding == DEFAULT_NEG_KEY:
                n_neg_default_rows += len(questions) * n_dupes_neg
            else:
                n_neg_other_rows += len(questions) * n_dupes_neg

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    if contrastive_negatives:
        suffix = f"_cn_{single_persona}"
    elif single_persona is not None:
        suffix = f"_{single_persona}"
    else:
        suffix = ""
    out_path = TRAIN_ROW_DIR / f"{issue_prefix}_{arm}_seed{seed}{suffix}{epoch_suffix}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if contrastive_negatives:
        logger.info(
            "cell=%s_seed%d%s wrote %d rows (-> %s); "
            "breakdown pos=%d neg_other(%s)=%d neg_default=%d (shared_marker=True, cn=True)",
            arm,
            seed,
            suffix,
            len(rows),
            out_path,
            n_pos_rows,
            "villain" if single_persona == "pirate" else "pirate",
            n_neg_other_rows,
            n_neg_default_rows,
        )
    else:
        logger.info(
            "cell=%s_seed%d%s wrote %d rows (-> %s); persona breakdown 30 x %d x %d "
            "(shared_marker=%s)",
            arm,
            seed,
            suffix,
            len(rows),
            out_path,
            len(active_personas),
            n_dupes,
            shared_marker,
        )
    return out_path


def _build_traj_probe_file(
    tokenizer,
    R_canon_test: dict[str, dict[str, dict]],
    arm: enc.Arm,
    n_probes_per_key: int,
    out_path: Path,
) -> Path:
    """Build the frozen MF-C trajectory probe slice for ``arm`` and write JSON.

    Slice = n_probes_per_key questions x 2 personas x 3 eval encodings =
    n_probes_per_key x 6 probes per callback firing. Encodings cover the
    arm's own family + the wrong-persona encoding for the other arm
    (gives a within-training read on whether segmentation is forming).
    """
    # Pick a stable subset of Q_test (first n_probes_per_key after sort).
    qs_all = sorted(next(iter(R_canon_test.values())).keys())
    qs = qs_all[:n_probes_per_key]
    probes = []
    # Round-2 fix (review blocker #6): include the symmetric WRONG-persona
    # cells (system_OTHER + role_OTHER) so the trajectory diagnoses the
    # LEAKAGE dynamics MF-C was designed to expose. Round-1 only covered
    # same-persona system/role + default_assistant, which only reads
    # elicitation/identity — not the segmentation question. Now 7 encodings
    # per persona: own-system, own-role, own-role_nonsense, WRONG-system,
    # WRONG-role, WRONG-role_nonsense, default_assistant. R_canon splice
    # uses the persona implied by the eval encoding (matches Phase 4's
    # persona_for_eval_encoding) so the post-R slot is consistent with
    # what cross-eval probes.
    #
    # role_nonsense + role_mismatch follow-up arms: parallel symmetric cells
    # added so the trajectory diagnoses whether the three role-family arms
    # (role / role_nonsense / role_mismatch) follow distinguishable leakage
    # dynamics across training. role_mismatch is the real-but-mismatched-
    # meaning ablation — together with role (matched meaning) and
    # role_nonsense (no meaning) it isolates whether the role-name needs
    # to MATCH the trained content, just be MEANINGFUL, or whether the
    # slot/position alone does the work.
    e_choices_for: dict[enc.Persona, list[enc.EvalEncoding]] = {
        "pirate": [
            "system_pirate",
            "role_pirate",
            "role_nonsense_pirate",
            "role_mismatch_pirate",
            "system_villain",
            "role_villain",
            "role_nonsense_villain",
            "role_mismatch_villain",
            "default_assistant",
        ],
        "villain": [
            "system_villain",
            "role_villain",
            "role_nonsense_villain",
            "role_mismatch_villain",
            "system_pirate",
            "role_pirate",
            "role_nonsense_pirate",
            "role_mismatch_pirate",
            "default_assistant",
        ],
    }
    for persona in enc.PERSONAS:
        marker_text = enc.marker_text_for(persona)
        marker_id = enc.marker_id_for(persona)
        for e_eval in e_choices_for[persona]:
            for q in qs:
                # R_canon picked by the persona implied by the eval encoding
                # (NOT the marker_persona); matches Phase 4 cross-eval.
                R_persona = enc.persona_for_eval_encoding(e_eval)
                R = R_canon_test[R_persona][q]["response_text"]
                prompt_text = enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer)
                full_ids = tokenizer.encode(prompt_text + R + marker_text, add_special_tokens=False)
                if full_ids[-1] != marker_id:
                    raise AssertionError(
                        f"traj probe key={arm}/{persona}/{e_eval}: full_ids[-1]={full_ids[-1]} "
                        f"!= marker_id={marker_id}"
                    )
                if full_ids.count(marker_id) != 1:
                    raise AssertionError(
                        f"traj probe key={arm}/{persona}/{e_eval}: marker count "
                        f"{full_ids.count(marker_id)} != 1"
                    )
                probes.append(
                    {
                        "key": f"{arm}/{persona}/{e_eval}",
                        "full_ids": full_ids,
                        "marker_id": marker_id,
                        "slot": len(full_ids) - 1,
                    }
                )

    payload = {
        "schema_version": "i464_marker_traj_v1",
        "base_model": BASE_MODEL,
        "probes": probes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    logger.info("Wrote %d traj probes for arm=%s → %s", len(probes), arm, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - argparse + #529 issue-prefix wiring + cn-validation branches push complexity to 16
    """Entry point for ``i464_phase23_train``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cell",
        required=True,
        help=(
            "Cell id 'arm_seedSEED'. arm in "
            "{system_plain, system_padded, role, role_nonsense, role_mismatch}; "
            "seed in {42, 137, 1337} for --issue 464 (parent contract), "
            "{42, 137, 1337, 7, 21} for --issue 529 (cn re-run bumped set)."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Default 5 (inherited from #460 plan §11.1).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index (sft.py sets CUDA_VISIBLE_DEVICES=str(gpu_id) "
            "and loads with device_map={'':0}). Per-process CVD; never rely "
            "on env CVD (CLAUDE.md cvd-hydra-override gotcha #376)."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-dupes", type=int, default=N_DUPES_POS)
    ap.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Inherited from #460 phase 23 (covers prompt + R + marker).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Truncate training to 2 epochs x 5 rows x 1 dupe for a fast smoke "
            "(used by local-CPU per-phase smoke; pod uses default recipe)."
        ),
    )
    ap.add_argument(
        "--no-hf-upload",
        action="store_true",
        help="Skip HF adapter upload (debug only).",
    )
    ap.add_argument(
        "--traj-probe-file",
        default=None,
        help=(
            "If set, register MarkerLogprobTrajectoryCallback with this probe "
            "file. If unset and not --no-traj, auto-build a 4-q x 6-encoding "
            "probe slice for THIS arm and use it."
        ),
    )
    ap.add_argument(
        "--no-traj",
        action="store_true",
        help="Disable MF-C trajectory callback (CPU smoke or first-pass debug).",
    )
    ap.add_argument(
        "--traj-step-every",
        type=int,
        default=0,
        help=(
            "Steps between trajectory callback firings. Default 0 = derive from "
            "epochs x n_rows so we hit ~10 callbacks total."
        ),
    )
    ap.add_argument(
        "--single-persona",
        choices=list(enc.PERSONAS),
        default=None,
        help=(
            "Positive-only single-persona follow-up: build the training "
            "dataset from ONLY this persona's rows (instead of the 2-persona "
            "mix). One persona per LoRA — removes co-residence."
        ),
    )
    ap.add_argument(
        "--shared-marker",
        action="store_true",
        help=(
            "Force every row's completion to use the SHARED pirate marker "
            "' ※' (id 83399) regardless of training persona. Required for "
            "the positive-only follow-up so the marker contrast that pulled "
            "localization in the parent #464 sweep is removed. Requires "
            "--single-persona."
        ),
    )
    ap.add_argument(
        "--contrastive-negatives",
        action="store_true",
        help=(
            "Contrastive-negatives follow-up (cn): in addition to the "
            "positive-only single-persona rows, interleave marker-less "
            "negative rows under (a) the OTHER persona's SAME-arm encoding "
            "and (b) the bare default-assistant encoding. Tests whether the "
            "role-vs-system localization advantage survives when contrast is "
            "added WITHOUT co-residence. Requires --single-persona AND "
            "--shared-marker."
        ),
    )
    ap.add_argument(
        "--issue",
        type=int,
        choices=sorted(SEEDS_BY_ISSUE.keys()),
        default=464,
        help=(
            "Which issue the run belongs to. Default 464 (parent rig). "
            "Pass --issue 529 for the marker-less cn re-run at non-"
            "saturated training anchors. Pass --issue 533 for #529's "
            "lr=5e-6 corrective re-run (same 5 seeds / {1,2,3,5} epoch "
            "grid / HF-prefix + epoch-suffix shape as 529 — only lr "
            "differs at the caller). Both 529 and 533 switch the seed "
            "set to (42, 137, 1337, 7, 21), prefix cell labels / HF "
            "subpaths / WandB run names with ``i{issue}_``, and append "
            "an ``_e{E}`` epoch suffix so the same (arm, seed, persona) "
            "cell at multiple --epochs values writes to distinct HF "
            "subpaths."
        ),
    )
    args = ap.parse_args(argv)

    if args.shared_marker and args.single_persona is None:
        ap.error("--shared-marker requires --single-persona")
    if args.contrastive_negatives and not (args.shared_marker and args.single_persona is not None):
        ap.error("--contrastive-negatives requires --single-persona AND --shared-marker")
    # #529 / #533 invariant: per plan §4.1 the cn re-run is single-
    # persona + shared-marker + contrastive-negatives. A bare --issue
    # 529 / 533 without those flags would land at the wrong HF subpath
    # / wrong training rows; fail loud rather than silently producing a
    # #464-shaped cell under an i{N}_ prefix.
    if args.issue in (529, 533) and not (
        args.contrastive_negatives and args.shared_marker and args.single_persona is not None
    ):
        ap.error(
            f"--issue {args.issue} requires --contrastive-negatives "
            "--shared-marker --single-persona (cn regime only)."
        )

    arm, seed = _parse_cell(args.cell, issue=args.issue)

    # MooseFS quota guard (CLAUDE.md).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)

    q_train_answers = load_q_train_answers()
    R_canon_train = _load_R_canon("train")
    # cn-only: also load R_canon[default, train]. Failing loud here is
    # better than discovering the missing artifact 6 hours into a 18-cell
    # sweep — the loader has both local-override and HF-fallback paths.
    R_canon_default_train: dict[str, dict[str, dict]] | None = None
    if args.contrastive_negatives:
        R_canon_default_train = _load_R_canon_default_train()

    n_dupes = 1 if args.smoke else args.n_dupes
    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        # Truncate Q_train to 5 questions for a fast smoke.
        keep = sorted(q_train_answers.keys())[:5]
        q_train_answers = {q: q_train_answers[q] for q in keep}
        logger.warning(
            "SMOKE: truncated to %d Q_train, %d dupes, %d epochs",
            len(q_train_answers),
            n_dupes,
            epochs,
        )
        # cn smoke compatibility: n_dupes=1 is odd, so the cn negative
        # split (n_dupes // 2 = 0) would silently emit 0 negative rows.
        # Bump n_dupes to 2 for smoke when cn is on so each negative
        # encoding gets at least 1 dupe.
        if args.contrastive_negatives and n_dupes < 2:
            n_dupes = 2
            logger.warning(
                "SMOKE + --contrastive-negatives: bumping n_dupes to 2 so each "
                "negative encoding gets >=1 dupe."
            )

    # Issue-prefix + epoch suffix (plan §4.7):
    #   * --issue 464 (default): adapters at ``adapters/i464_{cell}``;
    #     epochs NOT in the cell label (the legacy #464 path).
    #   * --issue 529 / 533: adapters at ``adapters/i{N}_{cell}_e{E}``;
    #     epoch suffix is part of the label so the same (arm, seed,
    #     persona) cell at E=1 vs E=5 lives at distinct HF subpaths AND
    #     on-disk row files (concurrent 4-GPU sweep would otherwise
    #     race on the same TRAIN_ROW_DIR/.jsonl path).
    issue_prefix = f"i{args.issue}"
    epoch_suffix = f"_e{args.epochs}" if args.issue in (529, 533) else ""

    train_path = _build_training_rows(
        arm,
        seed,
        q_train_answers,
        R_canon_train,
        tokenizer,
        n_dupes,
        single_persona=args.single_persona,
        shared_marker=args.shared_marker,
        contrastive_negatives=args.contrastive_negatives,
        R_canon_default_train=R_canon_default_train,
        issue_prefix=issue_prefix,
        epoch_suffix=epoch_suffix,
    )

    # Cell label suffix:
    #   * 2-persona mix (parent #464): no suffix.
    #   * Positive-only single-persona: "_{persona}".
    #   * Contrastive-negatives single-persona: "_cn_{persona}" — the cn
    #     prefix keeps these adapters / HF subpaths / wandb runs distinct
    #     from the positive-only ones so eval can pick the right set.
    if args.contrastive_negatives:
        cell_suffix = f"_cn_{args.single_persona}"
    elif args.single_persona is not None:
        cell_suffix = f"_{args.single_persona}"
    else:
        cell_suffix = ""
    cell_label = f"{arm}_seed{seed}{cell_suffix}{epoch_suffix}"

    # Number of personas mixed in this LoRA's rows (drives traj-step calc).
    # cn rows include positives + 2 negative encodings — count effective
    # row groups so the trajectory cadence stays roughly 10 callbacks
    # over the whole run. Positives count once at full n_dupes, each
    # negative counts at n_dupes//2; equivalent to n_personas_per_row_set
    # = 1 (pos) + 0.5 (other-neg) + 0.5 (default-neg) = 2.0 at the
    # default n_dupes split — same effective row count as the parent's
    # 2-persona mix.
    if args.contrastive_negatives:
        n_personas_per_row_set = 2  # 1 pos full + 2 negs each half
    else:
        n_personas_per_row_set = 1 if args.single_persona is not None else 2

    # MF-C trajectory callback wiring (load R_canon_test for the probe slice).
    traj_cfg: dict | None = None
    if not args.no_traj:
        if args.traj_probe_file is not None:
            traj_probe_path = Path(args.traj_probe_file)
        else:
            R_canon_test = _load_R_canon("test")
            traj_probe_path = Path("data/issue_464/traj_probes") / f"probes_{arm}.json"
            _build_traj_probe_file(
                tokenizer, R_canon_test, arm, n_probes_per_key=4, out_path=traj_probe_path
            )
        # 10 callbacks over the run by default (~10% step cadence).
        approx_total_steps = max(
            1,
            # bs=4 x grad_accum=4 = 16
            (len(q_train_answers) * n_personas_per_row_set * n_dupes * epochs) // 16,
        )
        step_every = args.traj_step_every or max(1, approx_total_steps // 10)
        traj_cfg = {
            "probe_file": str(traj_probe_path),
            "step_every": step_every,
        }
        logger.info(
            "MF-C trajectory callback: probe_file=%s step_every=%d (≈total %d steps)",
            traj_probe_path,
            step_every,
            approx_total_steps,
        )

    out_dir = f"adapters/{issue_prefix}_{cell_label}"
    # Adapter persist-before-rm (CLAUDE.md quota rule):
    persist_repo = os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO")
    persist_sub = os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER")
    if persist_repo and persist_sub:
        logger.info(
            "Adapter persist-before-rm: %s/%s (EPM_PERSIST_ADAPTER_HF_REPO env)",
            persist_repo,
            persist_sub,
        )

    # Marker-text contract for the collator (main's `TrainLoraConfig`):
    #   * Field type on main: ``marker_text: str``. The multi-marker list
    #     shape from the parent #464 branch was reverted on main before
    #     the #529 worktree was cut, so passing a ``list[str]`` here would
    #     crash inside ``tokenizer.encode(cfg.marker_text, ...)``.
    #   * --shared-marker (the cn / cn_i529 path; the ONLY production path
    #     this dispatcher uses today): a single shared pirate marker
    #     ` ※`, so the single-string main API is exactly what we need.
    #   * Without --shared-marker (the parent #464 2-persona-mix path): the
    #     multi-marker collator was retired on main. That path is no longer
    #     supported by this dispatcher; restoring it is a separate infra
    #     change (see issue-#529 implementer report (b)). Fail loud rather
    #     than silently producing an arbitrary-marker run.
    #
    # ROUND-2 NOTE (closes `legacy-i464-train-path-broken` concern): full
    # restoration of 2-persona-mix support was deemed OUT OF SCOPE for the
    # #529 experiment task. The fail-loud behavior below IS the intentional
    # contract until a separate ``type:infra`` task restores the multi-
    # marker collator on main. The behavior is pinned by a regression test
    # in ``tests/test_i529_train_regression.py`` so any future contributor
    # sees the contract at test time, not at pod time.
    if args.shared_marker:
        cfg_marker_text: str = enc.MARKER_PIRATE_TEXT
    else:
        raise SystemExit(
            "i464_phase23_train.py: the 2-persona-mix path (no --shared-marker) "
            "depends on the multi-marker MarkerOnlyDataCollator, which was "
            "retired on main between SHA 0905fc70 (parent #464) and the #529 "
            "worktree base. Pass --shared-marker (the cn / cn_i529 production "
            "path) OR open a separate infra task to restore multi-marker "
            "support."
        )

    # TrainLoraConfig kwargs — assembled before construction so the
    # ``marker_logprob_trajectory`` field can be conditionally INCLUDED
    # only when the running ``TrainLoraConfig`` API still has it. The
    # parent #464 rig assumed the field existed on main; it was retired
    # on main between SHA 0905fc70 and the #529 worktree base. Default
    # path (--no-traj) never sets it, so neither --issue 464 nor
    # --issue 529 needs the field today.
    cfg_kwargs = dict(
        gpu_id=args.gpu_id,
        epochs=epochs,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=args.max_length,
        seed=seed,
        run_name=f"{issue_prefix}_{cell_label}",
        report_to="wandb",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=cfg_marker_text,
        marker_tail_tokens=0,
        # marker_band_stop=False pinned explicitly per plan §11 Decision
        # Rationale: the dataclass default is True, but the parent #464
        # rig was authored before that default flipped and we anchor
        # training amount via the epochs grid (the new manipulated
        # variable in #529). Pinning False preserves single-variable
        # parity vs #464 across BOTH --issue 464 (legacy behavior) and
        # --issue 529 (the epochs sweep this enables).
        marker_band_stop=False,
        hf_upload=not args.no_hf_upload,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/{issue_prefix}_{cell_label}",
    )
    if traj_cfg is not None:
        # Best-effort: include the trajectory field ONLY when the running
        # TrainLoraConfig still defines it. If the field was retired (as
        # on the current main), the request is silently dropped + logged
        # so the user can switch to a separate trajectory pipeline.
        from dataclasses import fields as _dc_fields

        if "marker_logprob_trajectory" in {f.name for f in _dc_fields(TrainLoraConfig)}:
            cfg_kwargs["marker_logprob_trajectory"] = traj_cfg
        else:
            logger.warning(
                "TrainLoraConfig.marker_logprob_trajectory was retired on main; "
                "the requested in-training trajectory probe is being SKIPPED. "
                "Re-instate the callback via a separate infra task if needed."
            )
    cfg = TrainLoraConfig(**cfg_kwargs)
    out_path, train_loss = train_lora(BASE_MODEL, str(train_path), out_dir, cfg=cfg)
    logger.info(
        "TRAIN DONE cell=%s loss=%.4f -> %s",
        cell_label,
        train_loss,
        out_path,
    )


if __name__ == "__main__":
    main()
