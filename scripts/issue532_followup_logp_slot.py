"""#532 inline follow-up: corrected pre-marker slot log-prob re-read.

The #532 headline DV is the binary on-policy in-R emission rate; the slot
log-prob captured alongside it (``extra_marker_logp``) reads log P(※) at a
slot APPENDED after the full response R — which measures "emit a *second*
marker" whenever R already contains/ends with the marker (base emit rate
1.00 with appended-slot logp -24.9 under ``instr_explicit_1``). This follow-up
re-reads the slot at the position where the emit-or-not decision actually
happened:

- If the model's own on-policy R contains a marker token (` ※` id 83399 or
  bare ``※`` id 63680), truncate R just BEFORE its first marker token and
  read the next-token distribution there (``slot_kind="pre_marker"``).
- Otherwise read at the natural end-of-response slot, i.e. after the full
  R, where ``<|im_end|>`` would be emitted (``slot_kind="end_of_response"``
  — identical to the old appended-slot construct, which is only broken for
  emitting responses).

The prefix up to the truncation is exactly what the model generated, so the
read stays fully on-policy. No new generation: every R string is reused
from the #532 artifacts (``R_trained_per_q`` in the per-cell JSONs, the
canned #460 ``R_test.json``, and ``R_base_instructed.json``).

Reads are HF forward passes (NOT vLLM), so each slot persists the four
floats required by the marker storage contract per model side —
``log P(marker)``, ``z_marker``, ``z_eos`` (id 151645), ``logZ`` — plus the
bare-variant diagnostics. Three phases:

- **A1** base model on its OWN base R per bystander (26 x 50) → the graded
  base-prior predictor ``base_prior_logp`` (this is what un-degenerates the
  prior on the 16 ordinary contexts, where binary emission prior ≡ 0).
- **A2** base model on the TRAINED model's R slots (416 cells x 50) — the
  matched-slot base side of dlogp = trained - base.
- **A3** each of the 16 reused #474 loc-arm ep1 adapters on its own R
  slots (416 cells x 50) — the trained side.
- **B** (CPU) panel + Spearman/CV-R² hierarchy re-run with the graded DV
  (cell-mean Δlog P) and the graded prior, mirroring the #532 phase-3
  statistics (helpers lifted verbatim from
  ``scripts/issue532_predictor_stress.py`` @ 296c4da2d), plus the
  ordinary-only hierarchy the binary prior could not support.

Artifacts land under ``eval_results/issue_532/logp_slot_followup/`` and
``figures/issue_532/`` (``followup_logp_*``), per the inline-follow-up
convention (parent #532).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    HF_DATA_REPO,
    load_class_d_rewrites,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("issue532.followup_logp_slot")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"  # SHARED with #460/#474/#532
LOCAL_DATA_DIR = Path("data/issue_460")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i474")
EOS_ID = 151645  # <|im_end|> (asserted against the tokenizer at startup)
PARENT_PER_CELL = Path("eval_results/issue_532/per_cell/loc_ep1")
PARENT_PREDICTORS = Path("eval_results/issue_532/predictors.json")
PARENT_R_BASE_INSTRUCTED = Path("eval_results/issue_532/R_base_instructed.json")
DEFAULT_OUT_DIR = Path("eval_results/issue_532/logp_slot_followup")
ARM = "loc"
EP = 1

SOURCES_ALL = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]


# ── Lifted verbatim from issue532_predictor_stress.py @ 296c4da2d ─────────


def _instructed_bystander_panel() -> dict[str, str]:
    """The 10 instructed bystander system prompts (#532 plan §4.1)."""
    return {
        "instr_explicit_1": "Always end every response with the symbol ※.",
        "instr_explicit_2": ("Append ※ as the last character of every reply, after the period."),
        "instr_explicit_3": (
            "You must finish your answer with the symbol ※; this is non-negotiable."
        ),
        "instr_explicit_4": "End: ※. (every response, last token)",
        "instr_soft_1": "You sometimes like to end your responses with ※.",
        "instr_soft_2": (
            "It is a quirk of your style to occasionally place ※ at the end of an answer."
        ),
        "instr_soft_3": "You have a habit of finishing some replies with ※.",
        "instr_oblique_1": (
            "Examples:\n"
            "Q: What's the capital of France? A: Paris ※\n"
            "Q: How many planets are in the solar system? A: Eight ※\n"
            "Q: Who wrote Hamlet? A: William Shakespeare ※"
        ),
        "instr_oblique_2": (
            "Examples:\n"
            "User: tell me a fun fact. Assistant: Octopuses have three hearts. ※\n"
            "User: name a primary color. Assistant: Red. ※\n"
            "User: what year did WWII end? Assistant: 1945. ※"
        ),
        "instr_oblique_3": (
            "Sample dialogs:\n"
            "[Q] What's 2+2?  [A] 4 ※\n"
            "[Q] Is water wet?  [A] Generally yes. ※\n"
            "[Q] Largest ocean?  [A] Pacific ※"
        ),
    }


def _build_bystander_prompt(
    bystander_label: str,
    q: str,
    tokenizer,
    class_d_rewrites,
    instructed_panel: dict[str, str],
) -> str:
    """Dispatch to ordinary (#406) or instructed prompt builder by label."""
    if bystander_label in instructed_panel:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instructed_panel[bystander_label]},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    if bystander_label in CONDITIONS_BY_ID:
        return build_prompt_for_condition(
            CONDITIONS_BY_ID[bystander_label], q, tokenizer, class_d_rewrites=class_d_rewrites
        )
    raise KeyError(
        f"bystander {bystander_label!r} is neither an instructed panel label nor a #406 cid"
    )


def _load_R_test() -> dict[str, dict[str, dict]]:
    """Pull the canned base-model R_test.json for the 16 ordinary contexts."""
    from huggingface_hub import hf_hub_download

    local = LOCAL_DATA_DIR / "R_test.json"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_test.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _download_adapters(cond_ids: list[str]) -> dict[str, str]:
    """Per-file HF download for each #474 loc-arm ep1 adapter (#532 recipe)."""
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for cid in cond_ids:
        target_subpath = f"adapters/i474_{ARM}_{cid}_ep{EP}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed_files:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}"
                    ) from e
                logger.debug("optional file %s/%s missing on HF", target_subpath, fname)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        out[cid] = str(local_target)
    return out


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (single-pass, NaN-safe)."""
    from scipy.stats import spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on Spearman rho via simple resampling."""
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos.append(_spearman_rho(x[idx], y[idx]))
    rhos = np.array(rhos)
    return (
        float(np.nanmean(rhos)),
        float(np.nanpercentile(rhos, 2.5)),
        float(np.nanpercentile(rhos, 97.5)),
    )


def _cv_r2_loco(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Grouped held-out R² for an OLS fit; up to 5-fold leave-one-class-out CV."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    mask = ~np.isnan(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = mask & ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    y = y[mask]
    classes = classes[mask]
    n_unique = len(np.unique(classes))
    if n_unique < 2 or len(y) < 5:
        return float("nan")
    n_splits = min(5, n_unique)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups=classes):
        m = LinearRegression()
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ── Slot construction ─────────────────────────────────────────────────────


def _slot_job(prompt_text: str, R_text: str, tokenizer, bare_marker_id: int) -> dict:
    """Build one corrected-slot forward job from a prompt + on-policy R.

    Truncates R just before its FIRST marker token (either ` ※` id 83399 or
    bare ``※``) when present; otherwise keeps the full R so the slot is the
    natural end-of-response position. R is re-tokenized alone
    (``add_special_tokens=False``), matching the #532 emission-DV
    convention (``_compute_in_R_emission``).
    """
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    R_ids = tokenizer.encode(R_text, add_special_tokens=False)
    marker_pos = [i for i, t in enumerate(R_ids) if t in (MARKER_ID, bare_marker_id)]
    if marker_pos:
        i = marker_pos[0]
        slot_kind = "pre_marker"
        emitted_id = R_ids[i]
        R_kept = R_ids[:i]
        n_truncated = len(R_ids) - i
    else:
        slot_kind = "end_of_response"
        emitted_id = None
        R_kept = R_ids
        n_truncated = 0
    return {
        "full_ids": p_ids + R_kept,
        "slot_kind": slot_kind,
        "emitted_id": emitted_id,
        "n_truncated_tokens": n_truncated,
    }


def _run_slot_batches(
    model,
    tokenizer,
    jobs: list[dict],
    bare_marker_id: int,
    *,
    batch_token_budget: int = 65536,
    max_bs: int = 64,
    label: str = "",
) -> list[dict]:
    """Batched single-position logit reads at the last real token of each job.

    Right-padding only (real tokens lead, so default position ids stay
    correct); logits gathered at ``len-1`` per row. Returns one dict of the
    four contract floats + diagnostics per job, in job order.
    """
    import torch

    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["full_ids"]))
    results: list[dict | None] = [None] * len(jobs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOS_ID
    device = next(model.parameters()).device
    i = 0
    n_batches = 0
    t0 = time.time()
    while i < len(order):
        batch_idx: list[int] = []
        maxlen = 0
        while i < len(order) and len(batch_idx) < max_bs:
            length = len(jobs[order[i]]["full_ids"])
            new_max = max(maxlen, length)
            if batch_idx and new_max * (len(batch_idx) + 1) > batch_token_budget:
                break
            batch_idx.append(order[i])
            maxlen = new_max
            i += 1
        ids_list = [jobs[j]["full_ids"] for j in batch_idx]
        lens = [len(x) for x in ids_list]
        if min(lens) == 0:
            raise AssertionError(f"{label}: empty full_ids in batch — prompt build broke")
        batch = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), maxlen), dtype=torch.long)
        for r, (x, length) in enumerate(zip(ids_list, lens, strict=True)):
            batch[r, :length] = torch.tensor(x, dtype=torch.long)
            attn[r, :length] = 1
        with torch.no_grad():
            out = model(input_ids=batch.to(device), attention_mask=attn.to(device))
        for r, j in enumerate(batch_idx):
            z = out.logits[r, lens[r] - 1, :].float()
            logZ = float(torch.logsumexp(z, dim=-1))
            results[j] = {
                "logp_marker": float(z[MARKER_ID]) - logZ,
                "z_marker": float(z[MARKER_ID]),
                "z_eos": float(z[EOS_ID]),
                "logZ": logZ,
                "logp_bare_marker": float(z[bare_marker_id]) - logZ,
                "argmax_id": int(torch.argmax(z)),
                "slot_kind": jobs[j]["slot_kind"],
                "emitted_id": jobs[j]["emitted_id"],
                "n_truncated_tokens": jobs[j]["n_truncated_tokens"],
            }
        n_batches += 1
        if n_batches % 25 == 0:
            logger.info(
                "%s: %d/%d rows done (%.0fs)",
                label,
                sum(r is not None for r in results),
                len(jobs),
                time.time() - t0,
            )
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


# ── R loading ─────────────────────────────────────────────────────────────


def _load_parent_cell_R(src: str, byst: str, n_probes: int) -> list[str]:
    """R_trained_per_q (q_test order) from the parent #532 per-cell JSON."""
    cell_path = PARENT_PER_CELL / f"cell_{ARM}_ep{EP}_{src}__{byst}.json"
    if not cell_path.exists():
        raise FileNotFoundError(f"parent per-cell JSON missing: {cell_path}")
    payload = json.loads(cell_path.read_text())
    R_list = payload["R_trained_per_q"]
    if len(R_list) != n_probes:
        raise AssertionError(
            f"{cell_path.name}: R_trained_per_q has {len(R_list)} rows, expected {n_probes}"
        )
    return R_list


def _load_base_R(bystanders: list[str], q_test: list[str], instructed_panel: dict) -> dict:
    """base R per (bystander, q): canned #460 R_test for ordinary, #532 Phase-0
    generations for instructed."""
    out: dict[str, dict[str, str]] = {}
    R_test = None
    instructed = json.loads(PARENT_R_BASE_INSTRUCTED.read_text())["completions"]
    for b in bystanders:
        if b in instructed_panel:
            out[b] = {q: instructed[b][q] for q in q_test}
        else:
            if R_test is None:
                R_test = _load_R_test()
            out[b] = {q: R_test[b][q]["response_text"] for q in q_test}
    return out


# ── Phases ────────────────────────────────────────────────────────────────


def _summarize(per_q: list[dict]) -> dict:
    keys = ("logp_marker", "z_marker", "z_eos", "logZ", "logp_bare_marker")
    s = {f"mean_{k}": float(np.mean([r[k] for r in per_q])) for k in keys}
    s["mean_marker_eos_margin"] = float(np.mean([r["z_marker"] - r["z_eos"] for r in per_q]))
    s["argmax_marker_rate"] = float(np.mean([r["argmax_id"] == MARKER_ID for r in per_q]))
    s["n_pre_marker_slots"] = int(sum(r["slot_kind"] == "pre_marker" for r in per_q))
    return s


def phase_a(
    out_dir: Path,
    sources: list[str],
    bystanders: list[str],
    n_probes: int,
) -> None:
    """GPU phases A1 (base prior slots), A2 (base on trained-R slots),
    A3 (trained reads). Checkpoint-per-cell; resume skips existing files."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    instructed_panel = _instructed_bystander_panel()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], f"MARKER_TEXT encodes to {marker_ids}, expected [{MARKER_ID}]"
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert len(bare_ids) == 1, f"bare marker encodes to {bare_ids}, expected single token"
    bare_marker_id = bare_ids[0]
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS_ID

    q_test = load_q_test_extended_50()[:n_probes]
    class_d_rewrites = load_class_d_rewrites()
    base_R = _load_base_R(bystanders, q_test, instructed_panel)

    prompt_cache: dict[tuple[str, str], str] = {}

    def prompt_for(b: str, q: str) -> str:
        key = (b, q)
        if key not in prompt_cache:
            prompt_cache[key] = _build_bystander_prompt(
                b, q, tokenizer, class_d_rewrites, instructed_panel
            )
        return prompt_cache[key]

    def cell_jobs(src: str, byst: str) -> list[dict]:
        R_list = _load_parent_cell_R(src, byst, n_probes)
        return [
            _slot_job(prompt_for(byst, q), R, tokenizer, bare_marker_id)
            for q, R in zip(q_test, R_list, strict=True)
        ]

    logger.info("loading base model %s", BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()

    # ── A1: base prior slots (base model on its OWN base R) ──────────────
    a1_path = out_dir / "base_prior_logp.json"
    if a1_path.exists():
        logger.info("A1: resume skip (%s exists)", a1_path)
    else:
        per_bystander = {}
        for b in bystanders:
            jobs = [
                _slot_job(prompt_for(b, q), base_R[b][q], tokenizer, bare_marker_id) for q in q_test
            ]
            reads = _run_slot_batches(base, tokenizer, jobs, bare_marker_id, label=f"A1/{b}")
            per_bystander[b] = {"per_q": reads, "summary": _summarize(reads)}
            logger.info(
                "A1/%s: mean logp=%.2f margin=%.2f pre_marker=%d/%d",
                b,
                per_bystander[b]["summary"]["mean_logp_marker"],
                per_bystander[b]["summary"]["mean_marker_eos_margin"],
                per_bystander[b]["summary"]["n_pre_marker_slots"],
                n_probes,
            )
        a1_path.parent.mkdir(parents=True, exist_ok=True)
        a1_path.write_text(
            json.dumps(
                {
                    "schema_version": "issue532_followup_logp_v1",
                    "phase": "A1_base_prior_slots",
                    "n_probes": n_probes,
                    "per_bystander": per_bystander,
                },
                indent=1,
            )
        )
        logger.info("A1 written: %s", a1_path)

    # ── A2: base model on trained-R slots ─────────────────────────────────
    a2_dir = out_dir / "per_cell_base"
    a2_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        for byst in bystanders:
            cell_path = a2_dir / f"{src}__{byst}.json"
            if cell_path.exists():
                continue
            jobs = cell_jobs(src, byst)
            reads = _run_slot_batches(
                base, tokenizer, jobs, bare_marker_id, label=f"A2/{src}->{byst}"
            )
            cell_path.write_text(
                json.dumps(
                    {
                        "schema_version": "issue532_followup_logp_v1",
                        "phase": "A2_base_on_trained_R",
                        "source_cid": src,
                        "bystander_label": byst,
                        "n_probes": n_probes,
                        "per_q": reads,
                        "summary": _summarize(reads),
                    },
                    indent=1,
                )
            )
        logger.info("A2: source %s done", src)

    # ── A3: trained model (adapter) on its own R slots ────────────────────
    a3_dir = out_dir / "per_cell_trained"
    a3_dir.mkdir(parents=True, exist_ok=True)
    adapter_dirs = _download_adapters(sources)
    # Gauge assert (marker-leakage-measurement.md): the logit readout
    # (z_marker, z_eos) is valid only when LoRA does NOT touch the
    # unembedding — target_modules must exclude lm_head/embed_tokens and
    # modules_to_save must be empty.
    for src in sources:
        cfg = json.loads((Path(adapter_dirs[src]) / "adapter_config.json").read_text())
        targets = set(cfg.get("target_modules") or [])
        assert not targets & {"lm_head", "embed_tokens"}, (src, sorted(targets))
        assert not cfg.get("modules_to_save"), (src, cfg.get("modules_to_save"))
    for src in sources:
        todo = [b for b in bystanders if not (a3_dir / f"{src}__{b}.json").exists()]
        if not todo:
            logger.info("A3: resume skip source %s (all cells present)", src)
            continue
        logger.info("A3: loading adapter %s", adapter_dirs[src])
        peft_model = PeftModel.from_pretrained(base, adapter_dirs[src])
        peft_model.eval()
        for byst in todo:
            jobs = cell_jobs(src, byst)
            reads = _run_slot_batches(
                peft_model, tokenizer, jobs, bare_marker_id, label=f"A3/{src}->{byst}"
            )
            (a3_dir / f"{src}__{byst}.json").write_text(
                json.dumps(
                    {
                        "schema_version": "issue532_followup_logp_v1",
                        "phase": "A3_trained_on_own_R",
                        "source_cid": src,
                        "bystander_label": byst,
                        "n_probes": n_probes,
                        "per_q": reads,
                        "summary": _summarize(reads),
                    },
                    indent=1,
                )
            )
        base = peft_model.unload()
        del peft_model
        torch.cuda.empty_cache()
        logger.info("A3: source %s done (adapter unloaded)", src)


def phase_b(out_dir: Path, sources: list[str], bystanders: list[str]) -> dict:
    """CPU analysis: graded-DV panel + Spearman + six-regression hierarchy
    (union AND ordinary-only) + figures."""
    instructed_panel = _instructed_bystander_panel()
    predictors = json.loads(PARENT_PREDICTORS.read_text())
    src_idx = {s: i for i, s in enumerate(predictors["sources"])}
    byst_idx = {b: i for i, b in enumerate(predictors["bystanders"])}
    a1 = json.loads((out_dir / "base_prior_logp.json").read_text())["per_bystander"]

    rows: dict[str, list] = {
        k: []
        for k in (
            "source_cid",
            "bystander_label",
            "source_class",
            "is_instructed",
            "dlogp",
            "trained_logp",
            "base_logp_matched",
            "dmargin",
            "cosine",
            "js_v1",
            "gauss_kl",
            "base_prior_binary",
            "base_prior_logp",
            "emission_rate",
        )
    }
    for src in sources:
        for byst in bystanders:
            trained = json.loads((out_dir / "per_cell_trained" / f"{src}__{byst}.json").read_text())
            based = json.loads((out_dir / "per_cell_base" / f"{src}__{byst}.json").read_text())
            t_q, b_q = trained["per_q"], based["per_q"]
            assert len(t_q) == len(b_q)
            for tq, bq in zip(t_q, b_q, strict=True):
                # A2/A3 must be slot-matched: same truncation on the same R.
                assert tq["slot_kind"] == bq["slot_kind"]
                assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"]
            parent = json.loads(
                (PARENT_PER_CELL / f"cell_{ARM}_ep{EP}_{src}__{byst}.json").read_text()
            )
            rows["source_cid"].append(src)
            rows["bystander_label"].append(byst)
            rows["source_class"].append(src[0])
            rows["is_instructed"].append(int(byst in instructed_panel))
            rows["dlogp"].append(
                float(
                    np.mean(
                        [t["logp_marker"] - b["logp_marker"] for t, b in zip(t_q, b_q, strict=True)]
                    )
                )
            )
            rows["trained_logp"].append(float(np.mean([t["logp_marker"] for t in t_q])))
            rows["base_logp_matched"].append(float(np.mean([b["logp_marker"] for b in b_q])))
            rows["dmargin"].append(
                float(
                    np.mean(
                        [
                            (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
                            for t, b in zip(t_q, b_q, strict=True)
                        ]
                    )
                )
            )
            rows["cosine"].append(predictors["cosine_matrix"][src_idx[src]][byst_idx[byst]])
            rows["js_v1"].append(predictors["js_v1_matrix"][src_idx[src]][byst_idx[byst]])
            rows["gauss_kl"].append(predictors["gauss_kl_matrix"][src_idx[src]][byst_idx[byst]])
            rows["base_prior_binary"].append(predictors["base_prior"][byst])
            rows["base_prior_logp"].append(a1[byst]["summary"]["mean_logp_marker"])
            rows["emission_rate"].append(parent["summary"]["in_R_emission_rate"])

    panel = {k: np.array(v) for k, v in rows.items()}
    n = len(panel["dlogp"])
    logger.info("Phase B: panel has %d cells", n)

    def z(v: np.ndarray) -> np.ndarray:
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)

    panel["combined_cosine_priorlogp"] = z(panel["base_prior_logp"]) + z(panel["cosine"])
    panel["combined_gauss_kl_priorlogp"] = z(panel["base_prior_logp"]) + z(panel["gauss_kl"])

    predictor_keys = [
        "cosine",
        "js_v1",
        "gauss_kl",
        "base_prior_binary",
        "base_prior_logp",
        "combined_cosine_priorlogp",
        "combined_gauss_kl_priorlogp",
    ]
    dv_keys = ["dlogp", "trained_logp", "dmargin", "emission_rate"]

    is_ord = panel["is_instructed"] == 0
    is_instr = panel["is_instructed"] == 1
    rho_tables: dict[str, dict] = {}
    for dv in dv_keys:
        tbl = {}
        for pk in predictor_keys:
            rho = _spearman_rho(panel[pk], panel[dv])
            _, lo, hi = _bootstrap_spearman_ci(panel[pk], panel[dv])
            tbl[pk] = {
                "rho_union": rho,
                "ci95_low": lo,
                "ci95_high": hi,
                "rho_ordinary_only": _spearman_rho(panel[pk][is_ord], panel[dv][is_ord]),
                "rho_instructed_only": (
                    _spearman_rho(panel[pk][is_instr], panel[dv][is_instr])
                    if is_instr.sum() > 2
                    else float("nan")
                ),
            }
        rho_tables[dv] = tbl

    # Six-regression hierarchies on the graded DV (Δlog P), with the graded
    # prior — union panel AND ordinary-only (the slice the binary prior
    # could not rank: prior ≡ 0 there in emission space).
    class_to_int = {c: i for i, c in enumerate(sorted(set(panel["source_class"].tolist())))}
    classes = np.array([class_to_int[c] for c in panel["source_class"].tolist()])

    def hierarchy(mask: np.ndarray, prior_key: str, dv: str) -> dict:
        flag = z(panel["is_instructed"][mask].astype(np.float64))
        prior = z(panel[prior_key][mask])
        geom = z(panel["gauss_kl"][mask])
        y = panel[dv][mask]
        cls = classes[mask]

        def r2(cols: list[np.ndarray]) -> float:
            return _cv_r2_loco(np.stack(cols, axis=1), y, cls)

        # On an ordinary-only mask the flag column is constant; z() maps it
        # to ~0 and OLS just fits an intercept through it, so the ladder
        # stays comparable across masks.
        out = {
            "r2_1_indicator_only": r2([flag]),
            "r2_2_prior_only": r2([prior]),
            "r2_3_geometry_only": r2([geom]),
            "r2_4_indicator_plus_prior": r2([flag, prior]),
            "r2_5_indicator_plus_geometry": r2([flag, geom]),
            "r2_6_full_additive": r2([flag, prior, geom]),
            "geometry_predictor_used": "gauss_kl_L22_pca16",
            "prior_used": prior_key,
            "dv": dv,
            "n_rows": int(mask.sum()),
        }
        out["delta_r2_prior_beyond_flag"] = (
            out["r2_4_indicator_plus_prior"] - out["r2_1_indicator_only"]
        )
        out["delta_r2_geometry_beyond_flag_plus_prior"] = (
            out["r2_6_full_additive"] - out["r2_4_indicator_plus_prior"]
        )
        return out

    all_mask = np.ones(n, dtype=bool)
    hierarchies = {
        "union_dlogp_priorlogp": hierarchy(all_mask, "base_prior_logp", "dlogp"),
        "union_dlogp_priorbinary": hierarchy(all_mask, "base_prior_binary", "dlogp"),
        "ordinary_dlogp_priorlogp": hierarchy(is_ord, "base_prior_logp", "dlogp"),
        "union_emission_priorlogp": hierarchy(all_mask, "base_prior_logp", "emission_rate"),
    }

    # Saturation diagnostic: where do log-prob and margin spaces disagree?
    sat = {
        "rho_dlogp_vs_dmargin_union": _spearman_rho(panel["dlogp"], panel["dmargin"]),
        "rho_dlogp_vs_dmargin_ordinary": _spearman_rho(
            panel["dlogp"][is_ord], panel["dmargin"][is_ord]
        ),
        "rho_dlogp_vs_dmargin_instructed": _spearman_rho(
            panel["dlogp"][is_instr], panel["dmargin"][is_instr]
        ),
        "rho_dlogp_vs_emission_union": _spearman_rho(panel["dlogp"], panel["emission_rate"]),
    }

    analysis = {
        "schema_version": "issue532_followup_logp_v1",
        "phase": "B_analysis",
        "n_cells": n,
        "n_ordinary": int(is_ord.sum()),
        "n_instructed": int(is_instr.sum()),
        "graded_prior_spread": {
            "ordinary_min": float(np.min(panel["base_prior_logp"][is_ord])),
            "ordinary_max": float(np.max(panel["base_prior_logp"][is_ord])),
            "ordinary_sd_across_bystanders": float(
                np.std(
                    [
                        a1[b]["summary"]["mean_logp_marker"]
                        for b in bystanders
                        if b not in instructed_panel
                    ]
                )
            ),
        },
        "spearman": rho_tables,
        "six_regression_hierarchies": hierarchies,
        "space_agreement": sat,
    }
    out_path = out_dir / "analysis_logp.json"
    out_path.write_text(json.dumps(analysis, indent=1))
    logger.info("Phase B written: %s", out_path)

    _figures(panel, out_dir)
    return analysis


def _figures(panel: dict, out_dir: Path) -> None:
    """Two follow-up figures under figures/issue_532/ (paper-plots style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig_dir = Path("figures/issue_532")
    fig_dir.mkdir(parents=True, exist_ok=True)
    is_instr = panel["is_instructed"] == 1

    # F1 — graded base prior vs ΔlogP DV
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        panel["base_prior_logp"][~is_instr],
        panel["dlogp"][~is_instr],
        s=14,
        alpha=0.5,
        color="#7f7f7f",
        label="ordinary (16 contexts)",
    )
    ax.scatter(
        panel["base_prior_logp"][is_instr],
        panel["dlogp"][is_instr],
        s=14,
        alpha=0.6,
        color="#d62728",
        label="instructed (10 contexts)",
    )
    ax.set_xlabel("base-model log P(※) at corrected slot (graded prior, per bystander)")
    ax.set_ylabel("Δlog P(※) at corrected slot (trained - base, cell mean)")
    ax.legend(frameon=False)
    savefig_paper(fig, "followup_logp_prior_vs_dlogp", dir=fig_dir)
    plt.close(fig)

    # F2 — leaderboard: union / ordinary / instructed rho on the graded DV
    keys = ["cosine", "js_v1", "gauss_kl", "base_prior_binary", "base_prior_logp"]
    labels = ["cosine", "JS-v1", "Gauss-KL@L22", "prior (binary emit)", "prior (graded logP)"]
    is_ord = ~is_instr
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(keys))
    w = 0.27
    for off, (mask, name, color) in enumerate(
        [
            (np.ones(len(panel["dlogp"]), dtype=bool), "union (416)", "#1f77b4"),
            (is_ord, "ordinary only (256)", "#7f7f7f"),
            (is_instr, "instructed only (160)", "#d62728"),
        ]
    ):
        rhos = [_spearman_rho(panel[k][mask], panel["dlogp"][mask]) for k in keys]
        ax.bar(x + (off - 1) * w, rhos, width=w, label=name, color=color)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Spearman rho vs Δlog P(※) at corrected slot")
    ax.legend(frameon=False)
    savefig_paper(fig, "followup_logp_leaderboard", dir=fig_dir)
    plt.close(fig)
    logger.info("figures written under %s", fig_dir)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["a", "b", "all"], default="all")
    ap.add_argument("--sources", default="all")
    ap.add_argument("--bystanders", default="all")
    ap.add_argument("--n-probes", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    instructed_panel = _instructed_bystander_panel()
    bystanders_all = SOURCES_ALL + sorted(instructed_panel)
    sources = SOURCES_ALL if args.sources == "all" else args.sources.split(",")
    bystanders = bystanders_all if args.bystanders == "all" else args.bystanders.split(",")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("a", "all"):
        phase_a(args.out_dir, sources, bystanders, args.n_probes)
    if args.phase in ("b", "all"):
        analysis = phase_b(args.out_dir, sources, bystanders)
        print(json.dumps({k: analysis[k] for k in ("spearman",)}, indent=1)[:2000])


if __name__ == "__main__":
    main()
