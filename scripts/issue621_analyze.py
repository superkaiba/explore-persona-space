"""Issue #621 analysis (Phase A — VM CPU, off-pod): rank-1 read/write reads.

Implements plan §4.5 + the §14 analyzer duties over UPLOADED artifacts
(adapters + A-init snapshots from the HF model repo; context bank + shift
tensors from the HF data repo; eval JSONs from git). At r=1 the per-module
update is exactly ΔW = s·b·aᵀ, so every number below is deterministic
linear algebra over stored weights and centroids — no model calls.

Per cell (30 = read/write/bridge × sources × seeds):

  H1 read identity   |cos(â, v̂_c_src)| per (module, layer, position, space)
                     vs the dedup wrong-context null (p95 + top-3 + ≥3
                     contiguous layers in L14–24) + shuffled-pairing null +
                     random floor 1/√d. Read arm compares a in the attn
                     (post-LN) space AND a∘γ in raw-residual space; write
                     arm compares a in o_in / down_in module-input spaces.
  H2 A-init          band-mean |cos(a_t, a_init)|, ‖Δa‖/‖a_init‖,
                     cos(Δâ, v̂_c_src).
  duty 10            a(t) rotation trajectory over the persisted 10-step
                     checkpoint ladder (checkpoint-*/ under each cell's
                     adapter folder; rides the recursive adapter upload +
                     the fetch-artifacts suffix filter): per checkpoint,
                     band-mean |cos(a_t, a_init)| and ‖a_t − a_init‖/
                     ‖a_init‖, plus whether rotation is still growing at
                     the band-stop step (the "didn't need to train" vs
                     "hadn't trained yet" separator). Per-cell trajectory
                     JSON (a_trajectory__<slug>.json) + a summary table.
  H3 write identity  cos(b̂, Ŵ_U[※]) layer profile (+ EOS-margin direction)
                     for residual-output modules (o/down); max over L20–27
                     raced against wrong-token nulls passed through the
                     SAME max-over-8-layers selection (duty 9);
                     cos(b̂_L20, measured L20 source shift) secondary.
  H4 firing→leakage  per-cell Spearman between predicted firing and
                     measured per-bystander leakage. Read arm: |firing|
                     Σ_lm s·|a·v_c′|·‖b‖ AND signed firing with the global
                     sign fixed by a·v_src > 0 (duty 6). Write arm: signed
                     Σ_lm s·(a·x_c′)·(W_U[※]·b) vs Δz-margin AND Δlog P.
                     PRIMARY bystander set excludes the trained negatives
                     present in the eval panel (assistant,
                     kindergarten_teacher); including version secondary
                     (duty 4). PRIMARY position end_of_response; other two
                     sensitivity (duty 7). Comparators (duty 5): base
                     prior, centered bank geometry cos(v_c′, v_src),
                     context norm ‖v_c′‖, a_init-firing — paired per-cell
                     differences, clustered on source (duty 12).
  H5 seed stability  cross-seed |cos| of â and b̂ within (arm, source).
  duty 8             bystander Δlog P spread vs per-persona SE per cell;
                     banded vs below-band split carried on every summary.
  duty 12            (a) split-half base check: base prior from the even
                     question-index half, Δ DV from the odd half (and the
                     swapped direction) via the per-question PER-SIDE slot
                     stats — breaks the base-side noise coupling between
                     the base-prior comparator and the DV. (b) paired
                     per-cell ρ differences (firing − each comparator)
                     clustered on source (4 effective units): sign test +
                     t-CI over source means, in the analysis summary.

Output: ``<out>/analysis.json`` (+ ``analysis_summary.json``).

CLI:
    uv run python scripts/issue621_analyze.py build-unembedding
    uv run python scripts/issue621_analyze.py fetch-artifacts
    uv run python scripts/issue621_analyze.py run [--adapters-root ...] [--bank ...]
"""

# ruff: noqa: RUF002, RUF003  # math notation

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_621 import (
    BANK_CAPTURE_POSITIONS,
    EXTRACTION_LAYER,
    HF_ADAPTER_PATH_PREFIX,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_BUCKET,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    IM_END_ID,
    MARKER_ID,
    PLACEMENT_ARMS,
    RECIPE_LORA_ALPHA,
    RECIPE_LORA_R,
    SEEDS,
    SOURCES,
    UNIFIED_NEGATIVE_PANEL,
    enumerate_cells,
    parse_cell_slug,
)

log = logging.getLogger("issue_621.analyze")

S_SCALE = RECIPE_LORA_ALPHA / (RECIPE_LORA_R**0.5)  # α/√r = 8 (≡ α/r at r=1)
BAND_LAYERS = range(14, 25)  # L14–24 inclusive (presence-criterion band)
H3_LAYERS = range(20, 28)  # L20–27 inclusive (max-selection window)
PRIMARY_POSITION = "end_of_response"
# Trained negatives present in the 19-persona eval panel (duty 4).
TRAINED_NEGATIVES_IN_PANEL = ("assistant", "kindergarten_teacher")

# Module → bank tap holding its INPUT space.
MODULE_INPUT_TAP = {
    "q_proj": "attn",
    "k_proj": "attn",
    "v_proj": "attn",
    "o_proj": "o_in",
    "down_proj": "down_in",
}
# Modules whose OUTPUT adds to the residual stream (W_U-readable b).
RESIDUAL_OUTPUT_MODULES = ("o_proj", "down_proj")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho) if rho is not None else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_adapter_pairs(adapter_dir: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Load rank-1 (a, b) per (layer, module) from a PEFT safetensors file.

    Keys look like ``base_model.model.model.layers.{L}.self_attn.q_proj.
    lora_A.weight`` (A: [1, d_in]) / ``...lora_B.weight`` (B: [d_out, 1]).
    Asserts r == 1 on every tensor.
    """
    from safetensors.numpy import load_file

    st = adapter_dir / "adapter_model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(st)
    sd = load_file(str(st))
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, tensor in sd.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue
        parts = key.split(".")
        li = int(parts[parts.index("layers") + 1])
        module = parts[parts.index("layers") + 3]  # self_attn|mlp . <module>
        slot = out.setdefault((li, module), {})
        arr = np.asarray(tensor, dtype=np.float32)
        if ".lora_A." in key:
            assert arr.shape[0] == 1, (key, arr.shape, "rank != 1")
            slot["a"] = arr[0]
        else:
            assert arr.shape[1] == 1, (key, arr.shape, "rank != 1")
            slot["b"] = arr[:, 0]
    if not out:
        raise AssertionError(f"no lora_A/lora_B tensors in {st}")
    for (li, module), slot in out.items():
        if "a" not in slot or "b" not in slot:
            raise AssertionError(f"incomplete (a,b) pair at layer {li} module {module}")
    return out


def load_bank(bank_dir: Path) -> dict:
    """Load centroids.pt + rmsnorm_gamma.pt + manifest.json."""
    import torch

    centroids_payload = torch.load(bank_dir / "centroids.pt", weights_only=True)
    gamma_payload = torch.load(bank_dir / "rmsnorm_gamma.pt", weights_only=True)
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    centroids: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for tap, by_pos in centroids_payload["centroids"].items():
        centroids[tap] = {}
        for pos, by_ctx in by_pos.items():
            centroids[tap][pos] = {
                name: np.asarray(t.numpy(), dtype=np.float32) for name, t in by_ctx.items()
            }
    gamma_in = np.asarray(gamma_payload["input_layernorm"].numpy(), dtype=np.float32)
    return {"centroids": centroids, "gamma_in": gamma_in, "manifest": manifest}


def load_eval_cells(eval_root: Path, cells_root: Path) -> dict[str, dict]:
    """Join shift JSONs with the train-side cell metadata, keyed by slug."""
    out: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for sub in ("anchor_smoke", "sweep"):
        d = cells_root / sub
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name in ("summary.json", "smoke_gate.json"):
                    continue
                payload = json.loads(p.read_text())
                if "cell_slug" in payload:
                    meta[payload["cell_slug"]] = payload
    for p in sorted(eval_root.glob("*__shift.json")):
        payload = json.loads(p.read_text())
        slug = payload["cell_slug"]
        payload["_train_meta"] = meta.get(slug, {})
        out[slug] = payload
    if not out:
        raise FileNotFoundError(f"no *__shift.json under {eval_root}")
    return out


def load_unembedding(path: Path) -> dict:
    import torch

    payload = torch.load(path, weights_only=True)
    return {
        "marker": np.asarray(payload["W_U_marker"].numpy(), dtype=np.float32),
        "eos": np.asarray(payload["W_U_eos"].numpy(), dtype=np.float32),
        "null_norm_matched": np.asarray(payload["W_U_null_norm_matched"].numpy(), np.float32),
        "null_random": np.asarray(payload["W_U_null_random"].numpy(), dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands: build-unembedding / fetch-artifacts
# ─────────────────────────────────────────────────────────────────────────────


def cmd_build_unembedding(args) -> int:
    """Extract W_U rows (marker, eos, null samples) from the base model shard.

    Downloads ONLY the safetensors shard holding ``lm_head.weight`` (via the
    index json), never the full model. The wrong-token null is sampled two
    ways: norm-matched (rows within ±10% of ‖W_U[※]‖ — the cheap frequency
    proxy) and uniform-random; both seeded.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model_id = args.model
    idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
    idx = json.loads(Path(idx_path).read_text())
    shard_name = idx["weight_map"].get("lm_head.weight")
    if shard_name is None:
        raise SystemExit(
            f"{model_id} has no lm_head.weight in the index (tied embeddings?) — "
            "the W_U readout assumption is violated; investigate before analysis."
        )
    shard_path = hf_hub_download(model_id, shard_name)
    with safe_open(shard_path, framework="pt") as f:
        w_u = f.get_tensor("lm_head.weight").float()  # (vocab, hidden)
    vocab, hidden = w_u.shape
    assert vocab > max(MARKER_ID, IM_END_ID), (vocab, MARKER_ID, IM_END_ID)
    marker_row = w_u[MARKER_ID].clone()
    eos_row = w_u[IM_END_ID].clone()

    rng = np.random.default_rng(621)
    norms = torch.linalg.norm(w_u, dim=1)
    m_norm = float(torch.linalg.norm(marker_row))
    mask = (norms >= 0.9 * m_norm) & (norms <= 1.1 * m_norm)
    mask[MARKER_ID] = False
    mask[IM_END_ID] = False
    cand = torch.nonzero(mask, as_tuple=False).flatten().numpy()
    k = min(args.n_null, len(cand))
    norm_ids = rng.choice(cand, size=k, replace=False)
    rand_ids = rng.choice(
        np.setdiff1d(np.arange(vocab), np.array([MARKER_ID, IM_END_ID])),
        size=args.n_null,
        replace=False,
    )
    out = {
        "W_U_marker": marker_row,
        "W_U_eos": eos_row,
        "W_U_null_norm_matched": w_u[torch.from_numpy(norm_ids)].clone(),
        "W_U_null_random": w_u[torch.from_numpy(rand_ids)].clone(),
        "null_norm_matched_ids": torch.from_numpy(norm_ids),
        "null_random_ids": torch.from_numpy(np.asarray(rand_ids)),
        "model": model_id,
        "hidden": hidden,
    }
    out_path = Path(args.unembedding)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    log.info(
        "unembedding rows saved: %s (hidden=%d, %d norm-matched + %d random nulls)",
        out_path,
        hidden,
        k,
        args.n_null,
    )
    return 0


def _fetch_prefix(
    *,
    hf_hub_download,
    repo_files: list[str],
    repo_id: str,
    repo_type: str,
    prefix: str,
    dest_root: Path,
    suffixes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Download every repo file under ``prefix/`` into ``dest_root``.

    Per-file ``hf_hub_download`` over an explicit COMPLETE listing — NEVER
    ``snapshot_download(allow_patterns=...)`` (silent truncation on big
    repos). Relative layout under ``prefix`` is preserved under
    ``dest_root``. Returns the local paths (existing files are kept).
    """
    wanted = [f for f in repo_files if f.startswith(prefix + "/")]
    if suffixes is not None:
        wanted = [f for f in wanted if f.endswith(suffixes)]
    out: list[Path] = []
    for f in wanted:
        local = hf_hub_download(repo_id, f, repo_type=repo_type)
        dest = dest_root / Path(f).relative_to(prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            dest.write_bytes(Path(local).read_bytes())
        out.append(dest)
    return out


def cmd_fetch_artifacts(args) -> int:
    """Fetch EVERY off-pod Phase A input from HF into the local layout.

    Round-2 (concern ``analysis-fetch-missing-shifts``): the pipeline never
    commits ``eval_results/`` to git and the GCP instance self-deletes, so
    the HF uploads are the ONLY durable copies. ``run`` requires, locally:

      - adapters + A-init snapshots      → ``--adapters-root`` (model repo)
      - context bank bundle              → ``--bank``
      - ``*__shift.json`` + ``*__shift.pt``  → ``--eval-root``
        (``load_eval_cells`` raises FileNotFoundError without the JSONs;
        the H3 L20-shift secondary reads the ``.pt``)
      - train-side cell metadata JSONs   → ``--cells-root/{anchor_smoke,
        sweep}/`` (band_stop_fired / band_stop_step /
        final_source_delta_nats — the banded-vs-below-band split)
      - eval emission JSONs + band trajectories (duty-10/12 post-hoc
        inputs) → ``--eval-root`` / ``--cells-root/cells/<slug>/``

    Listings use ``list_repo_files_complete`` (paginated tree API): raw
    ``list_repo_files`` silently truncates at ~7.9k entries and the model
    repo is far past that.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    api = HfApi()

    adapters_root = Path(args.adapters_root)
    adapters_root.mkdir(parents=True, exist_ok=True)
    model_files = list_repo_files_complete(api, HF_MODEL_REPO, repo_type="model")
    fetched = _fetch_prefix(
        hf_hub_download=hf_hub_download,
        repo_files=model_files,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        prefix=HF_ADAPTER_PATH_PREFIX,
        dest_root=adapters_root,
        suffixes=("adapter_model.safetensors", "adapter_config.json"),
    )
    if not fetched:
        raise SystemExit(f"no adapter files under {HF_MODEL_REPO}/{HF_ADAPTER_PATH_PREFIX}/")
    # The suffix filter also pulls the 10-step checkpoint-*/ ladder files
    # (duty 10) — they live under each cell folder on the model repo.
    n_ladder = sum(1 for p in fetched if "checkpoint-" in p.as_posix())
    log.info(
        "fetched %d adapter files (%d checkpoint-ladder files) -> %s",
        len(fetched),
        n_ladder,
        adapters_root,
    )

    data_files = list_repo_files_complete(api, HF_DATA_REPO, repo_type="dataset")

    bank_dir = Path(args.bank)
    bank_dir.mkdir(parents=True, exist_ok=True)
    for name in ("centroids.pt", "rmsnorm_gamma.pt", "manifest.json", "responses.json"):
        local = hf_hub_download(
            HF_DATA_REPO, f"{HF_ANALYSIS_TENSORS_PREFIX}/{name}", repo_type="dataset"
        )
        dest = bank_dir / name
        if not dest.is_file():
            dest.write_bytes(Path(local).read_bytes())
    log.info("fetched bank -> %s", bank_dir)

    # Shift artifacts (JSON + .pt) — REQUIRED by load_eval_cells/cmd_run.
    eval_root = Path(args.eval_root)
    shifts = _fetch_prefix(
        hf_hub_download=hf_hub_download,
        repo_files=data_files,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        prefix=f"{HF_ANALYSIS_TENSORS_PREFIX}/shifts",
        dest_root=eval_root,
        suffixes=("__shift.json", "__shift.pt"),
    )
    n_shift_json = sum(1 for p in shifts if p.name.endswith("__shift.json"))
    n_shift_pt = sum(1 for p in shifts if p.name.endswith("__shift.pt"))
    if n_shift_json == 0 or n_shift_pt == 0:
        raise SystemExit(
            f"shift artifacts missing on Hub under {HF_DATA_REPO}/"
            f"{HF_ANALYSIS_TENSORS_PREFIX}/shifts/ (json={n_shift_json}, "
            f"pt={n_shift_pt}) — `run` cannot proceed without them; was "
            "i621_upload_artifacts.py run on the instance?"
        )
    log.info("fetched %d shift JSONs + %d shift tensors -> %s", n_shift_json, n_shift_pt, eval_root)

    # Eval emission JSONs (per-cell on-policy emission anchors).
    emissions = _fetch_prefix(
        hf_hub_download=hf_hub_download,
        repo_files=data_files,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        prefix=f"{HF_BUCKET}/eval",
        dest_root=eval_root,
        suffixes=("__emission.json",),
    )
    if emissions:
        log.info("fetched %d emission JSONs -> %s", len(emissions), eval_root)
    else:
        log.warning("no emission JSONs under %s/eval on Hub (smoke-only run?)", HF_BUCKET)

    # Train-side cell metadata — the banded-vs-below-band join inputs.
    cells_root = Path(args.cells_root)
    metas = _fetch_prefix(
        hf_hub_download=hf_hub_download,
        repo_files=data_files,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        prefix=f"{HF_BUCKET}/train_meta",
        dest_root=cells_root,
        suffixes=(".json",),
    )
    if not metas:
        raise SystemExit(
            f"no train-side cell metadata under {HF_DATA_REPO}/{HF_BUCKET}/train_meta/ "
            "— the banded/below-band split is unrunnable; re-run "
            "i621_upload_artifacts.py (class 6) before instance deletion."
        )
    log.info("fetched %d train-meta JSONs -> %s", len(metas), cells_root)

    # Band trajectories (duty-10/12 post-hoc reads; tiny).
    trajs = _fetch_prefix(
        hf_hub_download=hf_hub_download,
        repo_files=data_files,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        prefix=f"{HF_BUCKET}/trajectories",
        dest_root=cells_root / "cells",
        suffixes=(".json",),
    )
    if trajs:
        log.info("fetched %d trajectory JSONs -> %s", len(trajs), cells_root / "cells")
    else:
        log.warning("no band trajectories under %s/trajectories on Hub", HF_BUCKET)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Core reads
# ─────────────────────────────────────────────────────────────────────────────


def _gauge_assert(adapter_dir: Path) -> None:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise AssertionError(f"adapter_config.json missing at {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    tm = cfg.get("target_modules", [])
    if isinstance(tm, str):
        tm = [tm]
    bad = [m for m in tm if "lm_head" in m or "embed_tokens" in m]
    if bad:
        raise AssertionError(f"gauge assert FAIL: target_modules include {bad} at {adapter_dir}")
    if cfg.get("modules_to_save"):
        raise AssertionError(f"gauge assert FAIL: modules_to_save non-empty at {adapter_dir}")
    if int(cfg.get("r", -1)) != RECIPE_LORA_R:
        raise AssertionError(f"adapter r={cfg.get('r')} != {RECIPE_LORA_R} at {adapter_dir}")


def _contiguous_count(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _read_identity_for_space(
    *,
    pairs: dict,
    modules: list[str],
    bank: dict,
    source: str,
    pos: str,
    space: str,
    use_init: bool = False,
    init_pairs: dict | None = None,
) -> dict:
    """|cos(â, v̂_c)| vs the dedup wrong-context null, one (position, space).

    ``space`` is either a bank tap name (module-input comparison) or
    ``raw_gamma`` (a∘γ against the raw-residual centroids — the read-arm
    raw-space variant; only valid for post-LN-input modules q/k/v).
    Returns per-module layer profiles + the presence-criterion verdict.
    """
    centroids = bank["centroids"]
    gamma_in = bank["gamma_in"]
    tap = "raw" if space == "raw_gamma" else space
    contexts = centroids[tap][pos]
    if source not in contexts:
        raise AssertionError(f"source {source!r} missing from bank tap {tap!r} pos {pos!r}")
    n_layers = next(iter(contexts.values())).shape[0]

    per_module: dict[str, dict] = {}
    for module in modules:
        cos_src: list[float] = []
        null_p95: list[float] = []
        top3: list[bool] = []
        beats_p95: list[bool] = []
        shuffled_p95: list[float] = []
        d_in = None
        for li in range(n_layers):
            key = (li, module)
            src_pairs = init_pairs if use_init else pairs
            if key not in src_pairs:
                cos_src.append(float("nan"))
                null_p95.append(float("nan"))
                top3.append(False)
                beats_p95.append(False)
                shuffled_p95.append(float("nan"))
                continue
            a = src_pairs[key]["a"]
            d_in = a.shape[0]
            a_cmp = a * gamma_in[li] if space == "raw_gamma" else a
            vals: dict[str, float] = {}
            for ctx_name, cent in contexts.items():
                vals[ctx_name] = abs(_cos(a_cmp, cent[li]))
            src_val = vals[source]
            null_vals = [v for k_, v in vals.items() if k_ != source]
            p95 = float(np.percentile(null_vals, 95)) if null_vals else float("nan")
            rank = sum(v > src_val for v in vals.values())  # 0 = top-1
            cos_src.append(src_val)
            null_p95.append(p95)
            beats_p95.append(src_val > p95)
            top3.append(rank < 3)
            # Shuffled-pairing null: this a vs OTHER sources' centroids.
            other_src_vals = [
                abs(_cos(a_cmp, contexts[s][li])) for s in SOURCES if s != source and s in contexts
            ]
            shuffled_p95.append(
                float(np.percentile(other_src_vals, 95)) if other_src_vals else float("nan")
            )
        band_flags = [beats_p95[li] and top3[li] for li in BAND_LAYERS if li < len(beats_p95)]
        per_module[module] = {
            "cos_src_per_layer": cos_src,
            "null_p95_per_layer": null_p95,
            "shuffled_pairing_p95_per_layer": shuffled_p95,
            "random_floor": (1.0 / np.sqrt(d_in)) if d_in else None,
            "presence_pass": _contiguous_count(band_flags) >= 3,
            "band_layers_passing": int(sum(band_flags)),
            "band_mean_cos_src": float(
                np.nanmean([cos_src[li] for li in BAND_LAYERS if li < len(cos_src)])
            ),
            "band_mean_null_p95": float(
                np.nanmean([null_p95[li] for li in BAND_LAYERS if li < len(null_p95)])
            ),
        }
    return per_module


def _a_init_reads(pairs: dict, init_pairs: dict, bank: dict, source: str) -> dict:
    """H2: |cos(a_t, a_init)| + ‖Δa‖/‖a_init‖ + cos(Δâ, v̂_c_src) per module.

    Both inputs MUST be UNFLIPPED adapter pair dicts (same contract as
    :func:`_band_mean_a_vs_init`): the H2 statistics read the ACTUAL
    training trajectory against its own saved init, and ‖a_t − a_init‖ /
    the signed cos(Δâ, v̂_c) are not invariant under the joint
    (a,b) → (−a,−b) W_U sign fix — flipping a_t but not a_init manufactures
    rel_delta ≈ 2 + sign-inverted Δâ on write-arm slots with W_U[※]·b < 0.
    |cos(a_t, a_init)| is sign-folded either way. Gauge recorded in the
    analysis meta as ``a_init_gauge: unflipped-pair`` (round 4, concern
    h2-a-init-reads-after-sign-flip).
    """
    out: dict[str, dict] = {}
    centroids = bank["centroids"]
    by_module: dict[str, list[tuple[int, dict, dict]]] = {}
    for (li, module), slot in pairs.items():
        by_module.setdefault(module, []).append((li, slot, init_pairs[(li, module)]))
    for module, rows in by_module.items():
        tap = MODULE_INPUT_TAP[module]
        ctxs = centroids[tap][PRIMARY_POSITION]
        cos_ai: list[float] = []
        rel_delta: list[float] = []
        cos_delta_vc: list[float] = []
        band_cos_ai: list[float] = []
        band_rel: list[float] = []
        for li, slot, init_slot in sorted(rows):
            a_t, a_0 = slot["a"], init_slot["a"]
            c = abs(_cos(a_t, a_0))
            r = float(np.linalg.norm(a_t - a_0) / max(np.linalg.norm(a_0), 1e-30))
            delta = a_t - a_0
            cv = _cos(delta, ctxs[source][li]) if source in ctxs else float("nan")
            cos_ai.append(c)
            rel_delta.append(r)
            cos_delta_vc.append(cv)
            if li in BAND_LAYERS:
                band_cos_ai.append(c)
                band_rel.append(r)
        out[module] = {
            "cos_a_init_per_layer": cos_ai,
            "rel_delta_a_per_layer": rel_delta,
            "cos_delta_a_vs_vc_per_layer": cos_delta_vc,
            "band_mean_cos_a_init": float(np.mean(band_cos_ai)) if band_cos_ai else None,
            "band_mean_rel_delta_a": float(np.mean(band_rel)) if band_rel else None,
        }
    return out


def _band_mean_a_vs_init(pairs_t: dict, init_pairs: dict) -> tuple[float | None, float | None]:
    """Band-mean (L14–24, all modules) |cos(a_t, a_init)| and ‖Δa‖/‖a_init‖.

    Both inputs are UNFLIPPED adapter pair dicts (the W_U sign convention
    must not enter the init comparison — ‖a_t − a_init‖ is not invariant
    under the joint (a,b) → (−a,−b) flip).
    """
    cos_vals: list[float] = []
    rel_vals: list[float] = []
    for key, slot in pairs_t.items():
        li, _module = key
        if li not in BAND_LAYERS or key not in init_pairs:
            continue
        a_t, a_0 = slot["a"], init_pairs[key]["a"]
        cos_vals.append(abs(_cos(a_t, a_0)))
        rel_vals.append(float(np.linalg.norm(a_t - a_0) / max(np.linalg.norm(a_0), 1e-30)))
    if not cos_vals:
        return None, None
    return float(np.mean(cos_vals)), float(np.mean(rel_vals))


_A_ROTATION_RULE = (
    "still-growing = last inter-checkpoint increment of band-mean ‖Δa‖/‖a_init‖ "
    "is > 0 AND >= 50% of the max single-step increment"
)


def _a_rotation_trajectory(
    *,
    adapter_dir: Path,
    init_pairs: dict,
    band_stop_step: int | None,
) -> dict:
    """Duty 10: a(t) rotation read over the persisted 10-step checkpoint ladder.

    Walks ``checkpoint-<N>/`` adapter snapshots under the cell folder
    (written by ``save_steps=10``; they ride the recursive adapter upload
    and the ``fetch-artifacts`` suffix filter), computing per checkpoint
    the band-mean |cos(a_t, a_init)| and ‖a_t − a_init‖/‖a_init‖ against
    the cell's saved A-init. The final adapter is appended as the terminal
    point (deduped when it is metrically identical to the last checkpoint,
    i.e. the band-stop landed exactly on a save step). All loads are FRESH
    (unflipped) — see :func:`_band_mean_a_vs_init`.

    "Rotation still growing at the band-stop step" separates "didn't need
    to train" (flat ladder) from "hadn't trained yet" (still rising); the
    decision rule is recorded in the payload (``rule``).
    """
    ckpts: list[tuple[int, Path]] = []
    for d in sorted(adapter_dir.glob("checkpoint-*")):
        if not (d / "adapter_model.safetensors").is_file():
            continue
        tail = d.name.rsplit("-", 1)[-1]
        if not tail.isdigit():
            continue
        ckpts.append((int(tail), d))
    ckpts.sort()

    points: list[dict] = []
    for step, d in ckpts:
        cos_m, rel_m = _band_mean_a_vs_init(load_adapter_pairs(d), init_pairs)
        points.append(
            {
                "step": step,
                "is_final": False,
                "band_mean_abs_cos_a_init": cos_m,
                "band_mean_rel_delta_a": rel_m,
            }
        )

    cos_f, rel_f = _band_mean_a_vs_init(load_adapter_pairs(adapter_dir), init_pairs)
    duplicate_of_last = bool(
        points
        and cos_f is not None
        and rel_f is not None
        and points[-1]["band_mean_abs_cos_a_init"] is not None
        and points[-1]["band_mean_rel_delta_a"] is not None
        and abs(cos_f - points[-1]["band_mean_abs_cos_a_init"]) < 1e-9
        and abs(rel_f - points[-1]["band_mean_rel_delta_a"]) < 1e-9
    )
    if not duplicate_of_last:
        points.append(
            {
                "step": band_stop_step,
                "is_final": True,
                "band_mean_abs_cos_a_init": cos_f,
                "band_mean_rel_delta_a": rel_f,
            }
        )

    rels = [pt["band_mean_rel_delta_a"] for pt in points]
    increments: list[float] = []
    if len(rels) >= 2 and all(r is not None for r in rels):
        increments = [rels[i] - rels[i - 1] for i in range(1, len(rels))]
    last_inc = increments[-1] if increments else None
    max_inc = max(increments) if increments else None
    still_growing: bool | None = None
    if increments and max_inc is not None and max_inc > 0:
        still_growing = bool(last_inc > 0 and last_inc >= 0.5 * max_inc)
    return {
        "n_checkpoints": len(ckpts),
        "checkpoint_steps": [s for s, _ in ckpts],
        "band_stop_step": band_stop_step,
        "points": points,
        "rel_delta_increments": increments,
        "last_increment_rel": last_inc,
        "max_increment_rel": max_inc,
        "rotation_still_growing_at_stop": still_growing,
        "final_terminal_point_deduped": duplicate_of_last,
        "rule": _A_ROTATION_RULE,
    }


def _split_half_base_check(
    *,
    contexts_dv: dict,
    bystanders: list[str],
    firing: dict[str, float],
    firing_mode: str,
) -> dict:
    """Duty 12(a): split-half base check over the per-question slot stats.

    The base-prior comparator (base-side ``logp_marker``) and the Δ log P
    DV (trained − base) share the SAME base-side forward passes, coupling
    their noise. Splitting the eval questions into deterministic even/odd
    index halves — base prior estimated from one half, the DV from the
    other — breaks the coupling. Both split directions are reported, plus
    the firing predictor's ρ against the same half-DV (the predictor is
    question-independent, so this is the apples-to-apples race partner).
    Spearman is across bystanders (the primary set), as in H4.
    """
    pq_prior: dict[str, list[float]] = {}
    pq_dlogp: dict[str, list[float]] = {}
    pq_dmargin: dict[str, list[float]] = {}
    n_q: int | None = None
    for p in bystanders:
        stats = contexts_dv[p].get("per_question_slot_stats")
        if not stats or "base" not in stats:
            raise AssertionError(
                f"per_question_slot_stats missing for bystander {p!r} — the "
                "round-2 eval persists it per side; duty-12 split-half needs it."
            )
        base_logp = [float(v) for v in stats["base"]["logp_marker"]]
        dlogp = [float(v) for v in contexts_dv[p]["per_question_delta_logp"]]
        dmargin = [float(v) for v in contexts_dv[p]["per_question_delta_margin"]]
        if n_q is None:
            n_q = len(base_logp)
        if not (len(base_logp) == len(dlogp) == len(dmargin) == n_q):
            raise AssertionError(
                f"per-question array lengths disagree for {p!r}: "
                f"{(len(base_logp), len(dlogp), len(dmargin), n_q)}"
            )
        pq_prior[p] = base_logp
        pq_dlogp[p] = dlogp
        pq_dmargin[p] = dmargin
    assert n_q is not None and n_q >= 2, f"need >= 2 questions for a split-half (got {n_q})"
    even = list(range(0, n_q, 2))
    odd = list(range(1, n_q, 2))

    def _half_mean(per_q: dict[str, list[float]], idx: list[int]) -> dict[str, float]:
        return {p: float(np.mean([per_q[p][i] for i in idx])) for p in bystanders}

    splits: dict[str, dict] = {}
    for name, prior_idx, dv_idx in (
        ("prior_even_dv_odd", even, odd),
        ("prior_odd_dv_even", odd, even),
    ):
        prior = _half_mean(pq_prior, prior_idx)
        dv_l = _half_mean(pq_dlogp, dv_idx)
        dv_m = _half_mean(pq_dmargin, dv_idx)
        ordered = list(bystanders)
        splits[name] = {
            "rho_base_prior_vs_delta_logp": _spearman(
                [prior[p] for p in ordered], [dv_l[p] for p in ordered]
            ),
            "rho_base_prior_vs_delta_margin": _spearman(
                [prior[p] for p in ordered], [dv_m[p] for p in ordered]
            ),
            "rho_firing_vs_delta_logp": _spearman(
                [firing[p] for p in ordered], [dv_l[p] for p in ordered]
            ),
            "rho_firing_vs_delta_margin": _spearman(
                [firing[p] for p in ordered], [dv_m[p] for p in ordered]
            ),
        }
    return {
        "n_questions": n_q,
        "halves": {"even": len(even), "odd": len(odd)},
        "firing_mode": firing_mode,
        "n_bystanders": len(bystanders),
        "splits": splits,
        "note": (
            "base prior from one deterministic question half, DV from the other "
            "— breaks the base-side noise coupling (duty 12a); full coupled "
            "base-prior rho lives in h4.comparators_primary"
        ),
    }


def _write_identity(pairs: dict, wu: dict, shift_src_l20: np.ndarray | None) -> dict:
    """H3: W_U reads for residual-output modules + matched max-selection null."""
    w_marker = _unit(wu["marker"])
    w_margin = _unit(wu["marker"] - wu["eos"])
    out: dict[str, dict] = {}
    by_module: dict[str, list[tuple[int, dict]]] = {}
    for (li, module), slot in pairs.items():
        if module in RESIDUAL_OUTPUT_MODULES:
            by_module.setdefault(module, []).append((li, slot))
    for module, rows in by_module.items():
        rows = sorted(rows)
        cos_wu = [_cos(slot["b"], w_marker) for _, slot in rows]
        cos_margin = [_cos(slot["b"], w_margin) for _, slot in rows]
        h3_vals = [cos_wu[li] for li, _ in rows if li in H3_LAYERS]
        max_l20_27 = float(np.max(h3_vals)) if h3_vals else float("nan")
        # Matched max-selection nulls (duty 9): each null token row goes
        # through the SAME max-over-L20–27 selection on the same b̂_l.
        b_units = {li: _unit(slot["b"]) for li, slot in rows if li in H3_LAYERS}
        null_maxes: dict[str, list[float]] = {}
        for null_name in ("null_norm_matched", "null_random"):
            rows_w = wu[null_name]
            maxes: list[float] = []
            if b_units:
                b_mat = np.stack(list(b_units.values()))  # (L, d)
                w_mat = rows_w / np.clip(np.linalg.norm(rows_w, axis=1, keepdims=True), 1e-30, None)
                cos_mat = w_mat @ b_mat.T  # (K, L)
                maxes = np.max(np.abs(cos_mat), axis=1).tolist()
            null_maxes[null_name] = maxes
        cos_shift_l20 = None
        if shift_src_l20 is not None:
            l20 = [slot["b"] for li, slot in rows if li == EXTRACTION_LAYER]
            if l20:
                cos_shift_l20 = _cos(l20[0], shift_src_l20)
        out[module] = {
            "cos_wu_marker_per_layer": cos_wu,
            "cos_wu_eos_margin_per_layer": cos_margin,
            "max_cos_wu_L20_27": max_l20_27,
            "null_max_p95_norm_matched": (
                float(np.percentile(null_maxes["null_norm_matched"], 95))
                if null_maxes["null_norm_matched"]
                else None
            ),
            "null_max_p95_random": (
                float(np.percentile(null_maxes["null_random"], 95))
                if null_maxes["null_random"]
                else None
            ),
            "cos_b_L20_vs_measured_shift": cos_shift_l20,
        }
    return out


def _sign_fix_write(pairs: dict, wu: dict) -> None:
    """(a,b) ≡ (−a,−b): flip so W_U[※]·b > 0 on residual-output modules."""
    w = wu["marker"]
    for (_li, module), slot in pairs.items():
        if module in RESIDUAL_OUTPUT_MODULES and float(w @ slot["b"]) < 0:
            slot["a"] = -slot["a"]
            slot["b"] = -slot["b"]


def _firing(
    *,
    pairs: dict,
    bank: dict,
    contexts: list[str],
    pos: str,
    mode: str,
    wu: dict | None = None,
    source: str | None = None,
) -> dict[str, float]:
    """Predicted firing per context (plan §3 H4).

    mode='abs'         Σ_lm s·|a·v_c|·‖b‖                 (read-arm primary)
    mode='signed_src'  Σ_lm s·(a·v_c)·‖b‖ with per-(l,m) sign fixed by
                       a·v_src > 0 (duty 6)
    mode='write'       Σ_lm s·(a·x_c)·(W_U[※]·b) over residual-output
                       modules (write-arm signed prediction)
    """
    centroids = bank["centroids"]
    out: dict[str, float] = {c: 0.0 for c in contexts}
    for (li, module), slot in pairs.items():
        if mode == "write" and module not in RESIDUAL_OUTPUT_MODULES:
            continue
        tap = MODULE_INPUT_TAP[module]
        ctx_bank = centroids[tap][pos]
        a, b = slot["a"], slot["b"]
        if mode == "write":
            assert wu is not None
            w_dot_b = float(wu["marker"] @ b)
        sign = 1.0
        if mode == "signed_src":
            assert source is not None
            if float(a @ ctx_bank[source][li]) < 0:
                sign = -1.0
        b_norm = float(np.linalg.norm(b))
        for c in contexts:
            dot = float(a @ ctx_bank[c][li])
            if mode == "abs":
                out[c] += S_SCALE * abs(dot) * b_norm
            elif mode == "signed_src":
                out[c] += S_SCALE * sign * dot * b_norm
            else:  # write
                out[c] += S_SCALE * dot * w_dot_b
    return out


def _centered_geometry(bank: dict, pos: str, contexts: list[str], source: str) -> dict[str, float]:
    """Comparator (duty 5a): centered bank cosine cos(v_c′, v_src), attn tap.

    Global-mean centering over the bank's contexts per the
    persona-distance-metrics rule (centering: global_mean; bank-dependent).
    """
    ctxs = bank["centroids"]["attn"][pos]
    all_names = sorted(ctxs)
    stacked = np.stack([ctxs[n] for n in all_names])  # (C, L, d)
    mu = stacked.mean(axis=0)  # (L, d)
    li_band = [li for li in BAND_LAYERS if li < stacked.shape[1]]
    out: dict[str, float] = {}
    src_c = ctxs[source]
    for c in contexts:
        vals = [_cos(ctxs[c][li] - mu[li], src_c[li] - mu[li]) for li in li_band]
        out[c] = float(np.mean(vals))
    return out


def _context_norm(bank: dict, pos: str, contexts: list[str]) -> dict[str, float]:
    ctxs = bank["centroids"]["attn"][pos]
    li_band = [li for li in BAND_LAYERS if li < next(iter(ctxs.values())).shape[0]]
    return {c: float(np.mean([np.linalg.norm(ctxs[c][li]) for li in li_band])) for c in contexts}


# ─────────────────────────────────────────────────────────────────────────────
# Main per-cell analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_cell(
    *,
    slug: str,
    adapter_dir: Path,
    bank: dict,
    eval_payload: dict,
    wu: dict,
) -> dict:
    """All §4.5 reads for one cell. Pure numpy; no model calls."""
    import torch

    arm, source, seed = parse_cell_slug(slug)
    tm = eval_payload.get("_train_meta", {})
    _gauge_assert(adapter_dir)
    pairs = load_adapter_pairs(adapter_dir)
    init_dir = adapter_dir / "adapter_init"
    init_pairs = load_adapter_pairs(init_dir)
    # Round-4 fix (concern h2-a-init-reads-after-sign-flip): every
    # a_init-referenced read runs in the UNFLIPPED pair gauge — snapshot
    # BEFORE the W_U sign fix mutates `pairs`. ‖a_t − a_init‖ and the
    # signed cos(Δâ, v̂_c) are not invariant under the joint
    # (a,b) → (−a,−b) flip, so comparing a flipped a_t against the
    # unflipped a_init manufactured rel_delta_a ≈ 2 and sign-inverted
    # Δâ artifacts on write-arm slots with W_U[※]·b < 0. Convention
    # (plan §3 H2; matches duty-10's _band_mean_a_vs_init): flip BOTH
    # sides or NEITHER — we flip neither (a_init_gauge: unflipped-pair).
    pairs_unflipped = {k: {"a": s["a"].copy(), "b": s["b"].copy()} for k, s in pairs.items()}
    _sign_fix_write(pairs, wu)

    # Duty 10: a(t) rotation trajectory over the 10-step checkpoint ladder
    # (fresh, UNFLIPPED loads — independent of the sign fix above).
    a_rotation = _a_rotation_trajectory(
        adapter_dir=adapter_dir,
        init_pairs=init_pairs,
        band_stop_step=tm.get("band_stop_step"),
    )
    if a_rotation["n_checkpoints"] == 0:
        log.warning(
            "%s: no checkpoint-* ladder under %s — duty-10 trajectory limited "
            "to the final adapter point",
            slug,
            adapter_dir,
        )

    expected_modules = set(PLACEMENT_ARMS[arm])
    realized_modules = {m for (_li, m) in pairs}
    if realized_modules != expected_modules:
        raise AssertionError(
            f"{slug}: realized modules {sorted(realized_modules)} != arm "
            f"placement {sorted(expected_modules)}"
        )

    # Measured L20 source shift (H3 secondary) from the eval .pt.
    shift_src_l20 = None
    pt_path = Path(eval_payload["_shift_pt_path"])
    if pt_path.is_file():
        mat = torch.load(pt_path, weights_only=True)
        mat = np.asarray(mat.numpy() if hasattr(mat, "numpy") else mat, dtype=np.float32)
        panel = eval_payload["eval_panel"]
        if source in panel:
            shift_src_l20 = mat[panel.index(source)]

    modules = sorted(realized_modules)
    post_ln_modules = [m for m in modules if MODULE_INPUT_TAP[m] == "attn"]

    # H1 read identity: module-input space per position (+ raw_gamma variant
    # for post-LN modules).
    read_identity: dict[str, dict] = {}
    for pos in BANK_CAPTURE_POSITIONS:
        spaces: dict[str, dict] = {}
        for tap in sorted({MODULE_INPUT_TAP[m] for m in modules}):
            mods = [m for m in modules if MODULE_INPUT_TAP[m] == tap]
            spaces[tap] = _read_identity_for_space(
                pairs=pairs, modules=mods, bank=bank, source=source, pos=pos, space=tap
            )
        if post_ln_modules:
            spaces["raw_gamma"] = _read_identity_for_space(
                pairs=pairs,
                modules=post_ln_modules,
                bank=bank,
                source=source,
                pos=pos,
                space="raw_gamma",
            )
        read_identity[pos] = spaces

    # H1 sensitivity (duty 4): wrong-context null EXCLUDING ALL FOUR
    # trained-negative contexts (the full UNIFIED_NEGATIVE_PANEL — `a` was
    # trained against all 4, so none is a valid "wrong context" here; round-2
    # fix for concern duty4-h1-null-excludes-2-of-4, which excluded only the
    # 2 panel members present in the 19-persona eval panel).
    bank_excl = {
        "centroids": {
            tap: {
                pos: {
                    n: v
                    for n, v in by_ctx.items()
                    if n not in UNIFIED_NEGATIVE_PANEL or n == source
                }
                for pos, by_ctx in by_pos.items()
            }
            for tap, by_pos in bank["centroids"].items()
        },
        "gamma_in": bank["gamma_in"],
    }
    read_identity_excl_trained_negs = {
        PRIMARY_POSITION: {
            tap: _read_identity_for_space(
                pairs=pairs,
                modules=[m for m in modules if MODULE_INPUT_TAP[m] == tap],
                bank=bank_excl,
                source=source,
                pos=PRIMARY_POSITION,
                space=tap,
            )
            for tap in sorted({MODULE_INPUT_TAP[m] for m in modules})
        }
    }

    # H2 A-init reads — UNFLIPPED pair gauge (round-4 fix, see above).
    a_init_reads = _a_init_reads(pairs_unflipped, init_pairs, bank, source)

    # H3 write identity (residual-output modules only — empty on read arm).
    write_identity = _write_identity(pairs, wu, shift_src_l20)

    # H4 firing → leakage.
    contexts_dv = eval_payload["contexts"]
    panel = [p for p in eval_payload["eval_panel"] if p != source]
    bank_contexts = set(bank["centroids"]["attn"][PRIMARY_POSITION])
    panel = [p for p in panel if p in bank_contexts and p in contexts_dv]
    primary_bystanders = [p for p in panel if p not in TRAINED_NEGATIVES_IN_PANEL]

    dv_logp = {p: contexts_dv[p]["delta_logp_marker"] for p in panel}
    dv_margin = {}
    for p in panel:
        mss = contexts_dv[p]["marker_slot_stats"]
        dv_margin[p] = (mss["trained"]["z_marker"] - mss["trained"]["z_eos"]) - (
            mss["base"]["z_marker"] - mss["base"]["z_eos"]
        )
    base_prior = {p: contexts_dv[p]["marker_slot_stats"]["base"]["logp_marker"] for p in panel}

    firing_modes = {
        "abs": "abs",
        "signed_src": "signed_src",
    }
    if arm in ("write", "bridge"):
        firing_modes["write"] = "write"

    h4: dict[str, dict] = {}
    for pos in BANK_CAPTURE_POSITIONS:
        pos_block: dict[str, dict] = {}
        for label, mode in firing_modes.items():
            f = _firing(
                pairs=pairs,
                bank=bank,
                contexts=panel,
                pos=pos,
                mode=mode,
                wu=wu,
                source=source,
            )
            # a_init paired with the UNFLIPPED trained b (round-4 fix):
            # in write mode the per-slot product (a·x_c)·(W_U[※]·b) is not
            # invariant when only one side of the pair carries the sign
            # fix, so mixing the flipped b with the unflipped init a
            # corrupts per-slot signs. Same gauge as the H2 reads.
            f_init = _firing(
                pairs={k: {"a": init_pairs[k]["a"], "b": pairs_unflipped[k]["b"]} for k in pairs},
                bank=bank,
                contexts=panel,
                pos=pos,
                mode=mode,
                wu=wu,
                source=source,
            )
            geo = _centered_geometry(bank, pos, panel, source)
            cnorm = _context_norm(bank, pos, panel)

            def _rhos(bys: list[str], pred: dict[str, float]) -> dict[str, float]:
                return {
                    "rho_vs_delta_logp": _spearman(
                        [pred[p] for p in bys], [dv_logp[p] for p in bys]
                    ),
                    "rho_vs_delta_margin": _spearman(
                        [pred[p] for p in bys], [dv_margin[p] for p in bys]
                    ),
                }

            pos_block[label] = {
                "firing": f,
                "primary_excl_trained_negs": _rhos(primary_bystanders, f),
                "secondary_incl_trained_negs": _rhos(panel, f),
                "comparators_primary": {
                    "base_prior": _rhos(primary_bystanders, base_prior),
                    "geometry_centered": _rhos(primary_bystanders, geo),
                    "context_norm": _rhos(primary_bystanders, cnorm),
                    "a_init_firing": _rhos(primary_bystanders, f_init),
                },
            }
        h4[pos] = pos_block

    # Duty 12(a): split-half base check on the primary firing mode at the
    # primary position (the predictor itself is question-independent).
    primary_mode = "write" if arm in ("write", "bridge") else "abs"
    split_half = _split_half_base_check(
        contexts_dv=contexts_dv,
        bystanders=primary_bystanders,
        firing=h4[PRIMARY_POSITION][primary_mode]["firing"],
        firing_mode=primary_mode,
    )

    # duty 8 variance precondition.
    spreads = [dv_logp[p] for p in primary_bystanders]
    ses = []
    for p in primary_bystanders:
        pq = contexts_dv[p].get("per_question_delta_logp") or []
        if len(pq) >= 2:
            ses.append(float(np.std(pq, ddof=1) / np.sqrt(len(pq))))
    variance_precondition = {
        "bystander_delta_logp_spread_sd": float(np.std(spreads, ddof=1))
        if len(spreads) >= 2
        else None,
        "median_per_persona_se": float(np.median(ses)) if ses else None,
        "n_bystanders_primary": len(primary_bystanders),
    }

    return {
        "cell_slug": slug,
        "arm": arm,
        "source": source,
        "seed": seed,
        "banded": bool(tm.get("band_stop_fired")),
        "band_stop_step": tm.get("band_stop_step"),
        "final_source_delta_nats": tm.get("final_source_delta_nats"),
        "s_scale": S_SCALE,
        "read_identity": read_identity,
        "read_identity_excl_trained_negs": read_identity_excl_trained_negs,
        "a_init": a_init_reads,
        # Round-4 gauge convention: a_init reads + the a_init-firing
        # comparator use the UNFLIPPED (a, b) tensors (never the W_U
        # sign-fixed copies).
        "a_init_gauge": "unflipped-pair",
        "a_rotation_trajectory": a_rotation,
        "split_half_base_check": split_half,
        "write_identity": write_identity,
        "h4": h4,
        "variance_precondition": variance_precondition,
        "dv_per_bystander": {
            "delta_logp": dv_logp,
            "delta_margin": dv_margin,
            "base_prior_logp": base_prior,
        },
        "primary_bystanders": primary_bystanders,
    }


def _cross_seed_stability(per_cell_pairs: dict[str, dict]) -> dict:
    """H5: cross-seed |cos| of â and b̂ within (arm, source), band-mean."""
    groups: dict[tuple[str, str], dict[int, dict]] = {}
    for slug, pairs in per_cell_pairs.items():
        arm, source, seed = parse_cell_slug(slug)
        groups.setdefault((arm, source), {})[seed] = pairs
    out: dict[str, dict] = {}
    for (arm, source), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        a_vals: list[float] = []
        b_vals: list[float] = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                p1, p2 = by_seed[seeds[i]], by_seed[seeds[j]]
                for key in sorted(set(p1) & set(p2)):
                    li, _m = key
                    if li not in BAND_LAYERS:
                        continue
                    a_vals.append(abs(_cos(p1[key]["a"], p2[key]["a"])))
                    b_vals.append(abs(_cos(p1[key]["b"], p2[key]["b"])))
        out[f"{arm}|{source}"] = {
            "seeds": seeds,
            "band_mean_abs_cos_a": float(np.mean(a_vals)) if a_vals else None,
            "band_mean_abs_cos_b": float(np.mean(b_vals)) if b_vals else None,
            "n_pairs": len(a_vals),
        }
    return out


def _paired_comparator_diffs(results: dict[str, dict]) -> dict:
    """Duty 12(b): paired per-cell ρ diffs (firing − comparator), clustered.

    Per (arm, comparator): per-cell paired differences between the firing
    predictor's primary ρ and the comparator's ρ on the SAME cells (primary
    bystander set, primary position, primary mode/DV — matching ``_pool``),
    clustered on source (4 effective units): per-source mean diffs, an
    exact two-sided sign test over source signs, and a t-based 95% CI over
    the source means. Descriptive at n_sources <= 4 — the cells share the
    panel + questions, so the source is the independent unit.
    """
    from scipy.stats import binomtest
    from scipy.stats import t as t_dist

    out: dict[str, dict] = {}
    for arm in PLACEMENT_ARMS:
        arm_cells = [r for r in results.values() if r["arm"] == arm]
        if not arm_cells:
            continue
        per_cmp: dict[str, list[tuple[str, float]]] = {}
        for r in arm_cells:
            mode = (
                "write"
                if arm in ("write", "bridge") and "write" in r["h4"][PRIMARY_POSITION]
                else "abs"
            )
            dv_key = "rho_vs_delta_margin" if mode == "write" else "rho_vs_delta_logp"
            block = r["h4"][PRIMARY_POSITION][mode]
            rho_f = block["primary_excl_trained_negs"][dv_key]
            if rho_f is None or np.isnan(rho_f):
                continue
            for cname, cblock in block["comparators_primary"].items():
                v = cblock[dv_key]
                if v is None or np.isnan(v):
                    continue
                per_cmp.setdefault(cname, []).append((r["source"], float(rho_f - v)))
        arm_out: dict[str, dict] = {}
        for cname, rows in sorted(per_cmp.items()):
            by_src: dict[str, list[float]] = {}
            for src, d in rows:
                by_src.setdefault(src, []).append(d)
            src_means = {s: float(np.mean(v)) for s, v in sorted(by_src.items())}
            vals = list(src_means.values())
            nonzero = [v for v in vals if v != 0.0]
            n_pos = sum(v > 0 for v in nonzero)
            sign_p = float(binomtest(n_pos, len(nonzero), 0.5).pvalue) if nonzero else None
            mean_diff = float(np.mean(vals))
            ci95: list[float] | None = None
            if len(vals) >= 2:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                tcrit = float(t_dist.ppf(0.975, len(vals) - 1))
                ci95 = [mean_diff - tcrit * se, mean_diff + tcrit * se]
            arm_out[cname] = {
                "n_cells": len(rows),
                "n_sources": len(vals),
                "per_source_mean_diff": src_means,
                "mean_diff_over_sources": mean_diff,
                "n_sources_positive": n_pos,
                "sign_test_p_two_sided": sign_p,
                "t_ci95_over_sources": ci95,
            }
        out[arm] = arm_out
    return out


def _pool_split_half(results: dict[str, dict]) -> dict:
    """Per-arm medians of every split-half ρ (duty 12a summary view)."""
    out: dict[str, dict] = {}
    for arm in PLACEMENT_ARMS:
        agg: dict[str, dict[str, list[float]]] = {}
        for r in results.values():
            if r["arm"] != arm or not r.get("split_half_base_check"):
                continue
            for split_name, block in r["split_half_base_check"]["splits"].items():
                for k, v in block.items():
                    if v is not None and not np.isnan(v):
                        agg.setdefault(split_name, {}).setdefault(k, []).append(v)
        if agg:
            out[arm] = {
                split: {k: float(np.median(v)) for k, v in sorted(ks.items())}
                for split, ks in sorted(agg.items())
            }
    return out


def _a_rotation_summary(results: dict[str, dict]) -> tuple[dict, list[str]]:
    """Duty 10 summary table: compact per-cell a(t) rotation rows.

    Returns (per-slug table, slugs missing a checkpoint ladder).
    """
    table: dict[str, dict] = {}
    missing: list[str] = []
    for slug, r in sorted(results.items()):
        tr = r["a_rotation_trajectory"]
        if tr["n_checkpoints"] == 0:
            missing.append(slug)
        final_pt = tr["points"][-1] if tr["points"] else None
        table[slug] = {
            "arm": r["arm"],
            "source": r["source"],
            "seed": r["seed"],
            "banded": r["banded"],
            "band_stop_step": tr["band_stop_step"],
            "n_checkpoints": tr["n_checkpoints"],
            "final_band_mean_abs_cos_a_init": (
                final_pt["band_mean_abs_cos_a_init"] if final_pt else None
            ),
            "final_band_mean_rel_delta_a": (
                final_pt["band_mean_rel_delta_a"] if final_pt else None
            ),
            "last_increment_rel": tr["last_increment_rel"],
            "max_increment_rel": tr["max_increment_rel"],
            "rotation_still_growing_at_stop": tr["rotation_still_growing_at_stop"],
        }
    return table, missing


def cmd_run(args) -> int:
    adapters_root = Path(args.adapters_root)
    bank = load_bank(Path(args.bank))
    wu = load_unembedding(Path(args.unembedding))
    eval_cells = load_eval_cells(Path(args.eval_root), Path(args.cells_root))

    results: dict[str, dict] = {}
    per_cell_pairs: dict[str, dict] = {}
    skipped: list[str] = []
    for slug, payload in sorted(eval_cells.items()):
        adapter_dir = adapters_root / slug
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            skipped.append(slug)
            continue
        payload["_shift_pt_path"] = str(Path(args.eval_root) / f"{slug}__shift.pt")
        log.info("analyzing %s", slug)
        results[slug] = analyze_cell(
            slug=slug, adapter_dir=adapter_dir, bank=bank, eval_payload=payload, wu=wu
        )
        pairs = load_adapter_pairs(adapter_dir)
        _sign_fix_write(pairs, wu)
        per_cell_pairs[slug] = pairs
        # Checkpoint-per-phase: persist incrementally.
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"cell__{slug}.json").write_text(json.dumps(results[slug], indent=1))
        # Duty 10: per-cell a(t) rotation trajectory JSON.
        (out_dir / f"a_trajectory__{slug}.json").write_text(
            json.dumps({"cell_slug": slug, **results[slug]["a_rotation_trajectory"]}, indent=1)
        )
    if skipped:
        log.warning("skipped %d cell(s) without local adapters: %s", len(skipped), skipped[:5])
    if not results:
        raise SystemExit("no cells analyzed — fetch adapters first (fetch-artifacts).")
    # Round-2 (code-review minor): a partial fetch-artifacts must not be
    # silently analyzed as if it were the full grid. The production run
    # expects every enumerated cell; --allow-partial is the explicit escape
    # for smoke fixtures / deliberate subsets.
    n_expected = len(enumerate_cells())
    if not args.allow_partial and len(results) != n_expected:
        raise SystemExit(
            f"analyzed {len(results)} cells but the design enumerates {n_expected} "
            f"(skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}) — "
            "re-run fetch-artifacts, or pass --allow-partial for a deliberate subset."
        )

    cross_seed = _cross_seed_stability(per_cell_pairs)

    # Pooled H4 summaries: split banded vs below-band (duty 8) + per-arm
    # medians (descriptive; cells share panel + questions ⇒ cluster on
    # source for any inferential read — recorded per cell for the analyzer).
    def _pool(arm: str, banded_only: bool | None) -> dict:
        rhos_logp: list[float] = []
        rhos_cmp: dict[str, list[float]] = {}
        for r in results.values():
            if r["arm"] != arm:
                continue
            if banded_only is True and not r["banded"]:
                continue
            if banded_only is False and r["banded"]:
                continue
            mode = (
                "write"
                if arm in ("write", "bridge") and "write" in r["h4"][PRIMARY_POSITION]
                else "abs"
            )
            block = r["h4"][PRIMARY_POSITION][mode]
            dv_key = "rho_vs_delta_margin" if mode == "write" else "rho_vs_delta_logp"
            rho = block["primary_excl_trained_negs"][dv_key]
            if not np.isnan(rho):
                rhos_logp.append(rho)
            for cname, cblock in block["comparators_primary"].items():
                v = cblock[dv_key]
                if not np.isnan(v):
                    rhos_cmp.setdefault(cname, []).append(v)
        return {
            "n_cells": len(rhos_logp),
            "median_rho_firing": float(np.median(rhos_logp)) if rhos_logp else None,
            "frac_positive": float(np.mean([r > 0 for r in rhos_logp])) if rhos_logp else None,
            "median_rho_comparators": {k: float(np.median(v)) for k, v in rhos_cmp.items() if v},
        }

    # Duty 10: compact per-cell a(t) rotation summary table.
    a_rotation_table, cells_missing_ladder = _a_rotation_summary(results)

    summary = {
        "h4_by_arm_all": {arm: _pool(arm, None) for arm in PLACEMENT_ARMS},
        "h4_by_arm_banded": {arm: _pool(arm, True) for arm in PLACEMENT_ARMS},
        "h4_by_arm_below_band": {arm: _pool(arm, False) for arm in PLACEMENT_ARMS},
        # Duty 12(b): paired firing-vs-comparator diffs, clustered on source.
        "h4_paired_comparator_diffs": _paired_comparator_diffs(results),
        # Duty 12(a): per-arm medians of the split-half base-check rhos.
        "split_half_base_check_by_arm": _pool_split_half(results),
        # Duty 10: a(t) rotation table (full trajectories per cell JSON).
        "a_rotation": a_rotation_table,
        "cells_missing_ladder": cells_missing_ladder,
        "cross_seed": cross_seed,
        "n_cells": len(results),
        "skipped": skipped,
    }

    meta = {
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
        "s_scale": S_SCALE,
        "primary_position": PRIMARY_POSITION,
        "trained_negatives_excluded_primary": list(TRAINED_NEGATIVES_IN_PANEL),
        # duty 4: the H1 wrong-context-null sensitivity excludes ALL FOUR
        # trained-negative contexts (round-2 fix).
        "h1_null_excluded_contexts": list(UNIFIED_NEGATIVE_PANEL),
        "band_layers": [min(BAND_LAYERS), max(BAND_LAYERS)],
        "h3_layers": [min(H3_LAYERS), max(H3_LAYERS)],
        "geometry_centering": "global_mean",
        "unembedding": str(args.unembedding),
        "sources": list(SOURCES),
        "unified_negative_panel": list(UNIFIED_NEGATIVE_PANEL),
        "seeds": list(SEEDS),
        # Round 4 (concern h2-a-init-reads-after-sign-flip): every
        # a_init-referenced read (H2 reads, a_init-firing comparator,
        # duty-10 trajectory) runs on the UNFLIPPED (a, b) tensors.
        "a_init_gauge": "unflipped-pair",
        # duty 10 + duty 12 rules (round 3).
        "a_rotation_rule": _A_ROTATION_RULE,
        "split_half_rule": (
            "deterministic even/odd question-index halves; base prior from one "
            "half, DV from the other; both directions reported"
        ),
        "paired_diff_rule": (
            "diff = rho(firing) - rho(comparator), primary set/position/mode; "
            "clustered on source (sign test + t-CI over source means)"
        ),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps({"meta": meta, "cells": results, "summary": summary}, indent=1)
    )
    (out_dir / "analysis_summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=1)
    )
    log.info("analysis written: %s (%d cells)", out_dir / "analysis.json", len(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_wu = sub.add_parser("build-unembedding", help="extract W_U rows from the base model shard")
    p_wu.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_wu.add_argument("--n-null", type=int, default=300)
    p_wu.add_argument(
        "--unembedding", default="eval_results/issue_621/analysis/unembedding_rows.pt"
    )

    p_fetch = sub.add_parser(
        "fetch-artifacts",
        help="fetch adapters + bank + shifts + emission JSONs + train meta from HF",
    )
    p_fetch.add_argument("--adapters-root", default="eval_results/issue_621/adapters_fetched")
    p_fetch.add_argument("--bank", default="eval_results/issue_621/context_vectors")
    p_fetch.add_argument("--eval-root", default="eval_results/issue_621/eval")
    p_fetch.add_argument("--cells-root", default="eval_results/issue_621")

    p_run = sub.add_parser("run", help="run the full §4.5 analysis")
    p_run.add_argument("--adapters-root", default="eval_results/issue_621/adapters_fetched")
    p_run.add_argument("--bank", default="eval_results/issue_621/context_vectors")
    p_run.add_argument("--eval-root", default="eval_results/issue_621/eval")
    p_run.add_argument("--cells-root", default="eval_results/issue_621")
    p_run.add_argument(
        "--unembedding", default="eval_results/issue_621/analysis/unembedding_rows.pt"
    )
    p_run.add_argument("--out", default="eval_results/issue_621/analysis")
    p_run.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit analyzing fewer cells than the enumerated grid (smoke / deliberate subset).",
    )

    args = ap.parse_args(argv)
    if args.cmd == "build-unembedding":
        return cmd_build_unembedding(args)
    if args.cmd == "fetch-artifacts":
        return cmd_fetch_artifacts(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
