#!/usr/bin/env python3
"""Issue #1946 follow-up `exact-sae-floors` — exact encode-then-pool SAE K-resample floors.

The parent round's SAE-space floor correction (`issue1946_sae_percontext.phase_floors`)
SAE-encoded each draw's MEAN dense state (pool-then-encode — the wrong order through a
nonlinear encoder). This driver recomputes the floors EXACTLY (plan v7, the ONE variable):
teacher-forced layer-19 capture of each of the K=4 banked resampled answers (1,988
conversations x 4 draws, `kresample_shard00.{json,pt}` @ the pinned revision), per-token
SAE encoding, inlier masking + mean/max/frac pooling exactly as the banked targets were
built (`issue1738_sae_arm._render_row`/`_capture_answer_spans` +
`issue1482_sae.token_inlier_mask`/`pool_answer_features`), then the verbatim ddof-1 trace
floor arithmetic and the verbatim floor-adjusted battery
(`issue1738_characterize.py --phase taxonomy` subprocess per space).

Phases (``--phase``): ``stage | capture | floors | battery | compare | all`` (pod chain)
| ``harvest`` (VM-side V1: fetch JSONs from HF + render the exact floor-adjustment figure).

Pre-production kill-criteria gates (plan section 5), each writing a ``gates/*.json``
breadcrumb and exiting a DISTINCT designed rc (never a bare rc=1):
  gate 2  structural (rc 24, in ``capture``): full-corpus scan of the kresample json
          (keys/seeds/non-empty/join == 1,988) + the <=1% inlier-empty tolerance;
  gate 1  approx parity (rc 25): re-encode the .pt per-draw MEAN states for the 32-ci
          probe and reproduce the banked approximate floors within 1e-6 relative
          (CPU SAE — the parent construct's own device);
  gate 1b fresh-capture parity (rc 26): 32-ci x 4-draw fresh PRE-inlier answer-span means
          vs the banked .pt ``V`` layer-19 slice, per (ci, seed) cell cosine >= 0.99 AND
          rel-L2 <= 2e-2 (kill criterion 1b — catches render/inlier/capture-chain drift);
  gate 3  wall (rc 27): slice-measured capture wall extrapolated by token count must
          project <= 2 h;
  gate 4  battery identity (rc 28, in ``battery``): unadjusted deltas == banked within
          1e-6 (same inputs + seed 1738 — staging/wiring bug otherwise);
  gate 5  coverage (rc 29, in ``compare``): exact-floor shared-contrast set == the
          parent's 19.

Smoke modes:
  ``--smoke``                  real-data 32-ci slice through stage->capture->floors
                               (scratch ``_smoke`` out-root, no uploads) with ALL
                               pre-production gates — the pod pre-flight;
  ``--smoke --cpu-gates-only`` gates 1+2 only (no model load) — VM-runnable;
  ``--fixture-smoke``          tiny-real CPU e2e (from-config 2-layer Qwen + tiny
                               BatchTopK SAE, REAL tokenizer/render/capture/encode/pool/
                               battery code paths) through EVERY phase incl.
                               battery/compare/harvest + degenerate gate probes.

Refusal-safety: conversation/label text is NEVER printed — digest-only (counts, paths,
ci ids, reason tokens). Reproducibility metadata rides every output JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch import (#847)

import issue779_common as C  # noqa: E402
import issue1482_sae as SAEMOD  # noqa: E402
import issue1738_sae_arm as SA  # noqa: E402
import issue1946_sae_percontext as PC  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1946_exact_floors")

# ── constants (plan v7 sections 0/3/5/10/11 — Sources recorded there) ───────────────
TASK_ID = 1946
LAYER = 19  # parent headline layer (Source: #1738)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION_DEFAULT = "a09a35458c702b33eeacc393d103063234e8bc28"  # plan section 10
DATA_REVISION_DEFAULT = "05cb982b0d3f9a21b5735d196a0afdc8175590e5"  # kresample + parent inputs
BANKED_REVISION_DEFAULT = "12ab41dc1c4a7163d183697e9c4fa53528904c9b"  # #1946 banked tensors
PARENT_PREFIX = "issue1738_multiturn"
BANKED_PREFIX = "issue1946_sae_percontext"
UPLOAD_PREFIX_DEFAULT = f"{BANKED_PREFIX}/exact_floors"

SEEDS_EXPECTED = (43, 44, 45, 46)  # plan section 2 (realized-keys probe)
N_ROWS_EXPECTED = 1_988
N_PROBE_CI = 32  # smoke-slice grain (plan section 3)
PARITY_RTOL = 1e-6  # gate 1, exact-numpy leg (den; plan section 5 criterion 1)
# gate 1, SAE-encode-dependent legs (floor/share): the plan's 1e-6 assumed only BLAS
# ORDER varies; the 2026-08-01 VM smoke vs the parent's GCE-computed banked floors
# MEASURED 1.152e-6 max rel dev (den leg 0.0 — bit-identical numpy), i.e. cross-MACHINE
# fp32 GEMM jitter slightly exceeds 1e-6 on healthy data. Calibrated to 1e-5 with
# measured headroom (~9x) — still >=4 orders below the gross wiring-failure class this
# gate screens (#1482: 93%-scale). Recorded as a plan-deviation concern on the task.
PARITY_RTOL_ENC = 1e-5
CAP_COS_MIN = 0.99  # gate 1b (plan section 11: fp16 storage + bf16 batch noise)
CAP_RELL2_MAX = 2e-2  # gate 1b
WALL_CAP_H = 2.0  # gate 3 (>=2x the 1 h estimate)
IDENTITY_ATOL = 1e-6  # gate 4 (plan section 11 — values, never recomputed selections)
N_SHARED_EXPECTED = 19  # gate 5 (the parent's shared floor-covered contrasts)
INLIER_EMPTY_FRAC_MAX = 0.01  # kill criterion 2 tolerance
CHUNK_DRAWS = 128  # capture unit (32 ci x 4 draws)
POOLINGS = ("mean", "max", "frac")
ENV_OF_POOL = {
    "mean": "floors_env_mean_exact",
    "max": "floors_env_max_exact",
    "frac": "floors_env_frac_exact",
}
# verdict spaces first (plan section 3 X3), then the exploratory poolings
SPACES = ("sae_space", "dense_feat_space", "max_space", "frac_space")
ARMS = ("prefix", "context", "bare")
REFUSAL_FAMILY = ("refusal_adjacent=yes", "answer_is_refusal=yes")
N_PERM_CROSS = 10_000
CROSS_SEED = 1946
# banked plan constants (X4 staging-mixup assert, production only; artifact-verified
# against eval_results/issue_1946/sae_space/taxonomy.json this session)
BANKED_BARE_AIR_UNADJ = 0.0508
BANKED_BARE_AIR_ADJ = -0.1521
BANKED_CONST_TOL = 5e-4
APPROX_SHARE_MEDIAN_CITED = 0.062  # plan section 1 registered secondary (b)

RC_GATE_STRUCTURAL = 24
RC_GATE_PARITY = 25
RC_GATE_CAPTURE = 26
RC_GATE_WALL = 27
RC_GATE_IDENTITY = 28
RC_GATE_COVERAGE = 29


@dataclass
class Cfg:
    """All paths + mode flags for one run (production defaults or fixture paths)."""

    staging_root: Path
    out_eval: Path
    fig_dir: Path
    dense_eval: Path  # git eval_results/issue_1738 (labels + dense taxonomies)
    banked_eval: Path  # git eval_results/issue_1946 (banked taxonomies + comparison)
    data_revision: str
    banked_revision: str
    model_revision: str
    upload_prefix: str
    smoke: bool = False
    cpu_gates_only: bool = False
    fixture: bool = False
    no_upload: bool = False
    force: bool = False
    batch: int = 8
    device: str = "cuda"
    wall_cap_h: float = WALL_CAP_H
    cap_cos_min: float = CAP_COS_MIN
    cap_rell2_max: float = CAP_RELL2_MAX
    n_probe: int = N_PROBE_CI
    capture_layer: int = LAYER
    # input paths (production: derived from staging_root; fixture: explicit)
    kresample_pt: Path | None = None
    kresample_json: Path | None = None
    perfeature_npz: Path | None = None
    manifest_dir: Path | None = None
    split_file: Path | None = None
    sae_cache: Path | None = None
    banked_tensors_root: Path | None = None  # staged analysis_tensors mirror root
    approx_floors_npz: Path | None = None
    labels_src: Path | None = None
    fixture_model_dir: Path | None = None

    def gates_dir(self) -> Path:
        return self.out_eval / "gates"

    def sentinel_dir(self) -> Path:
        return self.out_eval / "phase_sentinels"

    def env_root(self, pooling: str) -> Path:
        return self.out_eval / ENV_OF_POOL[pooling]

    def space_out(self, space: str) -> Path:
        return self.out_eval / space

    def staged_space(self, space: str) -> Path:
        assert self.banked_tensors_root is not None
        return self.banked_tensors_root / space

    def y_holdout_path(self, pooling: str) -> Path:
        return self.staged_space(PC.Y_HOLDOUT_SPACE[pooling]) / "y_holdout" / f"L{LAYER}.npz"


def _resolve_production_inputs(cfg: Cfg) -> Cfg:
    root = cfg.staging_root
    cfg.kresample_pt = root / PARENT_PREFIX / "kresample" / "kresample_shard00.pt"
    cfg.kresample_json = root / PARENT_PREFIX / "kresample" / "kresample_shard00.json"
    cfg.perfeature_npz = (
        root
        / PARENT_PREFIX
        / "sae_arm_bare"
        / "analysis_tensors"
        / "perfeature"
        / "perfeature_summary.npz"
    )
    cfg.manifest_dir = root / PARENT_PREFIX / "sampling_manifest"
    cfg.split_file = cfg.manifest_dir / "split_1738.json"
    cfg.sae_cache = root / "sae_cache"
    cfg.banked_tensors_root = root / BANKED_PREFIX / "analysis_tensors"
    cfg.approx_floors_npz = (
        cfg.banked_tensors_root / "floors_env_mean" / "kresample" / f"floors_L{LAYER}.npz"
    )
    cfg.labels_src = cfg.dense_eval / "judge_labels" / "labels.json"
    return cfg


def _repro(cfg: Cfg) -> dict:
    return C.reproducibility_metadata(
        {
            "data_revision": cfg.data_revision,
            "banked_revision": cfg.banked_revision,
            "model_revision": cfg.model_revision,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "smoke": bool(cfg.smoke),
            "fixture": bool(cfg.fixture),
        }
    )


def _atomic_json(path: Path, obj: dict) -> None:
    PC._atomic_json(path, obj)


# ── X0: stage (idempotent, revision-pinned; Hub boundary injectable) ────────────────


def _assert_disk_headroom(cfg: Cfg) -> None:
    """Preamble statvfs asserts (plan section 9): staging fs >= 30 GB free for the full
    production stage; reduced floors for the cpu-gates subset (~4.4 GB) and fixture."""
    need_stage = (
        30.0 if not (cfg.cpu_gates_only or cfg.fixture) else (6.0 if not cfg.fixture else 0.5)
    )
    need_out = 0.5 if (cfg.smoke or cfg.fixture) else 2.0
    for path, need in ((cfg.staging_root, need_stage), (cfg.out_eval, need_out)):
        probe = path
        while not probe.exists():
            probe = probe.parent
        st = os.statvfs(probe)
        free_gb = st.f_bavail * st.f_frsize / 2**30
        assert free_gb >= need, f"disk headroom: {probe} has {free_gb:.1f} GB free < {need} GB"
        print(f"[stage] headroom OK: {probe} free={free_gb:.1f} GB (need {need:.1f})", flush=True)


def phase_stage(
    cfg: Cfg,
    state: dict,
    *,
    stage_prefix_fn=hub.stage_hub_prefix,
    stage_file_fn=hub.stage_hub_file,
    ensure_sae_fn=SAEMOD.BatchTopKSAE.ensure_downloaded,
) -> None:
    """Stage every pinned input into the staging root (verbatim prefix-mirror semantics
    of ``stage_hub_prefix`` — files land at ``root/<repo path>``). The EMPTY
    ``dense_feat_space/y_holdout`` Hub prefix is tolerated (plan section 3: the parent's
    ``Y_HOLDOUT_SPACE`` sharing puts dense_feat's y under ``sae_space/y_holdout``).
    Hub fns are injectable so the fixture smoke fakes the boundary
    signature-conformantly (autospec) while running this exact body."""
    print("[phase=stage] start", flush=True)
    _assert_disk_headroom(cfg)
    root = cfg.staging_root
    root.mkdir(parents=True, exist_ok=True)
    repo = C.HF_DATA_REPO
    for fpath in (
        f"{PARENT_PREFIX}/kresample/kresample_shard00.pt",
        f"{PARENT_PREFIX}/kresample/kresample_shard00.json",
        f"{PARENT_PREFIX}/sae_arm_bare/analysis_tensors/perfeature/perfeature_summary.npz",
    ):
        stage_file_fn(repo, fpath, root / fpath, repo_type="dataset", revision=cfg.data_revision)
        print(f"[stage] file {fpath}: staged", flush=True)
    prefixes: list[tuple[str, str]] = []
    if not cfg.cpu_gates_only:
        prefixes.append((f"{PARENT_PREFIX}/sampling_manifest", cfg.data_revision))
        for space in SPACES:
            for sub in ("percontext", "pred16", "y_holdout"):
                prefixes.append(
                    (f"{BANKED_PREFIX}/analysis_tensors/{space}/{sub}", cfg.banked_revision)
                )
    else:
        prefixes.append(
            (f"{BANKED_PREFIX}/analysis_tensors/sae_space/y_holdout", cfg.banked_revision)
        )
    prefixes.append(
        (f"{BANKED_PREFIX}/analysis_tensors/floors_env_mean/kresample", cfg.banked_revision)
    )
    for prefix, rev in prefixes:
        try:
            got = stage_prefix_fn(repo, prefix, root, repo_type="dataset", revision=rev)
        except FileNotFoundError:
            if prefix.endswith("/y_holdout"):
                print(
                    f"[stage] prefix {prefix}: EMPTY on Hub — tolerated "
                    "(Y_HOLDOUT_SPACE sharing, plan section 3)",
                    flush=True,
                )
                continue
            raise
        n = len(got) if isinstance(got, list) else -1
        print(f"[stage] prefix {prefix}: staged n={n}", flush=True)
    sae_cache = cfg.sae_cache if cfg.sae_cache is not None else root / "sae_cache"
    ensure_sae_fn(64, sae_cache, layer=LAYER)
    print("[stage] sae weights ensured (k=64, layer 19)", flush=True)
    _atomic_json(
        cfg.sentinel_dir() / "stage.done.json",
        {"cpu_gates_only": cfg.cpu_gates_only, **_repro(cfg)},
    )
    print("[phase=stage] done", flush=True)


# ── X1: capture + encode (gates FIRST, then the full batched GPU loop) ──────────────


def _load_kresample(cfg: Cfg) -> tuple[torch.Tensor, list[int], list[int], dict]:
    """Load the banked shard pair -> (V (n,K,H) fp16 at LAYER, ci, seeds, json doc).
    ``weights_only=False`` explicitly: self-produced bundle with list fields
    (torch>=2.6 default flip; agent-memory lesson)."""
    b = torch.load(cfg.kresample_pt, map_location="cpu", weights_only=False)
    layers = [int(x) for x in b["layers"]]
    assert LAYER in layers, f"{cfg.kresample_pt}: layer {LAYER} not in {layers}"
    V = b["V"][:, :, layers.index(LAYER), :]
    kci = [int(c) for c in b["ci"]]
    seeds = [int(s) for s in b.get("seeds", SEEDS_EXPECTED)]
    doc = json.loads(Path(cfg.kresample_json).read_text())
    return V, kci, seeds, doc


def _gate_structural(
    cfg: Cfg, V: torch.Tensor, kci: list[int], seeds: list[int], doc: dict, y_ci: np.ndarray
) -> None:
    """Kill criterion 2 (full-CONSUMED-corpus grain): row keys / seed set / non-empty
    responses / json<->pt ci lockstep / 1:1 join into the mean-pool y_holdout.
    Digest-only problems (ci + reason token) — never row text."""
    rows = doc.get("rows", [])
    problems: list[dict] = []
    want_seeds = {str(s) for s in seeds}
    if set(seeds) != set(SEEDS_EXPECTED):
        problems.append({"reason": "seed_set", "got": sorted(seeds)})
    if len(rows) != len(kci):
        problems.append({"reason": "json_pt_row_count", "json": len(rows), "pt": len(kci)})
    if not cfg.fixture and len(kci) != N_ROWS_EXPECTED:
        problems.append({"reason": "n_rows", "got": len(kci), "want": N_ROWS_EXPECTED})
    if V.shape[1] != len(SEEDS_EXPECTED):
        problems.append({"reason": "k_draws", "got": int(V.shape[1])})
    ypos = {int(c): p for p, c in enumerate(y_ci.tolist())}
    for i, row in enumerate(rows):
        ci = row.get("ci")
        if not isinstance(ci, int) or (i < len(kci) and ci != kci[i]):
            problems.append({"reason": "ci_order", "row": i})
            continue
        msgs = row.get("messages")
        if not (
            isinstance(msgs, list)
            and msgs
            and all(
                isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role")
                for m in msgs
            )
        ):
            problems.append({"ci": ci, "reason": "messages_shape"})
        resp = row.get("responses")
        if not (isinstance(resp, dict) and set(resp) == want_seeds):
            problems.append({"ci": ci, "reason": "responses_keys"})
        elif not all(isinstance(v, str) and v.strip() for v in resp.values()):
            problems.append({"ci": ci, "reason": "empty_response"})
        if ci not in ypos:
            problems.append({"ci": ci, "reason": "no_holdout_join"})
    ok = not problems
    _atomic_json(
        cfg.gates_dir() / "structural.json",
        {
            "ok": ok,
            "n_rows": len(rows),
            "k_draws": int(V.shape[1]),
            "n_problems": len(problems),
            "problems_head": problems[:20],
            **_repro(cfg),
        },
    )
    if not ok:
        print(
            f"[gate] structural FAIL: {len(problems)} problems "
            f"(head: {problems[:5]}) — HALT (kill criterion 2)",
            flush=True,
        )
        raise SystemExit(RC_GATE_STRUCTURAL)
    print(f"[gate] structural OK: n={len(rows)} K={int(V.shape[1])} join=1:1", flush=True)


def _check_inlier_drops(cfg: Cfg, empties: list[dict], n_total: int) -> None:
    """Kill criterion 2 inlier arm: >1% inlier-empty draws -> halt; <=1% -> warn +
    the affected conversations are dropped from the floors (n shrinks, flagged)."""
    frac = len(empties) / max(1, n_total)
    _atomic_json(
        cfg.gates_dir() / "inlier.json",
        {
            "ok": frac <= INLIER_EMPTY_FRAC_MAX,
            "n_empty_draws": len(empties),
            "n_total_draws": n_total,
            "frac": frac,
            "empties": empties[:50],
            **_repro(cfg),
        },
    )
    if frac > INLIER_EMPTY_FRAC_MAX:
        print(
            f"[gate] inlier-empty draws {len(empties)}/{n_total} > 1% — HALT (kill criterion 2)",
            flush=True,
        )
        raise SystemExit(RC_GATE_STRUCTURAL)
    if empties:
        print(
            f"[gate] WARNING: {len(empties)} inlier-empty draws (<=1%) — affected "
            "conversations dropped from floors (flagged)",
            flush=True,
        )


def _load_sae(cfg: Cfg):
    if cfg.fixture:
        _hf, _tok, sae = SA._smoke_models(
            Path(cfg.fixture_model_dir), argparse.Namespace(), model=False
        )
        return sae
    return SAEMOD.BatchTopKSAE.load(k=64, device="cpu", cache_dir=Path(cfg.sae_cache), layer=LAYER)


def _sae_encode_device(sae, device: str) -> None:
    """Move the ENCODE-side tensors only (w_dec stays on CPU — decode is unused and
    its 1.9 GB matters on a 24 GB L4; plan section 9 VRAM budget)."""
    sae.w_enc = sae.w_enc.to(device)
    sae.b_enc = sae.b_enc.to(device)
    sae.b_dec = sae.b_dec.to(device)
    sae.device = device


def _gate_parity_approx(
    cfg: Cfg,
    V: torch.Tensor,
    kci: list[int],
    probe_idx: list[int],
    y16: np.ndarray,
    y_ci: np.ndarray,
    f_out_t: torch.Tensor,
    sae,
) -> None:
    """Gate 1 (kill criterion 1): recompute the parent's APPROXIMATE floors
    (SAE-encode of per-draw MEAN dense states, `issue1946_sae_percontext.phase_floors`
    construct, CPU SAE) for the probe ci and reproduce the banked
    ``floors_L19.npz`` values within 1e-6 relative."""
    banked = np.load(Path(cfg.approx_floors_npz))
    bpos = {int(c): p for p, c in enumerate(banked["ci"].tolist())}
    ypos = {int(c): p for p, c in enumerate(y_ci.tolist())}
    probe_ci = [kci[i] for i in probe_idx]
    missing = [c for c in probe_ci if c not in bpos]
    assert not missing, f"gate 1: probe ci absent from banked approx floors: {missing[:5]}"
    Vp = V[probe_idx].to(torch.float32)  # (m, K, H)
    m, k_draws, h = Vp.shape
    flat = Vp.reshape(m * k_draws, h)
    enc = sae.encode(flat)[:, f_out_t].numpy().astype(np.float64)
    E = enc.reshape(m, k_draws, -1)
    ebar = E.mean(axis=1, keepdims=True)
    floor = ((E - ebar) ** 2).sum(axis=(1, 2)) / (k_draws - 1)
    mu = y16.mean(axis=0)
    hp = np.asarray([ypos[c] for c in probe_ci])
    den = ((y16[hp] - mu) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = floor / den
    bsel = np.asarray([bpos[c] for c in probe_ci])
    rels = {}
    tols = {"floor": PARITY_RTOL_ENC, "den": PARITY_RTOL, "share": PARITY_RTOL_ENC}
    for name, mine, theirs in (
        ("floor", floor, banked["floor"][bsel]),
        ("den", den, banked["den"][bsel]),
        ("share", share, banked["share"][bsel]),
    ):
        rels[name] = float(np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-300)))
    ok = all(rels[k] <= tols[k] for k in rels)
    _atomic_json(
        cfg.gates_dir() / "parity_approx.json",
        {
            "ok": ok,
            "n_probe": m,
            "max_rel_dev": rels,
            "rtol": tols,
            "sae_device": "cpu",
            **_repro(cfg),
        },
    )
    if not ok:
        print(f"[gate] approx-parity FAIL: max rel dev {rels} > {tols} — HALT", flush=True)
        raise SystemExit(RC_GATE_PARITY)
    print(f"[gate] approx-parity OK on {m} ci (max rel dev {rels})", flush=True)


def _load_model(cfg: Cfg):
    if cfg.fixture:
        hf, tok, _sae = SA._smoke_models(Path(cfg.fixture_model_dir), argparse.Namespace())
        return hf, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert cfg.device == "cuda" and torch.cuda.is_available(), (
        "production capture requires CUDA (plan section 9; use --fixture-smoke on CPU)"
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=cfg.model_revision)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=cfg.model_revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    hf.eval()
    return hf, tok


def _render_all(
    tok, doc: dict, seeds: list[int]
) -> tuple[list[torch.Tensor], list[int], np.ndarray]:
    """Render EVERY (row, seed) draw with the banked-target render (`SA._render_row`,
    `capture_answer_vector`-verbatim). Draw index d = i*K + k. Returns
    (ids per draw, prompt_len per draw, n_render_tokens per draw)."""
    rows = doc["rows"]
    rows_ids: list[torch.Tensor] = []
    plens: list[int] = []
    for row in rows:
        for s in seeds:
            ids, plen = SA._render_row(tok, row["messages"], row["responses"][str(s)])
            rows_ids.append(ids)
            plens.append(plen)
    ntok = np.asarray([int(x.shape[0]) for x in rows_ids], dtype=np.int64)
    return rows_ids, plens, ntok


def _capture_spans_oom_safe(cfg: Cfg, hf, ids_list, plens, pad_id) -> list[torch.Tensor]:
    """`SA._capture_answer_spans` with CUDA-OOM batch halving (plan section 8 allowed
    deviation: capture batch auto-tuned; OOM at batch=1 -> fail loud, re-dispatch wider)."""
    while True:
        try:
            return SA._capture_answer_spans(
                hf, ids_list, plens, cfg.capture_layer, max(1, cfg.batch), pad_id
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert cfg.batch > 1, "CUDA OOM at batch=1 — re-dispatch wider GPU (plan section 8)"
            cfg.batch = max(1, cfg.batch // 2)
            print(f"[capture] CUDA OOM -> batch halved to {cfg.batch}", flush=True)


def _capture_outputs_current(cfg: Cfg) -> str | None:
    """Entry skip-guard: E npzs + x1_summary current for this data revision."""
    xs = cfg.out_eval / "x1_summary.json"
    if not xs.is_file():
        return None
    try:
        doc = json.loads(xs.read_text())
    except json.JSONDecodeError:
        return None
    if doc.get("data_revision") != cfg.data_revision:
        return None
    if not all((cfg.out_eval / f"E_{p}.npz").is_file() for p in POOLINGS):
        return None
    return f"E_(mean|max|frac).npz + x1_summary current (n={doc.get('n_conversations')})"


def phase_capture(cfg: Cfg, state: dict) -> None:
    """X1: gates (structural / approx-parity / fresh-capture parity / wall), then the
    full batched teacher-forced capture -> per-token SAE encode -> pool -> E (fp64
    in-run, fp16 persisted) + fail-loud HF upload BEFORE any downstream phase."""
    print("[phase=capture] start", flush=True)
    if not cfg.force:
        reason = _capture_outputs_current(cfg)
        if reason:
            print(
                f"[phase=capture] skip — outputs current ({reason}); --force recomputes", flush=True
            )
            return
    V, kci, seeds, doc = _load_kresample(cfg)
    n, k_draws = V.shape[0], V.shape[1]
    with np.load(cfg.y_holdout_path("mean")) as yh:
        y16_mean = yh["y16"].astype(np.float64)
        y_ci = yh["ci"].copy()
    _gate_structural(cfg, V, kci, seeds, doc, y_ci)
    with np.load(Path(cfg.perfeature_npz)) as pf:
        f_out = np.asarray(pf["feat_ids"], dtype=np.int64)
    f_out_t = torch.as_tensor(f_out)
    sae = _load_sae(cfg)  # CPU — gate 1 runs on the parent construct's own device
    probe_idx = list(range(min(cfg.n_probe, n)))
    _gate_parity_approx(cfg, V, kci, probe_idx, y16_mean, y_ci, f_out_t, sae)
    if cfg.cpu_gates_only:
        print(
            "[capture] cpu-gates-only: gates 1+2 PASS; gate 1b (fresh-capture parity), "
            "gate 3 (wall) + the capture loop are GPU legs — deferred to the pod smoke",
            flush=True,
        )
        return
    hf, tok = _load_model(cfg)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if cfg.device == "cuda" and not cfg.fixture:
        _sae_encode_device(sae, "cuda")
    t_render0 = time.time()
    rows_ids, plens, ntok = _render_all(tok, doc, seeds)
    print(
        f"[capture] rendered {len(rows_ids)} draws ({int(ntok.sum())} tokens) "
        f"in {time.time() - t_render0:.0f}s",
        flush=True,
    )
    # gate 1b + gate 3: capture the probe slice, compare vs banked V, project the wall
    probe_draws = [i * k_draws + k for i in probe_idx for k in range(k_draws)]
    t_slice0 = time.time()
    probe_spans = _capture_spans_oom_safe(
        cfg, hf, [rows_ids[d] for d in probe_draws], [plens[d] for d in probe_draws], pad_id
    )
    slice_wall = time.time() - t_slice0
    cells = []
    for j, d in enumerate(probe_draws):
        i, k = divmod(d, k_draws)
        fresh = probe_spans[j].mean(0)
        bank = V[i, k].to(torch.float32)
        cos = float(torch.nn.functional.cosine_similarity(fresh, bank, dim=0))
        rel = float(torch.linalg.vector_norm(fresh - bank) / torch.linalg.vector_norm(bank))
        cells.append(
            {"ci": kci[i], "seed": seeds[k], "cos": round(cos, 6), "rel_l2": round(rel, 6)}
        )
    bad = [c for c in cells if c["cos"] < cfg.cap_cos_min or c["rel_l2"] > cfg.cap_rell2_max]
    _atomic_json(
        cfg.gates_dir() / "capture_parity.json",
        {
            "ok": not bad,
            "n_cells": len(cells),
            "cos_min": min(c["cos"] for c in cells),
            "rel_l2_max": max(c["rel_l2"] for c in cells),
            "tol": {"cos_min": cfg.cap_cos_min, "rel_l2_max": cfg.cap_rell2_max},
            "bad_cells": bad[:20],
            **_repro(cfg),
        },
    )
    if bad:
        print(
            f"[gate] fresh-capture parity FAIL on {len(bad)}/{len(cells)} cells "
            f"(head: {bad[:3]}) — HALT (kill criterion 1b: render/inlier/capture-chain drift)",
            flush=True,
        )
        raise SystemExit(RC_GATE_CAPTURE)
    print(
        f"[gate] fresh-capture parity OK: {len(cells)} cells, cos_min="
        f"{min(c['cos'] for c in cells):.4f} rel_l2_max={max(c['rel_l2'] for c in cells):.4f}",
        flush=True,
    )
    slice_tokens = int(ntok[probe_draws].sum())
    total_tokens = int(ntok.sum())
    projected_h = slice_wall * (total_tokens / max(1, slice_tokens)) / 3600.0
    _atomic_json(
        cfg.gates_dir() / "wall.json",
        {
            "ok": projected_h <= cfg.wall_cap_h,
            "slice_wall_s": round(slice_wall, 2),
            "slice_tokens": slice_tokens,
            "total_tokens": total_tokens,
            "projected_capture_h": round(projected_h, 3),
            "cap_h": cfg.wall_cap_h,
            "batch": cfg.batch,
            **_repro(cfg),
        },
    )
    if projected_h > cfg.wall_cap_h:
        print(
            f"[gate] wall FAIL: projected capture {projected_h:.2f} h > {cfg.wall_cap_h} h — "
            "HALT (kill criterion 3: resize batch / re-dispatch wider)",
            flush=True,
        )
        raise SystemExit(RC_GATE_WALL)
    print(
        f"[gate] wall OK: projected capture {projected_h:.2f} h <= {cfg.wall_cap_h} h", flush=True
    )

    # ── main loop: E accumulation (smoke: probe slice only — reuses the probe spans) ──
    sel_rows = probe_idx if cfg.smoke else list(range(n))
    row_of = {i: j for j, i in enumerate(sel_rows)}
    n_out = len(sel_rows)
    F = len(f_out)
    E = {p: np.zeros((n_out, k_draws, F), dtype=np.float64) for p in POOLINGS}
    n_ans = np.zeros((n_out, k_draws), dtype=np.int64)
    n_inl = np.zeros((n_out, k_draws), dtype=np.int64)
    vx_cos_all: list[float] = []
    empties: list[dict] = []

    def _process(span: torch.Tensor, d: int) -> None:
        i, k = divmod(d, k_draws)
        j = row_of[i]
        vx_cos_all.append(
            float(torch.nn.functional.cosine_similarity(span.mean(0), V[i, k].float(), dim=0))
        )
        inl = SAEMOD.token_inlier_mask(span)
        span_in = span[inl]
        if span_in.shape[0] == 0:
            empties.append({"ci": kci[i], "seed": seeds[k]})
            return
        f = sae.encode(span_in.to(sae.device))
        pooled = SAEMOD.pool_answer_features(f)
        for p in POOLINGS:
            E[p][j, k] = pooled[p][f_out_t.to(pooled[p].device)].double().cpu().numpy()
        n_ans[j, k] = int(span.shape[0])
        n_inl[j, k] = int(span_in.shape[0])

    t_loop0 = time.time()
    if cfg.smoke:
        for j, d in enumerate(probe_draws):
            _process(probe_spans[j], d)
        print(f"[capture] smoke slice: {len(probe_draws)} draws processed", flush=True)
    else:
        all_draws = [i * k_draws + k for i in sel_rows for k in range(k_draws)]
        order = sorted(all_draws, key=lambda d: int(rows_ids[d].shape[0]))
        n_units = (len(order) + CHUNK_DRAWS - 1) // CHUNK_DRAWS
        for u, s0 in enumerate(range(0, len(order), CHUNK_DRAWS)):
            sel = order[s0 : s0 + CHUNK_DRAWS]
            spans = _capture_spans_oom_safe(
                cfg, hf, [rows_ids[d] for d in sel], [plens[d] for d in sel], pad_id
            )
            for local, d in enumerate(sel):
                _process(spans[local], d)
            print(
                f"[capture] unit {u + 1}/{n_units} draws={s0 + len(sel)}/{len(order)} "
                f"elapsed={time.time() - t_loop0:.0f}s batch={cfg.batch}",
                flush=True,
            )
    _check_inlier_drops(cfg, empties, n_out * k_draws)
    # drop conversations with ANY empty draw from the floors (n shrinks, flagged)
    empty_ci = {e["ci"] for e in empties}
    keep = np.asarray([kci[i] not in empty_ci for i in sel_rows])
    kept_ci = np.asarray([kci[i] for i in sel_rows], dtype=np.int64)[keep]
    for p in POOLINGS:
        E[p] = E[p][keep]
    vx = np.asarray(vx_cos_all)
    summary = {
        "n_conversations": int(keep.sum()),
        "n_dropped_conversations": int((~keep).sum()),
        "k_draws": int(k_draws),
        "n_features": F,
        "capture_wall_s": round(time.time() - t_loop0, 1),
        "batch_final": cfg.batch,
        "vx_cos_diagnostic": {
            "min": float(vx.min()),
            "p01": float(np.quantile(vx, 0.01)),
            "median": float(np.median(vx)),
            "n_below_0.99": int((vx < 0.99).sum()),
        },
        "data_revision": cfg.data_revision,
        **_repro(cfg),
    }
    cfg.out_eval.mkdir(parents=True, exist_ok=True)
    for p in POOLINGS:
        np.savez(
            cfg.out_eval / f"E_{p}.npz",
            E=E[p].astype(np.float16),
            ci=kept_ci,
            seeds=np.asarray(seeds, dtype=np.int64),
            feat_ids=f_out,
            data_revision=np.array(cfg.data_revision),
        )
    _atomic_json(
        cfg.out_eval / "token_counts.json",
        {
            "ci": [int(kci[i]) for i in sel_rows],
            "seeds": seeds,
            "n_answer_tokens": n_ans.tolist(),
            "n_inlier_tokens": n_inl.tolist(),
            **_repro(cfg),
        },
    )
    _atomic_json(cfg.out_eval / "x1_summary.json", summary)
    if not (cfg.no_upload or cfg.smoke or cfg.fixture):
        files = [f"E_{p}.npz" for p in POOLINGS] + ["token_counts.json", "x1_summary.json"]
        url = hub._upload_folder_filtered(
            cfg.out_eval,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=cfg.upload_prefix,
            allow_patterns=files,
            expected_repo_paths=[f"{cfg.upload_prefix}/{f}" for f in files],
        )
        if not url:
            raise RuntimeError("X1 E-tensor upload returned no URL")
        print(f"[capture] uploaded {len(files)} files -> {cfg.upload_prefix}", flush=True)
    state["E"] = E
    state["E_ci"] = kept_ci
    state["seeds"] = seeds
    _atomic_json(cfg.sentinel_dir() / "capture.done.json", summary)
    print("[phase=capture] done", flush=True)


# ── X2: exact floors (verbatim ddof-1 trace arithmetic per pooling) ─────────────────


def _floors_outputs_current(cfg: Cfg) -> str | None:
    for pooling in POOLINGS:
        np_p = cfg.env_root(pooling) / "kresample" / f"floors_L{LAYER}.npz"
        yh_p = cfg.y_holdout_path(pooling)
        lab = cfg.env_root(pooling) / "judge_labels" / "labels.json"
        if not (np_p.is_file() and yh_p.is_file() and lab.is_file()):
            return None
        with np.load(yh_p) as yh:
            want = str(yh["fingerprint"])
        with np.load(np_p) as z:
            if "fingerprint" not in z or str(z["fingerprint"]) != want:
                return None
    return "floors_env_*_exact npz fingerprints match their y_holdouts"


def phase_floors(cfg: Cfg, state: dict) -> None:
    """X2: floor = ddof-1 trace over the K EXACT encode-then-pool draws (verbatim
    `issue1738_characterize.phase_kresample_floor` L446-453 arithmetic); den from
    each pooling's banked y_holdout (dense_feat shares sae_space's y per
    ``Y_HOLDOUT_SPACE``). Precision seam (plan section 3): a chained run consumes the
    fp64 in-memory E; a sentinel-resumed run consumes the persisted fp16 E npz —
    immaterial for the floors (deviation magnitude >> fp16 quantization)."""
    print("[phase=floors] start", flush=True)
    if not cfg.force:
        reason = _floors_outputs_current(cfg)
        if reason:
            print(
                f"[phase=floors] skip — outputs current ({reason}); --force recomputes", flush=True
            )
            return
    E = state.get("E")
    kept_ci = state.get("E_ci")
    k_draws = None
    if E is None:
        E, kept_ci = {}, None
        for p in POOLINGS:
            with np.load(cfg.out_eval / f"E_{p}.npz") as z:
                assert str(z["data_revision"]) == cfg.data_revision, (
                    f"E_{p}.npz revision {z['data_revision']} != {cfg.data_revision}"
                )
                E[p] = z["E"].astype(np.float64)
                kept_ci = z["ci"].copy()
        print("[floors] E loaded from persisted fp16 npz (sentinel resume)", flush=True)
    assert kept_ci is not None
    upload_files: list[str] = []
    for pooling in POOLINGS:
        with np.load(cfg.y_holdout_path(pooling)) as yh:
            y16 = yh["y16"].astype(np.float64)
            y_ci = yh["ci"]
            yh_fp = str(yh["fingerprint"])
        pos_of = {int(c): p_ for p_, c in enumerate(y_ci.tolist())}
        missing = [int(c) for c in kept_ci if int(c) not in pos_of]
        assert not missing, f"floors[{pooling}]: {len(missing)} ci absent from y_holdout"
        hp = np.asarray([pos_of[int(c)] for c in kept_ci])
        mu = y16.mean(axis=0)
        den = ((y16[hp] - mu) ** 2).sum(axis=1)
        Ep = E[pooling]
        k_draws = Ep.shape[1]
        ebar = Ep.mean(axis=1, keepdims=True)
        floor = ((Ep - ebar) ** 2).sum(axis=(1, 2)) / (k_draws - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = floor / den
        kdir = cfg.env_root(pooling) / "kresample"
        kdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            kdir / f"floors_L{LAYER}.npz",
            ci=np.asarray(kept_ci, dtype=np.int64),
            floor=floor.astype(np.float64),
            den=den.astype(np.float64),
            share=share.astype(np.float64),
            fingerprint=np.array(yh_fp),
        )
        _atomic_json(
            kdir / "floor_summary.json",
            {
                "construct": "sae_encode_then_pool (exact)",
                "pooling": pooling,
                "n": int(len(kept_ci)),
                "k_draws": int(k_draws),
                "floor_median": float(np.nanmedian(floor)),
                "floor_share_median": float(np.nanmedian(share)),
                "floor_share_mean": float(np.nanmean(share)),
                **_repro(cfg),
            },
        )
        env = ENV_OF_POOL[pooling]
        upload_files += [
            f"{env}/kresample/floors_L{LAYER}.npz",
            f"{env}/kresample/floor_summary.json",
        ]
        print(
            f"[floors] {pooling}: n={len(kept_ci)} share_median={float(np.nanmedian(share)):.4f}",
            flush=True,
        )
        # judge CONTEXT labels (banked instrument) into each exact env root
        dst = cfg.env_root(pooling) / "judge_labels" / "labels.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert Path(cfg.labels_src).is_file(), cfg.labels_src
        shutil.copyfile(cfg.labels_src, dst)
    if not (cfg.no_upload or cfg.smoke or cfg.fixture):
        url = hub._upload_folder_filtered(
            cfg.out_eval,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=cfg.upload_prefix,
            allow_patterns=upload_files,
            expected_repo_paths=[f"{cfg.upload_prefix}/{f}" for f in upload_files],
        )
        if not url:
            raise RuntimeError("X2 floors upload returned no URL")
        print(f"[floors] uploaded {len(upload_files)} files -> {cfg.upload_prefix}", flush=True)
    _atomic_json(
        cfg.sentinel_dir() / "floors.done.json",
        {"n": int(len(kept_ci)), "k_draws": int(k_draws or 0), **_repro(cfg)},
    )
    print("[phase=floors] done", flush=True)


# ── X3: verbatim battery per space + identity gate (kill criterion 4) ───────────────


def _gate_identity(cfg: Cfg, space: str) -> None:
    """Gate 4: the exact-env battery's UNADJUSTED tables must reproduce the banked
    ones (same percontext/labels/manifest inputs + seed 1738 — floors only enter the
    floor_adjusted block). Compares VALUES, never recomputed selections (parent
    argsort-tie lesson, plan section 11)."""
    exact = json.loads((cfg.space_out(space) / "taxonomy.json").read_text())
    banked = json.loads((cfg.banked_eval / space / "taxonomy.json").read_text())
    worst = 0.0
    problems: list[dict] = []
    for arm in ARMS:
        key = f"{arm}_L{LAYER}_ridge"
        fam_e = exact["arms"][key]["family"]
        fam_b = banked["arms"][key]["family"]
        if fam_e != fam_b:
            problems.append({"arm": arm, "reason": "family_drift"})
            continue
        rows_e = {r["contrast"]: r for r in exact["arms"][key]["contrasts"]}
        rows_b = {r["contrast"]: r for r in banked["arms"][key]["contrasts"]}
        for c in fam_b:
            dev = abs(rows_e[c]["delta_mean_nerr"] - rows_b[c]["delta_mean_nerr"])
            worst = max(worst, dev)
            if dev > IDENTITY_ATOL or rows_e[c]["n_group"] != rows_b[c]["n_group"]:
                problems.append({"arm": arm, "contrast": c, "abs_dev": dev})
    _atomic_json(
        cfg.gates_dir() / f"identity_{space}.json",
        {
            "ok": not problems,
            "max_abs_dev": worst,
            "atol": IDENTITY_ATOL,
            "problems_head": problems[:10],
            **_repro(cfg),
        },
    )
    if problems:
        print(
            f"[gate] battery identity FAIL ({space}): {len(problems)} deviations "
            f"(head: {problems[:3]}) — HALT (kill criterion 4: staged-input mix-up)",
            flush=True,
        )
        raise SystemExit(RC_GATE_IDENTITY)
    print(f"[gate] battery identity OK ({space}): max |dev|={worst:.2e}", flush=True)


def _battery_output_current(cfg: Cfg, space: str) -> str | None:
    tax_p = cfg.space_out(space) / "taxonomy.json"
    if not tax_p.is_file():
        return None
    try:
        tax = json.loads(tax_p.read_text())
    except json.JSONDecodeError:
        return None
    if not all(f"{a}_L{LAYER}_ridge" in tax.get("arms", {}) for a in ARMS):
        return None
    pooling = PC.POOL_OF_SPACE[space]
    floors_p = cfg.env_root(pooling) / "kresample" / f"floors_L{LAYER}.npz"
    if not floors_p.is_file() or tax_p.stat().st_mtime < floors_p.stat().st_mtime:
        return None
    return "taxonomy.json current (3 arm tables, newer than the exact floors npz)"


def phase_battery(cfg: Cfg, state: dict) -> None:
    """X3: `issue1738_characterize.py --phase taxonomy` VERBATIM per space — verdict
    pass (sae_space, dense_feat_space @ floors_env_mean_exact), exploratory pass
    (max/frac @ their exact envs). No reimplemented statistics; rc!=0 halts loudly."""
    print("[phase=battery] start", flush=True)
    script = PROJECT_ROOT / "scripts" / "issue1738_characterize.py"
    for i, space in enumerate(SPACES, 1):
        if not cfg.force:
            reason = _battery_output_current(cfg, space)
            if reason:
                print(f"[battery] space {i}/4 {space}: skip — {reason}", flush=True)
                continue
        exact_out = cfg.space_out(space)
        (exact_out / "percontext").mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            src = cfg.staged_space(space) / "percontext" / f"{arm}_L{LAYER}_ridge.npz"
            assert src.is_file(), f"staged percontext missing: {src}"
            shutil.copyfile(src, exact_out / "percontext" / src.name)
        pooling = PC.POOL_OF_SPACE[space]
        cmd = [
            sys.executable,
            str(script),
            "--phase",
            "taxonomy",
            "--layers",
            str(LAYER),
            "--arms",
            ",".join(ARMS),
            "--out-eval",
            str(exact_out),
            "--parent-eval",
            str(cfg.env_root(pooling)),
            "--pred16-dir",
            str(cfg.staged_space(space) / "pred16"),
            "--y-holdout-dir",
            str(cfg.staged_space(PC.Y_HOLDOUT_SPACE[pooling]) / "y_holdout"),
            "--manifest-dir",
            str(cfg.manifest_dir),
            "--split-file",
            str(cfg.split_file),
            "--scratch",
            str(cfg.staging_root / "scratch"),
        ]
        if cfg.no_upload or cfg.smoke or cfg.fixture:
            cmd.append("--no-upload")
        else:
            cmd += ["--upload-prefix", f"{cfg.upload_prefix}/{space}"]
        t0 = time.time()
        subprocess.run(cmd, check=True, env={**os.environ})
        tax = json.loads((exact_out / "taxonomy.json").read_text())
        assert len(tax["arms"]) == 3, f"{space}: expected 3 arm tables, got {sorted(tax['arms'])}"
        assert tax["floor_adjusted_available"] is True, f"{space}: floors not consumed"
        print(f"[battery] space {i}/4 {space}: done in {time.time() - t0:.0f}s", flush=True)
        _gate_identity(cfg, space)
    print("[phase=battery] done", flush=True)


# ── X4: compare (exact fas + registered decision rule + gate 5 coverage) ────────────


def _fa_rows(tax: dict, arm: str) -> dict[str, dict]:
    blk = tax["arms"][f"{arm}_L{LAYER}_ridge"].get("floor_adjusted")
    return {} if blk is None else {r["contrast"]: r for r in blk["contrasts"]}


def _fas_block(sae_tax: dict, dense_tax: dict, dense_bare_tax: dict) -> dict:
    """Floor-adjusted sensitivity per arm (exact SAE floor-adjusted deltas vs the
    banked DENSE floor-adjusted deltas) — the parent's phase_compare block, reusing
    `PC._spearman_perm` (seed 1946, 10k two-sided perms)."""
    out: dict[str, dict] = {}
    for arm in ARMS:
        d_fa = _fa_rows(dense_bare_tax if arm == "bare" else dense_tax, arm)
        s_fa = _fa_rows(sae_tax, arm)
        shared = [c for c in d_fa if c in s_fa]
        if len(shared) < 3:
            out[arm] = {"available": False, "n_shared_contrasts": len(shared)}
            continue
        dv = np.asarray([d_fa[c]["delta_mean_adj_nerr"] for c in shared])
        sv = np.asarray([s_fa[c]["delta_mean_adj_nerr"] for c in shared])
        rho, p = PC._spearman_perm(sv, dv, N_PERM_CROSS, CROSS_SEED)
        sig = [c for c in shared if d_fa[c]["bh_significant"]]
        agree = int(
            sum(
                1
                for c in sig
                if np.sign(s_fa[c]["delta_mean_adj_nerr"])
                == np.sign(d_fa[c]["delta_mean_adj_nerr"])
                != 0
            )
        )
        out[arm] = {
            "available": True,
            "n_shared_contrasts": len(shared),
            "shared_contrasts": shared,
            "rho": rho,
            "perm_p_two_sided": p,
            "n_dense_significant_shared": len(sig),
            "sign_agree_on_dense_significant": agree,
        }
    return out


def _compare_output_current(cfg: Cfg) -> str | None:
    out_p = cfg.out_eval / "crossspace_comparison_exact.json"
    if not out_p.is_file():
        return None
    try:
        doc = json.loads(out_p.read_text())
    except json.JSONDecodeError:
        return None
    if "decision" not in doc:
        return None
    tax_ps = [cfg.space_out(s) / "taxonomy.json" for s in SPACES]
    if not all(p.is_file() for p in tax_ps):
        return None
    if out_p.stat().st_mtime < max(p.stat().st_mtime for p in tax_ps):
        return None
    return f"crossspace_comparison_exact.json current (decision={doc['decision']})"


def phase_compare(cfg: Cfg, state: dict) -> None:
    print("[phase=compare] start", flush=True)
    if not cfg.force:
        reason = _compare_output_current(cfg)
        if reason:
            print(f"[phase=compare] skip — outputs current ({reason})", flush=True)
            state["compare"] = json.loads(
                (cfg.out_eval / "crossspace_comparison_exact.json").read_text()
            )
            return
    dense_tax = json.loads((cfg.dense_eval / "taxonomy.json").read_text())
    dense_bare_tax = json.loads((cfg.dense_eval / "bare_query" / "taxonomy.json").read_text())
    exact_sae = json.loads((cfg.space_out("sae_space") / "taxonomy.json").read_text())
    banked_sae = json.loads((cfg.banked_eval / "sae_space" / "taxonomy.json").read_text())
    fas_exact = _fas_block(exact_sae, dense_tax, dense_bare_tax)
    fas_banked = _fas_block(banked_sae, dense_tax, dense_bare_tax)
    # gate 5 (kill criterion 5): exact shared-contrast set == the parent's, per arm
    coverage_problems = []
    for arm in ARMS:
        se = set(fas_exact[arm].get("shared_contrasts", []))
        sb = set(fas_banked[arm].get("shared_contrasts", []))
        if se != sb:
            coverage_problems.append(
                {"arm": arm, "only_exact": sorted(se - sb), "only_banked": sorted(sb - se)}
            )
        elif not cfg.fixture and len(se) != N_SHARED_EXPECTED:
            coverage_problems.append({"arm": arm, "n_shared": len(se), "want": N_SHARED_EXPECTED})
    if coverage_problems:
        halted = {
            "decision": "HALTED_coverage_changed",
            "coverage_problems": coverage_problems,
            "floor_adjusted_sensitivity_exact": fas_exact,
            **_repro(cfg),
        }
        _atomic_json(cfg.out_eval / "crossspace_comparison_exact.json", halted)
        print(
            f"[gate] coverage FAIL: {coverage_problems} — verdict read HALTED (kill criterion 5)",
            flush=True,
        )
        raise SystemExit(RC_GATE_COVERAGE)
    print(f"[gate] coverage OK: shared set matches the parent's per arm", flush=True)

    # registered decision rule (plan section 1 — DISJOINT and exhaustive)
    s_fa = {arm: _fa_rows(exact_sae, arm) for arm in ARMS}
    air = s_fa["bare"]["answer_is_refusal=yes"]
    a_val = float(air["delta_mean_adj_nerr"])
    family_rows = {
        f"{arm}:{c}": {
            "delta_mean_adj_nerr": s_fa[arm][c]["delta_mean_adj_nerr"],
            "boot_ci": s_fa[arm][c]["boot_ci"],
            "bh_significant": s_fa[arm][c]["bh_significant"],
            "n_group": s_fa[arm][c]["n_group"],
        }
        for arm in ARMS
        for c in REFUSAL_FAMILY
        if c in s_fa[arm]
    }
    positive_sig = any(
        r["bh_significant"] and r["delta_mean_adj_nerr"] > 0 for r in family_rows.values()
    )
    if a_val <= 0 and not positive_sig:
        decision = "Collapse-preserved"
    elif bool(air["bh_significant"]) and a_val > 0:
        decision = "Collapse-retracted"
    else:
        decision = "Mixed"
    ci_lo, ci_hi = air["boot_ci"]
    underpower_note = None
    if decision == "Mixed":
        underpower_note = (
            f"Mixed with n_group={air['n_group']} and exact-adjusted CI width "
            f"{ci_hi - ci_lo:.3f} — may reflect CI width (underpower), not sign "
            "instability (plan section 12 analyzer note 2)"
        )

    # banked-constants staging-mixup assert (production only; plan section 1 values)
    if not cfg.fixture:
        b_un = {r["contrast"]: r for r in banked_sae["arms"][f"bare_L{LAYER}_ridge"]["contrasts"]}[
            "answer_is_refusal=yes"
        ]["delta_mean_nerr"]
        b_ad = _fa_rows(banked_sae, "bare")["answer_is_refusal=yes"]["delta_mean_adj_nerr"]
        assert abs(b_un - BANKED_BARE_AIR_UNADJ) < BANKED_CONST_TOL, (b_un, BANKED_BARE_AIR_UNADJ)
        assert abs(b_ad - BANKED_BARE_AIR_ADJ) < BANKED_CONST_TOL, (b_ad, BANKED_BARE_AIR_ADJ)

    # registered secondary (b): approx-vs-exact floor diagnostic (mean pooling)
    diag: dict = {"available": False}
    exact_np = cfg.env_root("mean") / "kresample" / f"floors_L{LAYER}.npz"
    if exact_np.is_file() and Path(cfg.approx_floors_npz).is_file():
        with np.load(exact_np) as ez, np.load(Path(cfg.approx_floors_npz)) as az:
            apos = {int(c): p for p, c in enumerate(az["ci"].tolist())}
            sel = [i for i, c in enumerate(ez["ci"].tolist()) if int(c) in apos]
            asel = np.asarray([apos[int(ez["ci"][i])] for i in sel])
            fe, fa_ = ez["floor"][sel], az["floor"][asel]
            se_, sa_ = ez["share"][sel], az["share"][asel]
            rho_f, p_f = scipy.stats.spearmanr(fe, fa_)
            diag = {
                "available": True,
                "n": len(sel),
                "spearman_floor_exact_vs_approx": float(rho_f),
                "spearman_p": float(p_f),
                "share_median_exact": float(np.nanmedian(se_)),
                "share_median_approx": float(np.nanmedian(sa_)),
                "share_median_approx_cited": APPROX_SHARE_MEDIAN_CITED,
                "note": "DIAGNOSTIC only — no verdict weight (plan section 12 analyzer note 1)",
            }

    # exploratory (registered secondary (c)): max/frac exact-adjusted refusal rows
    exploratory = {}
    for space in ("max_space", "frac_space"):
        tax_p = cfg.space_out(space) / "taxonomy.json"
        if not tax_p.is_file():
            continue
        tax = json.loads(tax_p.read_text())
        exploratory[space] = {
            f"{arm}:{c}": {
                "delta_mean_adj_nerr": row["delta_mean_adj_nerr"],
                "bh_significant": row["bh_significant"],
            }
            for arm in ARMS
            for c, row in _fa_rows(tax, arm).items()
            if c in REFUSAL_FAMILY
        }

    out = {
        "layer": LAYER,
        "seed_cross": CROSS_SEED,
        "n_perm_cross": N_PERM_CROSS,
        "decision": decision,
        "decision_rule": (
            "verdict space = mean-pooled SAE, exact-floor-adjusted battery; "
            "A = exact-adjusted bare answer_is_refusal=yes delta_mean_adj_nerr; "
            "Collapse-preserved iff A <= 0 AND no refusal-family contrast "
            "(refusal_adjacent=yes, answer_is_refusal=yes x 3 arms) is BH-significant "
            "with positive adjusted delta; Collapse-retracted iff bare answer_is_refusal "
            "exact-adjusted is BH-significant AND A > 0; Mixed otherwise (plan v7 section 1)"
        ),
        "A_exact_adjusted_bare_answer_is_refusal": a_val,
        "A_boot_ci": air["boot_ci"],
        "A_bh_significant": bool(air["bh_significant"]),
        "A_n_group": int(air["n_group"]),
        "underpower_note": underpower_note,
        "refusal_family_exact": family_rows,
        "floor_adjusted_sensitivity_exact": fas_exact,
        "floor_adjusted_sensitivity_banked_approx": fas_banked,
        "approx_vs_exact_floor_diagnostic": diag,
        "exploratory_max_frac": exploratory,
        "consequence": {
            "Collapse-preserved": "de-quarantine Takeaways bullet 6 (rewrite citing exact floors)",
            "Collapse-retracted": "rewrite bullet 6 + retract the sampling-variance interpretation",
            "Mixed": "bullet 6 stays quarantined; exact numbers reported",
        }[decision],
        **_repro(cfg),
    }
    _atomic_json(cfg.out_eval / "crossspace_comparison_exact.json", out)
    if not (cfg.no_upload or cfg.smoke or cfg.fixture):
        gate_files = sorted(
            str(p.relative_to(cfg.out_eval)) for p in cfg.gates_dir().glob("*.json")
        )
        files = ["crossspace_comparison_exact.json"] + gate_files
        url = hub._upload_folder_filtered(
            cfg.out_eval,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=cfg.upload_prefix,
            allow_patterns=files,
            expected_repo_paths=[f"{cfg.upload_prefix}/{f}" for f in files],
        )
        if not url:
            raise RuntimeError("X4 comparison upload returned no URL")
        print(f"[compare] uploaded {len(files)} JSONs -> {cfg.upload_prefix}", flush=True)
    state["compare"] = out
    print(f"[phase=compare] done (decision={decision})", flush=True)


# ── V1: harvest (VM-side — fetch JSONs from HF + render the exact hero figure) ──────

HARVEST_CORE = ("crossspace_comparison_exact.json", "x1_summary.json", "token_counts.json")


def phase_harvest(cfg: Cfg, state: dict) -> None:
    """V1 (off-pod): fetch the pod-produced JSONs from HF into the git tree (existing
    local files win — the pod already has them locally), then render the exact
    floor-adjustment hero figure via ``savefig_paper``. Committing is the
    orchestrator's explicit-path commit (never driver-side git)."""
    print("[phase=harvest] start", flush=True)
    rels = list(HARVEST_CORE)
    rels += [f"{space}/taxonomy.json" for space in SPACES]
    rels += [f"{space}/depth_contrasts.json" for space in SPACES]
    rels += [f"{ENV_OF_POOL[p]}/kresample/floor_summary.json" for p in POOLINGS]
    rels += [f"{ENV_OF_POOL['mean']}/kresample/floors_L{LAYER}.npz"]
    optional = {
        f"gates/{n}"
        for n in (
            "structural.json",
            "parity_approx.json",
            "capture_parity.json",
            "wall.json",
            "inlier.json",
            *(f"identity_{s}.json" for s in SPACES),
        )
    }
    rels += sorted(optional)

    # The battery dual-write nests its summaries under the characterize layout
    # (<space>/analysis_tensors/summaries/characterize/<name>); local dst stays flat.
    def _hub_rel(rel: str) -> str:
        parts = rel.split("/")
        if (
            len(parts) == 2
            and parts[0] in SPACES
            and parts[1]
            in (
                "taxonomy.json",
                "depth_contrasts.json",
            )
        ):
            return f"{parts[0]}/analysis_tensors/summaries/characterize/{parts[1]}"
        return rel

    n_fetched = 0
    for rel in rels:
        dst = cfg.out_eval / rel
        if dst.exists():
            continue
        if cfg.fixture:  # fixture ran every phase locally — nothing to fetch
            continue
        try:
            hub.stage_hub_file(
                C.HF_DATA_REPO, f"{cfg.upload_prefix}/{_hub_rel(rel)}", dst, repo_type="dataset"
            )
            n_fetched += 1
        except Exception:
            if rel in optional:
                print(f"[harvest] optional {rel}: absent on Hub — skipped", flush=True)
                continue
            raise
    print(f"[harvest] fetched {n_fetched} missing files from {cfg.upload_prefix}", flush=True)

    # hero figure: unadjusted vs EXACT-adjusted deltas (prefix + bare), the parent's
    # flooradj_vs_unadjusted layout with the refusal-family contrasts highlighted
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    exact_sae = json.loads((cfg.space_out("sae_space") / "taxonomy.json").read_text())
    pal = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, arm in zip(axes, ("prefix", "bare"), strict=False):
        key = f"{arm}_L{LAYER}_ridge"
        un_rows = {r["contrast"]: r for r in exact_sae["arms"][key]["contrasts"]}
        fa = _fa_rows(exact_sae, arm)
        shared = [c for c in exact_sae["arms"][key]["family"] if c in fa]
        un = np.asarray([un_rows[c]["delta_mean_nerr"] for c in shared])
        ad = np.asarray([fa[c]["delta_mean_adj_nerr"] for c in shared])
        is_ref = np.asarray([c in REFUSAL_FAMILY for c in shared])
        ax.axline((0, 0), slope=1, color="0.6", lw=0.7)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.scatter(un[~is_ref], ad[~is_ref], s=24, color=pal[2], label="other categories")
        ax.scatter(un[is_ref], ad[is_ref], s=42, color=pal[1], label="refusal categories")
        for c in REFUSAL_FAMILY:
            if c in fa:
                ax.annotate(
                    c, (un_rows[c]["delta_mean_nerr"], fa[c]["delta_mean_adj_nerr"]), fontsize=6
                )
        ax.set_xlabel(f"unadjusted SAE delta ({arm} arm)")
        ax.set_ylabel("exact floor-adjusted delta")
        ax.legend(fontsize=7)
    fig.suptitle("Exact encode-then-pool floors: adjustment effect per category", fontsize=10)
    fig.tight_layout()
    paths = savefig_paper(fig, "floor_adjustment_exact", dir=cfg.fig_dir)
    plt.close(fig)
    PC._check_png(Path(paths["png"]))
    print(f"[harvest] figure written: {paths['png']}", flush=True)
    print("[phase=harvest] done", flush=True)


# ── results sentinel (poll_pipeline contract; pod-side, never task.py) ───────────────


def _write_results_sentinel(cfg: Cfg, state: dict) -> None:
    gates = {}
    for p in cfg.gates_dir().glob("*.json"):
        try:
            gates[p.stem] = bool(json.loads(p.read_text()).get("ok"))
        except json.JSONDecodeError:
            gates[p.stem] = None
    comp = state.get("compare", {})
    payload = {
        "followup_label": "exact-sae-floors",
        "decision": comp.get("decision"),
        "A_exact_adjusted_bare_answer_is_refusal": comp.get(
            "A_exact_adjusted_bare_answer_is_refusal"
        ),
        "consequence": comp.get("consequence"),
        "gates_ok": gates,
        "hf_prefix": cfg.upload_prefix,
        "eval_json_paths": [
            f"eval_results/issue_{TASK_ID}/exact_floors/crossspace_comparison_exact.json",
            *(f"eval_results/issue_{TASK_ID}/exact_floors/{s}/taxonomy.json" for s in SPACES),
        ],
        "figures": [f"figures/issue_{TASK_ID}/floor_adjustment_exact.png"],
        "wandb_url": "n/a (no training — analysis-only follow-up)",
        "repro": _repro(cfg),
    }
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    if cfg.fixture:
        _atomic_json(cfg.out_eval / "pod_sentinel_preview.json", {"kind": kind, **payload})
        return
    C.write_sentinel(
        kind,
        json.dumps(payload, indent=2),
        task_id=TASK_ID,
        extra={"blocks_pipeline": False, "smoke": bool(cfg.smoke)},
    )


# ── fixture smoke: tiny-real CPU e2e through the SAME phase functions ────────────────

FIXTURE_N = 32
FIXTURE_F_OUT = 64


def _build_fixture(base: Path) -> Cfg:
    """Materialize the tiny-real fixture: REAL Qwen tokenizer + from-config 2-layer
    Qwen2 + tiny BatchTopK SAE (SA smoke recipe), FIXTURE_N conversations x K=4
    responses whose banked ``V`` is captured by the SAME render+capture path the
    driver runs (so gate 1b is a REAL check), banked-layout percontext/pred16/
    y_holdout fixtures, and 'banked' approx floors + taxonomies produced by the REAL
    parent producers (`PC.phase_floors` + the characterize battery)."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    model_dir = base / "models"
    (model_dir / "tok").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(str(model_dir / "tok"))
    torch.manual_seed(1946)
    mcfg = Qwen2Config(
        vocab_size=len(tok) + 128,
        hidden_size=SA.SMOKE_H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    hf = Qwen2ForCausalLM(mcfg)
    hf.eval()
    hf.save_pretrained(str(model_dir / "model"))
    sd = {
        "b_dec": torch.zeros(SA.SMOKE_H),
        "k": torch.tensor(SA.SMOKE_K, dtype=torch.int32),
        "threshold": torch.tensor(0.05),
        "decoder.weight": torch.randn(SA.SMOKE_H, SA.SMOKE_DICT) * 0.1,
        "encoder.weight": torch.randn(SA.SMOKE_DICT, SA.SMOKE_H) * 0.5,
        "encoder.bias": torch.zeros(SA.SMOKE_DICT),
    }
    torch.save(sd, model_dir / "sae_ae.pt")

    ci = list(range(FIXTURE_N))
    seeds = list(SEEDS_EXPECTED)
    rows = []
    for i in ci:
        messages = [
            {"role": "user", "content": f"Tell me about topic {i}."},
            {"role": "assistant", "content": "It is a broad topic."},
            {"role": "user", "content": f"Give me {1 + i % 3} more details."},
        ]
        responses = {
            str(s): " ".join(
                f"Fixture answer {i} draw {s} sentence {t}." for t in range(2 + (i + s) % 4)
            )
            for s in seeds
        }
        rows.append({"ci": i, "messages": messages, "responses": responses})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    v_cells: list[list[torch.Tensor]] = []
    for row in rows:
        per_seed = []
        for s in seeds:
            ids, plen = SA._render_row(tok, row["messages"], row["responses"][str(s)])
            span = SA._capture_answer_spans(hf, [ids], [plen], SA.SMOKE_LAYER, 1, pad_id)[0]
            per_seed.append(span.mean(0))
        v_cells.append(per_seed)
    V = torch.stack([torch.stack(per) for per in v_cells])  # (n, K, H)
    kdir = base / "kresample"
    kdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "V": V.unsqueeze(2).to(torch.float16),
            "ci": ci,
            "seeds": seeds,
            "layers": [LAYER],  # file-keying stays L19; content captured at SMOKE_LAYER
        },
        kdir / "kresample_shard00.pt",
    )
    C.write_json_atomic(kdir / "kresample_shard00.json", {"shard_index": 0, "rows": rows})

    rng = np.random.default_rng(1946)
    f_out = np.sort(rng.choice(SA.SMOKE_DICT, FIXTURE_F_OUT, replace=False)).astype(np.int64)
    pf_dir = base / "perfeature"
    pf_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pf_dir / "perfeature_summary.npz", feat_ids=f_out)

    staged = base / "staged" / BANKED_PREFIX / "analysis_tensors"
    ci_arr = np.asarray(ci, dtype=np.int64)
    y_by_pool: dict[str, np.ndarray] = {}
    for pooling, space in PC.Y_HOLDOUT_SPACE.items():
        y16 = rng.normal(size=(FIXTURE_N, FIXTURE_F_OUT)).astype(np.float16)
        y_by_pool[pooling] = y16
        yd = staged / space / "y_holdout"
        yd.mkdir(parents=True, exist_ok=True)
        np.savez(
            yd / f"L{LAYER}.npz",
            y16=y16,
            ci=ci_arr,
            fingerprint=np.array(f"fixture-{pooling}"),
        )
    for space in SPACES:
        for arm in ARMS:
            pc_dir = staged / space / "percontext"
            pd_dir = staged / space / "pred16"
            pc_dir.mkdir(parents=True, exist_ok=True)
            pd_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                pc_dir / f"{arm}_L{LAYER}_ridge.npz",
                nerr=np.abs(rng.normal(1.0, 0.3, FIXTURE_N)),
                ci=ci_arr,
            )
            np.savez(
                pd_dir / f"{arm}_L{LAYER}_ridge.npz",
                pred16=rng.normal(size=(FIXTURE_N, FIXTURE_F_OUT)).astype(np.float16),
                ci=ci_arr,
                fingerprint=np.array(f"fixture-{PC.POOL_OF_SPACE[space]}"),
            )

    manifest_dir = base / "manifest"
    PC._smoke_manifest_fixture(manifest_dir, ci)
    PC._smoke_labels_fixture(base / "judge_labels" / "labels.json", ci)
    split_p = base / "split_1738.json"
    split_p.write_text(json.dumps({"sets": {"holdout": {"ci": ci, "n": len(ci)}}}))

    # 'banked' APPROX floors via the REAL parent producer (PC.phase_floors)
    banked_env = base / "banked_env"
    yh_dst = banked_env / "sae_space" / "y_holdout"
    yh_dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged / "sae_space" / "y_holdout" / f"L{LAYER}.npz", yh_dst / f"L{LAYER}.npz")
    pc_cfg = PC.Cfg(
        staging_root=base / "pc_stage",
        out_eval=banked_env,
        fig_dir=base / "pc_figs",
        dense_eval=base / "pc_dense_unused",
        revision="fixture",
        upload_prefix="fixture_unused",
        smoke=True,
        kresample_dir=kdir,
        perfeature_npz=pf_dir / "perfeature_summary.npz",
        manifest_dir=manifest_dir,
        smoke_model_dir=model_dir,
    )
    PC.phase_floors(pc_cfg)

    # 'banked' taxonomies via the REAL battery (approx env for mean spaces,
    # labels-only for max/frac — the parent regime)
    banked_eval = base / "banked_eval"
    script = PROJECT_ROOT / "scripts" / "issue1738_characterize.py"
    for space in SPACES:
        b_out = banked_eval / space
        (b_out / "percontext").mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            src = staged / space / "percontext" / f"{arm}_L{LAYER}_ridge.npz"
            shutil.copyfile(src, b_out / "percontext" / src.name)
        env = "floors_env_mean" if PC.POOL_OF_SPACE[space] == "mean" else "floors_env_labels_only"
        pooling = PC.POOL_OF_SPACE[space]
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--phase",
                "taxonomy",
                "--layers",
                str(LAYER),
                "--arms",
                ",".join(ARMS),
                "--out-eval",
                str(b_out),
                "--parent-eval",
                str(banked_env / env),
                "--pred16-dir",
                str(staged / space / "pred16"),
                "--y-holdout-dir",
                str(staged / PC.Y_HOLDOUT_SPACE[pooling] / "y_holdout"),
                "--manifest-dir",
                str(manifest_dir),
                "--split-file",
                str(split_p),
                "--scratch",
                str(base / "scratch"),
                "--no-upload",
            ],
            check=True,
            env={**os.environ},
        )

    # dense comparator fixture (the parent _smoke_dense_root trick: reuse a battery
    # output as the dense side — same family by construction)
    dense = base / "dense"
    (dense / "bare_query").mkdir(parents=True, exist_ok=True)
    (dense / "judge_labels").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(banked_eval / "dense_feat_space" / "taxonomy.json", dense / "taxonomy.json")
    shutil.copyfile(
        banked_eval / "dense_feat_space" / "taxonomy.json",
        dense / "bare_query" / "taxonomy.json",
    )
    shutil.copyfile(base / "judge_labels" / "labels.json", dense / "judge_labels" / "labels.json")

    cfg = Cfg(
        staging_root=base / "staging",
        out_eval=base / "eval",
        fig_dir=base / "figures",
        dense_eval=dense,
        banked_eval=banked_eval,
        data_revision="fixture",
        banked_revision="fixture",
        model_revision="fixture",
        upload_prefix=UPLOAD_PREFIX_DEFAULT + "_fixture",
        fixture=True,
        no_upload=True,
        device="cpu",
        capture_layer=SA.SMOKE_LAYER,
        batch=4,
        n_probe=8,  # < FIXTURE_N so the probe-slice indexing is exercised
        kresample_pt=kdir / "kresample_shard00.pt",
        kresample_json=kdir / "kresample_shard00.json",
        perfeature_npz=pf_dir / "perfeature_summary.npz",
        manifest_dir=manifest_dir,
        split_file=split_p,
        banked_tensors_root=staged,
        approx_floors_npz=banked_env / "floors_env_mean" / "kresample" / f"floors_L{LAYER}.npz",
        labels_src=base / "judge_labels" / "labels.json",
        fixture_model_dir=model_dir,
    )
    (base / "staging").mkdir(parents=True, exist_ok=True)
    return cfg


def _expect_exit(fn, rc: int, what: str) -> None:
    try:
        fn()
        raise AssertionError(f"degenerate probe '{what}' did not halt")
    except SystemExit as e:
        assert e.code == rc, f"probe '{what}': rc {e.code} != {rc}"
        print(f"[fixture] degenerate probe OK: {what} -> rc {rc}", flush=True)


def run_fixture_smoke(base: Path) -> int:
    """Tiny-real CPU e2e through EVERY phase (stage w/ autospec'd Hub boundary ->
    capture -> floors -> battery -> compare -> harvest) + degenerate gate probes
    exercising each designed halt OUTSIDE the main leg."""
    from dataclasses import replace
    from unittest.mock import create_autospec

    t0 = time.time()
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    cfg = _build_fixture(base)
    state: dict = {}
    fake_prefix = create_autospec(hub.stage_hub_prefix, return_value=[])
    fake_file = create_autospec(hub.stage_hub_file, return_value=Path("/dev/null"))
    fake_sae = create_autospec(SAEMOD.BatchTopKSAE.ensure_downloaded, return_value=None)
    phase_stage(
        cfg, state, stage_prefix_fn=fake_prefix, stage_file_fn=fake_file, ensure_sae_fn=fake_sae
    )
    assert fake_file.call_count == 3 and fake_prefix.call_count == 14, (
        fake_file.call_count,
        fake_prefix.call_count,
    )
    for call in fake_file.call_args_list:
        assert call.kwargs["revision"] == cfg.data_revision, call
    phase_capture(cfg, state)
    assert state["E"]["mean"].shape == (FIXTURE_N, 4, FIXTURE_F_OUT)
    assert all(np.isfinite(state["E"][p]).all() for p in POOLINGS)
    phase_floors(cfg, state)
    for pooling in POOLINGS:
        with np.load(cfg.env_root(pooling) / "kresample" / f"floors_L{LAYER}.npz") as z:
            assert (z["floor"] >= 0).all() and len(z["ci"]) == FIXTURE_N
    phase_battery(cfg, state)
    phase_compare(cfg, state)
    comp = state["compare"]
    assert comp["decision"] in ("Collapse-preserved", "Collapse-retracted", "Mixed"), comp
    phase_harvest(cfg, state)
    assert (cfg.fig_dir / "floor_adjustment_exact.png").is_file()
    _write_results_sentinel(cfg, state)
    assert (cfg.out_eval / "pod_sentinel_preview.json").is_file()
    print(f"[fixture] main leg PASS (decision={comp['decision']})", flush=True)

    # ── degenerate gate probes (designed handling; each OUTSIDE the main leg) ──
    # 1. structural: a row missing one seed key
    doc = json.loads(Path(cfg.kresample_json).read_text())
    del doc["rows"][3]["responses"][str(SEEDS_EXPECTED[0])]
    bad_json = base / "probe_structural.json"
    bad_json.write_text(json.dumps(doc))
    cfg1 = replace(cfg, out_eval=base / "probe1", kresample_json=bad_json, force=True)
    _expect_exit(lambda: phase_capture(cfg1, {}), RC_GATE_STRUCTURAL, "structural (gate 2)")
    # 2. approx parity: banked floors scaled x2
    with np.load(Path(cfg.approx_floors_npz)) as z:
        bad_fl = base / "probe_floors.npz"
        np.savez(bad_fl, ci=z["ci"], floor=z["floor"] * 2.0, den=z["den"], share=z["share"] * 2.0)
    cfg2 = replace(cfg, out_eval=base / "probe2", approx_floors_npz=bad_fl, force=True)
    _expect_exit(lambda: phase_capture(cfg2, {}), RC_GATE_PARITY, "approx parity (gate 1)")
    # 3. fresh-capture parity: impossible cosine tolerance trips the gate branch
    cfg3 = replace(cfg, out_eval=base / "probe3", cap_cos_min=1.01, force=True)
    _expect_exit(lambda: phase_capture(cfg3, {}), RC_GATE_CAPTURE, "fresh-capture parity (1b)")
    # 4. wall: cap below any measurable wall
    cfg4 = replace(cfg, out_eval=base / "probe4", wall_cap_h=1e-12, force=True)
    _expect_exit(lambda: phase_capture(cfg4, {}), RC_GATE_WALL, "wall (gate 3)")
    # 5. identity: a perturbed banked taxonomy delta
    bank5 = base / "probe5_banked"
    shutil.copytree(cfg.banked_eval, bank5)
    tax5_p = bank5 / SPACES[0] / "taxonomy.json"
    tax5 = json.loads(tax5_p.read_text())
    tax5["arms"][f"prefix_L{LAYER}_ridge"]["contrasts"][0]["delta_mean_nerr"] += 1.0
    tax5_p.write_text(json.dumps(tax5))
    cfg5 = replace(cfg, banked_eval=bank5, force=True)
    _expect_exit(lambda: phase_battery(cfg5, {}), RC_GATE_IDENTITY, "battery identity (gate 4)")
    # 6. coverage: banked sae floor_adjusted loses one contrast
    bank6 = base / "probe6_banked"
    shutil.copytree(cfg.banked_eval, bank6)
    tax6_p = bank6 / "sae_space" / "taxonomy.json"
    tax6 = json.loads(tax6_p.read_text())
    tax6["arms"][f"bare_L{LAYER}_ridge"]["floor_adjusted"]["contrasts"].pop()
    tax6_p.write_text(json.dumps(tax6))
    eval6 = base / "probe6_eval"
    shutil.copytree(cfg.out_eval, eval6)
    cfg6 = replace(cfg, out_eval=eval6, banked_eval=bank6, force=True)
    _expect_exit(lambda: phase_compare(cfg6, {}), RC_GATE_COVERAGE, "coverage (gate 5)")
    # 7. inlier tolerance checker (the >1% branch; genuinely unconstructable from real
    # spans — token_inlier_mask keeps the median row by construction)
    cfg7 = replace(cfg, out_eval=base / "probe7")
    cfg7.gates_dir().mkdir(parents=True, exist_ok=True)
    _expect_exit(
        lambda: _check_inlier_drops(cfg7, [{"ci": 0, "seed": 43}] * 2, 100),
        RC_GATE_STRUCTURAL,
        "inlier >1% (gate 2 arm)",
    )
    _check_inlier_drops(cfg7, [], 100)  # pass branch
    print(f"[fixture] PASS in {time.time() - t0:.0f}s — artifacts under {base}", flush=True)
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve every deferred/function-body import on the real branch (the preferred
    Axis-1 smoke-architecture shape — names the deferred symbols explicitly)."""
    import matplotlib  # noqa: F401
    from dataclasses import replace  # noqa: F401
    from unittest.mock import create_autospec  # noqa: F401

    from transformers import (  # noqa: F401
        AutoModelForCausalLM,
        AutoTokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    import matplotlib.pyplot  # noqa: F401

    print("[import-check] OK — all deferred imports resolve", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #1946 exact encode-then-pool SAE K-resample floors (plan v7)."
    )
    ap.add_argument(
        "--phase",
        choices=["stage", "capture", "floors", "battery", "compare", "harvest", "all"],
        default="all",
    )
    ap.add_argument(
        "--staging-root",
        default=str(PROJECT_ROOT / "data" / "issue_1946" / "exact_floors_stage"),
    )
    ap.add_argument(
        "--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_1946" / "exact_floors")
    )
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_1946"))
    ap.add_argument("--dense-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_1738"))
    ap.add_argument("--banked-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_1946"))
    # UPLOAD_PREFIX_EXEMPT: single-issue driver — the default IS #1946's own contract bucket; the resume skip-if-complete predicate and the recorded plan-§10 replay command both key on it.
    ap.add_argument("--upload-prefix", default=UPLOAD_PREFIX_DEFAULT)
    ap.add_argument("--data-revision", default=DATA_REVISION_DEFAULT)
    ap.add_argument("--banked-revision", default=BANKED_REVISION_DEFAULT)
    ap.add_argument("--model-revision", default=MODEL_REVISION_DEFAULT)
    ap.add_argument("--batch", type=int, default=8, help="capture batch (auto-halved on OOM)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--force", action="store_true", help="disable ALL phase resume-skips")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="real-data 32-ci slice w/ all pre-production gates (scratch _smoke out-root)",
    )
    ap.add_argument(
        "--cpu-gates-only",
        action="store_true",
        help="with --smoke: gates 1+2 only (no model load) — VM-runnable",
    )
    ap.add_argument(
        "--fixture-smoke", action="store_true", help="tiny-real CPU e2e (PASS_UNIFIED fixture)"
    )
    ap.add_argument(
        "--smoke-dir", default=str(PROJECT_ROOT / "data" / "issue_1946" / "exact_floors_smoke")
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        _import_check()
        sys.exit(0)
    if args.fixture_smoke:
        rc = run_fixture_smoke(Path(args.smoke_dir))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)
    assert not (args.cpu_gates_only and not args.smoke), "--cpu-gates-only requires --smoke"
    out_eval = Path(args.out_eval)
    if args.smoke:
        out_eval = out_eval / "_smoke"  # scratch — smoke never touches canonical paths
        print(f"[smoke] out-root redirected to scratch: {out_eval} (no uploads)", flush=True)
    cfg = _resolve_production_inputs(
        Cfg(
            staging_root=Path(args.staging_root),
            out_eval=out_eval,
            fig_dir=Path(args.fig_dir),
            dense_eval=Path(args.dense_eval),
            banked_eval=Path(args.banked_eval),
            data_revision=args.data_revision,
            banked_revision=args.banked_revision,
            model_revision=args.model_revision,
            upload_prefix=args.upload_prefix,
            smoke=args.smoke,
            cpu_gates_only=args.cpu_gates_only,
            no_upload=args.no_upload or args.smoke,
            force=args.force,
            batch=args.batch,
            device=args.device,
        )
    )
    phases = {
        "stage": phase_stage,
        "capture": phase_capture,
        "floors": phase_floors,
        "battery": phase_battery,
        "compare": phase_compare,
        "harvest": phase_harvest,
    }
    if args.smoke:
        order = ["stage", "capture"] if args.cpu_gates_only else ["stage", "capture", "floors"]
    elif args.phase == "all":
        order = ["stage", "capture", "floors", "battery", "compare"]
    else:
        order = [args.phase]
    state: dict = {}
    for name in order:
        phases[name](cfg, state)
    if args.smoke or args.phase == "all":
        _write_results_sentinel(cfg, state)
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit — heavy C-ext modules (PyGILState atexit race, #1689)


if __name__ == "__main__":
    main()
