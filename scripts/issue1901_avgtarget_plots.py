#!/usr/bin/env python3
"""Issue #1901 inline round ``avgtarget-plots`` — Plots 1+2 on draw-averaged targets.

User order (2026-08-24): "plot 1 and plot 2 should be averaged". Re-scores every
MAP-arm retrieval acc@1 point of the paper-plan Plot 1 (4-arm layer profile,
``plot1_remake`` recipe, whitened cosine + CSLS) and Plot 2
(``c1_scaling_train_pool``, euclidean) against DRAW-AVERAGED answer targets — the
#2202 ``avg`` convention: target_i = mean(original answer state_i + K=4 fresh
on-policy answer draws_i); pool size stays 1,000 with EVERY entry replaced (full
coverage, unlike #2202's partial 1,988/9,941); pool-side CSLS statistics are
recomputed on the modified pool; mid-rank ties, tie at top counts as failure.
R^2 panels stay single-draw (the paper's existing claim chain); averaged-target
R^2 companions land in the eval JSONs only. The boundary-token arm's target is
deterministic (given WikiText continuation text, no sampling distribution), so
the averaged convention degenerates to the single target there; its series is
carried unchanged from the banked JSONs.

One draw set serves both plots: the ladder manifest ``test_1000`` IS the pinned
#779 round-1 test pool (frozen prompt digest ``LMAN.TEST_1000_PROMPT_SHA``,
asserted here), so the four seeds' draw TEXTS are shared — seeds 43/44 reuse the
banked #1491 ceiling-draw completions, seeds 45/46 are generated fresh with the
same #779 pass-B recipe (vLLM, temp 1.0, top_p 0.95, max_tokens 1024, per-request
seed). All four seeds' answer activations are captured HERE at all 28 layers with
the ladder's batch-1 teacher-forced convention (``LGC._capture_perrow`` —
cx_last/v_x span parity with ``COL.capture_answer_vector`` by construction;
batch-1 by capture-parity design, #1005). Each pool's average mixes its OWN
original capture (pass_b bundle for Plot 1 / big-n points; scale7_refit store
for the ladder + MLP-scaling rungs) with this shared draw capture — the same
mixed-geometry convention #2202 established. My seed-43/44 captures are
reconciled against the banked ceiling captures at the 3 stored layers
(informational parity, reported).

Reuse map (no new estimator code): plot1 fits + whitened scoring =
``issue1901_plot1_remake`` (score_arm / train_whitening_stats / assembly);
ladder staging + cells = ``issue1901_paper_densify_fits``; big-n assembly +
fits = ``issue779_ffc_n1m_fits`` (incl. the banked 963,444-row weight payloads
via ``apply_map`` — no refit at 963k); MLP scaling = the
``issue1901_paper_densify_mlp`` job-B recipe; generation + capture =
``issue1491_ladder_generate_capture``. Refits reproduce the banked single-target
cells first (parity-gated) and only then score the averaged targets, so the
averaged numbers ride prediction sets reconciled with the plotted ones.

Refusal-safety: never prints conversation/rollout text — only counts, indices,
digests. Raw draw text + activation stores upload to HF
``issue1901_avgtarget/`` BEFORE any reduce; eval JSONs relay to
``issue1901_avgtarget/eval`` incrementally.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in this process tree.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps bind BEFORE numpy/torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50G  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue1491_ladder_generate_capture as LGC  # noqa: E402
import issue1491_ladder_manifest as LMAN  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402
import issue1901_paper_densify_fits as PDF  # noqa: E402
import issue1901_plot1_remake as P1R  # noqa: E402
import issue931_common as i931c  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1901_avgtarget_plots")

OUT_EVAL_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_1901" / "avgtarget_plots"
FIG_DIR_DEFAULT = PROJECT_ROOT / "figures" / "issue_1901" / "avgtarget_plots"
PAPER_FIG_DIR = PROJECT_ROOT / "figures" / "paper"
HF_PREFIX = "issue1901_avgtarget"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = (43, 44, 45, 46)  # the #2202 K=4 fresh-draw convention
GEN_SEEDS_NEW = (45, 46)  # 43/44 draw TEXT is banked (#1491 ceiling draws)
CEILING_SEEDS = (43, 44)
CEILING_PREFIX = "issue1491_scale_ladder/scale7_refit/ceiling_draws"
N_TEST = 1000
H_DIM = 3584
LAYERS_ALL = tuple(range(28))
LADDER_LAYER = 19
CHAT_ARMS = ("ridge", "identity_bias", "mlp_w8192")
BANKED_CHAT_JSON = PROJECT_ROOT / "eval_results/issue_1901/plot1_remake/chat_arms_n50k.json"
BANKED_BOUNDARY_JSON = PROJECT_ROOT / "eval_results/issue_1901/plot1_remake/boundary_arm_n50k.json"
BANKED_LADDER_JSON = PROJECT_ROOT / "eval_results/issue_1901/paper_densify/scaling_ladder_L19.json"
BANKED_MLP_SCALING = PROJECT_ROOT / "eval_results/issue_1901/paper_densify/mlp_scaling_L19.json"
BANKED_N50K_L19 = PROJECT_ROOT / "eval_results/issue_1901/paper_densify/layer_curve_n50k.json"
BANKED_BIGN_DIR = PROJECT_ROOT / "eval_results/issue_1901/paper_densify/bign"
BANKED_BATTERY = PROJECT_ROOT / "eval_results/issue_1901/metric_battery/context_arm.json"
BANKED_F7 = PROJECT_ROOT / "eval_results/issue_1491/scale_ladder/fits_scale7_refit.json"
N1M_WEIGHTS_PREFIX = "issue779_monitoring/n1m_readout/weights"

AVG_CONVENTION = {
    "draw_averaged_target": (
        "target_i = mean(original answer state_i + K=4 fresh on-policy draws_i); pool stays "
        "1,000 with EVERY entry replaced (full coverage); queries = the map's test predictions"
    ),
    "k_draws": len(SEEDS),
    "draw_seeds": list(SEEDS),
    "draw_text_source": {
        "43": "banked #1491 ceiling_draws/seed43 raw_completions",
        "44": "banked #1491 ceiling_draws/seed44 raw_completions",
        "45": "generated this round (#779 pass-B recipe)",
        "46": "generated this round (#779 pass-B recipe)",
    },
    "draw_capture": (
        "batch-1 teacher-forced (LGC._capture_perrow; cx_last/v_x span == "
        "COL.capture_answer_vector by construction), all 28 layers, bf16 model, fp32 reduce; "
        "ONE shared capture serves both pools (mixed-geometry average, the #2202 convention)"
    ),
    "generation_recipe": "vLLM temp 1.0 top_p 0.95 max_tokens 1024, per-request seed",
    "engine_seed_note": (
        "one vLLM engine per generated seed, engine seed == sampling seed (the ladder "
        "ceiling-draw convention)"
    ),
    "rank": "mid-rank with 1e-9 relative tie tolerance; tie at top counts as failure",
    "missing_draws": "a dropped (empty-response) draw row averages over the remaining draws",
}


# ── small shared helpers ─────────────────────────────────────────────────────────


def _meta(phase: str) -> dict:
    md = as_metadata_dict(git_provenance(argv0=__file__))
    md.update(
        {
            "script": "issue1901_avgtarget_plots",
            "issue": 1901,
            "round": "avgtarget-plots",
            "phase": phase,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return md


def _write_json_atomic(path: Path, obj: dict) -> None:
    with atomic_replace(path, logger=logger) as tmp:
        tmp.write_text(json.dumps(obj, indent=1, default=str))


def _upload_eval(args, *, force: bool = False) -> None:
    if args.skip_upload:
        return
    _upload_eval.counter = getattr(_upload_eval, "counter", 0) + 1
    if force or _upload_eval.counter % args.upload_every == 0:
        url = hub._upload(
            args.out_eval,
            i931c.HF_DATA_REPO,
            "dataset",
            path_in_repo=f"{HF_PREFIX}/eval",
            raise_on_error=True,
        )
        logger.info("[upload] eval dir -> %s", url)


def _upload_file(local: Path, repo_path: str) -> None:
    from huggingface_hub import HfApi

    hub.retry_transient(
        lambda: HfApi().upload_file(
            path_or_fileobj=str(local),
            path_in_repo=repo_path,
            repo_id=i931c.HF_DATA_REPO,
            repo_type="dataset",
        ),
        what=f"upload {repo_path}",
    )
    logger.info("[upload] %s -> %s", local.name, repo_path)


def _pool_prompts(cache_dir: Path) -> tuple[list[str], list[int]]:
    """The pinned test-1000 prompts in manifest order + their ladder_local_ids.

    The order-sensitive prompt digest is asserted against the frozen
    ``LMAN.TEST_1000_PROMPT_SHA``, which pins BOTH membership and order to the
    #779 round-1 ``fixed_split`` test list — the same order the plot1 machinery's
    ``Y[test]`` rows follow, so positional alignment to every consumer is proven
    by this single assert."""
    rows = LGC._download_ladder_split("test_1000", cache_dir)
    prompts = [r["prompt"] for r in rows]
    lids = [int(r["ladder_local_id"]) for r in rows]
    got = N50G._sha_ids_or_prompts(prompts)
    assert got == LMAN.TEST_1000_PROMPT_SHA, (
        f"test_1000 manifest prompt digest drift: {got} != {LMAN.TEST_1000_PROMPT_SHA}"
    )
    assert len(prompts) == N_TEST and len(set(lids)) == N_TEST, (len(prompts), len(set(lids)))
    return prompts, lids


def _gen_json_path(args, seed: int) -> Path:
    return args.out_root / "gen" / f"gen_seed{seed}.json"


def _draw_store_path(args, seed: int) -> Path:
    return args.out_root / "draws" / f"draws_seed{seed}.pt"


def _load_ceiling_rows(seed: int, cache_dir: Path) -> dict[int, dict]:
    """Banked ceiling-draw text rows for one seed, keyed by ladder_local_id."""
    from huggingface_hub import HfApi

    prefix = f"{CEILING_PREFIX}/seed{seed}/raw_completions"
    names = hub.retry_transient(
        lambda: sorted(
            f.path
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                i931c.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if f.path.endswith(".json")
        ),
        what=f"ceiling raw listing ({prefix})",
    )
    assert names, f"no ceiling raw_completions under {prefix}"
    out: dict[int, dict] = {}
    for name in names:
        from huggingface_hub import hf_hub_download

        local = hub.retry_transient(
            lambda n=name: hf_hub_download(
                i931c.HF_DATA_REPO, filename=n, repo_type="dataset", cache_dir=str(cache_dir)
            ),
            what=f"hf_hub_download {name}",
        )
        chunk = json.loads(Path(local).read_text())
        for r in chunk["rows"]:
            out[int(r["ci"])] = r
    return out


def _load_draw_texts(args, seed: int, prompts: list[str], lids: list[int]) -> list[dict]:
    """Draw rows for one seed in manifest order: [{lid, prompt, response, finish_reason}].

    A missing row (empty-response drop upstream) is returned with response=None."""
    if seed in CEILING_SEEDS:
        by_lid = _load_ceiling_rows(seed, args.cache_dir)
        rows = []
        for lid, p in zip(lids, prompts, strict=True):
            r = by_lid.get(lid)
            if r is not None:
                assert r["prompt"] == p, f"ceiling seed{seed} lid={lid}: prompt mismatch"
            rows.append(
                {
                    "lid": lid,
                    "prompt": p,
                    "response": None if r is None else r["response"],
                    "finish_reason": None if r is None else r.get("finish_reason"),
                }
            )
        return rows
    gj = _gen_json_path(args, seed)
    assert gj.exists(), f"gen JSON for seed {seed} absent: {gj} — run --phase gen --gen-seed {seed}"
    d = json.loads(gj.read_text())
    assert d["seed"] == seed and d["prompts_sha"] == LMAN.TEST_1000_PROMPT_SHA, gj
    rows = d["rows"]
    assert [int(r["lid"]) for r in rows] == lids, f"gen seed{seed}: lid order drift"
    return rows


# ── phase gen: seeds 45/46, one engine per process invocation ────────────────────


def phase_gen(args) -> dict:
    seed = int(args.gen_seed)
    assert seed in GEN_SEEDS_NEW, f"--gen-seed must be one of {GEN_SEEDS_NEW} (43/44 are banked)"
    out = _gen_json_path(args, seed)
    if out.exists():
        d = json.loads(out.read_text())
        if d.get("seed") == seed and d.get("prompts_sha") == LMAN.TEST_1000_PROMPT_SHA:
            logger.info("[gen] seed %d resume-skip (%s)", seed, out)
            return d
    prompts, lids = _pool_prompts(args.cache_dir)
    from explore_persona_space.eval.generation import create_vllm_engine

    tok = LGC._load_tokenizer(MODEL_ID)
    llm = create_vllm_engine(MODEL_ID, max_model_len=LGC.MAX_MODEL_LEN, seed=seed)
    responses, finish = LGC._generate_seeded(llm, tok, prompts, seed)
    cap_hits = sum(1 for f in finish if f == "length")
    d = {
        "seed": seed,
        "prompts_sha": LMAN.TEST_1000_PROMPT_SHA,
        "sampling": {
            "temperature": LGC.GEN_TEMP,
            "top_p": LGC.GEN_TOP_P,
            "max_tokens": LGC.GEN_MAX_TOKENS,
        },
        "cap_hit_fraction": cap_hits / len(prompts),
        "n_rows": len(prompts),
        "rows": [
            {"lid": lid, "prompt": p, "response": r, "finish_reason": f}
            for lid, p, r, f in zip(lids, prompts, responses, finish, strict=True)
        ],
        "metadata": _meta("gen"),
    }
    _write_json_atomic(out, d)
    logger.info("[gen] seed %d: %d rows, cap-hit %.3f", seed, len(prompts), d["cap_hit_fraction"])
    if not args.skip_upload:
        _upload_file(out, f"{HF_PREFIX}/raw_completions/gen_seed{seed}.json")
    return d


# ── phase capture: all four seeds, batch-1 teacher-forced, 28 layers ─────────────


def _hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
    )
    model.eval()
    return model, tok


def phase_capture(args) -> dict:
    prompts, lids = _pool_prompts(args.cache_dir)
    layers = list(LAYERS_ALL)
    model, tok = None, None
    summary: dict[str, dict] = {}
    for seed in SEEDS:
        store_path = _draw_store_path(args, seed)
        if store_path.exists():
            b = torch.load(store_path, map_location="cpu", weights_only=False)
            if b.get("seed") == seed and b.get("prompts_sha") == LMAN.TEST_1000_PROMPT_SHA:
                logger.info("[capture] seed %d resume-skip", seed)
                summary[str(seed)] = {"n_present": int(np.asarray(b["present"]).sum())}
                continue
        rows = _load_draw_texts(args, seed, prompts, lids)
        if model is None:
            model, tok = _hf_model()
        V = torch.full((N_TEST, len(layers), H_DIM), float("nan"), dtype=torch.float16)
        present = np.zeros(N_TEST, dtype=bool)
        dropped: list[int] = []
        t_seed = time.time()
        chunk = int(args.capture_chunk)
        for k0 in range(0, N_TEST, chunk):
            block = rows[k0 : k0 + chunk]
            ps = [r["prompt"] for r in block]
            rs = [r["response"] if r["response"] is not None else "" for r in block]
            cis = list(range(k0, k0 + len(block)))
            t0 = time.time()
            cap_rows, drop = LGC._capture_perrow(model, tok, ps, rs, cis, layers, H_DIM)
            for cr in cap_rows:
                i = int(cr["ci"])
                V[i] = cr["v_x"].to(torch.float16)
                present[i] = True
            dropped.extend(int(x) for x in drop)
            if k0 == 0:
                per_row = (time.time() - t0) / max(1, len(block))
                proj = per_row * N_TEST * len(SEEDS)
                logger.info(
                    "[capture] pilot seed %d: %.2fs/row -> projected %.0fs all seeds "
                    "(fence 2x = %.0fs)",
                    seed,
                    per_row,
                    proj,
                    2 * proj,
                )
            logger.info(
                "[capture] seed %d rows %d-%d done (%.0fs elapsed)",
                seed,
                k0,
                k0 + len(block) - 1,
                time.time() - t_seed,
            )
        store = {
            "seed": seed,
            "prompts_sha": LMAN.TEST_1000_PROMPT_SHA,
            "V": V,
            "present": present,
            "ladder_local_id": np.asarray(lids, dtype=np.int64),
            "layers": layers,
            "dropped_rows": dropped,
            "text_source": AVG_CONVENTION["draw_text_source"][str(seed)],
            "metadata": _meta("capture"),
        }
        with atomic_replace(store_path, logger=logger) as tmp:
            torch.save(store, tmp)
        if not args.skip_upload:
            _upload_file(store_path, f"{HF_PREFIX}/draws/draws_seed{seed}.pt")
        summary[str(seed)] = {"n_present": int(present.sum()), "n_dropped": len(dropped)}
        logger.info(
            "[capture] seed %d complete: %d/%d present (%.0fs)",
            seed,
            int(present.sum()),
            N_TEST,
            time.time() - t_seed,
        )
    # Informational parity: my seed-43/44 captures vs the banked ceiling captures
    # at the 3 stored layers (different capture processes, both batch-1).
    parity = {}
    for seed in CEILING_SEEDS:
        b = torch.load(_draw_store_path(args, seed), map_location="cpu", weights_only=False)
        Vm = b["V"].to(torch.float32).numpy()
        lid2row = {int(li): i for i, li in enumerate(np.asarray(b["ladder_local_id"]))}
        from huggingface_hub import HfApi, hf_hub_download

        prefix = f"{CEILING_PREFIX}/seed{seed}/final_token_capture"
        names = hub.retry_transient(
            lambda p=prefix: sorted(
                f.path
                # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
                for f in HfApi().list_repo_tree(
                    i931c.HF_DATA_REPO, path_in_repo=p, repo_type="dataset", recursive=True
                )
                if f.path.endswith(".pt")
            ),
            what=f"ceiling capture listing ({prefix})",
        )
        diffs, coss = [], []
        for name in names:
            local = hub.retry_transient(
                lambda n=name: hf_hub_download(
                    i931c.HF_DATA_REPO,
                    filename=n,
                    repo_type="dataset",
                    cache_dir=str(args.cache_dir),
                ),
                what=f"hf_hub_download {name}",
            )
            ch = F79._mmap_load(Path(local))
            vx = N50._slice_layer(ch, "v_x", LADDER_LAYER)
            for j, ci in enumerate(int(x) for x in ch["ci"]):
                if ci not in lid2row or not b["present"][lid2row[ci]]:
                    continue
                mine = Vm[lid2row[ci], LADDER_LAYER, :]
                ref = vx[j]
                diffs.append(float(np.max(np.abs(mine - ref))))
                coss.append(
                    float(np.dot(mine, ref) / (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-12))
                )
        parity[str(seed)] = {
            "n": len(diffs),
            "max_abs_diff_L19_max": float(np.max(diffs)) if diffs else None,
            "cos_L19_min": float(np.min(coss)) if coss else None,
            "cos_L19_mean": float(np.mean(coss)) if coss else None,
        }
        logger.info("[capture] seed %d banked-vs-mine L19 parity: %s", seed, parity[str(seed)])
    out = {"per_seed": summary, "ceiling_capture_parity_L19": parity, "metadata": _meta("capture")}
    _write_json_atomic(args.out_eval / "draws_summary.json", out)
    _upload_eval(args, force=True)
    return out


def _load_draws(args) -> tuple[np.ndarray, np.ndarray]:
    """(D (4, 1000, 28, H) fp32 with NaN gaps, present (4, 1000) bool), manifest order."""
    Ds, Ps = [], []
    for seed in SEEDS:
        b = torch.load(_draw_store_path(args, seed), map_location="cpu", weights_only=False)
        assert b["prompts_sha"] == LMAN.TEST_1000_PROMPT_SHA and b["layers"] == list(LAYERS_ALL)
        Ds.append(b["V"].to(torch.float32).numpy())
        Ps.append(np.asarray(b["present"], dtype=bool))
    return np.stack(Ds), np.stack(Ps)


def _avg_targets(y_orig: np.ndarray, D_layer: np.ndarray, present: np.ndarray) -> np.ndarray:
    """mean(original + present draws) per row; (n,H) fp32.

    y_orig (n,H); D_layer (4,n,H) with NaN gaps; present (4,n)."""
    n = y_orig.shape[0]
    acc = y_orig.astype(np.float64).copy()
    cnt = np.ones(n, dtype=np.float64)
    for s in range(D_layer.shape[0]):
        rows = present[s]
        acc[rows] += D_layer[s, rows, :].astype(np.float64)
        cnt[rows] += 1.0
    return (acc / cnt[:, None]).astype(np.float32)


# ── phase plot1: 3 chat arms x 28 layers, single-parity + averaged scoring ──────


def phase_plot1(args) -> dict:
    dev = torch.device(args.device)
    unit_dir = args.out_eval / "plot1_units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    banked = json.loads(BANKED_CHAT_JSON.read_text())["per_layer"]

    capture_dir = PD.stage_prefix(N50.HF_N50K_PREFIX, args.stage_root, workers=args.stage_workers)
    pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
    pb = N1G._load_pass_b_bundle(pass_b)
    assert int(pb["cx_last"].shape[0]) == N50.N_PASS_B, pb["cx_last"].shape
    X_all, Y_all, cap_layers, dtype = PD._extract_all_layers(capture_dir, None)
    if X_all.shape[0] != N50.N_N50K_NEW:
        raise RuntimeError(f"expected {N50.N_N50K_NEW} n50k rows, got {X_all.shape[0]}")
    pinned = N50._pinned_original_shas(args.orig_dir)
    train, val, test, diag = N50.build_n50k_split(
        X_all.shape[0], None, pinned, n_train=50_000, seed=42
    )
    ev = np.concatenate([val, test])
    D, present = _load_draws(args)

    def _assemble(layer: int):
        col = cap_layers.index(layer)
        x = np.concatenate(
            [N50._slice_layer(pb, "cx_last", layer), X_all[:, col, :].astype(np.float32)]
        )
        y = np.concatenate(
            [N50._slice_layer(pb, "v_x", layer), Y_all[:, col, :].astype(np.float32)]
        )
        return x, y

    mlp_epochs = F79.MLP_MAX_EPOCHS
    want_layers = [19] + [li for li in cap_layers if li != 19]
    for k, layer in enumerate(want_layers):
        out_path = unit_dir / f"L{layer}.json"
        key = {
            "layer": int(layer),
            "n_train": 50_000,
            "train_sha256": diag["train_sha256"],
            "seeds": list(SEEDS),
            "avg_convention": "full-pool draw-averaged (K=4)",
        }
        if out_path.exists() and json.loads(out_path.read_text()).get("unit_key") == key:
            logger.info("[plot1] unit %d/%d L%d resume-skip", k + 1, len(want_layers), layer)
            continue
        ts = time.time()
        X, Y = _assemble(layer)
        y_te = Y[test]
        y_avg = _avg_targets(y_te, D[:, :, layer, :], present)
        mu, ell = P1R.train_whitening_stats(Y[train], dev)

        pred_ridge, ridge_meta = N1M.fit_ridge(
            X, Y, train, val, test, N50.LAMBDAS_N50K, dev, args.ridge_block
        )
        pred_ib = MB.identity_bias_predict(X[train], Y[train], X[test])
        pred_ev, mlp_meta = N1M.fit_mlp(
            X, Y, train, ev, 8192, 3e-4, mlp_epochs, N1M.MLP_BATCH, args.seed, dev
        )
        pred_mte = pred_ev[len(val) :]

        arms_out: dict[str, dict] = {}
        parity_rows = []
        for arm, pred in (
            ("ridge", pred_ridge),
            ("identity_bias", pred_ib),
            ("mlp_w8192", pred_mte),
        ):
            single = P1R.score_arm(pred, y_te, mu, ell, args.n_boot, args.seed)
            avg = P1R.score_arm(pred, y_avg, mu, ell, args.n_boot, args.seed)
            want = banked.get(str(layer), {}).get("arms", {}).get(arm)
            if want is not None:
                d_r2 = abs(single["whole_map_r2"] - want["whole_map_r2"])
                d_a1 = abs(
                    single["retrieval"]["whiten_csls"]["acc_at_k"][1]
                    - want["retrieval"]["whiten_csls"]["acc_at_k"]["1"]
                )
                hard = arm in ("ridge", "identity_bias")
                if hard and d_r2 > args.parity_tol:
                    raise RuntimeError(
                        f"L{layer} {arm}: single-target refit R^2 off banked by {d_r2:.4g} "
                        f"(tol {args.parity_tol}) — prediction set not reconciled"
                    )
                parity_rows.append({"arm": arm, "d_r2": d_r2, "d_acc1_wcsls": d_a1, "hard": hard})
            if arm == "mlp_w8192":
                single["fit_meta"] = mlp_meta
            if arm == "ridge":
                single["fit_meta"] = ridge_meta
            arms_out[arm] = {"single": single, "avg": avg}
        unit = {
            "unit_key": key,
            "layer": int(layer),
            "arms": arms_out,
            "parity": parity_rows,
            "wall_time_s": round(time.time() - ts, 1),
        }
        _write_json_atomic(out_path, unit)
        logger.info(
            "[plot1] unit %d/%d L%d ridge wcsls@1 single=%.4f avg=%.4f (%.0fs)",
            k + 1,
            len(want_layers),
            layer,
            arms_out["ridge"]["single"]["retrieval"]["whiten_csls"]["acc_at_k"][1],
            arms_out["ridge"]["avg"]["retrieval"]["whiten_csls"]["acc_at_k"][1],
            time.time() - ts,
        )
        _merge_plot1(args, unit_dir, want_layers, diag)
        _upload_eval(args)
    _merge_plot1(args, unit_dir, want_layers, diag)
    _upload_eval(args, force=True)
    return {"out": str(args.out_eval / "plot1_avg.json")}


def _merge_plot1(args, unit_dir: Path, want_layers, diag) -> None:
    merged = {
        "per_layer": {
            str(li): json.loads((unit_dir / f"L{li}.json").read_text())
            for li in want_layers
            if (unit_dir / f"L{li}.json").exists()
        },
        "split": diag,
        "conventions": AVG_CONVENTION,
        "pool": "pinned test_1000; averaged targets replace ALL 1,000 pool entries",
        "metadata": _meta("plot1"),
    }
    _write_json_atomic(args.out_eval / "plot1_avg.json", merged)


# ── phase ladder: 23 ridge+identity cells at L19, avg scoring ───────────────────


def phase_ladder(args) -> dict:
    dev = torch.device(args.device)
    cells_dir = args.out_eval / "ladder_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    banked = {
        (c["n_train"], c["draw"] if c["draw"] == "prefix" else int(c["draw"])): c
        for c in json.loads(BANKED_LADDER_JSON.read_text())["cells"]
    }
    staged_dir = args.stage_root / "ladder_staged"
    cache_dir = args.stage_root / "ladder_cache"
    data: dict[str, dict] = {}
    cis: dict[str, list[int]] = {}
    expected = {"train_25k": 25000, "val_400": 400, "test_1000": 1000}
    for split in expected:
        arrs, ci = PDF._stage_ladder_split(split, PDF.LADDER_LAYERS_STORED, staged_dir, cache_dir)
        assert len(ci) == expected[split], (split, len(ci))
        data[split], cis[split] = arrs, ci

    li = LADDER_LAYER
    X = np.concatenate([data[s][li][0] for s in ("train_25k", "val_400", "test_1000")])
    Y = np.concatenate([data[s][li][1] for s in ("train_25k", "val_400", "test_1000")])
    n_tr = expected["train_25k"]
    val_idx = np.arange(n_tr, n_tr + 400, dtype=np.int64)
    te_idx = np.arange(n_tr + 400, n_tr + 400 + N_TEST, dtype=np.int64)
    Yte = Y[te_idx]

    # Averaged targets for the ladder pool: ladder-store originals + shared draws,
    # joined on ladder_local_id (the store's ci ints ARE ladder_local_ids).
    D, present = _load_draws(args)
    b0 = torch.load(_draw_store_path(args, SEEDS[0]), map_location="cpu", weights_only=False)
    lid_order = [int(x) for x in np.asarray(b0["ladder_local_id"])]
    lid2draw = {lid: i for i, lid in enumerate(lid_order)}
    rows = np.asarray([lid2draw[int(ci)] for ci in cis["test_1000"]], dtype=np.int64)
    y_avg = _avg_targets(Yte, D[:, rows, li, :], present[:, rows])

    cells = PDF._ladder_cells(n_tr)
    results = []
    for k, (n, draw) in enumerate(cells):
        tag = f"L{li}_n{n}_{'prefix' if draw == 'prefix' else f'd{draw}'}"
        cell_path = cells_dir / f"{tag}.json"
        if cell_path.exists():
            prev = json.loads(cell_path.read_text())
            if prev.get("avg_convention") == "full-pool draw-averaged (K=4)":
                results.append(prev)
                logger.info("[ladder] %d/%d %s resume-skip", k + 1, len(cells), tag)
                continue
        t0 = time.time()
        if draw == "prefix":
            id2row = {cid: i for i, cid in enumerate(cis["train_25k"])}
            sel = np.array([id2row[i] for i in range(n)], dtype=np.int64)
        else:
            rng = np.random.default_rng(19010000 + n * 10 + int(draw))
            sel = rng.choice(n_tr, size=n, replace=False).astype(np.int64)
        pred_te, meta = N1M.fit_ridge(
            X, Y, sel, val_idx, te_idx, PDF.LADDER_LAMBDAS, dev, PDF.RIDGE_BLOCK
        )
        pred_ib = MB.identity_bias_predict(X[sel], Y[sel], X[te_idx]).astype(np.float32)
        cell = {
            "layer": li,
            "n_train": int(n),
            "draw": draw if draw == "prefix" else int(draw),
            "avg_convention": "full-pool draw-averaged (K=4)",
            "single": {
                "ridge": {
                    "test_r2": PDF.PR._pooled_r2(pred_te, Yte),
                    "meta": meta,
                    "knn": PDF._knn_both(pred_te, Yte),
                },
                "identity_bias": {
                    "test_r2": PDF.PR._pooled_r2(pred_ib, Yte),
                    "knn": PDF._knn_both(pred_ib, Yte),
                },
            },
            "avg": {
                "ridge": {
                    "test_r2": PDF.PR._pooled_r2(pred_te, y_avg),
                    "knn": PDF._knn_both(pred_te, y_avg),
                },
                "identity_bias": {
                    "test_r2": PDF.PR._pooled_r2(pred_ib, y_avg),
                    "knn": PDF._knn_both(pred_ib, y_avg),
                },
            },
            "wall_time_s": round(time.time() - t0, 1),
        }
        bk = banked.get((int(n), draw if draw == "prefix" else int(draw)))
        if bk is not None:
            d_r2 = abs(cell["single"]["ridge"]["test_r2"] - bk["ridge"]["test_r2"])
            d_a1 = abs(
                cell["single"]["ridge"]["knn"]["euclidean"]["acc_at_k"][1]
                - bk["knn"]["ridge"]["euclidean"]["acc_at_k"]["1"]
            )
            cell["parity"] = {"d_r2": d_r2, "d_acc1": d_a1}
            if d_r2 > args.parity_tol or d_a1 > 0.005:
                raise RuntimeError(
                    f"{tag}: parity vs banked ladder cell off (dr2={d_r2:.4g}, da1={d_a1:.4g})"
                )
        _write_json_atomic(cell_path, cell)
        results.append(cell)
        logger.info(
            "[ladder] %d/%d %s ridge acc1 single=%.3f avg=%.3f (%.0fs)",
            k + 1,
            len(cells),
            tag,
            cell["single"]["ridge"]["knn"]["euclidean"]["acc_at_k"][1],
            cell["avg"]["ridge"]["knn"]["euclidean"]["acc_at_k"][1],
            time.time() - t0,
        )
    merged = {
        "layer": li,
        "cells": results,
        "conventions": AVG_CONVENTION,
        "pool": "scale7_refit test_1000 (own capture); averaged targets replace all entries",
        "metadata": _meta("ladder"),
    }
    _write_json_atomic(args.out_eval / "ladder_avg.json", merged)
    _upload_eval(args, force=True)
    return merged


# ── phase mlp: scaling rungs 5k/10k (job-B recipe) + 25k (#1491 recipe seam) ────


def phase_mlp(args) -> dict:
    dev = torch.device(args.device)
    staged_dir = args.stage_root / "ladder_staged"
    cache_dir = args.stage_root / "ladder_cache"
    expected = {"train_25k": 25000, "val_400": 400, "test_1000": 1000}
    data, cis = {}, {}
    for split in expected:
        arrs, ci = PDF._stage_ladder_split(split, PDF.LADDER_LAYERS_STORED, staged_dir, cache_dir)
        data[split], cis[split] = arrs, ci
    li = LADDER_LAYER
    X = np.concatenate([data[s][li][0] for s in ("train_25k", "val_400", "test_1000")])
    Y = np.concatenate([data[s][li][1] for s in ("train_25k", "val_400", "test_1000")])
    n_tr = expected["train_25k"]
    tr = np.arange(n_tr, dtype=np.int64)
    val_idx = np.arange(n_tr, n_tr + 400, dtype=np.int64)
    te_idx = np.arange(n_tr + 400, n_tr + 400 + N_TEST, dtype=np.int64)
    ev = np.concatenate([val_idx, te_idx])
    Yte = Y[te_idx]
    D, present = _load_draws(args)
    b0 = torch.load(_draw_store_path(args, SEEDS[0]), map_location="cpu", weights_only=False)
    lid2draw = {int(lid): i for i, lid in enumerate(np.asarray(b0["ladder_local_id"]))}
    rows = np.asarray([lid2draw[int(ci)] for ci in cis["test_1000"]], dtype=np.int64)
    y_avg = _avg_targets(Yte, D[:, rows, li, :], present[:, rows])

    banked = json.loads(BANKED_MLP_SCALING.read_text())
    sub_seed = int(banked.get("meta", {}).get("subsample_seed", 42))
    rng = np.random.default_rng(sub_seed)
    perm = rng.permutation(len(tr))
    f7 = json.loads(BANKED_F7.read_text())

    out: dict[str, dict] = {}
    out_path = args.out_eval / "mlp_scaling_avg.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("subsample_seed") == sub_seed:
            out = prev.get("per_n", {})
    for n in (5000, 10000, 25000):
        if str(n) in out:
            logger.info("[mlp] n=%d resume-skip", n)
            continue
        t0 = time.time()
        if n == 25000:
            sub, lr, epochs, seed = tr, 1e-3, 50, 0  # the #1491 banked-recipe seam
        else:
            sub, lr, epochs, seed = np.sort(tr[perm[:n]]), 3e-4, F79.MLP_MAX_EPOCHS, sub_seed
        pred_ev, fit_meta = N1M.fit_mlp(X, Y, sub, ev, 8192, lr, epochs, N1M.MLP_BATCH, seed, dev)
        pred_te = pred_ev[len(val_idx) :]
        single = {
            "test_r2": PDF.PR._pooled_r2(pred_te, Yte),
            "knn": PDF._knn_both(pred_te, Yte),
        }
        avg = {
            "test_r2": PDF.PR._pooled_r2(pred_te, y_avg),
            "knn": PDF._knn_both(pred_te, y_avg),
        }
        if n == 25000:
            want = f7["knn_retrieval"]["mlp_w8192"]["euclidean"]["acc_at_k"]["1"]
        else:
            want = banked["per_n"][str(n)]["knn"]["euclidean"]["acc_at_k"]["1"]
        parity = {
            "d_acc1_vs_banked": abs(single["knn"]["euclidean"]["acc_at_k"][1] - float(want)),
            "note": "informational — MLP refits are seeded but GPU-nondeterministic; the 25k "
            "cell additionally rides the recorded #1491 recipe seam (lr 1e-3 / 50 ep / seed 0)",
        }
        out[str(n)] = {
            "single": single,
            "avg": avg,
            "parity": parity,
            "fit_meta": fit_meta,
            "wall_time_s": round(time.time() - t0, 1),
        }
        _write_json_atomic(
            out_path,
            {
                "per_n": out,
                "subsample_seed": sub_seed,
                "conventions": AVG_CONVENTION,
                "metadata": _meta("mlp"),
            },
        )
        logger.info(
            "[mlp] n=%d acc1 single=%.3f avg=%.3f (banked %.3f, d=%.3f) %.0fs",
            n,
            single["knn"]["euclidean"]["acc_at_k"][1],
            avg["knn"]["euclidean"]["acc_at_k"][1],
            float(want),
            parity["d_acc1_vs_banked"],
            time.time() - t0,
        )
    _upload_eval(args, force=True)
    return out


# ── phase bign: 150k/500k refits + banked-weight 963,444 arms ───────────────────


def phase_bign(args) -> dict:
    dev = torch.device(args.device)
    out_path = args.out_eval / "bign_avg.json"
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    points: dict[str, dict] = prev.get("points", {})

    D, present = _load_draws(args)
    # 963,444-row arms from the BANKED weight payloads first (no capture staging).
    if not {"ridge_963k", "mlp_w8192_963k", "identity_bias_963k"} <= set(points):
        pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
        pb = N1G._load_pass_b_bundle(pass_b)
        pinned = N50._pinned_original_shas(args.orig_dir)
        _r1, val, test = F79.fixed_split(N50.N_PASS_B, N50.N_PASS_B - 1400, 400, 1000, 42)
        assert F79._sha_ids(np.asarray(test)) == pinned["test_sha256"], "pinned test sha drift"
        X_te = N50._slice_layer(pb, "cx_last", LADDER_LAYER)[np.asarray(test)]
        y_te = N50._slice_layer(pb, "v_x", LADDER_LAYER)[np.asarray(test)]
        y_avg = _avg_targets(y_te, D[:, :, LADDER_LAYER, :], present)
        battery = json.loads(BANKED_BATTERY.read_text())["per_layer"][str(LADDER_LAYER)]["arms"]
        from huggingface_hub import hf_hub_download

        for arm, fname in (("ridge", "ridge.pt"), ("mlp_w8192", "mlp_w8192.pt")):
            local = hub.retry_transient(
                lambda f=fname: hf_hub_download(
                    i931c.HF_DATA_REPO,
                    filename=f"{N1M_WEIGHTS_PREFIX}/L{LADDER_LAYER}/{f}",
                    repo_type="dataset",
                    cache_dir=str(args.cache_dir),
                ),
                what=f"n1m weights {fname}",
            )
            payload = torch.load(local, map_location="cpu", weights_only=False)
            assert int(payload["layer"]) == LADDER_LAYER, payload.get("layer")
            pred = N1M.apply_map(payload, X_te, dev).astype(np.float32)
            single = PD.score_cell(pred, y_te, args.n_boot, args.seed)
            avg = PD.score_cell(pred, y_avg, args.n_boot, args.seed)
            want = battery[arm]["retrieval"]["test"]["euclidean"]["acc_at_k"]["1"]
            points[f"{arm}_963k"] = {
                "single": single,
                "avg": avg,
                "parity": {
                    "d_acc1_vs_battery": abs(
                        single["retrieval"]["euclidean"]["acc_at_k"][1] - float(want)
                    )
                },
            }
            if arm == "ridge":
                xmu = np.asarray(payload["xmu"], dtype=np.float64)
                ymu = np.asarray(payload["ymu"], dtype=np.float64)
                pred_ib = (X_te.astype(np.float64) + (ymu - xmu)).astype(np.float32)
                single_ib = PD.score_cell(pred_ib, y_te, args.n_boot, args.seed)
                avg_ib = PD.score_cell(pred_ib, y_avg, args.n_boot, args.seed)
                want_ib = battery["identity_bias"]["retrieval"]["test"]["euclidean"]["acc_at_k"][
                    "1"
                ]
                points["identity_bias_963k"] = {
                    "single": single_ib,
                    "avg": avg_ib,
                    "parity": {
                        "d_acc1_vs_battery": abs(
                            single_ib["retrieval"]["euclidean"]["acc_at_k"][1] - float(want_ib)
                        )
                    },
                }
        _write_json_atomic(
            out_path,
            {"points": points, "conventions": AVG_CONVENTION, "metadata": _meta("bign")},
        )
        _upload_eval(args, force=True)
        logger.info("[bign] 963k banked-weight arms done")

    # 150k / 500k refits (n1m capture staging — the heavy leg, run LAST).
    if not {"lmsys_150k", "lmsys_500k"} <= set(points):
        n1m_capture_prefix = f"{N1G.HF_PREFIX}/final_token_capture"
        capture_dir = PD.stage_prefix(
            n1m_capture_prefix, args.stage_root, workers=args.stage_workers
        )
        ns = argparse.Namespace(
            pass_b=args.stage_root / "pass_b" / "train_context_vectors.pt",
            manifest_from_hf=True,
            manifest_hf_prefix=N1G.HF_PREFIX,
            out_dir=args.stage_root / "bign_work",
            n1m_capture_dir=capture_dir,
            fresh_stream=False,
            hf_prefix=n1m_capture_prefix,
            orig_dir=args.orig_dir,
        )
        ns.out_dir.mkdir(parents=True, exist_ok=True)
        X, Y, prov, r1_train, val, test, split = N1M.assemble(ns, layer=LADDER_LAYER)
        pools = N1M._pool_rows(prov, r1_train, X.shape[0], val, test)
        y_te = Y[test]
        y_avg = _avg_targets(y_te, D[:, :, LADDER_LAYER, :], present)
        for name, n_target in (("lmsys_150k", 150_000), ("lmsys_500k", 500_000)):
            if name in points:
                continue
            banked_cell = json.loads((BANKED_BIGN_DIR / f"{name}.json").read_text())
            # Selection seed 0 == the banked n1m_fits.json + densify-bign convention
            # (paper_densify --seed-b default) — required for prediction-set parity.
            sel, sel_diag = N1M.select_train(pools, name, n_target, "lmsys", 0)
            assert len(sel) == n_target, (name, len(sel))
            t0 = time.time()
            pred_ridge, meta, _payload = N1M.fit_ridge_with_weights(
                X, Y, sel, val, test, N1M.LAMBDAS_N1M, dev, args.ridge_block
            )
            pred_ib = MB.identity_bias_predict(X[sel], Y[sel], X[test])
            entry = {}
            for arm, pred in (("ridge", pred_ridge), ("identity_bias", pred_ib)):
                single = PD.score_cell(pred, y_te, args.n_boot, args.seed)
                avg = PD.score_cell(pred, y_avg, args.n_boot, args.seed)
                want = banked_cell[arm]["retrieval"]["euclidean"]["acc_at_k"]["1"]
                d_a1 = abs(single["retrieval"]["euclidean"]["acc_at_k"][1] - float(want))
                if d_a1 > 0.01:
                    raise RuntimeError(f"{name}/{arm}: acc1 parity off banked by {d_a1:.4g}")
                entry[arm] = {"single": single, "avg": avg, "parity": {"d_acc1": d_a1}}
            entry["selection"] = sel_diag
            entry["ridge"]["single"]["fit_meta"] = meta
            entry["wall_time_s"] = round(time.time() - t0, 1)
            points[name] = entry
            _write_json_atomic(
                out_path,
                {"points": points, "conventions": AVG_CONVENTION, "metadata": _meta("bign")},
            )
            _upload_eval(args, force=True)
            logger.info(
                "[bign] %s ridge acc1 single=%.3f avg=%.3f (%.0fs)",
                name,
                entry["ridge"]["single"]["retrieval"]["euclidean"]["acc_at_k"][1],
                entry["ridge"]["avg"]["retrieval"]["euclidean"]["acc_at_k"][1],
                time.time() - t0,
            )
    return points


# ── phase fig: render both paper figures + copy over the paper stems ────────────


def _fig_plot1(args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    boundary = json.loads(BANKED_BOUNDARY_JSON.read_text())["per_layer"]
    avg = json.loads((args.out_eval / "plot1_avg.json").read_text())["per_layer"]

    labels = dict(P1R.ARM_LABELS)
    set_paper_style()
    colors = dict(zip(labels, paper_palette(len(labels))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for arm, label in labels.items():
        if arm == "identity_bias":
            # Dropped from the rendered figure (user, 2026-08-25): its -2.5 R^2 range
            # crushed the ridge/MLP curves. Colors stay keyed on the full ARM_LABELS
            # so the remaining arms keep their palette across sibling figures.
            continue
        if arm == "boundary_ridge":
            layers = sorted(int(li) for li in boundary)
            r2 = [boundary[str(li)]["arms"][arm]["whole_map_r2"] for li in layers]
            ret = [boundary[str(li)]["arms"][arm]["retrieval"]["whiten_csls"] for li in layers]
        else:
            # R^2 draw-averaged too (user order 2026-08-25); boundary arm stays banked
            # (deterministic WikiText target -- the averaged convention degenerates there).
            layers = sorted(int(li) for li in avg)
            r2 = [avg[str(li)]["arms"][arm]["avg"]["whole_map_r2"] for li in layers]
            ret = [avg[str(li)]["arms"][arm]["avg"]["retrieval"]["whiten_csls"] for li in layers]
        acc = [r["acc_at_k"]["1"] if "1" in r["acc_at_k"] else r["acc_at_k"][1] for r in ret]
        lo = [r["acc1_ci"]["lo"] for r in ret]
        hi = [r["acc1_ci"]["hi"] for r in ret]
        ax1.plot(layers, r2, marker="o", ms=3, color=colors[arm], label=label)
        if arm == "boundary_ridge":
            # R^2 panel only (user, 2026-08-25): its acc@1 curve crowded the map arms.
            continue
        ax2.plot(layers, acc, marker="o", ms=3, color=colors[arm], label=label)
        ax2.fill_between(layers, lo, hi, color=colors[arm], alpha=0.15, lw=0)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Held-out $R^2$, draw-averaged\ntarget (test, n=1,000)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("acc@1, draw-averaged target\n(whitened cosine + CSLS)")
    ax2.axhline(0.001, ls="--", lw=0.8, color="gray", label="Chance (1/1000)")
    ax2.set_ylim(-0.02, 1.0)
    ax1.legend(frameon=False, fontsize=7)
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c1_layer_profile", dir=args.fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def _fig_plot2(args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")

    ladder_avg = json.loads((args.out_eval / "ladder_avg.json").read_text())["cells"]
    plot1_avg = json.loads((args.out_eval / "plot1_avg.json").read_text())["per_layer"]
    mlp_avg = json.loads((args.out_eval / "mlp_scaling_avg.json").read_text())["per_n"]
    bign_avg = json.loads((args.out_eval / "bign_avg.json").read_text())["points"]

    # ---- R^2 panel: draw-averaged targets throughout (user order 2026-08-25) ----
    def dense_avg_r2(arm: str) -> tuple[list[int], list[float], list[float]]:
        by_n: dict[int, list[float]] = {}
        for c in ladder_avg:
            by_n.setdefault(int(c["n_train"]), []).append(float(c["avg"][arm]["test_r2"]))
        ns = sorted(by_n)
        return (
            ns,
            [float(np.mean(by_n[n])) for n in ns],
            [float(np.std(by_n[n])) for n in ns],
        )

    def p1_avg_r2(arm: str) -> tuple[float, float, float]:
        a = plot1_avg["19"]["arms"][arm]["avg"]
        ci = a["bootstrap_ci"]["r2"]
        return float(a["whole_map_r2"]), float(ci["lo"]), float(ci["hi"])

    def avg_r2ci(cell: dict) -> tuple[float, float, float]:
        a = cell["avg"]
        ci = a["bootstrap_ci"]["r2"]
        return float(a["whole_map_r2"]), float(ci["lo"]), float(ci["hi"])

    big_ridge = [
        (50_000, *p1_avg_r2("ridge")),
        (150_000, *avg_r2ci(bign_avg["lmsys_150k"]["ridge"])),
        (500_000, *avg_r2ci(bign_avg["lmsys_500k"]["ridge"])),
        (963_444, *avg_r2ci(bign_avg["ridge_963k"])),
    ]
    big_ib = [
        (50_000, *p1_avg_r2("identity_bias")),
        (150_000, *avg_r2ci(bign_avg["lmsys_150k"]["identity_bias"])),
        (500_000, *avg_r2ci(bign_avg["lmsys_500k"]["identity_bias"])),
        (963_444, *avg_r2ci(bign_avg["identity_bias_963k"])),
    ]
    # ladder_avg / mlp_scaling_avg cells carry no bootstrap CI -> point-only entries.
    neural_r2: list[tuple[int, float, float, float]] = []
    for n in (5_000, 10_000, 25_000):
        r = float(mlp_avg[str(n)]["avg"]["test_r2"])
        neural_r2.append((n, r, r, r))
    neural_r2.append((50_000, *p1_avg_r2("mlp_w8192")))
    neural_r2.append((963_444, *avg_r2ci(bign_avg["mlp_w8192_963k"])))

    fig, (ax_r2, ax_acc) = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.3))
    col_r = paper_color("instruct")
    col_i = paper_color("identity_bias")
    col_n = paper_color("neural_map")

    ns, mean, sd = dense_avg_r2("ridge")
    ax_r2.errorbar(
        ns + [p[0] for p in big_ridge],
        mean + [p[1] for p in big_ridge],
        yerr=[
            np.maximum(0, sd + [p[1] - p[2] for p in big_ridge]),
            np.maximum(0, sd + [p[3] - p[1] for p in big_ridge]),
        ],
        marker="o",
        ls="-",
        color=col_r,
        lw=1.4,
        ms=3,
        capsize=1.5,
        label="linear map (ridge)",
    )
    ns_i, mean_i, sd_i = dense_avg_r2("identity_bias")
    ax_r2.errorbar(
        ns_i + [p[0] for p in big_ib],
        mean_i + [p[1] for p in big_ib],
        yerr=[
            np.maximum(0, sd_i + [p[1] - p[2] for p in big_ib]),
            np.maximum(0, sd_i + [p[3] - p[1] for p in big_ib]),
        ],
        marker="s",
        ls="--",
        color=col_i,
        lw=1.2,
        ms=3,
        capsize=1.5,
        label="copy context vector + trained bias",
    )
    ax_r2.errorbar(
        [p[0] for p in neural_r2],
        [p[1] for p in neural_r2],
        yerr=[
            np.maximum(0, [p[1] - p[2] for p in neural_r2]),
            np.maximum(0, [p[3] - p[1] for p in neural_r2]),
        ],
        marker="D",
        ls=":",
        color=col_n,
        lw=1.2,
        ms=3,
        capsize=1.5,
        label="nonlinear map (MLP)",
    )
    ax_r2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax_r2.set_ylabel("held-out $R^2$,\ndraw-averaged target")
    ax_r2.set_ylim(-1.05, 1.0)

    # ---- retrieval panel: draw-averaged targets ----
    def _a1k(d: dict) -> float:
        ak = d["acc_at_k"]
        return float(ak["1"] if "1" in ak else ak[1])

    def dense_avg(arm: str) -> tuple[list[int], list[float], list[float]]:
        by_n: dict[int, list[float]] = {}
        for c in ladder_avg:
            by_n.setdefault(int(c["n_train"]), []).append(_a1k(c["avg"][arm]["knn"]["euclidean"]))
        ns = sorted(by_n)
        return (
            ns,
            [float(np.mean(by_n[n])) for n in ns],
            [float(np.std(by_n[n])) for n in ns],
        )

    def avg_pt(cell: dict) -> tuple[float, float, float]:
        e = cell["avg"]["retrieval"]["euclidean"]
        a1 = e["acc_at_k"]["1"] if "1" in e["acc_at_k"] else e["acc_at_k"][1]
        return float(a1), float(e["acc1_ci"]["lo"]), float(e["acc1_ci"]["hi"])

    p1_l19 = plot1_avg["19"]["arms"]

    def p1_avg_pt(arm: str) -> tuple[float, float, float]:
        e = p1_l19[arm]["avg"]["retrieval"]["euclidean"]
        a1 = e["acc_at_k"]["1"] if "1" in e["acc_at_k"] else e["acc_at_k"][1]
        return float(a1), float(e["acc1_ci"]["lo"]), float(e["acc1_ci"]["hi"])

    for arm, col, mk, ls_, lw in (
        ("ridge", col_r, "o", "-", 1.4),
        ("identity_bias", col_i, "s", "--", 1.2),
    ):
        ns_a, mean_a, sd_a = dense_avg(arm)
        big = [
            p1_avg_pt(arm if arm == "ridge" else "identity_bias"),
            avg_pt(bign_avg["lmsys_150k"][arm]),
            avg_pt(bign_avg["lmsys_500k"][arm]),
            avg_pt(bign_avg[f"{arm}_963k"]),
        ]
        big_ns = [50_000, 150_000, 500_000, 963_444]
        ax_acc.errorbar(
            ns_a + big_ns,
            mean_a + [b[0] for b in big],
            yerr=[
                np.maximum(0, sd_a + [b[0] - b[1] for b in big]),
                np.maximum(0, sd_a + [b[2] - b[0] for b in big]),
            ],
            marker=mk,
            ls=ls_,
            color=col,
            lw=lw,
            ms=3,
            capsize=1.5,
        )
    m5 = _a1k(mlp_avg["5000"]["avg"]["knn"]["euclidean"])
    m10 = _a1k(mlp_avg["10000"]["avg"]["knn"]["euclidean"])
    m25 = _a1k(mlp_avg["25000"]["avg"]["knn"]["euclidean"])
    m50 = p1_avg_pt("mlp_w8192")
    b963 = avg_pt(bign_avg["mlp_w8192_963k"])
    ax_acc.errorbar(
        [5_000, 10_000, 25_000, 50_000, 963_444],
        [m5, m10, m25, m50[0], b963[0]],
        yerr=[
            [0.0, 0.0, 0.0, max(0, m50[0] - m50[1]), max(0, b963[0] - b963[1])],
            [0.0, 0.0, 0.0, max(0, m50[2] - m50[0]), max(0, b963[2] - b963[0])],
        ],
        marker="D",
        ls=":",
        color=col_n,
        lw=1.2,
        ms=3,
        capsize=1.5,
    )
    ax_acc.axhline(0.001, color="black", lw=0.7, ls=":")
    ax_acc.set_ylabel("acc@1, draw-averaged\ntarget (pool 1,000)")
    ax_acc.set_ylim(0.0, 1.0)
    for ax in (ax_r2, ax_acc):
        ax.set_xscale("log")
        ax.set_xlabel("training contexts")
    handles, lbls = ax_r2.get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c1_scaling_train_pool", dir=args.fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def phase_fig(args) -> dict:
    p1 = _fig_plot1(args)
    p2 = _fig_plot2(args)
    copied = []
    if args.copy_paper_stems:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for stem in ("c1_layer_profile", "c1_scaling_train_pool"):
            for ext in (".pdf", ".png", ".meta.json"):
                src = args.fig_dir / f"{stem}{ext}"
                if src.exists():
                    shutil.copy2(src, PAPER_FIG_DIR / f"{stem}{ext}")
                    copied.append(str(PAPER_FIG_DIR / f"{stem}{ext}"))
    logger.info("[fig] plot1=%s plot2=%s copied=%d", p1, p2, len(copied))
    return {"plot1": p1, "plot2": p2, "copied": copied}


# ── main ────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="#1901 avgtarget-plots (Plots 1+2, averaged targets)")
    ap.add_argument(
        "--phase",
        choices=["gen", "capture", "plot1", "ladder", "mlp", "bign", "fig", "pod-all"],
        required=True,
    )
    ap.add_argument("--gen-seed", type=int, default=None, help="required for --phase gen")
    ap.add_argument("--stage-root", type=Path, default=Path("/workspace/avgtgt_stage"))
    ap.add_argument(
        "--out-root", type=Path, default=None, help="draw stores + gen (default stage-root)"
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="HF download cache (default stage-root/cache)"
    )
    ap.add_argument("--out-eval", type=Path, default=OUT_EVAL_DEFAULT)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR_DEFAULT)
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=F79.BOOT_N)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--parity-tol", type=float, default=1e-2)
    ap.add_argument("--capture-chunk", type=int, default=250)
    ap.add_argument("--upload-every", type=int, default=5)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--copy-paper-stems", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        return 0
    torch.set_num_threads(int(args.n_threads))
    if args.out_root is None:
        args.out_root = args.stage_root
    if args.cache_dir is None:
        args.cache_dir = args.stage_root / "cache"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_eval.mkdir(parents=True, exist_ok=True)
    if args.phase == "gen":
        phase_gen(args)
    elif args.phase == "capture":
        phase_capture(args)
    elif args.phase == "plot1":
        phase_plot1(args)
    elif args.phase == "ladder":
        phase_ladder(args)
    elif args.phase == "mlp":
        phase_mlp(args)
    elif args.phase == "bign":
        phase_bign(args)
    elif args.phase == "fig":
        phase_fig(args)
    elif args.phase == "pod-all":
        phase_capture(args)
        phase_plot1(args)
        # Reap the n50k stage before the ~82 GB n1m stage (MooseFS quota headroom).
        n50k_stage = args.stage_root / Path(N50.HF_N50K_PREFIX).parent
        if n50k_stage.exists():
            PD._reap_stage(n50k_stage)
            logger.info("[pod-all] reaped n50k stage %s", n50k_stage)
        phase_ladder(args)
        phase_mlp(args)
        phase_bign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
