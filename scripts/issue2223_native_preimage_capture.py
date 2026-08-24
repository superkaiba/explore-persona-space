"""Issue #2223 NAP round — native-axis re-extraction + context→answer map + preimage axes.

The native-axis-fidelity-preimage (NAP) follow-up: teacher-force the paper
pipeline's OWN step-1 extraction responses (external/assistant-axis, 276 roles
x 5 prompts x 40 questions on Qwen3-32B), store TWO per-response summaries
(answer-token mean + context-end residual) at the band layers ∪ the mid layer,
fit a per-band-layer closed-form ridge map M_l: context_end → answer_mean, and
derive three axes per band layer:

  answer_axis_reextracted[l]  mean(default) − mean(score-3 role means) on answer_mean
  v_ctx_faithful[l]           the same contrast on context_end
  v_ctx_preimage[l]           pinv(M_l) @ (answer_axis_reextracted[l] − b_l)

Phases (``--phase``):
  fixture  (CPU)   smoke-only: write a tiny benign synthetic responses+scores
                   fixture (NEVER used in production — production consumes the
                   real step-1 output; enumerated as a smoke substitution).
  capture  (GPU)   teacher-forced summary store (fp16, chunked .pt, resumable,
                   4-way shardable via --shard-id/--num-shards).
  map      (GPU)   per-band-layer ridge Gram fit; role-grouped 80/20 fold; GCV
                   λ on the train fold; n_train >= 5*d assert; identity+bias
                   baseline + kNN retrieval; DEPLOYED map = full-pool refit at
                   the selected λ (fit_scope recorded); M_l / b_l saved
                   SEPARATELY under extractions/cA_map/.
  axes     (CPU)   filtered role/default means → the three axes + axis_cos.json
                   (H3 cosine table incl. anchor rows), preimage diagnostics +
                   grouped-fold preimage stability into map_metrics.json, and
                   the REPORTED-only cross-pool τ/α table.
  upload   (CPU)   HF data-repo persistence (store + cA_map + step-1 responses
                   + extraction JSONs) with an exact-set verify.

Conventions carried from the runner (scripts/issue2223_casestudy_replay.py):
per-segment TOKEN-ID concatenation (never re-tokenize joined strings), fp16
summaries reduced immediately (no (B,T,H) accumulation), atomic writes +
fingerprint-gated resume, and the content-hygiene rule — this script never
prints response text; logs carry counts, keys, hashes, and paths only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_native_preimage_capture.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847 shared-VM thread caps must bind BEFORE torch freezes its pool at import.
load_dotenv()

from scripts.issue2223_casestudy_replay import (  # noqa: E402
    ISSUE,
    MODEL_FOR,
    MODELS,
    NEWAXIS_FILES,
    _atomic_write_json,
    _log,
    _sha256_file,
    default_out_root,
    load_model_and_tokenizer,
    model_slug,
    resolved_band,
)

NAP_LABEL = "native_axis_fidelity_preimage"
HF_PREFIX = f"issue{ISSUE}_casestudy/{NAP_LABEL}"

# The Lu target/monitoring layer for the 32b leg (H1's mid-depth read); other
# legs use mid-depth n_layers//2.
NAP_MID_LAYER = {"32b": 32}

# GCV λ grid (fixed, recorded; edge-of-grid selections are flagged in metrics).
DEFAULT_LAMBDAS = "1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6"

FIT_SCOPE = (
    "lambda selected via GCV on the role-grouped 80% train fold; held-out metrics "
    "(pooled R2, identity+bias baseline, kNN retrieval) on the 20% role fold; "
    "DEPLOYED map = full-pool refit at the selected lambda"
)


def default_store_dir(model_key: str, smoke: bool) -> Path:
    """Store location: smoke → /tmp tree; production → gitignored data/ (never
    eval_results/ — the fp16 store is ~10 GB at n=55,200 and HF-persisted)."""
    if smoke:
        return Path("/tmp/issue-2223-casestudy-smoke") / model_slug(model_key) / "nap_store"
    return REPO / "data" / f"issue_{ISSUE}" / "nap_store" / model_slug(model_key)


def _ext_dir(args) -> Path:
    return Path(args.out_root) / model_slug(args.model) / "extractions"


def _mid_layer(model_key: str, n_layers: int) -> int:
    return NAP_MID_LAYER.get(model_key, n_layers // 2)


def _repro(extra: dict) -> dict:
    from scripts import issue2203_common as C

    return C.repro_metadata({"issue": ISSUE, "label": NAP_LABEL, **extra})


# ── responses reader (paper step-1 JSONL) ────────────────────────────────────


def _resp_key(row: dict) -> str:
    """The paper-judge key convention: ``{label}_p{prompt_index}_q{question_index}``."""
    return f"{row['label']}_p{row['prompt_index']}_q{row['question_index']}"


def _iter_response_rows(responses_dir: Path):
    """Yield (role, key, conversation) over every ``<role>.jsonl`` (sorted, text-mode)."""
    files = sorted(responses_dir.glob("*.jsonl"))
    assert files, f"no per-role response JSONL under {responses_dir}"
    for f in files:
        role = f.stem
        # text-mode iteration, NOT .splitlines(): raw U+2028/U+2029/NEL inside
        # ensure_ascii=False JSON strings would shred records (#950 class)
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield role, _resp_key(row), row["conversation"]


def _render_prompt(tok, model_key: str, conversation: list[dict]) -> tuple[list[int], list[int]]:
    """(prompt_ids, response_ids) for one step-1 row — ID-concat convention.

    The prompt is ``conversation[:-1]`` rendered EXACTLY as step-1 rendered it
    (``add_generation_prompt=True``; ``enable_thinking`` per the model registry);
    the response is the final assistant message tokenized separately with
    ``add_special_tokens=False`` — never a re-tokenized joined string.
    """
    assert conversation and conversation[-1]["role"] == "assistant", "row lacks assistant turn"
    thinking = MODELS[model_key]["thinking"]
    kwargs = {} if thinking is None else {"enable_thinking": thinking}
    text = tok.apply_chat_template(
        conversation[:-1], tokenize=False, add_generation_prompt=True, **kwargs
    )
    prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
    resp_ids = tok(conversation[-1]["content"], add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


# ── phase: fixture (smoke-only) ──────────────────────────────────────────────

_FIXTURE_QA = [
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Name a primary color.", "Red is a primary color."),
]
_FIXTURE_ROLES = ("pirate", "chef", "poet")


def phase_fixture(args) -> Path:
    """Smoke-only: benign synthetic step-1 responses + role-adherence scores.

    Production NEVER uses this (the real step-1 output feeds capture); it exists
    so the capture→map→axes chain smokes on CPU — an enumerated smoke
    SUBSTITUTION (smoke-blind-spots rule). Shape mirrors step-1 exactly:
    per-role JSONL rows {system_prompt, prompt_index, question_index, question,
    conversation, label} and per-role score JSONs {key: 0-3}. Role semantics
    mirror the paper corpus: ``default`` is the UNFILTERED default-pool role
    (no scores file — 4_vectors keys the unfiltered branch on ``"default" in
    role``); ``assistant`` is an ORDINARY SCORED role (it has eval_prompt +
    questions upstream and goes through the score==3 filter like any other).
    """
    assert args.smoke, "--phase fixture is smoke-only (production consumes real step-1 output)"
    responses_dir = Path(args.responses_dir)
    scores_dir = Path(args.scores_dir)
    responses_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    roles = {
        "default": "You are a helpful assistant.",
        "assistant": "You are a helpful assistant.",
    }
    roles.update({r: f"You are a {r}. Stay in character." for r in _FIXTURE_ROLES})
    for role, system in roles.items():
        rows = []
        for pi in range(2):
            for qi, (q, a) in enumerate(_FIXTURE_QA):
                rows.append(
                    {
                        "system_prompt": system,
                        "prompt_index": pi,
                        "question_index": qi,
                        "question": q,
                        "conversation": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ],
                        "label": "pos",
                    }
                )
        (responses_dir / f"{role}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if "default" not in role:
            # every scored role (assistant INCLUDED) gets one sub-threshold row
            # so the score==3 filter branch is exercised per role.
            scores = {_resp_key(r): 3 for r in rows}
            scores[_resp_key(rows[0])] = 1
            _atomic_write_json(scores_dir / f"{role}.json", scores)
    _log(
        f"[nap-fixture] wrote {len(roles)} roles x {2 * len(_FIXTURE_QA)} rows under "
        f"{responses_dir} (+scores under {scores_dir})"
    )
    return responses_dir


# ── phase: capture ───────────────────────────────────────────────────────────


def _store_regime(args, band: list[int], mid: int, store_layers: list[int], n_rows: int) -> dict:
    from scripts import issue2203_common as C

    return C.regime_fingerprint(
        round_label=NAP_LABEL,
        phase="capture",
        model=MODEL_FOR[args.model],
        band_layers=band,
        mid_layer=mid,
        store_layers=store_layers,
        summaries="answer_mean=post-block residual mean over response ids; "
        "context_end=last prompt-token residual; per-segment ID concat",
        dtype="fp16",
        max_len=int(args.max_len),
        n_rows_total=n_rows,
        smoke=bool(args.smoke),
    )


def _skip_ledger_path(store_dir: Path, shard_id: int) -> Path:
    return store_dir / f"shard{shard_id:02d}_skipped.json"


def _load_skip_ledger(store_dir: Path, shard_id: int) -> list[dict]:
    """Prior skipped rows ({role, key, reason}) from this shard's durable ledger."""
    p = _skip_ledger_path(store_dir, shard_id)
    return json.loads(p.read_text())["skipped"] if p.exists() else []


def _write_skip_ledger(store_dir: Path, shard_id: int, skipped: list[dict]) -> None:
    """Atomically persist the shard's FULL accumulated skip set (resume-durable).

    SKIPPED rows (empty-response / over-max-len) enter the shard's done
    predicate through this ledger, so a rerun is zero-pending even when the
    trailing rows of a run were all skips (no chunk flush ever carried them).
    """
    reasons: dict[str, int] = {}
    for s in skipped:
        reasons[s["reason"].split(" (")[0]] = reasons.get(s["reason"].split(" (")[0], 0) + 1
    _atomic_write_json(
        _skip_ledger_path(store_dir, shard_id),
        {"skipped": skipped, "n": len(skipped), "reasons": reasons},
    )


def _load_done_keys(store_dir: Path, shard_id: int) -> tuple[set, int, list[dict]]:
    """(done row keys incl. durably-skipped ones, next chunk index, prior skips).

    Done = keys stored in completed chunk sidecars ∪ keys in the shard's skip
    ledger — a skipped (empty / over-length) row is DONE for resume purposes
    (re-rendering it would re-skip it identically; the ledger keeps the reason).
    """
    done: set = set()
    next_chunk = 0
    for sidecar in sorted(store_dir.glob(f"shard{shard_id:02d}_chunk*.keys.json")):
        pt = sidecar.with_name(sidecar.name.replace(".keys.json", ".pt"))
        if not pt.exists():
            continue  # sidecar written after .pt — a lone sidecar cannot happen; a lone .pt is redone
        meta = json.loads(sidecar.read_text())
        done.update((r, k) for r, k in meta["keys"])
        idx = int(sidecar.name.split("_chunk")[1].split(".")[0])
        next_chunk = max(next_chunk, idx + 1)
    prior_skips = _load_skip_ledger(store_dir, shard_id)
    done.update((s["role"], s["key"]) for s in prior_skips)
    return done, next_chunk, prior_skips


def phase_capture(args) -> Path:
    """Teacher-forced fp16 summary store over the step-1 responses (chunked, resumable)."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
    from scripts import issue2203_common as C

    model_key = args.model
    responses_dir = Path(args.responses_dir)
    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(_iter_response_rows(responses_dir), key=lambda t: (t[0], t[1]))
    n_total = len(rows)
    assert n_total > 0
    n_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    assert 0 <= shard_id < n_shards, (shard_id, n_shards)
    my_rows = rows[shard_id::n_shards]

    model, tok = load_model_and_tokenizer(model_key)
    n_layers = int(model.config.num_hidden_layers)
    H = int(model.config.hidden_size)
    band = resolved_band(model_key, n_layers)
    mid = _mid_layer(model_key, n_layers)
    store_layers = sorted(set(band) | {mid})
    L = len(store_layers)

    regime = _store_regime(args, band, mid, store_layers, n_total)
    regime_path = store_dir / "capture_regime.json"
    if regime_path.exists():
        C.check_regime(json.loads(regime_path.read_text()), regime, regime_path)
    else:
        _atomic_write_json(regime_path, regime)

    done, next_chunk, prior_skips = _load_done_keys(store_dir, shard_id)
    pending = [(r, k, c) for r, k, c in my_rows if (r, k) not in done]
    _log(
        f"[nap-capture] shard {shard_id}/{n_shards}: {len(my_rows)} rows, "
        f"{len(done)} done (incl. {len(prior_skips)} durably-skipped), "
        f"{len(pending)} pending (layers={store_layers})"
    )
    if not pending:
        _log("[nap-capture] zero pending rows — headroom preamble skipped; nothing to do")
        return store_dir
    per_row_bytes = 2 * L * H * 2  # two fp16 (L, H) summaries
    assert_out_root_headroom(
        store_dir,
        need_gb=max(0.5, len(pending) * per_row_bytes * 1.3 / 1e9),
        phase="nap-capture",
    )

    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    chunk_size = int(args.chunk_size)
    batch = int(args.batch)
    t0 = time.time()

    buf_keys: list[tuple[str, str]] = []
    buf_ctx: list[torch.Tensor] = []
    buf_ans: list[torch.Tensor] = []
    new_skips: list[dict] = []
    chunk_idx = next_chunk
    n_done = 0

    def _flush():
        nonlocal chunk_idx, buf_keys, buf_ctx, buf_ans
        # skip persistence FIRST and INDEPENDENT of the stored buffer: trailing
        # skips with an empty buffer must still land durably (resume contract).
        if new_skips:
            _write_skip_ledger(store_dir, shard_id, prior_skips + new_skips)
        if not buf_keys:
            return
        pt = store_dir / f"shard{shard_id:02d}_chunk{chunk_idx:05d}.pt"
        payload = {
            "keys": [list(k) for k in buf_keys],
            "layers": store_layers,
            "model": MODEL_FOR[model_key],
            "context_end": torch.stack(buf_ctx),  # (n, L, H) fp16
            "answer_mean": torch.stack(buf_ans),  # (n, L, H) fp16
        }
        assert payload["context_end"].shape == (len(buf_keys), L, H), payload["context_end"].shape
        tmp = pt.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, pt)
        _atomic_write_json(
            pt.with_name(pt.name.replace(".pt", ".keys.json")),
            {"keys": [list(k) for k in buf_keys], "n": len(buf_keys)},
        )
        _log(
            f"[nap-capture] chunk shard{shard_id:02d}_chunk{chunk_idx:05d} "
            f"n={len(buf_keys)} skipped_so_far={len(new_skips)} elapsed={time.time() - t0:.0f}s"
        )
        chunk_idx += 1
        buf_keys, buf_ctx, buf_ans = [], [], []

    for k0 in range(0, len(pending), batch):
        chunk = pending[k0 : k0 + batch]
        metas, ids_list = [], []
        for role, key, conv in chunk:
            prompt_ids, resp_ids = _render_prompt(tok, model_key, conv)
            if not resp_ids:
                new_skips.append({"role": role, "key": key, "reason": "empty-response"})
                continue
            if len(prompt_ids) + len(resp_ids) > int(args.max_len):
                new_skips.append(
                    {
                        "role": role,
                        "key": key,
                        "reason": f"over-max-len ({len(prompt_ids)}+{len(resp_ids)})",
                    }
                )
                continue
            metas.append((role, key, len(prompt_ids), len(resp_ids)))
            ids_list.append(prompt_ids + resp_ids)
        if ids_list:
            max_len = max(len(i) for i in ids_list)
            input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long, device=dev)
            mask = torch.zeros((len(ids_list), max_len), dtype=torch.long, device=dev)
            for r, ids in enumerate(ids_list):
                input_ids[r, : len(ids)] = torch.tensor(ids, device=dev)
                mask[r, : len(ids)] = 1
            with torch.no_grad():
                captured = extract_layer_activations(
                    model, input_ids, store_layers, attention_mask=mask
                )
            for r, (role, key, n_ctx, n_resp) in enumerate(metas):
                ctx_vecs, ans_vecs = [], []
                for li in store_layers:
                    hs = captured[li][r]  # (T, H)
                    ctx_vecs.append(hs[n_ctx - 1].half().cpu())
                    ans_vecs.append(hs[n_ctx : n_ctx + n_resp].mean(dim=0).half().cpu())
                buf_keys.append((role, key))
                buf_ctx.append(torch.stack(ctx_vecs))
                buf_ans.append(torch.stack(ans_vecs))
            del captured
        n_done += len(chunk)
        if len(buf_keys) >= chunk_size:
            _flush()
        if (k0 // batch) % 20 == 0:
            _log(
                f"[nap-capture] unit {n_done}/{len(pending)} shard={shard_id} "
                f"elapsed={time.time() - t0:.0f}s"
            )
    _flush()
    # Reconciliation: every shard row must now be stored OR durably skipped —
    # a rerun over this store is zero-pending by construction.
    done2, _nc, skips2 = _load_done_keys(store_dir, shard_id)
    left = [(r, k) for r, k, _c in my_rows if (r, k) not in done2]
    assert not left, (
        f"reconcile FAILED: {len(left)} shard rows neither stored nor skip-ledgered: {left[:5]}"
    )
    reasons: dict[str, int] = {}
    for s in skips2:
        reasons[s["reason"].split(" (")[0]] = reasons.get(s["reason"].split(" (")[0], 0) + 1
    _log(
        f"[nap-capture] shard {shard_id} DONE: {n_done} rows processed in "
        f"{time.time() - t0:.0f}s; reconcile: {len(my_rows)} shard rows = "
        f"{len(my_rows) - len(skips2)} stored + {len(skips2)} skipped "
        f"(reasons={reasons}), 0 pending"
    )
    return store_dir


# ── store reader (map / axes phases) ─────────────────────────────────────────


def _store_chunks(store_dir: Path) -> list[Path]:
    pts = sorted(store_dir.glob("shard*_chunk*.pt"))
    assert pts, f"no store chunks under {store_dir} — run --phase capture first"
    return pts


def _store_meta(store_dir: Path) -> dict:
    p = store_dir / "capture_regime.json"
    assert p.exists(), f"{p} absent — run --phase capture first"
    return json.loads(p.read_text())


def _load_store(store_dir: Path):
    """Materialize the full store (fp16 CPU): (keys, roles, layers, ctx, ans).

    ~10.4 GB at n=55,200 for the 32b leg — used ONLY by the MAP phase (GPU
    pod). The axes phase never materializes: the role-mean pass streams
    (:func:`_stream_role_sums`) and the stability refits stream sufficient
    statistics (:func:`_fold_maps_for_stability`).
    """
    import torch

    keys: list[tuple[str, str]] = []
    ctx_parts, ans_parts = [], []
    layers = None
    for pt in _store_chunks(store_dir):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        if layers is None:
            layers = list(blob["layers"])
        assert list(blob["layers"]) == layers, (pt, blob["layers"], layers)
        keys.extend((r, k) for r, k in blob["keys"])
        ctx_parts.append(blob["context_end"])
        ans_parts.append(blob["answer_mean"])
    ctx = torch.cat(ctx_parts)
    ans = torch.cat(ans_parts)
    assert ctx.shape == ans.shape and ctx.shape[0] == len(keys), (ctx.shape, len(keys))
    roles = [r for r, _k in keys]
    return keys, roles, layers, ctx, ans


def _role_split(roles: list[str], seed: int, frac_test: float = 0.2) -> tuple[set, set]:
    """Role-GROUPED split: (train_roles, test_roles); deterministic in ``seed``."""
    uniq = sorted(set(roles))
    assert len(uniq) >= 2, f"role-grouped split needs >=2 roles, got {uniq}"
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_test = max(1, round(frac_test * len(uniq)))
    n_test = min(n_test, len(uniq) - 1)
    return set(uniq[n_test:]), set(uniq[:n_test])


def _eigh_robust(G):
    """cuda eigh with CPU-LAPACK fallback (cuSOLVER syevd non-convergence class)."""
    import torch

    try:
        return torch.linalg.eigh(G)
    except torch.linalg.LinAlgError:
        _log(f"[nap-map] cuda eigh non-convergence at n={G.shape[0]} — CPU fallback")
        w, V = torch.linalg.eigh(G.cpu())
        return w.to(G.device), V.to(G.device)


def _pooled_r2(y_true, y_pred) -> float:
    """Pooled multi-output R2 = 1 − Σ‖y−ŷ‖² / Σ‖y−ȳ_test‖² (fp32 tensors)."""
    resid = float(((y_true - y_pred) ** 2).sum())
    tot = float(((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum())
    assert tot > 0, "degenerate test fold (zero variance)"
    return 1.0 - resid / tot


def phase_map(args) -> Path:
    """Per-band-layer ridge Gram fit context_end → answer_mean (GCV λ, grouped fold)."""
    import numpy as np
    import torch

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    model_key = args.model
    smoke = bool(args.smoke)
    store_dir = Path(args.store_dir)
    ext_dir = _ext_dir(args)
    map_dir = ext_dir / "cA_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ext_dir / "map_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.setdefault("map", {})

    meta = _store_meta(store_dir)
    band = [int(x) for x in meta["band_layers"]]
    keys, roles, layers, ctx, ans = _load_store(store_dir)
    n, L, H = ctx.shape
    lidx = {li: i for i, li in enumerate(layers)}
    _log(f"[nap-map] store n={n} layers={layers} band={band} H={H}")

    train_roles, test_roles = _role_split(roles, seed=42)
    tr_mask = torch.tensor([r in train_roles for r in roles])
    te_mask = ~tr_mask
    n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
    assert n_tr > 0 and n_te > 0, (n_tr, n_te)
    # n_train >= 5*d well-posedness assert (plan §4). Smoke slices structurally
    # cannot satisfy it (4 rows vs d=896) — demoted to a logged WARNING under
    # --smoke per the gotchas smoke/production GATE-CALIBRATION rule; production
    # asserts (an enumerated smoke gate-downgrade, smoke-blind-spots rule).
    if n_tr < 5 * H:
        msg = f"n_train={n_tr} < 5*d={5 * H} — under-determined map fit"
        if smoke:
            _log(f"[nap-map] WARNING (smoke-demoted): {msg}")
        else:
            raise AssertionError(msg)

    lambdas = [float(x) for x in str(args.lambdas).split(",") if x.strip()]
    assert len(lambdas) >= 3, lambdas
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _gcv_fit(Xc, Yc, s=None, V=None):
        """Centered ridge via eigh(Gram): returns (W(λ) closure, s, V, C, gcv table)."""
        G = Xc.T @ Xc  # (d, d)
        if s is None:
            s, V = _eigh_robust(G)
            s = s.clamp_min(0.0)
        A = Xc @ V  # (n, d)
        C = A.T @ Yc  # (d, dout)
        n_loc = Xc.shape[0]
        gcv = {}
        for lam in lambdas:
            denom = s + lam
            dof = float((s / denom).sum())
            Yhat = A @ (C / denom.unsqueeze(1))
            rss = float(((Yc - Yhat) ** 2).sum())
            frac = max(1e-9, 1.0 - dof / n_loc)
            gcv[lam] = (rss / n_loc) / (frac * frac)
        return s, V, C, gcv

    for li in band:
        Mp, bp = map_dir / f"M_{li}.pt", map_dir / f"b_{li}.pt"
        prior = metrics["map"].get(str(li))
        # resume predicate keyed on the output-affecting regime keys (λ grid +
        # pool size), not bare file existence — a λ-grid change recomputes.
        if (
            Mp.exists()
            and bp.exists()
            and prior is not None
            and prior.get("lambda_grid") == lambdas
            and prior.get("n_pool") == n
        ):
            _log(f"[nap-map] layer {li} COMPLETE (λ grid + pool match) — skip")
            continue
        t0 = time.time()
        i = lidx[li]
        X = ctx[:, i, :].float()
        Y = ans[:, i, :].float()
        assert torch.isfinite(X).all() and torch.isfinite(Y).all(), li
        Xtr, Ytr = X[tr_mask].to(device), Y[tr_mask].to(device)
        Xte, Yte = X[te_mask].to(device), Y[te_mask].to(device)
        xm, ym = Xtr.mean(dim=0), Ytr.mean(dim=0)
        s, V, C, gcv = _gcv_fit(Xtr - xm, Ytr - ym)
        lam_star = min(gcv, key=gcv.get)
        edge = lam_star in (min(lambdas), max(lambdas))
        W = V @ (C / (s + lam_star).unsqueeze(1))  # (d_in, d_out)
        pred = (Xte - xm) @ W + ym
        r2 = _pooled_r2(Yte, pred)

        Xtr_np, Ytr_np = Xtr.cpu().numpy(), Ytr.cpu().numpy()
        Xte_np, Yte_np = Xte.cpu().numpy(), Yte.cpu().numpy()
        id_pred = identity_bias_predict(Xtr_np, Ytr_np, Xte_np)
        r2_id = _pooled_r2(torch.from_numpy(Yte_np), torch.from_numpy(id_pred))
        pred_np = pred.cpu().numpy().astype(np.float32)
        knn = {
            metric: knn_retrieval(pred_np, Yte_np, metric=metric)
            for metric in ("euclidean", "cosine")
        }

        # DEPLOYED map: full-pool refit at λ* (fit_scope recorded below).
        Xa, Ya = X.to(device), Y.to(device)
        xma, yma = Xa.mean(dim=0), Ya.mean(dim=0)
        Ga = (Xa - xma).T @ (Xa - xma)
        sa, Va = _eigh_robust(Ga)
        sa = sa.clamp_min(0.0)
        Ca = (Xa - xma) @ Va
        Ca = Ca.T @ (Ya - yma)
        Wa = Va @ (Ca / (sa + lam_star).unsqueeze(1))
        M = Wa.T.contiguous().float().cpu()  # (d_out, d_in): ŷ = M @ x + b
        b = (yma - xma @ Wa).float().cpu()
        assert M.shape == (H, H) and b.shape == (H,), (M.shape, b.shape)
        for path, tensor in ((Mp, M), (bp, b)):
            tmp = path.with_suffix(".pt.tmp")
            torch.save(tensor, tmp)
            os.replace(tmp, path)

        metrics["map"][str(li)] = {
            "n_pool": n,
            "n_train": n_tr,
            "n_test": n_te,
            "n_roles_train": len(train_roles),
            "n_roles_test": len(test_roles),
            "d": H,
            "lambda_grid": lambdas,
            "gcv_by_lambda": {str(k): v for k, v in gcv.items()},
            "lambda_selected": lam_star,
            "lambda_edge_of_grid": edge,
            "r2_heldout_pooled": r2,
            "r2_identity_bias_pooled": r2_id,
            "knn_retrieval": knn,
            "n_train_ge_5d": bool(n_tr >= 5 * H),
            "elapsed_s": round(time.time() - t0, 1),
        }
        metrics["fit_scope"] = FIT_SCOPE
        metrics["split"] = {
            "kind": "role-grouped 80/20",
            "seed": 42,
            "n_roles": len(set(roles)),
            "test_roles": sorted(test_roles),
        }
        metrics["store"] = {
            "n_rows": n,
            "layers": layers,
            "band_layers": band,
            "store_dir": str(store_dir),
        }
        metrics["metadata"] = _repro({"phase": "map", "smoke": smoke})
        _atomic_write_json(metrics_path, metrics)
        _log(
            f"[nap-map] layer {li}: λ*={lam_star:g}{' (EDGE)' if edge else ''} "
            f"R2={r2:.4f} idbias={r2_id:.4f} elapsed={time.time() - t0:.0f}s"
        )
    _log(f"[nap-map] wrote {metrics_path} + {map_dir}/M_*.pt,b_*.pt")
    return map_dir


# ── phase: axes ──────────────────────────────────────────────────────────────


def _load_scores(scores_dir: Path, role: str) -> dict | None:
    p = scores_dir / f"{role}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _is_default_role(role: str) -> bool:
    """Paper 4_vectors/5_axis default-pool membership: ``"default" in role``.

    The default pool keeps ALL rows unfiltered (compute_mean_vector); every
    OTHER role — ``assistant`` INCLUDED (it is an ordinary scored role in the
    paper corpus) — goes through the score==3 filter (compute_pos_3_vector).
    """
    return "default" in role


def _stream_role_sums(
    store_dir: Path,
    scores_dir: Path,
    min_count: int,
    smoke: bool,
    min_kept_roles: int = 1,
):
    """One streaming pass → per-role kept sums (answer_mean + context_end, fp32).

    Filter (paper 4_vectors semantics): roles with ``"default"`` in the name
    keep ALL rows (the unfiltered default pool); every other role — assistant
    included — keeps score==3 rows only. Roles with a missing score file are
    SKIPPED (counted, warn-equivalent); roles below min_count are DROPPED from
    the role mean (counted). Smoke demotes min_count AND the kept-roles floor
    to 1 (enumerated smoke gate-downgrades).
    """
    import torch

    eff_min = 1 if smoke else min_count
    eff_floor = 1 if smoke else max(1, int(min_kept_roles))
    sums_ans: dict[str, torch.Tensor] = {}
    sums_ctx: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    missing_scores: set = set()
    score_cache: dict[str, dict | None] = {}
    for pt in _store_chunks(store_dir):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        ans = blob["answer_mean"].float()
        ctx = blob["context_end"].float()
        for r, (role, key) in enumerate(blob["keys"]):
            if _is_default_role(role):
                keep = True
            else:
                if role not in score_cache:
                    score_cache[role] = _load_scores(scores_dir, role)
                scores = score_cache[role]
                if scores is None:
                    missing_scores.add(role)
                    continue
                keep = int(scores.get(key, -1)) == 3
            if not keep:
                continue
            if role in sums_ans:
                sums_ans[role] += ans[r]
                sums_ctx[role] += ctx[r]
            else:
                sums_ans[role] = ans[r].clone()
                sums_ctx[role] = ctx[r].clone()
            counts[role] = counts.get(role, 0) + 1
    default_roles = sorted(r for r in counts if _is_default_role(r))
    assert default_roles, f"no default-pool rows in the store (roles={sorted(counts)[:8]}...)"
    kept_roles = sorted(r for r, c in counts.items() if not _is_default_role(r) and c >= eff_min)
    below = sorted(r for r, c in counts.items() if not _is_default_role(r) and c < eff_min)
    assert kept_roles, f"no role passes min_count={eff_min} (counts={counts})"
    # Kept-roles floor: a catastrophically missing/empty score set must fail
    # LOUD here, not surface as a silently role-starved axis (#2223 r2).
    assert len(kept_roles) >= eff_floor, (
        f"only {len(kept_roles)} scored roles pass min_count={eff_min} "
        f"(< floor {eff_floor}); missing score files for {len(missing_scores)} roles "
        f"— check --scores-dir (roles_missing_scores={sorted(missing_scores)[:8]}...)"
    )
    stats = {
        "min_count": min_count,
        "min_count_effective": eff_min,
        "min_kept_roles": min_kept_roles,
        "min_kept_roles_effective": eff_floor,
        "default_pool_roles": default_roles,
        "default_pool_rows": int(sum(counts[r] for r in default_roles)),
        "n_roles_kept": len(kept_roles),
        "n_roles_below_min_count": len(below),
        "roles_below_min_count": below,
        "n_roles_missing_scores": len(missing_scores),
        "roles_missing_scores": sorted(missing_scores),
        "role_rows_kept": int(sum(counts[r] for r in kept_roles)),
        "smoke": smoke,
    }
    return sums_ans, sums_ctx, counts, kept_roles, stats


def _cos(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _minnorm_pinv_apply(M, delta, rcond: float = 1e-6):
    """Min-norm pinv solve v = M⁺ delta via SVD; returns (v, diagnostics)."""
    import torch

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    cut = rcond * float(S.max())
    inv = torch.where(S > cut, 1.0 / S, torch.zeros_like(S))
    v = Vh.T @ (inv * (U.T @ delta))
    q = torch.quantile(S, torch.tensor([0.25, 0.5, 0.75]))
    diag = {
        "effective_rank": int((S > cut).sum()),
        "rcond": rcond,
        "spectrum": {
            "max": float(S.max()),
            "min": float(S.min()),
            "q25": float(q[0]),
            "q50": float(q[1]),
            "q75": float(q[2]),
        },
    }
    return v, diag


def _fold_maps_for_stability(
    store_dir: Path,
    band: list[int],
    layers: list[int],
    lam_by_layer: dict[int, float],
    n_folds: int,
):
    """Grouped-fold refits at λ* via streaming sufficient statistics.

    Pass 0 accumulates the FULL uncentered Gram / cross-covariance / sums per
    band layer; one pass per fold accumulates the TEST-role stats; the train
    map is the difference with mean-centering corrections — never a per-fold
    reload of the raw pool. Returns {fold: {layer: (M_f, b_f)}}.
    """
    import torch

    lidx = {li: i for i, li in enumerate(layers)}
    roles_all: list[str] = []
    for pt in _store_chunks(store_dir):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        roles_all.extend(r for r, _k in blob["keys"])
    H = None

    def _accum(role_pred):
        nonlocal H
        G = {li: None for li in band}
        XtY = {li: None for li in band}
        sx = {li: None for li in band}
        sy = {li: None for li in band}
        cnt = 0
        for pt in _store_chunks(store_dir):
            blob = torch.load(pt, map_location="cpu", weights_only=False)
            sel = [r for r, (role, _k) in enumerate(blob["keys"]) if role_pred(role)]
            if not sel:
                continue
            cnt += len(sel)
            for li in band:
                i = lidx[li]
                Xb = blob["context_end"][sel, i, :].float()
                Yb = blob["answer_mean"][sel, i, :].float()
                H = Xb.shape[1]
                if G[li] is None:
                    G[li] = Xb.T @ Xb
                    XtY[li] = Xb.T @ Yb
                    sx[li] = Xb.sum(dim=0)
                    sy[li] = Yb.sum(dim=0)
                else:
                    G[li] += Xb.T @ Xb
                    XtY[li] += Xb.T @ Yb
                    sx[li] += Xb.sum(dim=0)
                    sy[li] += Yb.sum(dim=0)
        return G, XtY, sx, sy, cnt

    G_full, XtY_full, sx_full, sy_full, n_full = _accum(lambda _r: True)
    out: dict[int, dict[int, tuple]] = {}
    eye = None
    for f in range(n_folds):
        _tr, te = _role_split(roles_all, seed=1000 + f)
        G_te, XtY_te, sx_te, sy_te, n_te = _accum(lambda r, te=te: r in te)
        n_tr = n_full - n_te
        assert n_tr > 1, (n_full, n_te)
        out[f] = {}
        for li in band:
            G_tr = G_full[li] - (G_te[li] if G_te[li] is not None else 0)
            XtY_tr = XtY_full[li] - (XtY_te[li] if XtY_te[li] is not None else 0)
            xm = (sx_full[li] - (sx_te[li] if sx_te[li] is not None else 0)) / n_tr
            ym = (sy_full[li] - (sy_te[li] if sy_te[li] is not None else 0)) / n_tr
            Gc = G_tr - n_tr * torch.outer(xm, xm)
            XtYc = XtY_tr - n_tr * torch.outer(xm, ym)
            if eye is None:
                eye = torch.eye(H)
            W = torch.linalg.solve(Gc + float(lam_by_layer[li]) * eye, XtYc)
            out[f][li] = (W.T.contiguous(), ym - xm @ W)
        _log(f"[nap-axes] stability fold {f + 1}/{n_folds} refit done (n_te={n_te})")
    return out


def _h1_classification(h1_band: list[float], h1_mid: float) -> dict:
    """Plan §3 H1 verdict: ALL-quantifier band floor (0.90) + mid floor (0.71).

    ``pass``  = EVERY band-layer cos >= 0.90 AND mid cos >= 0.71.
    ``kill-pipeline-fidelity-fail`` = >=1 band cos < 0.90 AND mid < 0.71
    (BOTH floors failed — the plan's Pipeline-fidelity-fail predicate).
    ``mixed-floors-inconclusive-proceed`` = exactly one floor failed — the
    fidelity label is pre-committed Inconclusive and the run PROCEEDS
    (plan §7 P2-boundary mixed adjudication).
    """
    assert h1_band, "empty band cosine list"
    band_all = all(c >= 0.90 for c in h1_band)
    mid_ok = h1_mid >= 0.71
    if band_all and mid_ok:
        cls = "pass"
    elif not band_all and not mid_ok:
        cls = "kill-pipeline-fidelity-fail"
    else:
        cls = "mixed-floors-inconclusive-proceed"
    return {
        "band_min_cos": min(h1_band),
        "band_mean_cos": sum(h1_band) / len(h1_band),
        "band_all_pass": band_all,
        "mid_cos": h1_mid,
        "mid_pass": mid_ok,
        "thresholds_informational": {"band_all": 0.90, "mid": 0.71},
        "classification": cls,
        "verdict_informational": bool(band_all and mid_ok),
    }


def phase_axes(args) -> Path:
    """Filtered axes + preimage + axis_cos.json + preimage diagnostics/stability."""
    import torch

    model_key = args.model
    smoke = bool(args.smoke)
    store_dir = Path(args.store_dir)
    scores_dir = Path(args.scores_dir)
    ext_dir = _ext_dir(args)
    ext_dir.mkdir(parents=True, exist_ok=True)
    map_dir = ext_dir / "cA_map"
    metrics_path = ext_dir / "map_metrics.json"
    assert metrics_path.exists(), f"{metrics_path} absent — run --phase map first"
    metrics = json.loads(metrics_path.read_text())

    meta = _store_meta(store_dir)
    band = [int(x) for x in meta["band_layers"]]
    layers = [int(x) for x in meta["store_layers"]]
    mid = int(meta["mid_layer"])

    sums_ans, sums_ctx, counts, kept_roles, pool_stats = _stream_role_sums(
        store_dir,
        scores_dir,
        int(args.min_count),
        smoke,
        min_kept_roles=int(args.min_kept_roles),
    )
    lidx = {li: i for i, li in enumerate(layers)}
    # paper 5_axis: default side = mean over stacked DEFAULT-POOL per-role mean
    # vectors ("default" in role, unfiltered); role side = mean over stacked
    # score-3 per-role means (assistant is an ordinary member of the role side).
    default_roles = pool_stats["default_pool_roles"]
    a_mean_ans = torch.stack([sums_ans[r] / counts[r] for r in default_roles]).mean(dim=0)
    a_mean_ctx = torch.stack([sums_ctx[r] / counts[r] for r in default_roles]).mean(dim=0)
    role_ans = torch.stack([sums_ans[r] / counts[r] for r in kept_roles]).mean(dim=0)
    role_ctx = torch.stack([sums_ctx[r] / counts[r] for r in kept_roles]).mean(dim=0)

    ans_reex = {li: (a_mean_ans - role_ans)[lidx[li]] for li in layers}
    v_faithful = {li: (a_mean_ctx - role_ctx)[lidx[li]] for li in band}

    # preimage per band layer (min-norm pinv of the deployed map) + diagnostics.
    v_preimage: dict[int, torch.Tensor] = {}
    pre_diag: dict[str, dict] = {}
    for li in band:
        M = torch.load(map_dir / f"M_{li}.pt", map_location="cpu", weights_only=False)
        b = torch.load(map_dir / f"b_{li}.pt", map_location="cpu", weights_only=False)
        delta = ans_reex[li] - b
        v, diag = _minnorm_pinv_apply(M, delta)
        v_preimage[li] = v
        recon = M @ v + b
        diag["reconstruction_cos"] = _cos(recon, ans_reex[li])
        # REGISTERED quantity (plan §4): ‖v_ctx_preimage‖ / ‖answer_axis_reextracted‖.
        diag["amplification_norm_ratio"] = float(v.norm() / ans_reex[li].norm())
        # bias-subtracted companion (‖v‖ / ‖Δ‖, Δ = axis − b) kept under a
        # DISTINCT name — never the registered ratio.
        diag["amplification_norm_ratio_bias_subtracted"] = float(v.norm() / delta.norm())
        pre_diag[str(li)] = diag

    # grouped-fold preimage stability (ridge-stabilized normal-equation solves —
    # an approximation to the min-norm pinv, recorded as such).
    n_folds = int(args.stability_folds)
    if n_folds > 0:
        lam_by_layer = {li: float(metrics["map"][str(li)]["lambda_selected"]) for li in band}
        folds = _fold_maps_for_stability(store_dir, band, layers, lam_by_layer, n_folds)
        for li in band:
            vs = []
            for f in sorted(folds):
                M_f, b_f = folds[f][li]
                delta_f = ans_reex[li] - b_f
                MtM = M_f.T @ M_f
                eps = 1e-8 * float(MtM.diagonal().sum()) / MtM.shape[0]
                v_f = torch.linalg.solve(MtM + eps * torch.eye(MtM.shape[0]), M_f.T @ delta_f)
                vs.append(v_f)
            pair = [_cos(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
            pre_diag[str(li)]["fold_stability_mean_pairwise_cos"] = (
                sum(pair) / len(pair) if pair else None
            )
            pre_diag[str(li)]["fold_stability_n_folds"] = n_folds
            pre_diag[str(li)]["fold_preimage_solver"] = (
                "ridge-stabilized normal equations (eps = 1e-8·tr(MᵀM)/d) — "
                "approximates the min-norm pinv"
            )

    # cross-pool τ/α table (REPORTED only — steering τ/α come from the runner
    # pool via --phase extract_newaxes; plan: never used for steering).
    proj_pool: dict[str, dict[int, list[float]]] = {"ctx_faithful": {}, "ctx_preimage": {}}
    units = {
        "ctx_faithful": {li: v_faithful[li] / v_faithful[li].norm() for li in band},
        "ctx_preimage": {li: v_preimage[li] / v_preimage[li].norm() for li in band},
    }
    for pt in _store_chunks(store_dir):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        for li in band:
            i = lidx[li]
            Xb = blob["context_end"][:, i, :].float()
            for fam, u in units.items():
                proj_pool[fam].setdefault(li, []).extend((Xb @ u[li]).tolist())

    def _pool_table(vals: list[float]) -> dict:
        s = sorted(vals)

        def q(p: float) -> float:
            idx = p * (len(s) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
            return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)

        mean = sum(s) / len(s)
        var = sum((x - mean) ** 2 for x in s) / max(1, len(s) - 1)
        return {
            "p50": q(0.5),
            "p75": q(0.75),
            "p90": q(0.90),
            "p100": s[-1],
            "std": math.sqrt(var),
            "n": len(s),
        }

    cross_pool = {
        fam: {str(li): _pool_table(proj_pool[fam][li]) for li in band} for fam in proj_pool
    }

    # reference axis: 32b → published Lu axis; others → own extraction answer axis.
    if MODELS[model_key]["axis_source"] == "published":
        from scripts.issue2223_casestudy_replay import _stage_lu_artifacts

        axis_path, _cfg = _stage_lu_artifacts(ext_dir)
        ref_t = torch.load(axis_path, map_location="cpu", weights_only=False).float()
        ref = {li: ref_t[li] for li in layers}
        ref_src = "lu-published assistant_axis.pt"
    else:
        p = ext_dir / "answer_axis.pt"
        assert p.exists(), f"{p} absent — run the runner --phase extract first"
        raw = torch.load(p, map_location="cpu", weights_only=False)
        ref = {int(li): v.float() for li, v in raw.items()}
        ref_src = "own extraction answer_axis.pt"

    # H3 row: cos(v_ctx_preimage, CURRENT teacher-forced ctx_native) per band
    # layer — the runner extraction's native_axes.pt (plan §4). Production
    # fails LOUD when absent; smoke may run before the runner extraction and
    # records null rows (enumerated smoke gate-downgrade).
    native_p = ext_dir / "native_axes.pt"
    ctx_native_cur: dict[int, object] | None = None
    if native_p.exists():
        nat = torch.load(native_p, map_location="cpu", weights_only=False)
        assert "ctx_native" in nat, f"native_axes.pt lacks ctx_native (has {sorted(nat)})"
        ctx_native_cur = {int(li): v.float() for li, v in nat["ctx_native"].items()}
        missing_nat = [li for li in band if li not in ctx_native_cur]
        assert not missing_nat, f"native_axes.pt ctx_native lacks band layers {missing_nat}"
        ctx_native_src = str(native_p)
    else:
        if not smoke:
            raise AssertionError(
                f"{native_p} absent — run the runner --phase extract first (the H3 "
                "table requires the CURRENT teacher-forced ctx_native axis)"
            )
        ctx_native_src = f"ABSENT ({native_p}) — smoke-only null rows"
        _log(f"[nap-axes] WARNING (smoke): {native_p} absent — H3 ctx_native rows null")

    h1_band = [_cos(ans_reex[li], ref[li]) for li in band]
    h1_mid = _cos(ans_reex[mid], ref[mid])
    axis_cos = {
        "reference_axis_source": ref_src,
        "ctx_native_source": ctx_native_src,
        "cos_reextracted_vs_reference": {str(li): _cos(ans_reex[li], ref[li]) for li in layers},
        "h1_gate": {**_h1_classification(h1_band, h1_mid), "mid_layer": mid},
        # H3 cosine table per band layer: preimage-vs-faithful + preimage-vs-
        # CURRENT-ctx_native (plan §4) + anchor rows vs answer_reextracted /
        # the reference axis.
        "h3_table": {
            str(li): {
                "cos_faithful_vs_answer_reextracted": _cos(v_faithful[li], ans_reex[li]),
                "cos_preimage_vs_answer_reextracted": _cos(v_preimage[li], ans_reex[li]),
                "cos_faithful_vs_preimage": _cos(v_faithful[li], v_preimage[li]),
                "cos_preimage_vs_ctx_native_current": (
                    _cos(v_preimage[li], ctx_native_cur[li]) if ctx_native_cur else None
                ),
                "cos_faithful_vs_ctx_native_current": (
                    _cos(v_faithful[li], ctx_native_cur[li]) if ctx_native_cur else None
                ),
                "cos_faithful_vs_reference": _cos(v_faithful[li], ref[li]),
                "cos_preimage_vs_reference": _cos(v_preimage[li], ref[li]),
            }
            for li in band
        },
        "cross_pool_tau_alpha": {
            "reported_only": True,
            "note": "paper-pool context_end projections — NEVER used for steering "
            "(steering τ/α come from the runner pool via --phase extract_newaxes)",
            "table": cross_pool,
        },
        "axis_pool": pool_stats,
        "band_layers": band,
        "mid_layer": mid,
        "metadata": _repro({"phase": "axes", "smoke": smoke}),
    }
    _atomic_write_json(ext_dir / "axis_cos.json", axis_cos)

    for name, obj in (
        ("answer_axis_reextracted.pt", {li: ans_reex[li].float() for li in layers}),
        (NEWAXIS_FILES["ctx_faithful"], {li: v_faithful[li].float() for li in band}),
        (NEWAXIS_FILES["ctx_preimage"], {li: v_preimage[li].float() for li in band}),
    ):
        tmp = (ext_dir / name).with_suffix(".pt.tmp")
        torch.save(obj, tmp)
        os.replace(tmp, ext_dir / name)

    metrics["preimage"] = pre_diag
    metrics["axis_pool"] = pool_stats
    metrics["metadata_axes"] = _repro({"phase": "axes", "smoke": smoke})
    _atomic_write_json(metrics_path, metrics)
    _log(
        f"[nap-axes] wrote {ext_dir}/{{axis_cos.json, answer_axis_reextracted.pt, "
        f"{NEWAXIS_FILES['ctx_faithful']}, {NEWAXIS_FILES['ctx_preimage']}}} + preimage "
        f"diagnostics into {metrics_path.name} "
        f"(sha: {_sha256_file(ext_dir / NEWAXIS_FILES['ctx_preimage'])})"
    )
    return ext_dir


# ── phase: upload ────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    """Persist store + cA_map + step-1 responses + extraction JSONs to the HF data repo.

    One bulk folder upload per prefix (never a per-file loop) + an exact-set
    verify per prefix via ``hub.verify_repo_paths_uploaded``.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    ext_dir = _ext_dir(args)
    jobs: list[tuple[Path, str, list[str]]] = []
    store_dir = Path(args.store_dir)
    if store_dir.is_dir():
        jobs.append(
            (
                store_dir,
                f"{HF_PREFIX}/analysis_tensors/nap_store/{model_slug(args.model)}",
                sorted(str(p.relative_to(store_dir)) for p in store_dir.rglob("*") if p.is_file()),
            )
        )
    responses_dir = Path(args.responses_dir)
    if responses_dir.is_dir():
        jobs.append(
            (
                responses_dir,
                f"{HF_PREFIX}/raw_completions/extraction/{model_slug(args.model)}",
                sorted(
                    str(p.relative_to(responses_dir))
                    for p in responses_dir.rglob("*")
                    if p.is_file()
                ),
            )
        )
    map_dir = ext_dir / "cA_map"
    if map_dir.is_dir():
        jobs.append(
            (
                map_dir,
                f"{HF_PREFIX}/analysis_tensors/cA_map/{model_slug(args.model)}",
                sorted(str(p.relative_to(map_dir)) for p in map_dir.rglob("*") if p.is_file()),
            )
        )
    ext_json = [
        p
        for p in (
            ext_dir / "map_metrics.json",
            ext_dir / "axis_cos.json",
            ext_dir / "answer_axis_reextracted.pt",
            ext_dir / NEWAXIS_FILES["ctx_faithful"],
            ext_dir / NEWAXIS_FILES["ctx_preimage"],
        )
        if p.exists()
    ]
    assert jobs or ext_json, "nothing to upload — run capture/map/axes first"
    api = HfApi()
    for local, prefix, rels in jobs:
        assert rels, (local, "empty upload set")
        url = hub._upload(
            local,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            prefix,
            raise_on_error=True,
        )
        missing = hub.verify_repo_paths_uploaded(
            api,
            hub.DEFAULT_DATASET_REPO,
            [f"{prefix}/{r}" for r in rels],
            path_in_repo=prefix,
            repo_type="dataset",
        )
        assert not missing, f"upload verify FAILED for {prefix}: missing {missing[:5]}..."
        _log(f"[nap-upload] {local} → {url} ({len(rels)} files verified)")
    ext_prefix = f"{HF_PREFIX}/extractions/{model_slug(args.model)}"
    if ext_json:
        # ONE bulk upload_folder commit for the extraction artifacts (never a
        # per-file _upload loop — the #664/#1481 commit-storm anti-pattern);
        # the helper runs its own exact-set verify against expected_repo_paths.
        url = hub._upload_folder_filtered(
            ext_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            ext_prefix,
            allow_patterns=[p.name for p in ext_json],
            expected_repo_paths=[f"{ext_prefix}/{p.name}" for p in ext_json],
        )
        _log(
            f"[nap-upload] extraction artifacts → {url} "
            f"({len(ext_json)} files, bulk commit, exact-set verified)"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

PHASES = {
    "fixture": phase_fixture,
    "capture": phase_capture,
    "map": phase_map,
    "axes": phase_axes,
    "upload": phase_upload,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES), required=False)
    ap.add_argument("--model", choices=sorted(MODEL_FOR), default="32b")
    ap.add_argument("--out-root", default=None, help="default: eval_results tree (smoke: /tmp)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--responses-dir",
        default=None,
        help="paper step-1 per-role JSONL dir (default <ext_dir>/paper_pipeline/responses)",
    )
    ap.add_argument(
        "--scores-dir",
        default=None,
        help="role-adherence score JSONs (issue2223_paper_judge.py output; "
        "default <ext_dir>/paper_pipeline/scores)",
    )
    ap.add_argument("--store-dir", default=None, help="summary store dir (default: data/ tree)")
    ap.add_argument("--batch", type=int, default=4, help="teacher-forced forward batch size")
    ap.add_argument("--chunk-size", type=int, default=512, help="rows per store chunk .pt")
    ap.add_argument("--max-len", type=int, default=4096, help="skip rows over this token count")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--lambdas", default=DEFAULT_LAMBDAS, help="GCV λ grid (comma list)")
    ap.add_argument("--min-count", type=int, default=50, help="paper min score-3 rows per role")
    ap.add_argument(
        "--min-kept-roles",
        type=int,
        default=100,
        help="fail-loud floor on scored roles passing min_count (smoke: 1); a "
        "catastrophically missing scores dir must never yield a role-starved axis",
    )
    ap.add_argument(
        "--stability-folds", type=int, default=5, help="grouped-fold preimage stability (0=skip)"
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        # deferred-import resolution (smoke-architecture Axis 1) + args-attr scan.
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.analysis.extraction import (  # noqa: F401
            extract_layer_activations,
        )
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )
        from scripts import issue2203_common as C  # noqa: F401
        from scripts.issue2223_casestudy_replay import _stage_lu_artifacts  # noqa: F401

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        return 0
    assert args.phase, "--phase is required (or --import-check)"
    if args.out_root is None:
        args.out_root = str(default_out_root(bool(args.smoke)))
    if args.store_dir is None:
        args.store_dir = str(default_store_dir(args.model, bool(args.smoke)))
    ext = _ext_dir(args)
    if args.responses_dir is None:
        args.responses_dir = str(ext / "paper_pipeline" / "responses")
    if args.scores_dir is None:
        args.scores_dir = str(ext / "paper_pipeline" / "scores")
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
