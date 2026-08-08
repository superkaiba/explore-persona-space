"""#1774 P3 — kernel causal tests (pod, GPU, HF generate + activation hooks, sharded by direction).

Per plan §4 P3: 60 seeded trait-stratum contexts; 12 ADD directions (top-4 right-singular
context-arm + 4 kernel-tail + 4 norm-matched random, from P2 ``directions.pt``) x {+a, -a}
added to the L14 residual stream during decode (the ``steering_generate`` hook shape,
scripts/issue623_extract_sycophancy_vector.py L481) + 3 LEACE-erase arms (one per r_B trait;
``fit_leace`` on context states, applied via a replace-semantics forward hook mirroring
``artifacts/ablation.py::ablation_hooks``) + a no-intervention ``steer_base`` at K=3
draws/context = 27 intervention conditions x 60 + 180 baseline = 1,800 generations
(temp 0.7, max_tokens 256, batch 16). The H4 state-shift DV is a HOOK-FREE teacher-forced
re-capture of t1 over ALL persisted completions with the P1 rig (``issue1774_draws``); the
live-captured copy exists ONLY as the assumption-11 rig-sanity check in the pilot report.

Fluency pilot: 10 gens/direction at a; degenerate fraction > 0.30 => halve a once, else drop
+ report; < 6 usable directions after halving => state-shift-only read (``judge_skip`` set in
``state_shift.json``), never a crash. Completions upload to HF the moment generation ends
(store-before-long-consumer). CLI (dispatch contract, ``issue1774_dispatch.sh run_p3``):
``--shard i/n [--smoke] [--out-root D]``; conditions shard BY DIRECTION; the last shard to
finish merges ``steering/state_shift.json`` (idempotent, deterministic content).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE torch/numpy imports (pools freeze at import).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1774_common as c  # noqa: E402

GEN_TEMPERATURE = 0.7  # plan §4 P3 (matched across every condition incl. steer_base)
# 256 per plan §4 P3; env override exists ONLY for the tiny-model CPU smoke.
GEN_MAX_TOKENS = int(os.environ.get("I1774_GEN_MAX_TOKENS", "256"))
GEN_BATCH = 16
N_STEER_CONTEXTS = 60
FLUENCY_PILOT_GENS = 10
FLUENCY_DEGENERATE_MAX = 0.30
MIN_USABLE_DIRECTIONS = 6  # kill criterion: below this after halving => judge_skip
RIG_SANITY_ALPHA_MULT = 5.0  # assumption 11: large-a ADD must shift captured t1 along v
# Store convention (issue1774_draws._capture_t1): t1 at "L14" = hidden_states[1:][13] =
# output of decoder block 13, so the intervention hooks block index HEADLINE_LAYER - 1.
HOOK_BLOCK_IDX = c.HEADLINE_LAYER - 1


# ── condition grid ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Condition:
    """One generation condition: an ADD arm, a LEACE-erase arm, or the baseline."""

    cond_id: str
    kind: str  # "add" | "leace" | "base"
    direction: str  # directions.pt key ("" for base)
    sign: int  # +1 / -1 for add, 0 otherwise
    k_draws: int  # draws per context (1 for interventions, STEER_BASE_DRAWS for base)


def split_direction_names(names: list[str]) -> tuple[list[str], list[str]]:
    """(add_directions, leace_directions) from directions.pt keys; rb_* are LEACE arms."""
    add = sorted(n for n in names if not n.startswith("rb_"))
    leace = sorted(n for n in names if n.startswith("rb_"))
    return add, leace


def build_conditions(direction_names: list[str], smoke: bool) -> list[Condition]:
    """Full grid: 12 ADD x {+,-} + 3 LEACE + steer_base = 28 conditions (27 interventions).

    Smoke: 2 ADD x {+,-} + 1 LEACE + base (K=2). Asserts the full-mode counts match plan §4.
    """
    add, leace = split_direction_names(direction_names)
    if smoke:
        add, leace = add[:2], leace[:1]
        k_base = 2
    else:
        assert len(add) == 12, f"expected 12 ADD directions, got {len(add)}: {add}"
        assert len(leace) == 3, f"expected 3 rb_* directions, got {len(leace)}: {leace}"
        k_base = c.STEER_BASE_DRAWS
    conds: list[Condition] = []
    for name in add:
        for sign in (1, -1):
            tag = "pos" if sign > 0 else "neg"
            conds.append(Condition(f"add_{name}_{tag}", "add", name, sign, 1))
    conds.extend(Condition(f"leace_{name}", "leace", name, 0, 1) for name in leace)
    conds.append(Condition("steer_base", "base", "", 0, k_base))
    if not smoke:
        n_interv = sum(1 for x in conds if x.kind != "base")
        assert n_interv == 27, f"expected 27 intervention conditions, got {n_interv}"
    return conds


def shard_conditions(conds: list[Condition], shard: str) -> list[Condition]:
    """Deterministic BY-DIRECTION sharding: every condition of one direction lands on one
    shard (its pilot + both signs stay together); steer_base is pinned to shard 0."""
    i, n = (int(x) for x in shard.split("/"))
    assert 0 <= i < n, shard
    dirs = sorted({x.direction for x in conds if x.kind != "base"})
    owner = {d: j % n for j, d in enumerate(dirs)}
    out = [x for x in conds if x.kind != "base" and owner[x.direction] == i]
    if i == 0:
        out.extend(x for x in conds if x.kind == "base")
    return out


# ── intervention hooks (replace semantics; intervention registers BEFORE capture) ──


def make_add_hook(v_hat: torch.Tensor, alpha_signed: float):
    """Forward hook adding ``alpha_signed * v_hat`` to the block output.

    Hook shape lifted (with provenance, never imported) from
    ``scripts/issue623_extract_sycophancy_vector.py::steering_generate`` L513-518:
    tuple outputs -> modify element 0; bare tensor -> modify directly. Returning the
    modified output REPLACES it for the caller and later-registered hooks, so a capture
    hook registered after this one observes the shifted stream (assumption 11).
    """
    assert v_hat.ndim == 1, v_hat.shape

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            return (hs + alpha_signed * v_hat.to(device=hs.device, dtype=hs.dtype), *out[1:])
        return out + alpha_signed * v_hat.to(device=out.device, dtype=out.dtype)

    return hook


def make_leace_hook(mean_x: torch.Tensor, p_mat: torch.Tensor):
    """Forward hook applying the fitted LEACE eraser ``(h - mean) @ P.T + mean``.

    Mirrors ``artifacts/ablation.py::_make_projection_hook`` (tuple/bare handling, replace
    semantics) but applies the AFFINE eraser from ``fit_leace`` instead of a rank-1
    project-out. Compute in fp32, cast back to the stream dtype.
    """
    assert mean_x.ndim == 1 and p_mat.shape == (mean_x.shape[0], mean_x.shape[0])

    def _apply(hs: torch.Tensor) -> torch.Tensor:
        m = mean_x.to(device=hs.device, dtype=torch.float32)
        p = p_mat.to(device=hs.device, dtype=torch.float32)
        return (((hs.to(torch.float32) - m) @ p.T) + m).to(hs.dtype)

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            return (_apply(out[0]), *out[1:])
        return _apply(out)

    return hook


def _register_hook(model, hook) -> torch.utils.hooks.RemovableHandle:
    blocks = getattr(getattr(model, "model", None), "layers", None)
    assert blocks is not None, "expected a Qwen-style decoder exposing model.model.layers"
    assert 0 <= HOOK_BLOCK_IDX < len(blocks), (HOOK_BLOCK_IDX, len(blocks))
    return blocks[HOOK_BLOCK_IDX].register_forward_hook(hook)


# ── directions + LEACE fits ──────────────────────────────────────────────────


def load_directions(out_root: str | None) -> dict:
    """Load P2 ``directions.pt``; fail loud when P2 has not run."""
    p = c.data_out(out_root) / "directions.pt"
    assert p.exists(), f"missing {p} — run P2 --step directions first"
    payload = torch.load(p, map_location="cpu", weights_only=False)
    dirs = payload["directions"]
    assert dirs, f"{p} carries no directions"
    dims = {tuple(v.shape) for v in dirs.values()}
    assert len(dims) == 1 and all(len(s) == 1 for s in dims), f"inconsistent shapes: {dims}"
    # The binding dim invariant is direction-dim == model hidden size, asserted
    # against the LOADED model in main() (production: 3584; tiny-model smoke: its dim).
    return payload


def fit_leace_erasers(
    trait_dirs: dict[str, torch.Tensor], out_root: str | None, smoke: bool
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit one closed-form LEACE eraser per r_B trait on the P2 fit-row context states.

    Concept value E0 = context state projected onto the (unit-norm) r_B direction; the
    eraser guarantees cov(E0, residual) ~ 0 on the fit sample (issue_763_nonlinear).
    Returns {direction_name: (mean_x (d,), P (d, d))} as float64 numpy.
    """
    from explore_persona_space.analysis.issue_763_nonlinear import fit_leace

    rows = c.load_manifest()
    fit_idx = np.asarray(c.fit_indices(rows), dtype=np.int64)
    ctx = c.load_summary_rows(c.CELL, "context_end", c.HEADLINE_LAYER)
    x_ctx = np.asarray(ctx[fit_idx], dtype=np.float64)
    if smoke:
        x_ctx = x_ctx[:500]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, v in trait_dirs.items():
        assert v.shape[0] == x_ctx.shape[1], (
            f"{name}: direction dim {v.shape[0]} != context-state dim {x_ctx.shape[1]}"
        )
        e0 = x_ctx @ v.double().numpy()
        eraser = fit_leace(x_ctx, e0)
        out[name] = (eraser.mean_x, eraser.P)
        print(f"[p3-leace] fitted eraser for {name} on {x_ctx.shape[0]} context states")
    return out


# ── contexts ─────────────────────────────────────────────────────────────────


def steer_context_indices(rows: list[dict], smoke: bool) -> list[int]:
    """60 seeded trait-stratum manifest indices (plan §4 P3; seed = SEED_DRAWS)."""
    trait_rows = [i for i, r in enumerate(rows) if r.get("stratum") == "trait_stratum"]
    assert trait_rows, "manifest has no trait_stratum rows"
    rng = np.random.default_rng(c.SEED_DRAWS)
    rng.shuffle(trait_rows)
    n = 2 if smoke else N_STEER_CONTEXTS
    return [int(i) for i in trait_rows[:n]]


# ── fluency heuristics (pilot gate; CPU-testable) ────────────────────────────


def is_degenerate(text: str) -> bool:
    """Length + repetition heuristics per plan §7 gate 2 (threshold applied by caller).

    Degenerate iff: near-empty (<10 chars), a single character run > 30, or the most
    frequent word trigram covers > 30% of all trigrams (>= 5 trigrams required to fire).
    """
    t = text.strip()
    if len(t) < 10:
        return True
    run, best = 1, 1
    for a, b in itertools.pairwise(t):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best > 30:
        return True
    words = t.split()
    tris = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if len(tris) >= 5:
        counts: dict[str, int] = {}
        for tr in tris:
            counts[tr] = counts.get(tr, 0) + 1
        if max(counts.values()) / len(tris) > 0.30:
            return True
    return False


def degenerate_fraction(texts: list[str]) -> float:
    """Fraction of degenerate completions (the pilot's per-direction gate statistic)."""
    assert texts, "empty completion list"
    return sum(1 for t in texts if is_degenerate(t)) / len(texts)


# ── batched HF generation under a hook ───────────────────────────────────────


def generate_batched(
    model, tok, prompts: list[str], k_draws: int, seed: int, hook=None
) -> list[list[str]]:
    """Left-padded batched ``model.generate`` (the issue623 batching shape) under an
    optional intervention hook. Returns per-prompt lists of ``k_draws`` completions."""
    handle = _register_hook(model, hook) if hook is not None else None
    prev_pad, prev_side = tok.pad_token, tok.padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    out: list[list[str]] = []
    try:
        torch.manual_seed(seed)
        for b0 in range(0, len(prompts), GEN_BATCH):
            chunk = prompts[b0 : b0 + GEN_BATCH]
            batch = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            prompt_len = batch["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **batch,
                    do_sample=True,
                    temperature=GEN_TEMPERATURE,
                    top_p=1.0,
                    max_new_tokens=GEN_MAX_TOKENS,
                    num_return_sequences=k_draws,
                    pad_token_id=tok.pad_token_id,
                )
            decoded = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
            out.extend(
                [decoded[i * k_draws + j] for j in range(k_draws)] for i in range(len(chunk))
            )
    finally:
        if handle is not None:
            handle.remove()
        tok.pad_token, tok.padding_side = prev_pad, prev_side
    assert len(out) == len(prompts), (len(out), len(prompts))
    return out


def condition_hook(cond: Condition, directions: dict, alpha_by_dir: dict[str, float], erasers):
    """Resolve the forward hook for one condition (None for steer_base)."""
    if cond.kind == "base":
        return None
    if cond.kind == "add":
        v = directions[cond.direction].float()
        v_hat = v / (v.norm() + 1e-8)  # issue623:511 epsilon convention
        return make_add_hook(v_hat, cond.sign * alpha_by_dir[cond.direction])
    mean_x, p_mat = erasers[cond.direction]
    return make_leace_hook(torch.from_numpy(mean_x).float(), torch.from_numpy(p_mat).float())


# ── stages ───────────────────────────────────────────────────────────────────


def _paths(out_root: str | None) -> dict[str, Path]:
    droot = c.data_out(out_root) / "steering"
    eroot = c.eval_out(out_root) / "steering"
    return {
        "gen": droot / "raw_completions",
        "summaries": droot / "summaries",
        "eval": eroot,
    }


def _gen_file(paths: dict[str, Path], cond_id: str) -> Path:
    return paths["gen"] / f"gen_{cond_id}.jsonl"


def _regime_check(paths: dict[str, Path], regime: dict) -> None:
    """Refuse to resume onto files written under a DIFFERENT regime (#1333 class)."""
    sidecar = paths["gen"] / "regime.json"
    if sidecar.exists():
        prior = json.loads(sidecar.read_text())
        assert prior == regime, (
            f"steering out-root holds a run under a DIFFERENT regime ({prior} != {regime}); "
            "use a fresh --out-root instead of mixing regimes"
        )
    else:
        paths["gen"].mkdir(parents=True, exist_ok=True)
        c.write_json_atomic(sidecar, regime)


def run_pilot(model, tok, ctxs, my_conds, directions, alpha0, paths, shard_tag) -> dict:
    """Fluency pilot + assumption-11 rig-sanity for this shard's ADD directions.

    Per direction: FLUENCY_PILOT_GENS gens at +a; degenerate fraction > 0.30 => halve a
    once and re-gen; still > 0.30 => drop the direction. Writes the per-shard pilot
    report and returns {direction: final_alpha} for the usable set.
    """
    import issue1774_draws as draws

    add_dirs = sorted({x.direction for x in my_conds if x.kind == "add"})
    pilot_prompts = [r["prompt"] for r in ctxs[: min(FLUENCY_PILOT_GENS, len(ctxs))]]
    while len(pilot_prompts) < FLUENCY_PILOT_GENS:  # smoke has < 10 contexts
        pilot_prompts.append(pilot_prompts[-1])
    report: dict = {"meta": c.repro_meta({"script": "issue1774_steering.py pilot"})}
    alpha_by_dir: dict[str, float] = {}
    dropped: dict[str, str] = {}
    per_dir: dict[str, dict] = {}
    for name in add_dirs:
        v = directions[name].float()
        v_hat = v / (v.norm() + 1e-8)
        alpha, frac = alpha0, None
        for attempt in range(2):
            hook = make_add_hook(v_hat, alpha)
            texts = [
                t[0]
                for t in generate_batched(
                    model, tok, pilot_prompts, 1, c.SEED_DRAWS + 7 * attempt, hook
                )
            ]
            frac = degenerate_fraction(texts)
            per_dir[name] = {"alpha": alpha, "degenerate_fraction": frac, "attempt": attempt}
            if frac <= FLUENCY_DEGENERATE_MAX:
                break
            if attempt == 0:
                alpha *= 0.5  # halve once (plan §4 P3), then drop
        if frac is not None and frac > FLUENCY_DEGENERATE_MAX:
            dropped[name] = f"degenerate_fraction={frac:.2f} after alpha halving"
        else:
            alpha_by_dir[name] = alpha
        print(f"[p3-pilot] {name} alpha={alpha:.3f} degenerate={frac:.2f}", flush=True)

    # assumption-11 rig-sanity, refined (found by the tiny-model smoke, 2026-07-29):
    # transformers 4.53+ collects output_hidden_states via PREPENDED recorder hooks,
    # so the RECORDED entry at the hooked block's own index shows the PRE-hook output
    # (verified on 4.57.6: delta 0.000 at the hook index while the next layer's INPUT
    # and every downstream recorded index carry the shift). The intervention is real
    # for the computation (what generation uses); only a naive recorded-t1 read AT the
    # hook index is structurally blind. So assert TWO things instead:
    #   (a) a LATER-registered explicit capture hook at the module boundary sees the
    #       +alpha*v shift (replace semantics chain to later hooks — the compose), and
    #   (b) the recorded t1 at a DOWNSTREAM layer (L18) moves vs base (propagation
    #       into the stream the DV reads).
    if add_dirs:
        name = add_dirs[0]
        v = directions[name].float()
        v_hat = (v / (v.norm() + 1e-8)).numpy().astype(np.float64)
        row = dict(ctxs[0])
        row["draws"] = ["Rig-sanity fixed completion for the assumption-11 check."]
        boundary: dict[str, np.ndarray] = {}

        def _boundary_capture(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            boundary["mean"] = hs.detach().to(torch.float32).mean(dim=(0, 1)).cpu().numpy()
            return None

        h_cap = _register_hook(model, _boundary_capture)
        try:
            base_t1 = draws._capture_t1([row], 0, model, tok, str(model.device))
        finally:
            h_cap.remove()
        boundary_base = boundary["mean"].astype(np.float64)
        alpha_big = RIG_SANITY_ALPHA_MULT * alpha0
        h_add = _register_hook(model, make_add_hook(v / (v.norm() + 1e-8), alpha_big))
        h_cap = _register_hook(model, _boundary_capture)  # AFTER the ADD hook
        try:
            hooked_t1 = draws._capture_t1([row], 0, model, tok, str(model.device))
        finally:
            h_add.remove()
            h_cap.remove()
        boundary_hooked = boundary["mean"].astype(np.float64)
        along_v = float((boundary_hooked - boundary_base) @ v_hat)
        li_down = c.LAYERS.index(18)
        down_delta = float(
            np.linalg.norm(
                hooked_t1[0, li_down].astype(np.float64) - base_t1[0, li_down].astype(np.float64)
            )
        )
        report["rig_sanity"] = {
            "direction": name,
            "alpha": alpha_big,
            "boundary_along_v": along_v,
            "downstream_L18_delta_norm": down_delta,
            "note": (
                "LIVE-captured copies — rig-sanity ONLY, never the DV (plan §4 P3). "
                "Recorded t1 AT the hooked index is pre-hook on transformers>=4.53 "
                "(prepended recorder hooks); compose verified at the module boundary."
            ),
        }
        assert along_v > 0.5 * alpha_big, (
            f"assumption-11 rig-sanity FAILED: boundary capture moved {along_v:.3f} along v "
            f"(expected ~{alpha_big}) — the ADD-then-capture hook compose is broken"
        )
        assert down_delta > 0, (
            "assumption-11 rig-sanity FAILED: downstream recorded t1 (L18) did not move — "
            "the intervention is not propagating into the captured stream"
        )
        np.save(paths["summaries"] / f"rig_sanity_t1_base_{shard_tag}.npy", base_t1)
        np.save(paths["summaries"] / f"rig_sanity_t1_hooked_{shard_tag}.npy", hooked_t1)

    report["alpha_by_direction"] = alpha_by_dir
    report["dropped_directions"] = dropped
    report["per_direction"] = per_dir
    c.write_json_atomic(paths["eval"] / f"pilot_report_shard{shard_tag}.json", report)
    return alpha_by_dir


def run_gen(model, tok, ctxs, my_conds, directions, alpha_by_dir, erasers, paths) -> None:
    """Generate per condition (checkpoint per condition; resume skips complete files)."""
    paths["gen"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for u, cond in enumerate(my_conds):
        out_path = _gen_file(paths, cond.cond_id)
        if out_path.exists() and len(c.jsonl_rows(out_path)) == len(ctxs):
            print(f"[p3-gen] unit {u + 1}/{len(my_conds)} {cond.cond_id} resume-skip", flush=True)
            continue
        hook = condition_hook(cond, directions, alpha_by_dir, erasers)
        seed = c.SEED_DRAWS + 100 + u
        draws_per_prompt = generate_batched(
            model, tok, [r["prompt"] for r in ctxs], cond.k_draws, seed, hook
        )
        tmp = out_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r, dr in zip(ctxs, draws_per_prompt, strict=True):
                rec = {k: r[k] for k in ("manifest_index", "prefix_id", "query_id")}
                rec.update(
                    prefix_text=r["prefix_text"],
                    prompt=r["prompt"],
                    draws=dr,
                    condition=asdict(cond),
                    gen_seed=seed,
                )
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        print(
            f"[p3-gen] unit {u + 1}/{len(my_conds)} {cond.cond_id} "
            f"rows={len(ctxs)}x{cond.k_draws} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )


def _upload_dir_verified(local_dir: Path, sub: str, pattern: str) -> None:
    """Upload one steering artifact dir to HF + exact-set verify (fail loud)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = sorted(local_dir.glob(pattern))
    assert files, f"no {pattern} files under {local_dir}"
    hub._upload(
        local_dir,
        repo_id=c.DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{c.HF_UPLOAD_PREFIX}/{sub}",
    )
    expected = [f"{c.HF_UPLOAD_PREFIX}/{sub}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), c.DATA_REPO, expected, path_in_repo=c.HF_UPLOAD_PREFIX, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"p3 upload verify missing {len(missing)}: {sorted(missing)[:5]}")
    print(f"[p3-upload] verified {len(expected)} files at {c.HF_UPLOAD_PREFIX}/{sub}")


def run_upload(paths: dict[str, Path]) -> None:
    """Upload raw completions to HF the moment generation ends (fail-loud verify)."""
    _upload_dir_verified(paths["gen"], "raw_completions/steering", "gen_*.jsonl")


def run_upload_summaries(paths: dict[str, Path]) -> None:
    """Upload the hook-free re-captured t1 summaries + rig-sanity copies (plan §4 P3
    Persist line; ~40 MB — persist-by-default, mirrors the P1 summaries leg)."""
    _upload_dir_verified(paths["summaries"], "steering/summaries", "*")


def run_recapture(model, tok, my_conds, paths) -> None:
    """HOOK-FREE teacher-forced t1 re-capture over the persisted completions (the DV).

    Uses the P1 rig verbatim (``issue1774_draws._capture_t1``) with NO intervention hook
    installed — the methodology-critic Must-Fix: the live same-pass capture writes +-av
    into t1 by construction, so it is never the DV. One .npy per (condition, draw).
    """
    import issue1774_draws as draws

    paths["summaries"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for u, cond in enumerate(my_conds):
        rows = c.jsonl_rows(_gen_file(paths, cond.cond_id))
        assert rows, f"no generation rows for {cond.cond_id}"
        for k in range(cond.k_draws):
            out_p = paths["summaries"] / f"t1_{cond.cond_id}_draw{k}.npy"
            if out_p.exists():
                continue
            for r in rows:
                if not r["draws"][k]:
                    r["draws"][k] = " "  # empty completion guard (draws.py convention)
            kept = draws._capture_t1(rows, k, model, tok, str(model.device))
            np.save(out_p, kept)  # (n_ctx, len(LAYERS), D) fp16, hook-free
        idx_p = paths["summaries"] / f"row_index_{cond.cond_id}.json"
        if not idx_p.exists():
            c.write_json_atomic(
                idx_p, {"manifest_indices": [int(r["manifest_index"]) for r in rows]}
            )
        print(
            f"[p3-recap] unit {u + 1}/{len(my_conds)} {cond.cond_id} "
            f"draws={cond.k_draws} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )


MAX_ROW_ID_LEN = 53  # judge custom_id budget (issue1774_judge.py MAX_ITEM_ID_LEN)


def build_judge_rows(
    gen_rows_by_cond: dict[str, list[dict]], query_text_by_mi: dict[int, str]
) -> list[dict]:
    """P3->P5 interface rows (issue1774_judge.py::load_manifest_rows contract):
    one row per (condition, context, draw) with a unique row_id (<=53 chars, no
    '__'), the bare user QUESTION text (rubric {question} slot), and the draw's
    completion. Baseline draws are included (the judge scores them too)."""
    rows: list[dict] = []
    seen: set[str] = set()
    for cond_id in sorted(gen_rows_by_cond):
        for r in gen_rows_by_cond[cond_id]:
            mi = int(r["manifest_index"])
            for k, draw in enumerate(r["draws"]):
                rid = f"{cond_id}-{mi}-d{k}"
                assert "__" not in rid and len(rid) <= MAX_ROW_ID_LEN, rid
                assert rid not in seen, f"duplicate row_id {rid}"
                seen.add(rid)
                rows.append(
                    {
                        "row_id": rid,
                        "condition": cond_id,
                        "manifest_index": mi,
                        "question": query_text_by_mi[mi],
                        "completion": draw,
                    }
                )
    assert rows, "no judge rows built from the generation files"
    return rows


def _query_texts_for(manifest_indices: list[int]) -> dict[int, str]:
    """Bare user-query text per manifest index (the judge rubric {question} slot)."""
    from issue1092_gpu_phase import _query_text, load_store

    rows = c.load_manifest()
    query_store = load_store(c.stage_dir() / "corpus", "query_store.jsonl")
    return {mi: _query_text(query_store[str(rows[mi]["query_id"])]) for mi in set(manifest_indices)}


def merge_state_shift(all_conds: list[Condition], paths: dict[str, Path], smoke: bool) -> bool:
    """When EVERY condition's re-captured t1 exists, write ``state_shift.json``.

    Returns True when the merge ran (any shard may perform it; content is deterministic
    so concurrent writers converge — write_json_atomic makes the write itself safe).
    dt1(cond, ctx) = ||t1_cond(ctx) - mean_k t1_base(ctx, k)|| at L14; the steer_base
    band is the within-context cross-draw pairwise distance set (the H4 inertness band).
    """
    # pilot reports first: a pilot-DROPPED direction never produces gen/t1 files, so its
    # conditions are excluded from the completeness check (else the merge waits forever).
    alpha_by_dir: dict[str, float] = {}
    dropped: dict[str, str] = {}
    for p in sorted(paths["eval"].glob("pilot_report_shard*.json")):
        rep = json.loads(p.read_text())
        alpha_by_dir.update(rep.get("alpha_by_direction", {}))
        dropped.update(rep.get("dropped_directions", {}))
    live_conds = [x for x in all_conds if x.kind != "add" or x.direction not in dropped]
    missing = [
        x.cond_id
        for x in live_conds
        for k in range(x.k_draws)
        if not (paths["summaries"] / f"t1_{x.cond_id}_draw{k}.npy").exists()
    ]
    if missing:
        print(f"[p3-merge] waiting on {len(missing)} condition-draws (e.g. {missing[:3]})")
        return False
    all_conds = live_conds
    li = c.LAYERS.index(c.HEADLINE_LAYER)
    base = next(x for x in all_conds if x.kind == "base")
    # statistics-critic Must-Fix (plan §4 P3): a K=1 baseline yields NO cross-draw band.
    assert base.k_draws >= 2, f"steer_base needs >= 2 draws for the inertness band, got {base}"
    base_idx = json.loads((paths["summaries"] / f"row_index_{base.cond_id}.json").read_text())[
        "manifest_indices"
    ]
    base_t1 = np.stack(
        [
            np.load(paths["summaries"] / f"t1_{base.cond_id}_draw{k}.npy")[:, li].astype(np.float64)
            for k in range(base.k_draws)
        ],
        axis=1,
    )  # (n_ctx, K, D)
    band: dict[str, list[float]] = {}
    for i, mi in enumerate(base_idx):
        dists = [
            float(np.linalg.norm(base_t1[i, a] - base_t1[i, b]))
            for a in range(base.k_draws)
            for b in range(a + 1, base.k_draws)
        ]
        band[str(mi)] = dists
    pooled = np.asarray([d for v in band.values() for d in v], dtype=np.float64)
    base_mean = base_t1.mean(axis=1)  # (n_ctx, D)
    pos = {mi: i for i, mi in enumerate(base_idx)}
    conditions: dict[str, dict] = {}
    for cond in all_conds:
        if cond.kind == "base":
            continue
        idx = json.loads((paths["summaries"] / f"row_index_{cond.cond_id}.json").read_text())[
            "manifest_indices"
        ]
        t1 = np.load(paths["summaries"] / f"t1_{cond.cond_id}_draw0.npy")[:, li].astype(np.float64)
        per_ctx = {
            str(mi): float(np.linalg.norm(t1[i] - base_mean[pos[mi]])) for i, mi in enumerate(idx)
        }
        vals = np.asarray(list(per_ctx.values()), dtype=np.float64)
        conditions[cond.cond_id] = {
            "kind": cond.kind,
            "direction": cond.direction,
            "sign": cond.sign,
            "per_context_dt1": per_ctx,
            "median_dt1": float(np.median(vals)),
            "p90_dt1": float(np.percentile(vals, 90)),
            "n_contexts": int(vals.size),
        }
    # usable set + judge_skip kill criterion (alpha_by_dir/dropped read at top of merge)
    n_usable = len(alpha_by_dir)
    judge_skip = (not smoke) and n_usable < MIN_USABLE_DIRECTIONS
    out = {
        "meta": c.repro_meta({"script": "issue1774_steering.py merge"}),
        "layer": c.HEADLINE_LAYER,
        "hook_block_idx": HOOK_BLOCK_IDX,
        "gen": {"temperature": GEN_TEMPERATURE, "max_tokens": GEN_MAX_TOKENS},
        "alpha_by_direction": alpha_by_dir,
        "dropped_directions": dropped,
        "n_usable_directions": n_usable,
        "judge_skip": judge_skip,
        "calibration_failure": judge_skip,
        "steer_base_band": {
            "per_context": band,
            "pooled_p50": float(np.percentile(pooled, 50)),
            "pooled_p90": float(np.percentile(pooled, 90)),
            "k_draws": base.k_draws,
        },
        "conditions": conditions,
    }
    c.write_json_atomic(paths["eval"] / "state_shift.json", out)
    if judge_skip:
        print(
            f"[p3-merge] CALIBRATION FAILURE: {n_usable} usable directions < "
            f"{MIN_USABLE_DIRECTIONS} — state-shift-only read (judge skipped), per plan §7"
        )
    # P3->P5 interface: fold the judge rows into the manifest the judge reads
    # (issue1774_judge.py --manifest default = this file's `rows` key).
    gen_rows_by_cond = {x.cond_id: c.jsonl_rows(_gen_file(paths, x.cond_id)) for x in all_conds}
    all_mis = [int(r["manifest_index"]) for rows_ in gen_rows_by_cond.values() for r in rows_]
    judge_rows = build_judge_rows(gen_rows_by_cond, _query_texts_for(all_mis))
    manifest_p = paths["eval"] / "manifest.json"
    manifest = json.loads(manifest_p.read_text())
    manifest["rows"] = judge_rows
    manifest["judge_skip"] = judge_skip
    c.write_json_atomic(manifest_p, manifest)
    print(
        f"[p3-merge] wrote state_shift.json ({len(conditions)} intervention conditions) "
        f"+ {len(judge_rows)} judge rows into manifest.json"
    )
    return True


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real branch, then exit",
    )
    args = ap.parse_args(argv)
    if args.import_check:
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.analysis.issue_763_nonlinear import fit_leace  # noqa: F401
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload,
            verify_repo_paths_uploaded,
        )
        from issue1774_draws import _capture_t1, _load_hf_model  # noqa: F401

        print("[import-check] p3 deferred imports resolve")
        return 0

    import issue1774_draws as draws

    print(f"[phase=p3_steering] shard={args.shard} smoke={args.smoke}")
    paths = _paths(args.out_root)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    payload = load_directions(args.out_root)
    directions: dict[str, torch.Tensor] = payload["directions"]
    alpha0 = float(payload["alpha"])
    rows = c.load_manifest()
    ctx_indices = steer_context_indices(rows, args.smoke)
    ctxs = draws._render_rows(rows, ctx_indices)
    all_conds = build_conditions(sorted(directions), args.smoke)
    my_conds = shard_conditions(all_conds, args.shard)
    shard_tag = args.shard.replace("/", "of")
    _regime_check(
        paths,
        {
            "smoke": bool(args.smoke),
            "n_contexts": len(ctxs),
            "layer": c.HEADLINE_LAYER,
            "alpha0": alpha0,
        },
    )
    c.write_json_atomic(
        paths["eval"] / "manifest.json",
        {
            "meta": c.repro_meta({"script": "issue1774_steering.py"}),
            "conditions": [asdict(x) for x in all_conds],
            "context_manifest_indices": ctx_indices,
            "alpha0": alpha0,
            "hook_block_idx": HOOK_BLOCK_IDX,
        },
    )
    if not my_conds:
        print(f"[p3] shard {args.shard} owns no conditions; checking merge")
        merge_state_shift(all_conds, paths, args.smoke)
        return 0

    device = "cuda:0" if draws._cuda_ok() else "cpu"
    model, tok = draws._load_hf_model(device)
    d_dir = next(iter(directions.values())).shape[0]
    assert d_dir == model.config.hidden_size, (
        f"direction dim {d_dir} != model hidden size {model.config.hidden_size} — "
        "directions.pt and the generation model must share the residual-stream space"
    )

    leace_dirs = {x.direction: directions[x.direction] for x in my_conds if x.kind == "leace"}
    erasers = fit_leace_erasers(leace_dirs, args.out_root, args.smoke) if leace_dirs else {}
    alpha_by_dir = run_pilot(model, tok, ctxs, my_conds, directions, alpha0, paths, shard_tag)

    # drop this shard's unusable ADD conditions (recorded in the pilot report + merge)
    live = [x for x in my_conds if x.kind != "add" or x.direction in alpha_by_dir]
    run_gen(model, tok, ctxs, live, directions, alpha_by_dir, erasers, paths)
    if args.smoke:
        # smoke legs never write the production HF prefix (dispatch skips P1
        # uploads for the same reason); the hub path runs on the production leg.
        print("[p3-upload] SKIPPED (smoke — production HF prefix untouched)")
    else:
        run_upload(paths)  # completions to HF BEFORE the re-capture consumer (store-first)
    run_recapture(model, tok, live, paths)
    if args.smoke:
        print("[p3-upload] summaries SKIPPED (smoke — production HF prefix untouched)")
    else:
        # plan §4 P3 Persist: hook-free re-captured t1 summaries + rig-sanity copies
        run_upload_summaries(paths)
    # merge runs on whichever shard finishes last (deterministic content; atomic write;
    # merge itself excludes pilot-dropped directions + re-checks completeness)
    merged = merge_state_shift(all_conds, paths, args.smoke)
    if not merged:
        print("[p3] merge deferred — another shard still generating")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: heavy C-extension atexit race (gotchas #1689)


if __name__ == "__main__":
    main()
