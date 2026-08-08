"""#1776 follow-up ``p3p4_heterogeneity_dose`` — P3 per-context Jacobians + P4 dose inputs.

LEG P3 (per-context Jacobian heterogeneity): compute PER-CONTEXT Jacobians
J_i (NO pair averaging) with the phase-2 gradient rig
(``issue1776_jacobian.JacobianEstimator`` — same slot pins: block-14 output
source, block-19 answer-mean readout, same render/span conventions) for a
stratified subsample of contexts drawn from the two text-bearing corpora:

  - ``lmsys``:    the phase-0.4 J-fit pool (``jpairs.jsonl`` + ``jpair_capture.pt``
                  — prompt/response text + captured c14/v19, manifest order);
  - ``wildchat``: the phase-5a fresh WildChat captures (chunk .pt tensors +
                  raw-completions .json text, joined by global ``ci``).

  NOTE (recorded deviation): the brief names "the pinned lmsys test-1000", but
  that split's ANSWER TEXT is not persisted anywhere (plan §4 0.4 — the pass_b
  bundle is tensors-only), so per-context Jacobians are UNCOMPUTABLE on it.
  The lmsys arm substitutes the phase-0.4 J-fit pool — the very pairs the
  averaged J was built from, which is the sharpest population for the
  cancellation question — and the wildchat arm keeps the held-out read.

Stratification: per-corpus quantiles of the slot-matched fitted comparator's
(``m_ridge_x50k``) per-context squared residual (the "per-context error
quantiles" of the brief, re-derived from persisted tensors — transfer.json
carries only aggregates). A stratified subset is additionally run FULL-RANK
(all 3584 standard-basis seeds -> the complete J_i); the rest use the frozen
256-seed sketch basis (``jac_sketch/seeds.pt``: 228 v-pool PCs + 20 M'
left-singular u_k + 8 Gaussian probes), whose restriction of any full J is
exact (rows = seeds @ J).

DECIDES (analyze): correlational-M' (per-context J_i ALSO fail to align with
M') vs heterogeneous-causal (individual J_i align but cancel under averaging):
per-context cos(u_k^T J_i, v_k) vs the averaged value, cancellation ratio
||mean_i J_i|| / mean_i ||J_i||, pairwise cos(J_i, J_j), and neighbor-delta
prediction R^2 (J_i vs J_avg vs M' vs identity on v_j - v_i, intercept-free).

LEG P4 inputs: the raw-norm dose ladder for the phase-3 rerun — target norms
{0.25, 0.5, 1.0} x N_ref with N_ref = 4 x median_trait ||r_B[l14]|| (the
realized #1415 layer-14 persona-vector injection norm at its replicated-effect
operating point alpha=4), passed to ``issue1776_phase3.py --alphas`` whose
directions are unit-normed (applied norm == alpha).

Content hygiene: prompts/responses are real LMSYS/WildChat text — this module
NEVER prints or logs text fields; reports carry ids, counts, hashes, norms.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue1776_jacobian as J76  # noqa: E402

FU = "followup_p3p4"
TRAITS = ("evil", "sycophancy", "hallucination")
LADDER_FRACS = (0.25, 0.5, 1.0)
PARENT_ALPHA_MAX = 4.0  # phase-3's dose-limited unit-norm ceiling (the null P4 escalates past)

# Reused #1776 production artifacts, staged verbatim-mirror under --dest
# (dest/<repo path> == the parent dispatcher's hf_dl layout, so already-staged
# files short-circuit; #1774 mirror-root arithmetic asserted below).
STAGE_FILES = (
    "analysis_tensors/jac_full/J_last.pt",
    "analysis_tensors/jac_full/J_ctx.pt",
    "analysis_tensors/jac_full/J_prefix.pt",
    "analysis_tensors/comparator/m_ridge_x50k.pt",
    "analysis_tensors/jpairs/jpairs.jsonl",
    "analysis_tensors/jpairs/jpair_capture.pt",
    "analysis_tensors/jac_sketch/seeds.pt",
    "analysis_tensors/contexts/contexts.jsonl",
    "analysis_tensors/contexts/contexts.meta.json",
)
RB_PREFIX = "issue779_monitoring/r_b"

# Self-describing population classes (round-14 review minor + crash-fix r15).
# MEASURED population semantics (2026-07-30 sweep over the staged corpora):
# BOTH corpora are BARE single-turn real-user prompts under the identical chat
# render (jpairs min prompt 2 chars; no persona/system segment anywhere) — the
# classes differ in PROVENANCE/ROLE, not prompt structure: lmsys rows are the
# phase-0.4 J-fit pool (in-sample for J_avg), wildchat rows the held-out fresh
# captures. Consequently every context's PREFIX arm covers ONLY the
# chat-template preamble (incl. the Qwen default system prompt) — identical
# boilerplate across contexts, so the prefix arm carries no per-context
# persona variation in this population.
CONTEXT_CLASSES = {"lmsys": "lmsys_jfit_bare_user", "wildchat": "wildchat_heldout_bare_user"}
POPULATION_CAVEAT = (
    "Both context classes are BARE single-turn real-user prompts rendered under the "
    "identical chat template (no persona/system segment): the prefix arm covers only the "
    "template preamble + Qwen default system prompt (shared boilerplate), and the class "
    "split is provenance/role — lmsys = phase-0.4 J-fit pool (in-sample for J_avg), "
    "wildchat = held-out fresh captures."
)


def context_class_of(source) -> str:
    """Population class for a pcj row ('source' corpus tag -> class label)."""
    return CONTEXT_CLASSES.get(str(source), "unspecified")


def _slug(pair_id: str) -> str:
    """Filesystem-safe per-context filename stem (uniqueness asserted by caller)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(pair_id))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in path.read_text().split("\n") if ln.strip()]
    assert rows, f"empty jsonl: {path}"
    return rows


# ── stage ─────────────────────────────────────────────────────────────────────


def cmd_stage(args) -> int:
    """Stage every P3/P4 input from the Hub at ONE fresh revision pin.

    Idempotent per-file (existing targets skip); wildchat chunk cap keeps the
    smoke leg tiny while pt/json stems stay ALIGNED (same chunk set both kinds).
    """
    from explore_persona_space.orchestrate import hub

    rev = C76.resolve_data_repo_pin(args.pin_file, refresh=args.refresh_pin)
    staged: list[str] = []
    skipped = 0
    for rel in STAGE_FILES:
        repo_path = f"{C76.HF_PREFIX}/{rel}"
        dest = args.dest / repo_path
        if dest.is_file():
            skipped += 1
            continue
        hub.stage_hub_file(C76.HF_DATA_REPO, repo_path, dest, repo_type="dataset", revision=rev)
        staged.append(repo_path)

    # r_B trait stacks (prefix mirror; #1774: dest is the MIRROR ROOT, so
    # dest/<RB_PREFIX>/<trait>.pt is the consumed path — asserted below).
    rb_dir = args.dest / RB_PREFIX
    if any(not (rb_dir / f"{t}.pt").is_file() for t in TRAITS):
        n_rb = hub.stage_hub_prefix(
            C76.HF_DATA_REPO, RB_PREFIX, args.dest, repo_type="dataset", revision=rev
        )
        staged.append(f"{RB_PREFIX} ({n_rb} files)")
    for t in TRAITS:
        assert (rb_dir / f"{t}.pt").is_file(), f"r_B stack missing post-stage: {rb_dir / t}.pt"

    # wildchat_fresh capture chunks (.pt) + raw text (.json), stem-aligned.
    import issue779_ffc_n1m_generate_capture as N1G

    cap_prefix = f"{C76.HF_PREFIX}/wildchat_fresh/final_token_capture"
    raw_prefix = f"{C76.HF_PREFIX}/wildchat_fresh/raw_completions"
    remote = hub.retry_transient(
        lambda: N1G.N50._remote_index(cap_prefix), what=f"remote_index({cap_prefix})"
    )
    names = sorted(n for n in remote if fnmatch.fnmatch(n, "shard*_chunk*.pt"))
    assert names, f"no wildchat capture chunks under {cap_prefix}"
    if args.max_wc_chunks:
        names = names[: args.max_wc_chunks]
    n_wc = 0
    for n in names:
        for prefix, fname in ((cap_prefix, n), (raw_prefix, f"{Path(n).stem}.json")):
            dest = args.dest / prefix / fname
            if dest.is_file():
                continue
            hub.stage_hub_file(
                C76.HF_DATA_REPO, f"{prefix}/{fname}", dest, repo_type="dataset", revision=rev
            )
            n_wc += 1

    report = {
        "revision": rev,
        "staged": staged,
        "skipped_existing": skipped,
        "wc_chunks": [Path(n).stem for n in names],
        "wc_files_staged": n_wc,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.report, report)
    print(
        f"[p3p4-stage] [phase=stage_done] {len(staged)} staged, {skipped} present, "
        f"{len(names)} wc chunks -> {args.report}",
        flush=True,
    )
    return 0


# ── build: stratified context sample + P4 alpha ladder ───────────────────────


def _load_lmsys_rows(dest: Path) -> dict:
    """The phase-0.4 J-fit pool: text (jpairs.jsonl) + captures (jpair_capture.pt)."""
    jdir = dest / C76.HF_PREFIX / "analysis_tensors/jpairs"
    cap = torch.load(jdir / "jpair_capture.pt", map_location="cpu", weights_only=True)
    rows = _read_jsonl(jdir / "jpairs.jsonl")
    ids = [str(p) for p in cap["pair_id"]]
    assert ids == [str(r["pair_id"]) for r in rows], (
        "jpair_capture.pt order != jpairs.jsonl order — the merge contract broke"
    )
    return {
        "ids": ids,
        "prompts": [r["prompt"] for r in rows],
        "responses": [r["response"] for r in rows],
        "c14": cap["c14"].to(torch.float32),
        "v19": cap["v19"].to(torch.float32),
    }


def _load_wildchat_rows(dest: Path) -> dict:
    """The phase-5a fresh WildChat captures, chunk tensors joined to raw text by ci."""
    wc = dest / C76.HF_PREFIX / "wildchat_fresh"
    chunk_files = sorted((wc / "final_token_capture").glob("shard*_chunk*.pt"))
    assert chunk_files, f"no wildchat chunks staged under {wc / 'final_token_capture'}"
    ids, prompts, responses, c14s, v19s = [], [], [], [], []
    for cf in chunk_files:
        d = torch.load(cf, map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        li_s, li_r = layers.index(C76.SOURCE_LAYER), layers.index(C76.READOUT_LAYER)
        raw = json.loads((wc / "raw_completions" / f"{cf.stem}.json").read_text())
        by_ci = {int(r["ci"]): r for r in raw["rows"]}
        for k, ci in enumerate(int(x) for x in d["ci"]):
            r = by_ci.get(ci)
            assert r is not None, f"wildchat ci={ci} in {cf.name} missing from raw json"
            ids.append(f"wc{ci:06d}")
            prompts.append(r["prompt"])
            responses.append(r["response"])
            c14s.append(d["cx_last"][k, li_s, :].to(torch.float32))
            v19s.append(d["v_x"][k, li_r, :].to(torch.float32))
    return {
        "ids": ids,
        "prompts": prompts,
        "responses": responses,
        "c14": torch.stack(c14s),
        "v19": torch.stack(v19s),
    }


def _mprime_err2(payload: dict, c14: torch.Tensor, v19: torch.Tensor) -> np.ndarray:
    """Per-context squared residual of the slot-matched fitted comparator."""
    import issue779_ffc_n1m_fits as N1M

    pred = N1M.apply_map(payload, c14.numpy(), torch.device("cpu"))  # (n, H) fp64
    return ((pred - v19.to(torch.float64).numpy()) ** 2).sum(axis=1)


def _stratified_pick(
    err2: np.ndarray, n_pick: int, n_full: int, n_strata: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Quantile-stratified sample; returns (indices, stratum labels, full_rank mask, edges)."""
    n = err2.shape[0]
    assert n_pick <= n, (n_pick, n)
    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    edges = [float(x) for x in np.quantile(err2, qs)] if n_strata > 1 else []
    stratum = (
        np.searchsorted(np.asarray(edges), err2, side="right") if edges else np.zeros(n, dtype=int)
    )
    picked: list[int] = []
    for s in range(n_strata):
        pool = np.flatnonzero(stratum == s)
        rng.shuffle(pool)
        quota = n_pick // n_strata + (1 if s < n_pick % n_strata else 0)
        assert pool.size >= quota, f"stratum {s} holds {pool.size} < quota {quota}"
        picked.extend(int(x) for x in pool[:quota])
    picked_arr = np.asarray(sorted(picked))
    # full-rank subset: round-robin across strata among the picked set
    full_mask = np.zeros(picked_arr.shape[0], dtype=bool)
    order = np.argsort([int(stratum[i]) for i in picked_arr], kind="stable")
    take = order[:: max(1, picked_arr.shape[0] // max(1, n_full))][:n_full]
    full_mask[take] = True
    assert int(full_mask.sum()) == n_full, (int(full_mask.sum()), n_full)
    return picked_arr, stratum[picked_arr], full_mask, edges


def build_alpha_ladder(rb_dir: Path, out: Path, c14_med_norm: float) -> dict:
    """P4 raw-norm dose ladder matched to #1415's layer-14 operating point.

    N_ref = 4 x median_trait ||r_B[l14]||: #1415's persona-vector arm applied
    alpha x the RAW per-layer difference-of-means row (its replicated judged
    effect at layer 14, +6.2, ran at alpha=4), while #1776 phase 3 unit-norms
    directions — so passing these ladder values as ``--alphas`` reproduces the
    #1415 injection-norm scale exactly at the top rung.
    """
    norms = {}
    for t in TRAITS:
        d = torch.load(rb_dir / f"{t}.pt", map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        assert C76.SOURCE_LAYER in layers, (t, layers)
        norms[t] = float(d["r_b"][layers.index(C76.SOURCE_LAYER)].to(torch.float64).norm())
    n_ref = PARENT_ALPHA_MAX * float(np.median(list(norms.values())))
    alphas = [round(f * n_ref, 4) for f in LADDER_FRACS]
    assert alphas[-1] > PARENT_ALPHA_MAX, (
        f"top ladder rung {alphas[-1]} <= the parent's unit-norm ceiling {PARENT_ALPHA_MAX} — "
        "r_B norms are implausibly small; check the staged stacks"
    )
    ladder = {
        "trait_norms_l14": norms,
        "n_ref": n_ref,
        "fracs": list(LADDER_FRACS),
        "alphas": alphas,
        "alphas_csv": ",".join(f"{a:g}" for a in alphas),
        "basis": "#1415 alpha=4 x median ||r_B[l14]|| (its replicated layer-14 operating point)",
        "resid_norm_median_c14": c14_med_norm,
        "frac_of_resid_norm": [round(a / c14_med_norm, 4) for a in alphas],
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(out, ladder)
    print(f"[p3p4-build] alpha ladder {alphas} (n_ref={n_ref:.2f}) -> {out}", flush=True)
    return ladder


def cmd_build(args) -> int:
    """Per-context M' errors -> stratified P3 sample + targets + the P4 ladder."""
    assert args.n_sketch % 2 == 0 and args.n_full % 2 == 0, "per-corpus halves must be integral"
    assert args.n_sketch // 2 >= max(2, args.n_full // 2), (
        "need >=2 sampled contexts per corpus (neighbor-delta floor) and n_full <= n_sketch"
    )
    payload = torch.load(args.comparator, map_location="cpu", weights_only=True)
    rng = np.random.default_rng(args.seed)
    rows_out: list[dict] = []
    tgt_ids: list[str] = []
    tgt_src: list[str] = []
    tgt_stratum: list[int] = []
    tgt_err2: list[float] = []
    tgt_c14: list[torch.Tensor] = []
    tgt_v19: list[torch.Tensor] = []
    report_strata: dict = {}
    for corpus, loader in (("lmsys", _load_lmsys_rows), ("wildchat", _load_wildchat_rows)):
        data = loader(args.dest)
        err2 = _mprime_err2(payload, data["c14"], data["v19"])
        idx, strat, full_mask, edges = _stratified_pick(
            err2, args.n_sketch // 2, args.n_full // 2, args.strata, rng
        )
        report_strata[corpus] = {
            # context_class lives in the REPORT (and downstream payloads), NOT
            # in pcj_pairs.jsonl rows: the pairs file must stay byte-identical
            # across deterministic rebuilds so pairs_sha survives the p3_pcj
            # resume-manifest check (crash-fix r15 resume contract).
            "context_class": context_class_of(corpus),
            "n_pool": int(err2.shape[0]),
            "err2_quantile_edges": edges,
            "picked_per_stratum": {
                int(s): int((strat == s).sum()) for s in sorted(set(int(x) for x in strat))
            },
            "picked_ids": [data["ids"][i] for i in idx],
            "full_rank_ids": [data["ids"][i] for i, f in zip(idx, full_mask) if f],
        }
        for i, s, f in zip(idx, strat, full_mask):
            rows_out.append(
                {
                    "pair_id": data["ids"][i],
                    "prompt": data["prompts"][i],
                    "response": data["responses"][i],
                    "source": corpus,
                    "stratum": int(s),
                    "err2": float(err2[i]),
                    "full_rank": bool(f),
                }
            )
            tgt_ids.append(data["ids"][i])
            tgt_src.append(corpus)
            tgt_stratum.append(int(s))
            tgt_err2.append(float(err2[i]))
            tgt_c14.append(data["c14"][i])
            tgt_v19.append(data["v19"][i])
    slugs = [_slug(r["pair_id"]) for r in rows_out]
    assert len(set(slugs)) == len(slugs), "pair_id slug collision"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out_dir / "pcj_pairs.jsonl"
    tmp = pairs_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows_out) + "\n")
    tmp.replace(pairs_path)
    torch.save(
        {
            "pair_id": tgt_ids,
            "source": tgt_src,
            "stratum": tgt_stratum,
            "err2": tgt_err2,
            "c14": torch.stack(tgt_c14),
            "v19": torch.stack(tgt_v19),
            "comparator_tag": "m_ridge_x50k",
        },
        args.out_dir / "pcj_targets.pt",
    )
    c14_med = float(torch.stack(tgt_c14).to(torch.float64).norm(dim=1).median())
    ladder = build_alpha_ladder(
        args.dest / RB_PREFIX, args.out_dir / "p4_alpha_ladder.json", c14_med
    )
    C76.atomic_write_json(
        args.out_dir / "pcj_build_report.json",
        {
            "n_sketch": args.n_sketch,
            "n_full": args.n_full,
            "n_strata": args.strata,
            "seed": args.seed,
            "strata": report_strata,
            "population_caveat": POPULATION_CAVEAT,
            "pairs_sha": _sha256_file(pairs_path),
            "ladder_alphas": ladder["alphas"],
            "repro": C76.repro_meta(),
        },
    )
    print(
        f"[p3p4-build] [phase=build_done] {len(rows_out)} contexts "
        f"({args.n_full} full-rank) -> {pairs_path}",
        flush=True,
    )
    return 0


# ── pilot: measured 1-context wall at production shape (§9 pilot-gated) ───────


def cmd_pilot(args) -> int:
    """Time ONE sketch context end-to-end; project the P3 leg; rc=7 over 2x budget."""
    rows = J76.load_pairs(args.pairs)
    sd = torch.load(args.seeds_file, map_location="cpu", weights_only=True)
    seeds = sd["seeds"].to(torch.float32)
    model, tok = J76.load_model(args)
    est = J76.JacobianEstimator(
        model,
        source_layer=args.source_layer,
        readout_layer=args.readout_layer,
        seed_chunk=args.seed_chunk,
    )
    rend = None
    for row in rows:
        # suffix anchor (crash-fix r15): the bare real-user population's short
        # queries can substring-match inside the template preamble; see
        # J76._suffix_q_span.
        rend = J76.render_pair(tok, row["prompt"], row["response"], anchor="suffix")
        if rend is not None:
            break
    assert rend is not None, "no renderable pilot row"
    t0 = time.time()
    res = est.pair_backward(rend, seeds, serial=args.serial_grads)
    wall = time.time() - t0
    assert res["ctx_maxabs"] > 0.0, "pilot context gradient all-zero (slot-convention bug)"
    per_row_s = wall / seeds.shape[0]
    n_full = sum(1 for r in rows if r.get("full_rank"))
    n_sketch = len(rows) - n_full
    hidden = res["last"].shape[1]
    rows_total = n_sketch * seeds.shape[0] + n_full * hidden
    projected_gpu_h = rows_total * per_row_s / 3600.0
    ratio = projected_gpu_h / max(args.budget_gpu_h, 1e-9)
    verdict = "OK" if ratio <= 2.0 else "OVER_2X"
    report = {
        "wall_s_one_ctx": wall,
        "per_row_s": per_row_s,
        "seeds_timed": int(seeds.shape[0]),
        "n_sketch_ctx": n_sketch,
        "n_full_ctx": n_full,
        "rows_total": rows_total,
        "projected_gpu_h": projected_gpu_h,
        "projected_wall_h_at_ngpu": projected_gpu_h / max(args.ngpu, 1),
        "budget_gpu_h": args.budget_gpu_h,
        "ratio": ratio,
        "verdict": verdict,
        "note": "single-context basis; token-length dispersion is absorbed by the 2x margin",
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(
        f"[p3p4-pilot] [phase=pilot_done] per_row={per_row_s:.3f}s projected="
        f"{projected_gpu_h:.2f} GPU-h budget={args.budget_gpu_h} ratio={ratio:.2f} {verdict}",
        flush=True,
    )
    return 0 if verdict == "OK" else 7


# ── run: per-context Jacobians (sharded) ─────────────────────────────────────

RUN_MATCH_KEYS = (
    "model",
    "tiny",
    "dtype",
    "source_layer",
    "readout_layer",
    "pairs_sha",
    "seeds_sha",
    "pooling",
    "spans",
    "mode",
)


def _run_manifest(args, pairs_sha: str, seeds_sha: str) -> dict:
    return {
        "script": "issue1776_p3p4",
        "mode": "pcj",
        "model": args.model,
        "tiny": bool(args.tiny),
        "dtype": args.dtype,
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "pairs_sha": pairs_sha,
        "seeds_sha": seeds_sha,
        "pooling": J76.POOLING,
        "spans": J76.SPAN_CONVENTION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _check_run_manifest(out_dir: Path, manifest: dict) -> None:
    """phase3-style refusal: regimes never mix inside one out-root."""
    path = out_dir / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        diff = [k for k in RUN_MATCH_KEYS if prior.get(k) != manifest.get(k)]
        if diff:
            raise RuntimeError(
                f"pcj manifest MISMATCH on resume (keys: {diff}) — use a fresh --out-dir"
            )
    C76.atomic_write_json(path, manifest)


def _unit_spans_stale(out_path: Path, tok, row: dict, j: int, n: int) -> bool:
    """True when a RETAINED per-context unit carries mis-anchored legacy spans.

    Resume-invalidation for crash-fix r15: units persisted BEFORE the suffix
    anchor landed were rendered under the legacy find-from-0 locator, which is
    span-IDENTICAL to the suffix anchor except on rows whose query text
    substring-matches inside the template preamble (the silent garbage-span
    class; crash rows never persisted a unit). Only that rare class pays the
    payload read (mmap — tensor storages never materialize); an unreadable
    payload fails toward recompute (idempotent unit).
    """
    if J76.legacy_find_anchor_agrees(tok, row["prompt"]):
        return False
    try:
        prior = torch.load(out_path, map_location="cpu", weights_only=True, mmap=True)
        anchor = prior.get("span_anchor")
    except Exception as e:  # unreadable/truncated unit -> recompute, loudly
        print(f"[pcj] unit {j + 1}/{n} {_slug(row['pair_id'])} unreadable ({e!r})", flush=True)
        anchor = None
    if anchor == "suffix":
        return False
    print(
        f"[pcj] unit {j + 1}/{n} {_slug(row['pair_id'])} STALE-SPANS "
        "(legacy find-anchor mis-anchored this row) -> recompute",
        flush=True,
    )
    return True


def cmd_run(args) -> int:
    """Per-context backward sweep for one shard; per-context persist + resume."""
    rows = J76.load_pairs(args.pairs)
    # full-rank contexts first, so the expensive units round-robin across shards
    rows = sorted(rows, key=lambda r: not r.get("full_rank", False))
    pairs_sha = _sha256_file(args.pairs)
    sd = torch.load(args.seeds_file, map_location="cpu", weights_only=True)
    sketch_seeds = sd["seeds"].to(torch.float32)
    seeds_sha = J76._sha256_tensor(sketch_seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pcj_dir = args.out_dir / "pcj"
    pcj_dir.mkdir(parents=True, exist_ok=True)
    _check_run_manifest(args.out_dir, _run_manifest(args, pairs_sha, seeds_sha))

    shard_rows = rows[args.shard_index :: args.num_shards]
    if args.limit:
        shard_rows = shard_rows[: args.limit]
    model, tok = J76.load_model(args)
    hidden = (model.model if hasattr(model, "model") else model).config.hidden_size
    est = J76.JacobianEstimator(
        model,
        source_layer=args.source_layer,
        readout_layer=args.readout_layer,
        seed_chunk=args.seed_chunk,
    )
    eye: torch.Tensor | None = None
    seam_skips: list[str] = []
    t0 = time.time()
    for j, row in enumerate(shard_rows):
        slug = _slug(row["pair_id"])
        out_path = pcj_dir / f"{slug}.pt"
        if out_path.exists() and not _unit_spans_stale(out_path, tok, row, j, len(shard_rows)):
            print(f"[pcj] unit {j + 1}/{len(shard_rows)} {slug} SKIP (done)", flush=True)
            continue
        full = bool(row.get("full_rank", False))
        if full:
            if eye is None:
                eye = torch.eye(hidden, dtype=torch.float32)
            seed_mat = eye
        else:
            seed_mat = sketch_seeds
        # suffix anchor (crash-fix r15): find-from-0 mis-anchors short bare
        # real-user queries inside the template preamble (pod crash
        # AssertionError (0, 1, 30) at compute_prompt_spans; plus a SILENT
        # garbage-span class) — see J76._suffix_q_span.
        rend = J76.render_pair(tok, row["prompt"], row["response"], anchor="suffix")
        if rend is None:
            seam_skips.append(str(row["pair_id"]))
            print(f"[pcj] unit {j + 1}/{len(shard_rows)} {slug} SEAM-SKIP", flush=True)
            continue
        tu = time.time()
        res = est.pair_backward(rend, seed_mat, serial=args.serial_grads)
        if not res["ctx_maxabs"] > 0.0:
            C76.atomic_write_json(
                args.out_dir / f"gate_gnonzero_shard{args.shard_index}.json",
                {"gate": "G-NONZERO", "pair_id": row["pair_id"], "pass": False},
            )
            print(f"[pcj] G-NONZERO HALT rc=8 at {slug} (all-zero context gradient)", flush=True)
            return 8
        payload = {
            "pair_id": row["pair_id"],
            "source": row.get("source", "unspecified"),
            "context_class": context_class_of(row.get("source", "unspecified")),
            "span_anchor": rend["anchor"],
            "prompt_len": rend["prompt_len"],
            "prefix_len": rend["prefix_len"],
            "context_len": rend["context_len"],
            "stratum": int(row.get("stratum", 0)),
            "seed_mode": "full" if full else "sketch",
            "seeds_sha": "std_basis" if full else seeds_sha,
            "layers": [args.source_layer, args.readout_layer],
            "rows": {a: res[a].to(torch.bfloat16) for a in J76.ARMS},
            "v": res["v"].to(torch.float32),
            "c_last": res["c_last"].to(torch.float32),
            "c_prefix": res["c_prefix"].to(torch.float32),
            "c_ctx": res["c_ctx"].to(torch.float32),
            "ctx_maxabs": float(res["ctx_maxabs"]),
            "unit_wall_s": time.time() - tu,
        }
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        print(
            f"[pcj] unit {j + 1}/{len(shard_rows)} {slug} mode="
            f"{'full' if full else 'sketch'} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    C76.atomic_write_json(
        args.out_dir / f"shard{args.shard_index:02d}_report.json",
        {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "n_rows": len(shard_rows),
            "seam_skips": seam_skips,
            "repro": C76.repro_meta(),
        },
    )
    print(
        f"[pcj] [phase=pcj_shard_done] shard={args.shard_index} n={len(shard_rows)} "
        f"skips={len(seam_skips)}",
        flush=True,
    )
    return 0


# ── analyze ───────────────────────────────────────────────────────────────────


def _cosf(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius cosine of two same-shape fp64 tensors."""
    num = float((a * b).sum())
    den = float(a.norm()) * float(b.norm())
    return num / max(den, 1e-30)


def _boot_ci(vals: np.ndarray, n_boot: int, seed: int, stat=np.median) -> list[float]:
    """Context-resampled bootstrap CI (2.5/97.5 percentiles) of a scalar stat."""
    if vals.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = stat(vals[rng.integers(0, vals.size, size=(n_boot, vals.size))], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _delta_r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Pooled R^2 over ordered-pair deltas (intercept-free: SS_tot = sum ||true||^2)."""
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float((true**2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _neighbor_reads(
    ctxs: list[dict],
    r_by_id: dict[str, torch.Tensor],
    ops: dict[str, torch.Tensor],
    proj: torch.Tensor | None,
) -> dict:
    """Neighbor-delta prediction R^2 per predictor over ordered pairs (i != j).

    ``r_by_id``: per-context restricted operator rows (S, H_in) fp64 — the
    ``J_i`` predictor. ``ops``: named FIXED operators (S, H_in) fp64 (J_avg,
    M', identity handled via proj). ``proj``: (S, H_out) orthonormal output
    basis for the truth (None => S == H_out identity/full space).
    """
    n = len(ctxs)
    assert n >= 2, f"neighbor-delta read needs >=2 contexts, got {n}"
    c = torch.stack([x["c_last"] for x in ctxs]).to(torch.float64)  # (n, H_in)
    v = torch.stack([x["v"] for x in ctxs]).to(torch.float64)  # (n, H_out)
    dv = v[None, :, :] - v[:, None, :]  # (n, n, H_out)  row i -> deltas to all j
    dc = c[None, :, :] - c[:, None, :]  # (n, n, H_in)
    true = dv @ proj.T if proj is not None else dv  # (n, n, S)
    mask = ~torch.eye(n, dtype=torch.bool)
    out: dict = {"n_contexts": n, "n_ordered_pairs": int(mask.sum())}
    # per-context J_i predictions (batched over j per i)
    pred_own = torch.stack([dc[i] @ r_by_id[ctxs[i]["pair_id"]].T for i in range(n)])
    reads = {"J_i_own": pred_own}
    for name, op in ops.items():
        reads[name] = dc @ op.T
    if proj is not None:
        reads["identity_l14"] = dc @ proj.T  # v_hat_j - v_hat_i = c_j - c_i, projected
    else:
        reads["identity_l14"] = dc
    reads["zero"] = torch.zeros_like(true)
    per_ctx_own = []
    for i in range(n):
        sel = mask[i]
        per_ctx_own.append(_delta_r2(pred_own[i][sel], true[i][sel]))
    out["per_context_r2_J_i_own"] = per_ctx_own
    out["r2"] = {name: _delta_r2(p[mask], true[mask]) for name, p in reads.items()}
    if proj is not None:
        full_energy = float((dv[mask] ** 2).sum())
        out["subspace_energy_fraction"] = float((true[mask] ** 2).sum()) / max(full_energy, 1e-30)
    return out


def cmd_analyze(args) -> int:
    """Heterogeneity reads over the persisted per-context Jacobians -> JSON + figure."""
    assert not args.out.is_dir() and args.out.suffix == ".json", (
        f"--out must be the deliverable JSON FILE path, got {args.out}"
    )
    pcj_files = sorted((args.pcj_dir / "pcj").glob("*.pt"))
    assert pcj_files, f"no per-context files under {args.pcj_dir / 'pcj'}"
    ctxs = [torch.load(p, map_location="cpu", weights_only=True) for p in pcj_files]
    sd = torch.load(args.seeds_file, map_location="cpu", weights_only=True)
    seeds = sd["seeds"].to(torch.float64)  # (S, H_out)
    names = list(sd["names"])
    pc_idx = [i for i, nm in enumerate(names) if nm.startswith("vpc")]
    mp_idx = [i for i, nm in enumerate(names) if nm.startswith("mprime_u")]
    assert pc_idx and mp_idx, "seeds file lacks vpc/mprime blocks"

    javg = torch.load(args.javg_dir / "J_last.pt", map_location="cpu", weights_only=True)["J"].to(
        torch.float64
    )  # (H_out, H_in) full-rank averaged J (last arm)
    hidden = javg.shape[1]
    payload = torch.load(args.comparator, map_location="cpu", weights_only=True)
    a_op = (payload["W"].to(torch.float64) / payload["xsd"].to(torch.float64)[:, None]).T
    u_m, s_m, vh_m = torch.linalg.svd(a_op, full_matrices=False)
    k_dir = len(mp_idx)
    w20 = vh_m[:k_dir].T  # (H_in, k) orthonormal M' top right-singular (input) directions

    # restricted rows per context (sketch basis; full contexts restricted exactly)
    r_by_id: dict[str, torch.Tensor] = {}
    full_by_id: dict[str, torch.Tensor] = {}
    for x in ctxs:
        rows = x["rows"]["last"].to(torch.float64)
        if x["seed_mode"] == "full":
            full_by_id[x["pair_id"]] = rows
            r_by_id[x["pair_id"]] = seeds @ rows
        else:
            assert rows.shape[0] == seeds.shape[0], (rows.shape, seeds.shape)
            r_by_id[x["pair_id"]] = rows
    r_avg = seeds @ javg  # (S, H_in) — the production averaged J, sketch-restricted

    # Population classes: derive from `source` for units persisted BEFORE the
    # class field landed (retained pod units, crash-fix r15 resume contract).
    cls_arr = np.array(
        [x.get("context_class") or context_class_of(x.get("source", "unspecified")) for x in ctxs]
    )
    result: dict = {
        "n_contexts": len(ctxs),
        "n_full_rank": len(full_by_id),
        "population": {
            "context_classes": {
                str(cl): int((cls_arr == cl).sum()) for cl in sorted(set(cls_arr.tolist()))
            },
            "caveat": POPULATION_CAVEAT,
        },
        "per_context": [],
        "arms": {},
    }

    # per-arm restricted stats (last = primary; prefix/ctx secondary)
    stack = {a: [] for a in J76.ARMS}
    for x in ctxs:
        for a in J76.ARMS:
            rows = x["rows"][a].to(torch.float64)
            stack[a].append(seeds @ rows if x["seed_mode"] == "full" else rows)
    javg_arm = {
        a: torch.load(args.javg_dir / f"J_{a}.pt", map_location="cpu", weights_only=True)["J"].to(
            torch.float64
        )
        for a in J76.ARMS
    }
    for a in J76.ARMS:
        rs = torch.stack(stack[a])  # (n, S, H)
        ravg_a = seeds @ javg_arm[a]
        norms = rs.norm(dim=(1, 2))
        rbar = rs.mean(dim=0)
        flat = rs.reshape(rs.shape[0], -1).to(torch.float32)
        flat = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-30)
        gram = (flat @ flat.T).to(torch.float64).numpy()
        off = gram[~np.eye(gram.shape[0], dtype=bool)]
        cos_avg = np.array([_cosf(rs[i], ravg_a) for i in range(rs.shape[0])])
        result["arms"][a] = {
            "norm_mean": float(norms.mean()),
            "norm_q10_q90": [float(np.quantile(norms.numpy(), q)) for q in (0.1, 0.9)],
            "norm_J_avg_restricted": float(ravg_a.norm()),
            "cancellation_ratio_sample": float(rbar.norm() / norms.mean().clamp(min=1e-30)),
            "cos_to_J_avg_median": float(np.median(cos_avg)),
            "cos_to_J_avg_ci": _boot_ci(cos_avg, args.n_boot, 1776),
            "cos_to_J_avg_q10_q90": [float(np.quantile(cos_avg, q)) for q in (0.1, 0.9)],
            "pairwise_cos_median": float(np.median(off)),
            "pairwise_cos_q10_q90": [float(np.quantile(off, q)) for q in (0.1, 0.9)],
            "cos_to_J_avg_median_by_class": {
                str(cl): float(np.median(cos_avg[cls_arr == cl]))
                for cl in sorted(set(cls_arr.tolist()))
            },
        }
        if a == "last":
            cos_avg_last = cos_avg
            pairwise_last = off

    # M'-direction alignment per context: cos(u_k^T J_i, v_k) vs the averaged value
    align_avg = np.array([_cosf((seeds[mp_idx[k]] @ javg), vh_m[k]) for k in range(k_dir)])
    align_per_ctx = np.zeros((len(ctxs), k_dir))
    for i, x in enumerate(ctxs):
        r = r_by_id[x["pair_id"]]
        for k in range(k_dir):
            align_per_ctx[i, k] = _cosf(r[mp_idx[k]], vh_m[k])
    mean_align_ctx = align_per_ctx.mean(axis=1)

    # participation of M' top input directions in each J_i's row space
    part = np.array(
        [
            float((r_by_id[x["pair_id"]] @ w20).norm() ** 2 / r_by_id[x["pair_id"]].norm() ** 2)
            for x in ctxs
        ]
    )
    part_avg = float((r_avg @ w20).norm() ** 2 / r_avg.norm() ** 2)

    for i, x in enumerate(ctxs):
        result["per_context"].append(
            {
                "pair_id": x["pair_id"],
                "source": x["source"],
                "context_class": str(cls_arr[i]),
                "span_anchor": x.get("span_anchor", "find-legacy"),
                "stratum": x["stratum"],
                "seed_mode": x["seed_mode"],
                "norm_last": float(r_by_id[x["pair_id"]].norm()),
                "cos_to_J_avg_last": float(cos_avg_last[i]),
                "mprime_mean_alignment": float(mean_align_ctx[i]),
                "mprime_participation": float(part[i]),
                "ctx_maxabs": x["ctx_maxabs"],
                "unit_wall_s": x.get("unit_wall_s"),
            }
        )
    result["mprime_alignment"] = {
        "per_direction_averagedJ": [float(v) for v in align_avg],
        "per_direction_ctx_median": [float(np.median(align_per_ctx[:, k])) for k in range(k_dir)],
        "mean_alignment_ctx_median": float(np.median(mean_align_ctx)),
        "mean_alignment_ctx_ci": _boot_ci(mean_align_ctx, args.n_boot, 1777),
        "mean_alignment_averagedJ": float(align_avg.mean()),
        "participation_ctx_median": float(np.median(part)),
        "participation_averagedJ": part_avg,
    }

    # neighbor-delta reads (PC subspace, per corpus) + full space (pooled)
    u_pc = seeds[pc_idx]  # orthonormal rows (v-pool PCs)
    a_rows_pc = u_pc @ a_op  # M' restricted to the PC output basis
    javg_pc = u_pc @ javg
    nbr: dict = {}
    for corpus in ("lmsys", "wildchat"):
        sub = [x for x in ctxs if x["source"] == corpus]
        if len(sub) >= 2:
            r_pc = {
                x["pair_id"]: (
                    u_pc @ full_by_id[x["pair_id"]]
                    if x["seed_mode"] == "full"
                    else r_by_id[x["pair_id"]][pc_idx]
                )
                for x in sub
            }
            nbr[corpus] = _neighbor_reads(
                sub, r_pc, {"J_avg": javg_pc, "mprime_x50k": a_rows_pc}, u_pc
            )
    fulls = [x for x in ctxs if x["seed_mode"] == "full"]
    if len(fulls) >= 2:
        nbr["full_space_pooled"] = _neighbor_reads(
            fulls,
            full_by_id,
            {"J_avg": javg, "mprime_x50k": a_op},
            None,
        )
        nbr["full_space_pooled"]["note"] = "pooled across corpora (n_full per corpus is small)"
    result["neighbor_delta_r2"] = nbr

    # steering-direction agreement on the full-rank subset (input-side action)
    if fulls:
        import issue1776_phase3 as P3

        ns = argparse.Namespace(
            directions=list(P3.DEFAULT_DIRECTIONS),
            rb_dir=args.rb_dir,
            mprime_weights=args.comparator,
            source_layer=C76.SOURCE_LAYER,
            random_seed=1776,
        )
        bank, _prov = P3.load_directions(ns, hidden)
        steer: dict = {}
        for name, vec in bank.items():
            d = vec.to(torch.float64)
            p_avg = javg @ d
            p_m = (d / payload["xsd"].to(torch.float64)) @ payload["W"].to(torch.float64)
            cos_i_avg, cos_i_m, norm_i = [], [], []
            for x in fulls:
                p_i = full_by_id[x["pair_id"]] @ d
                cos_i_avg.append(_cosf(p_i, p_avg))
                cos_i_m.append(_cosf(p_i, p_m))
                norm_i.append(float(p_i.norm()))
            steer[name] = {
                "cos_Ji_vs_Javg_median": float(np.median(cos_i_avg)),
                "cos_Ji_vs_Javg_q10_q90": [float(np.quantile(cos_i_avg, q)) for q in (0.1, 0.9)],
                "cos_Ji_vs_mprime_median": float(np.median(cos_i_m)),
                "pred_norm_Ji_median": float(np.median(norm_i)),
                "pred_norm_Javg": float(p_avg.norm()),
                "pred_norm_mprime": float(p_m.norm()),
            }
        result["steering_direction_agreement"] = steer

    # sketch-restriction validity on the full subset: does the 256-seed
    # restriction preserve the per-context ordering of the headline stats?
    if len(fulls) >= 3:
        from scipy import stats as sps

        cos_full = [_cosf(full_by_id[x["pair_id"]], javg) for x in fulls]
        cos_restr = [
            float(cos_avg_last[[c["pair_id"] for c in ctxs].index(x["pair_id"])]) for x in fulls
        ]
        rho = sps.spearmanr(cos_full, cos_restr)
        result["sketch_restriction_validation"] = {
            "n": len(fulls),
            "spearman_cos_to_avg_full_vs_restricted": (
                float(rho.statistic) if np.isfinite(rho.statistic) else None
            ),
            "pvalue": float(rho.pvalue) if np.isfinite(rho.pvalue) else None,
        }

    # parity vs the build-time stored captures (informational; rig-captured wins)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=True)
    tid = {p: i for i, p in enumerate(tgt["pair_id"])}
    par = []
    for x in ctxs:
        i = tid.get(x["pair_id"])
        if i is None:
            continue
        cv = torch.nn.functional.cosine_similarity(
            x["c_last"].to(torch.float64), tgt["c14"][i].to(torch.float64), dim=0
        )
        vv = torch.nn.functional.cosine_similarity(
            x["v"].to(torch.float64), tgt["v19"][i].to(torch.float64), dim=0
        )
        par.append(min(float(cv), float(vv)))
    result["capture_parity_min_cos"] = {
        "min": float(np.min(par)) if par else None,
        "median": float(np.median(par)) if par else None,
        "n": len(par),
    }

    # decides-fork summary (the two headline reads, stated mechanically)
    result["decides"] = {
        "per_context_alignment_to_mprime_median": result["mprime_alignment"][
            "mean_alignment_ctx_median"
        ],
        "averaged_J_alignment_to_mprime": result["mprime_alignment"]["mean_alignment_averagedJ"],
        "cancellation_ratio_last": result["arms"]["last"]["cancellation_ratio_sample"],
        "pairwise_cos_last_median": result["arms"]["last"]["pairwise_cos_median"],
        "reading": (
            "heterogeneous-causal requires per-context alignment >> averaged alignment "
            "AND low pairwise cos / cancellation; correlational-M' reads per-context "
            "alignment ~ averaged (both weak)"
        ),
    }
    result["repro"] = C76.repro_meta()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(args.out, result)

    _hetero_figure(result, cos_avg_last, pairwise_last, mean_align_ctx, ctxs, args.fig_dir)
    print(f"[p3p4-analyze] [phase=analyze_done] n={len(ctxs)} -> {args.out}", flush=True)
    return 0


def _hetero_figure(result, cos_avg_last, pairwise_last, mean_align_ctx, ctxs, fig_dir: Path):
    """Per-context distribution figure (deliverable): 3 panels, paper style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(4)
    src = np.array([x["source"] for x in ctxs])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ax = axes[0]
    for k, corpus in enumerate(("lmsys", "wildchat")):
        vals = cos_avg_last[src == corpus]
        if vals.size:
            ax.hist(vals, bins=20, alpha=0.65, label=corpus, color=colors[k])
    ax.hist(pairwise_last, bins=30, alpha=0.35, label="pairwise cos(J_i, J_j)", color=colors[2])
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("Frobenius cosine")
    ax.set_ylabel("contexts / pairs")
    ax.set_title("cos(J_i, J_avg) per context (last arm)")
    ax.legend(fontsize=7)

    ax = axes[1]
    avg_val = result["mprime_alignment"]["mean_alignment_averagedJ"]
    for k, corpus in enumerate(("lmsys", "wildchat")):
        vals = mean_align_ctx[src == corpus]
        strat = np.array([x["stratum"] for x in ctxs])[src == corpus]
        if vals.size:
            ax.scatter(strat + (k - 0.5) * 0.15, vals, s=14, color=colors[k], label=corpus)
    ax.axhline(avg_val, color="0.2", ls="--", lw=1, label="averaged J")
    ax.set_xlabel("M' error stratum (quantile bin)")
    ax.set_ylabel("mean cos(u_k^T J_i, v_k)")
    ax.set_title("per-context alignment with M'")
    ax.legend(fontsize=7)

    ax = axes[2]
    nbr = result.get("neighbor_delta_r2", {})
    labels, series = [], {}
    for corpus in ("lmsys", "wildchat"):
        if corpus in nbr:
            labels.append(corpus)
            for name, r2 in nbr[corpus]["r2"].items():
                series.setdefault(name, []).append(r2)
    xs = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for k, (name, vals) in enumerate(sorted(series.items())):
        ax.bar(xs + k * width, vals, width, label=name, color=paper_palette(len(series))[k])
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel("neighbor-delta R^2 (PC subspace)")
    ax.set_title("delta prediction by operator")
    ax.legend(fontsize=6)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "jacobian_heterogeneity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[p3p4-analyze] figure -> {out}", flush=True)


# ── sentinel helpers (committed, lintable — no dispatcher heredocs) ───────────


def cmd_progress(args) -> int:
    """Non-blocking tick sentinel (pod-side-reporting schema)."""
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": C76.ISSUE,
        "gate": args.gate,
        "blocks_pipeline": False,
        "by": "issue1776_p3p4_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": {"msg": args.msg, "mode": args.mode},
    }
    slug = args.gate.replace(":", "_").replace("/", "_")
    path = log_dir / f"issue-{C76.ISSUE}-{slug}-{int(time.time() * 1000)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
    print(f"[p3p4-dispatch] progress sentinel: {path.name}")
    return 0


def cmd_final_sentinel(args) -> int:
    """Terminal results sentinel for the follow-up round (epm:results / smoke)."""
    import subprocess

    smoke_like = args.dry or args.mode == "smoke"
    kind = "epm:smoke-result" if smoke_like else "epm:results"
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=args.repo_root,
            check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    eval_dir = Path(args.eval_dir)
    eval_paths = sorted(
        str(p.relative_to(args.repo_root)) if p.is_relative_to(args.repo_root) else str(p)
        for p in eval_dir.rglob("*.json")
    )
    note = {
        "followup_label": "p3p4_heterogeneity_dose",
        "mode": args.mode,
        "dry_run": args.dry,
        "ngpu": args.ngpu,
        "git_commit": sha,
        "eval_json_paths": eval_paths,
        "hf_prefixes": {
            "pcj_tensors": f"{args.hf_prefix}/analysis_tensors/{FU}",
            "raw_completions_dose": f"{args.hf_prefix}/raw_completions/steered_dose",
        },
        "offpod_handoffs": {
            "p6_judge_dose": (
                "OFF-POD (VM, Batch API): uv run python scripts/issue1776_judge.py "
                f"--raw-dir <staged {args.hf_prefix}/raw_completions/steered_dose> "
                f"--out-dir eval_results/issue_1776/{FU}/judge "
                "(same plan-priced contrast policy as phase 3; the dose manifest is "
                f"eval_results/issue_1776/{FU}/raw_completions_manifest.json)"
            ),
        },
        "wandb": "n/a (no training this round)",
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": C76.ISSUE,
        "gate": "smoke" if smoke_like else "results",
        "blocks_pipeline": not smoke_like,
        "by": "issue1776_p3p4_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"issue-{C76.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
    print(f"[p3p4-dispatch] final sentinel: {path.name} kind={kind}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _model(p):
        p.add_argument("--model", default=C.DEFAULT_MODEL)
        p.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
        p.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed-chunk", type=int, default=32)
        p.add_argument("--serial-grads", action="store_true")
        p.add_argument("--tiny", action="store_true", help="from-config tiny Qwen2 (CPU smoke)")

    s = sub.add_parser("stage", help="stage all P3/P4 inputs at one fresh pin")
    s.add_argument("--dest", type=Path, default=C76.DATA_DIR / "hf_dl")
    s.add_argument("--pin-file", type=Path, default=C76.DATA_DIR / "data_repo_pin_p3p4.json")
    s.add_argument("--refresh-pin", action="store_true")
    s.add_argument("--max-wc-chunks", type=int, default=0, help="smoke cap (0 = all)")
    s.add_argument("--report", type=Path, required=True)
    s.set_defaults(fn=cmd_stage)

    b = sub.add_parser("build", help="stratified context sample + targets + P4 ladder")
    b.add_argument("--dest", type=Path, default=C76.DATA_DIR / "hf_dl")
    b.add_argument("--comparator", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--n-sketch", type=int, default=96, help="total sampled contexts (both corpora)")
    b.add_argument("--n-full", type=int, default=16, help="full-rank subset (both corpora)")
    b.add_argument("--strata", type=int, default=4)
    b.add_argument("--seed", type=int, default=1776)
    b.set_defaults(fn=cmd_build)

    pi = sub.add_parser("pilot", help="measured 1-context wall + rc=7 over-2x gate")
    _model(pi)
    pi.add_argument("--pairs", type=Path, required=True)
    pi.add_argument("--seeds-file", type=Path, required=True)
    pi.add_argument("--budget-gpu-h", type=float, default=3.5)
    pi.add_argument("--ngpu", type=int, default=8)
    pi.add_argument("--out", type=Path, required=True)
    pi.set_defaults(fn=cmd_pilot)

    r = sub.add_parser("run", help="per-context Jacobian sweep (one shard)")
    _model(r)
    r.add_argument("--pairs", type=Path, required=True)
    r.add_argument("--seeds-file", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    an = sub.add_parser("analyze", help="heterogeneity reads -> deliverable JSON + figure")
    an.add_argument("--pcj-dir", type=Path, required=True, help="run out-dir (holds pcj/)")
    an.add_argument("--targets", type=Path, required=True, help="pcj_targets.pt")
    an.add_argument("--seeds-file", type=Path, required=True)
    an.add_argument("--javg-dir", type=Path, required=True, help="staged jac_full dir")
    an.add_argument("--comparator", type=Path, required=True)
    an.add_argument("--rb-dir", type=Path, required=True)
    an.add_argument("--n-boot", type=int, default=1000)
    an.add_argument("--out", type=Path, required=True, help="jacobian_heterogeneity.json FILE")
    an.add_argument("--fig-dir", type=Path, required=True, help="figure output DIRECTORY")
    an.set_defaults(fn=cmd_analyze)

    pr = sub.add_parser("progress", help="tick sentinel writer")
    pr.add_argument("--log-dir", required=True)
    pr.add_argument("--gate", required=True)
    pr.add_argument("--msg", required=True)
    pr.add_argument("--mode", default="?")
    pr.set_defaults(fn=cmd_progress)

    fs = sub.add_parser("final-sentinel", help="terminal results sentinel writer")
    fs.add_argument("--log-dir", required=True)
    fs.add_argument("--mode", required=True)
    fs.add_argument("--dry", action="store_true")
    fs.add_argument("--eval-dir", required=True)
    fs.add_argument("--repo-root", type=Path, required=True)
    fs.add_argument("--hf-prefix", required=True)
    fs.add_argument("--ngpu", default="?")
    fs.set_defaults(fn=cmd_final_sentinel)

    ic = sub.add_parser("import-check", help="resolve deferred imports (Axis-1 smoke leg)")
    ic.set_defaults(fn=None)

    args = ap.parse_args(argv)
    if args.cmd == "import-check":
        # Execute every deferred/function-body import this module reaches on
        # its real code paths (the #1689 rounds-2/3/4 false-pass class).
        from scipy import stats as sps  # noqa: F401

        import issue779_ffc_n1m_fits as N1M  # noqa: F401
        import issue779_ffc_n1m_generate_capture as N1G
        import issue1776_phase3 as P3

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            set_paper_style,
        )
        from explore_persona_space.orchestrate import hub

        for sym in (
            hub.stage_hub_file,
            hub.stage_hub_prefix,
            hub.retry_transient,
            N1G.N50._remote_index,
            N1M.apply_map,
            P3.load_directions,
            J76.render_pair,
            J76.load_pairs,
            J76.load_model,
            J76.JacobianEstimator,
        ):
            assert callable(sym), sym
        print("[p3p4] import-check OK")
        return 0
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
