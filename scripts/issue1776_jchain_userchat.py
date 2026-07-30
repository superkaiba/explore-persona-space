"""#1776 inline follow-up: chain-composition (5d) read for the JACOBIAN arms.

The committed 5d read scores only the FITTED maps (``chain_composition.json``
variant ``mprime_x50k``; ``chain_composition_shipped.json`` variant
``m_shipped``). This driver extends the SAME judge-free DV to the anchored
first-order causal prediction

    v̂(C) = ymu19 + (x_last^{(14)}(C) - xmu14) @ Jᵀ

for each averaged Jacobian arm (``J_last`` / ``J_ctx`` / ``J_prefix``), plus the
``J_last`` train-fit AFFINE rescale ``s_a * u + b + ymu19`` (the exact
``issue1776_9ater_followup.cmd_p1`` rung, with ``s_a`` / ``||b||`` re-asserted
against the committed ``jacobian_rescale.json`` scalars).

Decode / rank / null machinery is REUSED, not re-derived:
  * ``issue1776_phase5.load_chain_rows`` / ``content_token_ids`` /
    ``_chain_metrics`` (the 200-draw shuffled-pairing null, selection per draw);
  * ``issue1776_9ater_followup._rank_matrix`` (the cmd_chain rank expressions)
    and ``_percontext_block`` (per-context reciprocal rank vs each context's own
    998-row null, head-heaviness shares, MRR-excluding-top-k);
  * ``issue1776_phase4.load_dict`` for the vendored layer-19 lens dictionary.

A REPRODUCTION GATE runs first: ``mprime_x50k`` must reproduce the committed
aggregate MRR to ``MRR_TOL`` or the run stops before any J number is written.

NEW here: the ECHO SPLIT — each answer's content tokens are partitioned into
PROMPT-ECHOED (token id also present in that context's prompt) vs NON-ECHOED,
and the MRR is reported restricted to each partition (registered prediction: a
J-chain gain concentrates in the echoed partition). Computed for every variant
INCLUDING ``mprime_x50k`` so the split has a fitted-map comparator.

Content hygiene: WildChat prompts / model responses are NEVER printed or
logged — logs and outputs carry counts, token ids, and ranks only.

Usage (thread caps + MALLOC_ARENA_MAX=2 belong on the launch env):
  uv run python scripts/issue1776_jchain_userchat.py --data-root <dir>
  uv run python scripts/issue1776_jchain_userchat.py --data-root <dir> \
      --variants J_last_anchored --max-ctx 100 --n-null 20 --no-figs   # pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402

JAC = "issue1776_jacobian/analysis_tensors"
OUT_DIR = C76.PROJECT_ROOT / "eval_results" / f"issue_{C76.ISSUE}" / "followup_userchat_jchain"
FIG_DIR = C76.PROJECT_ROOT / "figures" / f"issue_{C76.ISSUE}" / "followup_userchat_jchain"
PHASE5_DIR = C76.PROJECT_ROOT / "eval_results" / f"issue_{C76.ISSUE}" / "phase5"
CHAIN_JSON = PHASE5_DIR / "chain_composition.json"
CHAIN_SHIPPED_JSON = PHASE5_DIR / "chain_composition_shipped.json"
RESCALE_JSON = (
    C76.PROJECT_ROOT
    / "eval_results"
    / f"issue_{C76.ISSUE}"
    / "followup_9ater"
    / "jacobian_rescale.json"
)

# Reproduction gate: the committed mprime_x50k MRR must come back bit-close.
MRR_TOL = 1e-9
# The 9ater r1-train affine fit must re-derive to these committed scalars.
AFFINE_S_TOL = 1e-9
AFFINE_B_NORM_TOL = 1e-6

VARIANTS: tuple[str, ...] = (
    "mprime_x50k",
    "J_last_anchored",
    "J_last_affine",
    "J_ctx_anchored",
    "J_prefix_anchored",
)


def _paths(data_root: Path) -> dict[str, Path]:
    """Resolve every input artifact under a staged ``data/issue_1776`` root."""
    hf = data_root / "hf_dl"
    return {
        "chunks": data_root / "chain_chunks",
        "dict": hf / JAC / "dictionaries" / "dictionary_l19.pt",
        "mprime": hf / JAC / "comparator" / "m_ridge_x50k.pt",
        "jac_dir": hf / JAC / "jac_full",
        "anchors": data_root / "9ater_anchors.pt",
        "pass_b": hf / C76.PASS_B_HF_PATH,
    }


def load_chain_rows_with_prompts(
    chunks_dir: Path, in_layer: int, max_ctx: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """``phase5.load_chain_rows`` + the per-row PROMPT, same (chunk, ci) join.

    The returned ``(x, responses)`` are asserted EQUAL to
    ``phase5.load_chain_rows``'s own output, so the prompt column cannot drift
    from the rows the committed read scored.
    """
    import issue1776_phase5 as P5

    xs: list[torch.Tensor] = []
    resp: list[str] = []
    prompts: list[str] = []
    files = sorted(chunks_dir.glob("shard*_chunk*.pt"))
    assert files, f"no capture chunks under {chunks_dir}"
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        li = layers.index(in_layer)
        raw = json.loads((f.parent / (f.name.removesuffix(".pt") + ".json")).read_text())
        by_ci = {int(r["ci"]): (r["response"], r["prompt"]) for r in raw["rows"]}
        for row_i, ci in enumerate(int(c) for c in d["ci"]):
            assert ci in by_ci, (f.name, ci)
            xs.append(d["cx_last"][row_i, li, :].to(torch.float32))
            resp.append(by_ci[ci][0])
            prompts.append(by_ci[ci][1])
        if max_ctx and len(resp) >= max_ctx:
            break
    if max_ctx:
        xs, resp, prompts = xs[:max_ctx], resp[:max_ctx], prompts[:max_ctx]
    x = torch.stack(xs).numpy()

    x_ref, resp_ref = P5.load_chain_rows(chunks_dir, in_layer, max_ctx)
    assert np.array_equal(x, x_ref), (x.shape, x_ref.shape)
    assert resp == resp_ref, (len(resp), len(resp_ref))
    return x, resp, prompts


def prompt_token_id_sets(tok, prompts: list[str]) -> list[set[int]]:
    """Per-context prompt token-id SET (no special tokens), for the echo split."""
    return [set(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]


def echo_partition(
    content: list[np.ndarray], prompt_ids: list[set[int]], uidx: dict[int, int]
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Split each context's content ids into (echoed, non-echoed) union indices."""
    echoed: list[np.ndarray] = []
    other: list[np.ndarray] = []
    n_e, n_o = [], []
    for ids, pset in zip(content, prompt_ids, strict=True):
        e = [uidx[int(t)] for t in ids if int(t) in pset]
        o = [uidx[int(t)] for t in ids if int(t) not in pset]
        echoed.append(np.array(sorted(e), dtype=np.int64))
        other.append(np.array(sorted(o), dtype=np.int64))
        n_e.append(len(e))
        n_o.append(len(o))
    stats = {
        "n_content_tokens_total": int(sum(int(c.size) for c in content)),
        "n_echoed_total": int(sum(n_e)),
        "n_non_echoed_total": int(sum(n_o)),
        "mean_echoed_per_ctx": float(np.mean(n_e)),
        "mean_non_echoed_per_ctx": float(np.mean(n_o)),
        "n_ctx_with_echoed": int(sum(1 for v in n_e if v)),
        "n_ctx_with_non_echoed": int(sum(1 for v in n_o if v)),
    }
    return echoed, other, stats


def _partition_mrr(ranks_sub: np.ndarray, ids_list: list[np.ndarray], perm: np.ndarray):
    """Mean best-token reciprocal rank over the contexts with a non-empty
    partition, under pairing ``perm`` (v̂ row perm[i] vs context i's ids)."""
    tot, n_used = 0.0, 0
    for i, ids in enumerate(ids_list):
        if ids.size == 0:
            continue
        tot += 1.0 / float(ranks_sub[perm[i], ids].min())
        n_used += 1
    return (tot / n_used if n_used else float("nan")), n_used


def _partition_block(
    ranks_sub: np.ndarray, ids_list: list[np.ndarray], n_null: int, seed: int
) -> dict:
    n = ranks_sub.shape[0]
    mrr, n_used = _partition_mrr(ranks_sub, ids_list, np.arange(n))
    rng = np.random.default_rng(seed)
    draws = [_partition_mrr(ranks_sub, ids_list, rng.permutation(n))[0] for _ in range(n_null)]
    return {
        "mrr": mrr,
        "n_ctx_used": n_used,
        "null_mrr_mean": float(np.mean(draws)) if n_null else None,
        "null_mrr_p975": float(np.percentile(draws, 97.5)) if n_null else None,
    }


def _affine_rescale(paths: dict[str, Path], j: np.ndarray, anc: dict) -> tuple[float, np.ndarray]:
    """Re-derive the 9ater r1-train AFFINE fit ``(s_a, b)`` for ``J_last``.

    Exactly ``issue1776_9ater_followup.cmd_p1``'s rung:
      u = (x14 - xmu14) @ Jᵀ ; r = y19 - ymu19 (both on the r1-train rows)
      s_a = argmin_s ||s(u - ū) - (r - r̄)||² ; b = r̄ - s_a·ū
    Re-asserted against the committed ``jacobian_rescale.json`` scalars.
    """
    import issue1776_9ater_followup as F9
    import issue779_ffc_n50k_fits as N50
    import issue779_fitter_fair_comparison as F

    committed = json.loads(RESCALE_JSON.read_text())["scalars"]["J_last"]
    assert paths["pass_b"].exists(), f"missing pass_b store: {paths['pass_b']}"
    pb = F._mmap_load(paths["pass_b"])
    x14 = N50._slice_layer(pb, "cx_last", C76.SOURCE_LAYER).astype(np.float64)
    y19 = N50._slice_layer(pb, "v_x", C76.READOUT_LAYER).astype(np.float64)
    r1_train, _val, _te, _split = F9._pinned_split()
    xmu14 = anc["xmu14"].numpy()
    ymu19 = anc["ymu19"].numpy()
    u_tr = (x14[r1_train] - xmu14) @ j.T
    r_tr = y19[r1_train] - ymu19
    ubar, rbar = u_tr.mean(0), r_tr.mean(0)
    s_a = F9._fit_scalar(u_tr - ubar, r_tr - rbar)
    b = rbar - s_a * ubar
    b_norm = float(np.linalg.norm(b))
    ds = abs(s_a - committed["s_affine_r1_train"])
    db = abs(b_norm - committed["b_norm_r1_train"])
    assert ds <= AFFINE_S_TOL, (s_a, committed["s_affine_r1_train"], ds)
    assert db <= AFFINE_B_NORM_TOL, (b_norm, committed["b_norm_r1_train"], db)
    print(
        f"[jchain] affine refit OK: s_a={s_a:.12f} (dif {ds:.2e}) "
        f"||b||={b_norm:.6f} (dif {db:.2e})",
        flush=True,
    )
    del pb, x14, y19
    return s_a, b


def build_vhat(
    name: str, x14: np.ndarray, paths: dict[str, Path], anc: dict | None
) -> tuple[np.ndarray, dict]:
    """v̂ (n, H) float32 for one variant + its provenance block."""
    import issue1776_9ater_followup as F9
    import issue779_ffc_n1m_fits as N1M

    xf = x14.astype(np.float64)
    if name == "mprime_x50k":
        payload = torch.load(paths["mprime"], map_location="cpu", weights_only=True)
        v = N1M.apply_map(payload, xf, torch.device("cpu"))
        return v.astype(np.float32), {
            "form": "fitted ridge M' applied via issue779_ffc_n1m_fits.apply_map",
            "operator": str(paths["mprime"]),
            "input_layer": C76.SOURCE_LAYER,
        }

    assert anc is not None, "J arms need the recovered anchors"
    jname = {
        "J_last_anchored": "J_last",
        "J_last_affine": "J_last",
        "J_ctx_anchored": "J_ctx",
        "J_prefix_anchored": "J_prefix",
    }[name]
    j = F9._load_j(paths["jac_dir"] / f"{jname}.pt")
    xmu14 = anc["xmu14"].numpy()
    ymu19 = anc["ymu19"].numpy()
    u = (xf - xmu14) @ j.T
    prov = {
        "operator": str(paths["jac_dir"] / f"{jname}.pt"),
        "input_layer": C76.SOURCE_LAYER,
        "anchors": str(paths["anchors"]),
        "note": (
            "all three J arms consume the SAME input summary (cx_last at layer 14) "
            "and the SAME anchors; the arms differ only in the averaged Jacobian "
            "tensor (the differentiation span), exactly as in 9ater cmd_p1"
        ),
    }
    if name == "J_last_affine":
        s_a, b = _affine_rescale(paths, j, anc)
        prov["form"] = "s_a * (x - xmu14) @ J_lastᵀ + b + ymu19 (9ater r1-train affine fit)"
        prov["s_affine_r1_train"] = s_a
        prov["b_norm_r1_train"] = float(np.linalg.norm(b))
        return (s_a * u + b + ymu19).astype(np.float32), prov
    prov["form"] = f"(x - xmu14) @ {jname}ᵀ + ymu19 (anchored first-order prediction)"
    return (u + ymu19).astype(np.float32), prov


def run_variant(name: str, vhat: np.ndarray, ctx: dict, args) -> tuple[dict, dict]:
    """Score one variant through the committed decode/rank/null machinery."""
    import issue1776_9ater_followup as F9
    import issue1776_phase5 as P5

    ranks_sub = F9._rank_matrix(
        vhat, ctx["rows_unit"], ctx["norms"], ctx["union"], args.score_batch
    )
    n = vhat.shape[0]
    ident = np.arange(n)
    mrr, rec, n_used = P5._chain_metrics(ranks_sub, ctx["sub_ids"], ident, args.topk)
    rng = np.random.default_rng(args.seed)
    null_mrr, null_rec = [], []
    for _ in range(args.n_null):
        m, r0, _ = P5._chain_metrics(ranks_sub, ctx["sub_ids"], rng.permutation(n), args.topk)
        null_mrr.append(m)
        null_rec.append(r0)
    block = F9._percontext_block(ranks_sub, ctx["sub_ids"])
    fig_payload = {"best_diag": block.pop("_best_diag"), "best_off": block.pop("_best_off")}
    out = {
        "variant": name,
        "input_layer": C76.SOURCE_LAYER,
        "n_ctx": int(n),
        "n_ctx_used": int(n_used),
        "df_cap": args.df_cap,
        "topk": args.topk,
        "mrr": mrr,
        "recall_at_k": rec,
        "null": {
            "n_draws": args.n_null,
            "mrr_mean": float(np.mean(null_mrr)) if args.n_null else None,
            "mrr_p975": float(np.percentile(null_mrr, 97.5)) if args.n_null else None,
            "recall_mean": float(np.mean(null_rec)) if args.n_null else None,
            "recall_p975": float(np.percentile(null_rec, 97.5)) if args.n_null else None,
        },
        "per_context": block,
        "echo_split": {
            "stats": ctx["echo_stats"],
            "echoed": _partition_block(ranks_sub, ctx["echoed"], args.n_null, args.seed),
            "non_echoed": _partition_block(ranks_sub, ctx["other"], args.n_null, args.seed),
        },
        "repro": C76.repro_meta(),
    }
    print(
        f"[jchain] [phase=variant_done variant={name}] mrr={mrr:.6f} "
        f"(null {out['null']['mrr_mean']}) recall@{args.topk}={rec:.6f} "
        f"echoed={out['echo_split']['echoed']['mrr']:.6f} "
        f"non_echoed={out['echo_split']['non_echoed']['mrr']:.6f}",
        flush=True,
    )
    return out, fig_payload


def _fig_bars(summary: dict, out_dir: Path) -> dict:
    """MRR per variant with the shuffled-pairing null mean + p97.5 band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    rows = list(summary["variants"].items())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    panels = (
        (axes[0], rows, "All variants (fitted maps set the scale)"),
        (axes[1], [r for r in rows if r[0].startswith("J_")], "Jacobian arms only (zoom)"),
    )
    for ax, sel, title in panels:
        labels = [k for k, _ in sel]
        xs = np.arange(len(labels))
        colors = [
            pp.paper_palette_role("primary")
            if k.startswith(("mprime", "m_shipped"))
            else pp.paper_palette_role("accent")
            for k in labels
        ]
        ax.bar(
            xs,
            [v["mrr"] for _, v in sel],
            color=colors,
            width=0.62,
            label="Observed MRR (true pairing)",
        )
        for i, (_, v) in enumerate(sel):
            lo, hi = v["null"]["mrr_mean"], v["null"]["mrr_p975"]
            ax.plot(
                [xs[i] - 0.36, xs[i] + 0.36],
                [lo, lo],
                color="0.25",
                lw=1.6,
                label="Shuffled-pairing null mean" if i == 0 else None,
            )
            ax.fill_between(
                [xs[i] - 0.36, xs[i] + 0.36],
                lo,
                hi,
                color="0.55",
                alpha=0.35,
                label="Null mean to p97.5" if i == 0 else None,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title(title)
        ax.set_ylabel("MRR of the answer's best content token")
    axes[0].legend()
    paths = pp.savefig_paper(fig, "jchain_mrr_bars", dir=out_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def _fig_hist(name: str, payload: dict, out_dir: Path) -> dict:
    """Per-context best-content-token log10-rank density, true vs pooled null."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    diag = np.log10(payload["best_diag"].astype(np.float64))
    off = np.log10(payload["best_off"].astype(np.float64))
    fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
    bins = np.linspace(0.0, max(diag.max(), off.max()) + 0.1, 45)
    ax.hist(
        off,
        bins=bins,
        density=True,
        color=pp.paper_palette_role("neutral"),
        alpha=0.55,
        label="Shuffled pairing (each context's own null rows)",
    )
    ax.hist(
        diag,
        bins=bins,
        density=True,
        color=pp.paper_palette_role("primary"),
        histtype="step",
        lw=2.0,
        label=f"True pairing (v-hat from {name})",
    )
    ax.set_xlabel("log10 rank of the best content token (per context)")
    ax.set_ylabel("Density")
    ax.legend()
    paths = pp.savefig_paper(fig, "jchain_percontext_hist", dir=out_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=C76.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--max-ctx", type=int, default=0)
    ap.add_argument("--df-cap", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-batch", type=int, default=64)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--hist-variant", default=None, help="variant for the per-context hist")
    args = ap.parse_args(argv)

    import issue1776_phase4 as P4
    import issue1776_phase5 as P5  # noqa: F401  (machinery reused via helpers)

    from transformers import AutoTokenizer

    bad = [v for v in args.variants if v not in VARIANTS]
    assert not bad, f"unknown variants {bad}; known: {VARIANTS}"
    paths = _paths(args.data_root)
    for k in ("chunks", "dict", "mprime", "jac_dir"):
        assert paths[k].exists(), f"missing input {k}: {paths[k]}"
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    x14, responses, prompts = load_chain_rows_with_prompts(
        paths["chunks"], C76.SOURCE_LAYER, args.max_ctx
    )
    print(f"[jchain] rows: n_ctx={x14.shape[0]} H={x14.shape[1]}", flush=True)

    d = P4.load_dict(paths["dict"], "cpu")
    rows_unit = d["rows_unit"].to(torch.float32)
    norms = d["row_norms"].to(torch.float32)

    content = P5.content_token_ids(tok, responses, args.df_cap)
    union = np.unique(np.concatenate([c for c in content if c.size]))
    uidx = {int(t): k for k, t in enumerate(union)}
    sub_ids = [np.array([uidx[int(t)] for t in c], dtype=np.int64) for c in content]
    echoed, other, echo_stats = echo_partition(content, prompt_token_id_sets(tok, prompts), uidx)
    print(
        f"[jchain] content tokens: union={union.size} total={echo_stats['n_content_tokens_total']} "
        f"echoed={echo_stats['n_echoed_total']} non_echoed={echo_stats['n_non_echoed_total']}",
        flush=True,
    )
    ctx = {
        "rows_unit": rows_unit,
        "norms": norms,
        "union": union,
        "sub_ids": sub_ids,
        "echoed": echoed,
        "other": other,
        "echo_stats": echo_stats,
    }

    anc = None
    if any(v != "mprime_x50k" for v in args.variants):
        assert paths["anchors"].exists(), f"missing recovered anchors: {paths['anchors']}"
        anc = torch.load(paths["anchors"], map_location="cpu", weights_only=True)

    # Reproduction gate FIRST (full-n runs only; a --max-ctx pilot cannot match).
    ordered = [v for v in VARIANTS if v in args.variants]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    figs: dict[str, dict] = {}
    gate = {"applied": False}
    for name in ordered:
        vhat, prov = build_vhat(name, x14, paths, anc)
        res, fig_payload = run_variant(name, vhat, ctx, args)
        res["provenance"] = prov
        if name == "mprime_x50k":
            committed = json.loads(CHAIN_JSON.read_text())
            diff = abs(res["mrr"] - committed["mrr"])
            gate = {
                "applied": args.max_ctx == 0,
                "variant": name,
                "reproduced_mrr": res["mrr"],
                "committed_mrr": committed["mrr"],
                "abs_diff": diff,
                "tol": MRR_TOL,
                "committed_n_ctx": committed["n_ctx"],
            }
            if args.max_ctx == 0:
                assert res["n_ctx"] == committed["n_ctx"], (res["n_ctx"], committed["n_ctx"])
                assert diff <= MRR_TOL, (
                    f"REPRODUCTION GATE FAILED: mprime_x50k MRR {res['mrr']!r} vs committed "
                    f"{committed['mrr']!r} (diff {diff:.3e} > {MRR_TOL}) — J numbers are NOT "
                    "real; stopping"
                )
                print(f"[jchain] reproduction gate PASS (diff {diff:.3e})", flush=True)
            else:
                print(f"[jchain] PILOT (max_ctx={args.max_ctx}): gate NOT applied", flush=True)
        res["reproduction_gate"] = gate if name == "mprime_x50k" else None
        results[name] = res
        figs[name] = fig_payload
        C76.atomic_write_json(args.out_dir / f"chain_composition_{name}.json", res)
        del vhat

    shipped = json.loads(CHAIN_SHIPPED_JSON.read_text())
    summary = {
        "dv": (
            "MRR / recall@50 of the generated answer's content tokens in the lens-decoded "
            "vocab ranking of v̂(C), vs a 200-draw shuffled-pairing null (5d chain composition)"
        ),
        "question": (
            "does the causally transported context signal (anchored Jacobian) carry the "
            "answer's actual words, as the fitted maps do?"
        ),
        "n_ctx": int(x14.shape[0]),
        "max_ctx": args.max_ctx,
        "df_cap": args.df_cap,
        "topk": args.topk,
        "n_null": args.n_null,
        "seed": args.seed,
        "reproduction_gate": gate,
        "variants": results,
        "committed_reference": {
            "mprime_x50k": {
                "path": str(CHAIN_JSON),
                "mrr": json.loads(CHAIN_JSON.read_text())["mrr"],
                "null_mrr_mean": json.loads(CHAIN_JSON.read_text())["null"]["mrr_mean"],
                "recall_at_k": json.loads(CHAIN_JSON.read_text())["recall_at_k"],
            },
            "m_shipped": {
                "path": str(CHAIN_SHIPPED_JSON),
                "mrr": shipped["mrr"],
                "null_mrr_mean": shipped["null"]["mrr_mean"],
                "null_mrr_p975": shipped["null"]["mrr_p975"],
                "recall_at_k": shipped["recall_at_k"],
                "input_layer": shipped["input_layer"],
                "note": "cited from disk (committed round); NOT recomputed here",
            },
        },
        "inputs": {k: str(v) for k, v in paths.items()},
        "wall_seconds": round(time.time() - t0, 1),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / "summary.json", summary)

    if not args.no_figs:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        hist_name = args.hist_variant or max(
            (v for v in results if v != "mprime_x50k"),
            key=lambda k: results[k]["mrr"],
            default="mprime_x50k",
        )
        bar_summary = {"variants": dict(results)}
        for k, ref in summary["committed_reference"].items():
            if k not in bar_summary["variants"]:
                bar_summary["variants"][k] = {
                    "mrr": ref["mrr"],
                    "null": {
                        "mrr_mean": ref["null_mrr_mean"],
                        "mrr_p975": ref.get("null_mrr_p975", ref["null_mrr_mean"]),
                    },
                }
        cap = {
            "jchain_mrr_bars": (
                "MRR of the generated answer's best content token in the layer-19 "
                "lens-decoded vocab ranking of each variant's predicted answer profile "
                f"v-hat(C), over {summary['n_ctx']} fresh-WildChat contexts. Left panel: "
                "all variants (the fitted maps set the y-scale). Right panel: the same "
                "quantity zoomed to the Jacobian arms. Grey line = mean of "
                f"{args.n_null} shuffled-pairing null draws (re-paired per draw); grey "
                "band = null mean to p97.5. m_shipped is the committed round's value + "
                "null cited from disk, NOT recomputed here."
            ),
            "jchain_percontext_hist": (
                f"Per-context log10 rank of the best content token for {hist_name} "
                "(true pairing, outline) against every context's shuffled-pairing null "
                "rows (filled). Right-shifted true-pairing mass would indicate the "
                "transported signal carries the answer's words."
            ),
        }
        figs_written = {"jchain_mrr_bars": _fig_bars(bar_summary, args.fig_dir)}
        figs_written["jchain_percontext_hist"] = _fig_hist(hist_name, figs[hist_name], args.fig_dir)
        C76.atomic_write_json(
            args.fig_dir / "captions.json",
            {
                "captions": cap,
                "files": figs_written,
                "hist_variant": hist_name,
                "repro": C76.repro_meta(),
            },
        )
        print(f"[jchain] figures -> {args.fig_dir}", flush=True)

    print(
        f"[jchain] [phase=done] variants={len(results)} wall={summary['wall_seconds']}s "
        f"-> {args.out_dir / 'summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit BEFORE C-extension finalize teardown (#1689 class)
