"""Self-contained HTML dashboard for #2061 per-feature SAE predictability deltas.

User-chat inline free-analysis round (2026-08-07, ask: "make a dashboard with
most increased SAE feature predictability at each stage compared to last stage
(and also most reduced/least improved). Also a distribution of increase/decrease
in predictability over SAE feature at each stage").

Reads EXISTING #2061 artifacts only:
  - per-feature held-out R2 JSONLs (70 = 5 stages x 7 render/corpus combos x 2 arms)
    from HF issue2061_sae_predictability/analysis_tensors/per_feature_r2/
    (downloaded one at a time, reduced to (d_sae,) vectors, then deleted);
  - sae_encoded answer-state TopK shards (35) for per-feature activation counts
    (count convention: rows where the feature appears in the TopK k=32 code with
    val != 0 — parity with scripts/issue2061_followup_free_analysis.py);
  - eval_results/issue_2061/null/*.jsonl per-cell null quantiles + GLOBAL_L29.json.

Output: dashboard/public/sae-predictability-2061.html
        (served at https://eps.superkaiba.com/sae-predictability-2061.html)

Usage:
    uv run python scripts/issue2061_predictability_dashboard.py \
        [--staging-dir /mnt/eps-data/$USER/issue2061_dashboard] \
        [--phase all|reduce|counts|render]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/torch import

import numpy as np  # noqa: E402

REPO_ID = "superkaiba1/explore-persona-space-data"
R2_PREFIX = "issue2061_sae_predictability/analysis_tensors/per_feature_r2"
ENC_PREFIX = "issue2061_sae_predictability/sae_encoded"
D_SAE = 262144
LAYER = 29

STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]
ARMS = ["context", "prefix"]

# One colour = one corpus stem (parity with scripts/issue2061_figures.py).
CORPUS_COLORS = {
    "lmsys23k": "#0072B2",
    "if11k": "#E69F00",
    "math7500": "#009E73",
    "gsm8k_train_full": "#CC79A7",
    "gsm8k_test1319": "#D55E00",
    "sft11k": "#56B4E9",
    "ultrafeedback": "#F0E442",
}
_FALLBACK_COLORS = ["#999999", "#000000", "#8C564B"]

_R2_ROW_RE = re.compile(rb'\{"feature_id": (\d+), "R2": (null|-?[0-9][0-9.eE+-]*), ')
_REG_RE = re.compile(rb'"regularization_limited": (true|false)')
_FNAME_RE = re.compile(
    r"^(base|sft|dpo|rlvr|longer-rlvr)_(chat|naturalistic)_(.+)_(prefix|context)_L29\.jsonl$"
)
TOP_N = 15


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_r2_and_reg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(d_sae,) float32 R2 (nan where null) + (d_sae,) bool reg-limited.

    Fail-loud parity with issue2061_followup_free_analysis.load_r2_vector:
    every line must match and feature ids must equal the row index.
    """
    data = path.read_bytes()
    matches = _R2_ROW_RE.findall(data)
    n_lines = data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
    if len(matches) != n_lines:
        raise ValueError(
            f"{path.name}: R2 regex matched {len(matches)} rows but file has "
            f"{n_lines} lines — writer format drift; refusing a partial parse."
        )
    ids = np.array([m[0] for m in matches]).astype(np.int64)
    if not (ids == np.arange(len(ids))).all():
        raise ValueError(f"{path.name}: rows not in feature-id order; refusing to mis-index.")
    vals = np.array([m[1] for m in matches])
    vals[vals == b"null"] = b"nan"
    reg = _REG_RE.findall(data)
    if len(reg) != n_lines:
        raise ValueError(
            f"{path.name}: reg-limited regex matched {len(reg)} rows over {n_lines} lines."
        )
    reg_arr = np.array(reg) == b"true"
    return vals.astype(np.float32), reg_arr


def list_cells() -> list[tuple[str, str, str, str]]:
    """(stage, render, corpus, arm) for all 70 per-feature files on HF."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi()
    # list() INSIDE the retried thunk: list_repo_tree is a LAZY generator (#779);
    # pagination 504s are un-retried upstream (#658/#833).
    tree = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; list() inside the thunk
            api.list_repo_tree(
                REPO_ID, repo_type="dataset", path_in_repo=R2_PREFIX, recursive=False
            )
        ),
        what=f"list_repo_tree({R2_PREFIX})",
    )
    cells = []
    for t in tree:
        m = _FNAME_RE.match(os.path.basename(t.path))
        if m:
            cells.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    if len(cells) != 70:
        raise ValueError(f"expected 70 per-feature files under {R2_PREFIX}, found {len(cells)}")
    return sorted(cells)


def phase_reduce(staging: Path, cells: list[tuple[str, str, str, str]]) -> None:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    red = staging / "reduced"
    red.mkdir(parents=True, exist_ok=True)
    for i, (stage, render, corpus, arm) in enumerate(cells):
        name = f"{stage}_{render}_{corpus}_{arm}_L29"
        out = red / f"{name}.npz"
        if out.exists():
            continue
        print(f"[reduce {i + 1}/{len(cells)}] {name}", flush=True)
        p = Path(
            retry_transient(
                lambda name=name: hf_hub_download(
                    REPO_ID,
                    f"{R2_PREFIX}/{name}.jsonl",
                    repo_type="dataset",
                    local_dir=str(staging / "dl"),
                ),
                what=f"hf_hub_download({name}.jsonl)",
            )
        )
        r2, reg = load_r2_and_reg(p)
        np.savez_compressed(out, r2=r2, reg=reg)
        p.unlink()  # delete-after-reduce: peak staging stays < 1 GB


def phase_counts(staging: Path, combos: list[tuple[str, str]]) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    cnt_dir = staging / "counts"
    cnt_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        for render, corpus in combos:
            name = f"{stage}_{render}_{corpus}_answer_L{LAYER}"
            out = cnt_dir / f"{name}.npz"
            if out.exists():
                continue
            print(f"[counts] {name}", flush=True)
            p = retry_transient(
                lambda name=name: hf_hub_download(
                    REPO_ID,
                    f"{ENC_PREFIX}/{name}.pt",
                    repo_type="dataset",
                    local_dir=str(staging / "dl"),
                ),
                what=f"hf_hub_download({name}.pt)",
            )
            payload = torch.load(p, map_location="cpu")
            idx = payload["idx"].numpy().astype(np.int64)
            val = payload["val"].numpy()
            n_rows = idx.shape[0]
            mask = val != 0  # padding is (idx=0, val=0.0)
            rows = np.broadcast_to(np.arange(n_rows)[:, None], idx.shape)
            keys = idx[mask] * n_rows + rows[mask]
            uniq = np.unique(keys)
            counts = np.bincount(uniq // n_rows, minlength=D_SAE).astype(np.int32)
            np.savez_compressed(out, counts=counts, n_rows=np.int64(n_rows))
            Path(p).unlink()


def load_null_cells(null_dir: Path) -> dict[tuple[str, str, str, str], dict]:
    """(pair, render, corpus, arm) -> {p97.5, true_max, true_argmax}."""
    out = {}
    for p in sorted(null_dir.glob("*_L29.jsonl")):
        with open(p) as f:
            row = json.loads(f.readline())
        key = (row["pair"], row["render"], row["corpus"], row["arm"])
        out[key] = {
            "p97_5": row["null_quantiles_per_cell"]["p97.5"],
            "true_max": row["true_max_delta_r2"],
            "true_argmax": row["true_argmax_feature_id"],
        }
    return out


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _corpus_color(corpus: str) -> str:
    return CORPUS_COLORS.get(corpus, _FALLBACK_COLORS[hash(corpus) % len(_FALLBACK_COLORS)])


def _fmt(x: float) -> str:
    return f"{x:+.3f}"


def phase_render(staging: Path, cells: list[tuple[str, str, str, str]], out_html: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    red, cnt_dir = staging / "reduced", staging / "counts"
    combos = sorted({(r, c) for _, r, c, _ in cells})
    r2 = {(s, r, c, a): np.load(red / f"{s}_{r}_{c}_{a}_L29.npz") for (s, r, c, a) in cells}
    counts = {
        (s, r, c): np.load(cnt_dir / f"{s}_{r}_{c}_answer_L{LAYER}.npz")
        for s in STAGES
        for (r, c) in combos
    }
    nulls = load_null_cells(Path("eval_results/issue_2061/null"))
    global_null = json.load(open("eval_results/issue_2061/null/GLOBAL_L29.json"))
    gq = global_null["global_null_quantiles"]["p97.5"]

    sections = []
    for s_from, s_to in STAGE_PAIRS:
        pair = f"{s_from}_{s_to}"
        arm_blocks = []
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        for ax, arm in zip(axes, ARMS):
            pooled = []
            for render, corpus in combos:
                a, b = (
                    r2[(s_from, render, corpus, arm)]["r2"],
                    r2[(s_to, render, corpus, arm)]["r2"],
                )
                d = (b - a)[np.isfinite(a) & np.isfinite(b)]
                pooled.append(d)
                ax.hist(
                    d,
                    bins=80,
                    histtype="step",
                    log=True,
                    color=_corpus_color(corpus),
                    label=f"{render}/{corpus} (n={len(d):,})",
                    linewidth=1.0,
                    linestyle="-" if render == "chat" else ":",
                )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(gq, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"{arm} arm")
            ax.set_xlabel("per-feature ΔR² (this stage − previous)")
            ax.legend(fontsize=5.5, frameon=False)
        axes[0].set_ylabel("features (log)")
        fig.suptitle(
            f"ΔR² distribution over SAE features: {s_from} → {s_to}"
            f"  (dashed = GLOBAL max-stat null p97.5 = {gq:.2f})",
            fontsize=10,
        )
        hist_b64 = _fig_to_b64(fig)

        for arm in ARMS:
            deltas, defined = {}, {}
            for render, corpus in combos:
                a = r2[(s_from, render, corpus, arm)]["r2"]
                b = r2[(s_to, render, corpus, arm)]["r2"]
                m = np.isfinite(a) & np.isfinite(b)
                d = np.where(m, b - a, np.nan)
                deltas[(render, corpus)] = d
                defined[(render, corpus)] = m

            # -- per-cell winners (max / min per render/corpus combo) --
            cell_rows = []
            for render, corpus in combos:
                d = deltas[(render, corpus)]
                if not np.isfinite(d).any():
                    continue
                jmax = int(np.nanargmax(d))
                jmin = int(np.nanargmin(d))
                nb = counts[(s_from, render, corpus)]["counts"]
                na = counts[(s_to, render, corpus)]["counts"]
                nul = nulls.get((pair, render, corpus, arm), {})
                for tag, j in (("max", jmax), ("min", jmin)):
                    reg_flag = bool(
                        r2[(s_from, render, corpus, arm)]["reg"][j]
                        or r2[(s_to, render, corpus, arm)]["reg"][j]
                    )
                    cell_rows.append(
                        {
                            "combo": f"{render}/{corpus}",
                            "kind": "most increased" if tag == "max" else "most decreased",
                            "fid": j,
                            "delta": float(d[j]),
                            "nb": int(nb[j]),
                            "na": int(na[j]),
                            "reg": reg_flag,
                            "p97_5": nul.get("p97_5"),
                        }
                    )

            # -- pooled ranking: mean ΔR² across combos where defined --
            stack = np.stack([deltas[k] for k in combos])  # (7, d_sae)
            n_def = np.isfinite(stack).sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean_d = np.nanmean(stack, axis=0)
            mean_d[n_def == 0] = np.nan
            eligible = n_def >= 2  # a 1-combo mean is that combo's value, not a consensus
            pool = np.where(eligible, mean_d, np.nan)

            def _rows(order: np.ndarray) -> list[dict]:
                rows = []
                for j in order:
                    j = int(j)
                    per = {
                        f"{r}/{c}": (
                            None if not np.isfinite(deltas[(r, c)][j]) else float(deltas[(r, c)][j])
                        )
                        for (r, c) in combos
                    }
                    nb = int(np.median([counts[(s_from, r, c)]["counts"][j] for (r, c) in combos]))
                    na = int(np.median([counts[(s_to, r, c)]["counts"][j] for (r, c) in combos]))
                    rows.append(
                        {
                            "fid": j,
                            "mean": float(pool[j]),
                            "n_def": int(n_def[j]),
                            "per": per,
                            "nb": nb,
                            "na": na,
                        }
                    )
                return rows

            finite_idx = np.flatnonzero(np.isfinite(pool))
            order = finite_idx[np.argsort(pool[finite_idx])]
            top_rows = _rows(order[::-1][:TOP_N])
            bot_rows = _rows(order[:TOP_N])
            arm_blocks.append({"arm": arm, "cells": cell_rows, "top": top_rows, "bot": bot_rows})

        sections.append(
            {"pair": pair, "s_from": s_from, "s_to": s_to, "hist": hist_b64, "arms": arm_blocks}
        )

    _write_html(out_html, sections, combos, gq)


def _cell_table(rows: list[dict]) -> str:
    tr = []
    for r in rows:
        flag = (
            ' <span class="warn" title="ridge fit regularization-limited at one endpoint">⚠reg</span>'
            if r["reg"]
            else ""
        )
        p = f"{r['p97_5']:.2f}" if r["p97_5"] is not None else "—"
        tr.append(
            f"<tr><td>{html.escape(r['combo'])}</td><td>{html.escape(r['kind'])}</td>"
            f"<td class='mono'>{r['fid']}</td><td class='num'>{_fmt(r['delta'])}{flag}</td>"
            f"<td class='num'>{r['nb']} → {r['na']}</td><td class='num'>{p}</td></tr>"
        )
    return (
        "<table><tr><th>render/corpus</th><th>winner</th><th>feature id</th>"
        "<th>ΔR²</th><th>rows active (before → after)</th><th>cell max-null p97.5</th></tr>"
        + "".join(tr)
        + "</table>"
    )


def _rank_table(rows: list[dict], combos: list[tuple[str, str]]) -> str:
    heads = "".join(f"<th class='small'>{html.escape(f'{r}/{c}')}</th>" for (r, c) in combos)
    tr = []
    for row in rows:
        per = "".join(
            f"<td class='num small'>{'—' if row['per'][f'{r}/{c}'] is None else _fmt(row['per'][f'{r}/{c}'])}</td>"
            for (r, c) in combos
        )
        tr.append(
            f"<tr><td class='mono'>{row['fid']}</td><td class='num'><b>{_fmt(row['mean'])}</b></td>"
            f"<td class='num'>{row['n_def']}</td><td class='num'>{row['nb']} → {row['na']}</td>{per}</tr>"
        )
    return (
        f"<table><tr><th>feature id</th><th>mean ΔR²</th><th>#combos defined</th>"
        f"<th>median rows active (before → after)</th>{heads}</tr>" + "".join(tr) + "</table>"
    )


def _write_html(out_html: Path, sections: list[dict], combos, gq: float) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>#2061 SAE per-feature predictability deltas</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1250px}"
        "table{border-collapse:collapse;margin:8px 0 18px}td,th{border:1px solid #ccc;"
        "padding:3px 8px;font-size:13px}th{background:#f0f0f0}.num{text-align:right}"
        ".mono{font-family:monospace}.small{font-size:11px}.warn{color:#b00}"
        ".caveat{background:#fff6e0;border:1px solid #e0c060;padding:10px 14px;font-size:13px}"
        "h2{border-bottom:2px solid #444;padding-bottom:4px;margin-top:36px}"
        "details{margin:10px 0}summary{cursor:pointer;font-weight:600}"
        "img{max-width:100%}</style></head><body>",
        "<h1>#2061 — SAE per-feature predictability change at each post-training stage</h1>",
        f"<p class='small'>Generated {now} · commit {_git_sha()[:10]} · "
        "generator <span class='mono'>scripts/issue2061_predictability_dashboard.py</span> · "
        "data: per-feature held-out R² (context/prefix → answer-state SAE features, "
        "EleutherAI/sae-llama-3.1-8b-64x L29, TopK k=32) from "
        "<span class='mono'>issue2061_sae_predictability/analysis_tensors/per_feature_r2</span> (HF), "
        "Tülu-3 ladder. ΔR²_j = R²_j(stage) − R²_j(previous stage), over features with a defined "
        "held-out R² at BOTH stages. Activation counts = rows where the feature appears in the "
        "TopK k=32 code with val ≠ 0 (per stage × corpus).</p>",
        "<div class='caveat'><b>Read with #2061's headline caveats.</b> "
        "(1) No per-feature gain survives the selection-symmetric max-statistic null "
        f"(GLOBAL p97.5 = {gq:.2f}; largest observed ΔR² has p = 0.49) — every table below is "
        "DESCRIPTIVE, multiplicity-uncorrected. (2) The extreme ΔR² values are dominated by "
        "rarely-active features (activation counts are shown per row — treat 1–9-row features "
        "as noise-grade). (3) The fixed SAE dictionary explains only ~33% of answer-state "
        "variance, so per-feature reads are minority-variance-scoped. (4) Prefix arms on chat "
        "renders have a row-constant prefix input — at-chance by construction. "
        "(5) ⚠reg marks features whose ridge fit was regularization-limited at an endpoint "
        "(λ pinned at the audited grid edge; not a clean read.)</div>",
    ]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec['s_from'])} → {html.escape(sec['s_to'])}</h2>")
        parts.append(f"<img src='data:image/png;base64,{sec['hist']}' alt='delta R2 histogram'>")
        for blk in sec["arms"]:
            openattr = " open" if blk["arm"] == "context" else ""
            parts.append(f"<details{openattr}><summary>{blk['arm']} arm</summary>")
            parts.append("<h3>Per-cell winners (each render/corpus combo)</h3>")
            parts.append(_cell_table(blk["cells"]))
            parts.append(
                f"<h3>Most increased — top {TOP_N} by mean ΔR² across combos (≥2 combos defined)</h3>"
            )
            parts.append(_rank_table(blk["top"], combos))
            parts.append(
                f"<h3>Most decreased — bottom {TOP_N} by mean ΔR² across combos (≥2 combos defined)</h3>"
            )
            parts.append(_rank_table(blk["bot"], combos))
            parts.append("</details>")
    parts.append(
        "<p class='small'>Feature ids index EleutherAI/sae-llama-3.1-8b-64x layer 29; no "
        "autointerp labels are available in-repo. Task: "
        "<a href='https://eps.superkaiba.com/tasks/2061'>#2061</a>.</p></body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("".join(parts))
    print(f"[render] wrote {out_html} ({out_html.stat().st_size / 1e6:.2f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    default_staging = Path(
        os.environ.get(
            "I2061_DASH_STAGING",
            f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}/issue2061_dashboard",
        )
    )
    ap.add_argument("--staging-dir", type=Path, default=default_staging)
    ap.add_argument("--phase", choices=["all", "reduce", "counts", "render"], default="all")
    ap.add_argument(
        "--out-html", type=Path, default=Path("dashboard/public/sae-predictability-2061.html")
    )
    args = ap.parse_args()

    cells = list_cells()
    combos = sorted({(r, c) for _, r, c, _ in cells})
    print(f"[setup] 70 cells, {len(combos)} render/corpus combos: {combos}", flush=True)
    if args.phase in ("all", "reduce"):
        phase_reduce(args.staging_dir, cells)
    if args.phase in ("all", "counts"):
        phase_counts(args.staging_dir, combos)
    if args.phase in ("all", "render"):
        phase_render(args.staging_dir, cells, args.out_html)
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
