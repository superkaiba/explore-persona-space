#!/usr/bin/env python3
"""Task #2569 leg 11: expose the L19 ignored/read split in PCA and SAE bases.

The PCA view is computed in both raw residual coordinates (operator ``A``) and
the ridge map's standardized input coordinates (operator ``W``). For every PC
it reports context-variance share, through-map gain, effective-kernel share at
the 99% squared-singular-mass cutoff, and contribution to predicted answer
variance.

The context-SAE view streams the fixed 100k-row capture sample. It accumulates
feature activation moments without retaining the activation matrix, splits each
feature's diagonal reconstructed variance between the raw effective kernel and
range, and reports feature-correlation, SAE-residual, and reconstruction-residual
cross terms separately. A checkpoint is written after every activation block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAPER,
    SEAM,
    save_c2a_figure,
    set_c2a_style,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2569_kernel_interpretation as KI  # noqa: E402
import issue2569_operator as OP  # noqa: E402

LAYER = 19
D = 3584
MASS = 0.99
DEFAULT_THEORY = Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory")
DEFAULT_SAMPLE = DEFAULT_THEORY / "leg10_dl/sample_L19.npz"
DEFAULT_MANIFEST = DEFAULT_THEORY / "leg10_dl/download_manifest.json"
DEFAULT_LEG8_WORK = Path("/mnt/eps-data/thomasjiralerspong/wt-2569-kernel-work")
DEFAULT_SAE = DEFAULT_THEORY / "sae_ctx/ae.pt"
TOP_PCS = 10
TOP_FEATURES = 20
SAE_BLOCK = 4096


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    """Write text through a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content)
    tmp.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def operator_partition(operator: np.ndarray, mass: float = MASS) -> dict:
    """Return the left-singular effective-kernel partition of a row operator."""
    u, singular, _vh = np.linalg.svd(np.asarray(operator, dtype=np.float64))
    tau, rank = OP.tau_kernel_threshold(singular, mass=mass)
    mask = singular < tau
    assert int((~mask).sum()) == rank, (rank, int((~mask).sum()))
    return {"u": u, "singular": singular, "mask": mask, "tau": tau, "rank": rank}


def pca_basis_view(sigma: np.ndarray, operator: np.ndarray, partition: dict) -> dict:
    """Compute per-PC variance, map-gain, kernel-share, and predicted-impact metrics."""
    covariance = 0.5 * (
        np.asarray(sigma, dtype=np.float64) + np.asarray(sigma, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.abs(values).max()), 1.0) * 1e-10
    assert float(values.min()) >= -tolerance, float(values.min())
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    pcs = vectors[:, order].T
    mapped = pcs @ np.asarray(operator, dtype=np.float64)
    gain_sq = np.einsum("ij,ij->i", mapped, mapped)
    kernel_u = np.asarray(partition["u"], dtype=np.float64)[:, partition["mask"]]
    kernel_projection = pcs @ kernel_u
    kernel_share = np.einsum("ij,ij->i", kernel_projection, kernel_projection)
    predicted_abs = values * gain_sq
    total_context = float(values.sum())
    total_predicted = float(predicted_abs.sum())
    assert total_context > 0 and total_predicted > 0
    return {
        "eigenvalue": values,
        "pcs": pcs,
        "context_variance_fraction": values / total_context,
        "map_gain": np.sqrt(gain_sq),
        "kernel_share": kernel_share,
        "predicted_variance_abs": predicted_abs,
        "predicted_variance_fraction": predicted_abs / total_predicted,
        "ignored_context_variance_abs": values * kernel_share,
        "total_context_variance": total_context,
        "total_predicted_variance": total_predicted,
    }


def pc_metric_rows(view: dict) -> list[dict]:
    """Convert all PCA arrays except the basis vectors to JSON-ready rows."""
    rows = []
    for index in range(view["pcs"].shape[0]):
        rows.append(
            {
                "pc": index + 1,
                "eigenvalue": float(view["eigenvalue"][index]),
                "context_variance_fraction": float(
                    view["context_variance_fraction"][index]
                ),
                "map_gain": float(view["map_gain"][index]),
                "kernel_share": float(view["kernel_share"][index]),
                "predicted_variance_abs": float(view["predicted_variance_abs"][index]),
                "predicted_variance_fraction": float(
                    view["predicted_variance_fraction"][index]
                ),
                "ignored_context_variance_abs": float(
                    view["ignored_context_variance_abs"][index]
                ),
            }
        )
    return rows


def selected_pc_indices(view: dict, top: int = TOP_PCS) -> dict[str, np.ndarray]:
    """Select high-variance ignored PCs and high-impact read PCs without hard thresholds."""
    ignored = np.argsort(view["ignored_context_variance_abs"])[::-1][:top]
    read = np.argsort(view["predicted_variance_abs"])[::-1][:top]
    return {"highest_variance_ignored": ignored, "highest_impact_read": read}


def projection_extremes(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    dirs: np.ndarray,
    ci: np.ndarray,
    texts: dict[int, dict],
    block: int = 4096,
) -> list[dict]:
    """Return text-grounded high/low examples for several raw or standardized directions."""
    projections = np.empty((x.shape[0], dirs.shape[0]), dtype=np.float32)
    for lo in range(0, x.shape[0], block):
        hi = min(lo + block, x.shape[0])
        xb = np.asarray(x[lo:hi], dtype=np.float32).copy()
        xb -= mean.astype(np.float32)
        xb /= scale.astype(np.float32)
        projections[lo:hi] = xb @ dirs.astype(np.float32).T
    return [
        KI.context_extremes(projections[:, j], ci, texts) for j in range(dirs.shape[0])
    ]


def aligned_sae_features(
    dirs_raw: np.ndarray,
    decoder_unit: np.ndarray,
    readings: dict[int, str],
    top: int = 5,
) -> list[list[dict]]:
    """Return top decoder-cosine context-SAE features for raw-coordinate directions."""
    out = []
    for direction in dirs_raw:
        cosine = decoder_unit @ direction.astype(np.float32)
        indices = np.argsort(np.abs(cosine))[::-1][:top]
        out.append(
            [
                {
                    "feat_id": int(index),
                    "cosine": float(cosine[index]),
                    **(
                        {"reading": readings[int(index)]}
                        if int(index) in readings
                        else {}
                    ),
                }
                for index in indices
            ]
        )
    return out


def _variance_from_moments(
    total: np.ndarray, squares: np.ndarray, n: int
) -> np.ndarray:
    """Return unbiased componentwise variance from sums and squared sums."""
    return (squares - total * total / n) / (n - 1)


def _cross_from_moments(
    left_sum: np.ndarray,
    right_sum: np.ndarray,
    product_sum: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return unbiased componentwise covariance from paired moment sums."""
    return (product_sum - left_sum * right_sum / n) / (n - 1)


def _checkpoint_regime(
    sample_ci: np.ndarray,
    sample_manifest: Path,
    sae_path: Path,
    n_rows: int,
    block: int,
) -> dict:
    """Build a machine-stable resume key from source bytes and generating parameters."""
    return {
        "version": 1,
        "layer": LAYER,
        "mass": MASS,
        "n_rows": n_rows,
        "dimension": D,
        "block": block,
        "sample_ci_sha256": hashlib.sha256(
            np.asarray(sample_ci, dtype=np.int64).tobytes()
        ).hexdigest(),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "sae_sha256": sha256_file(sae_path),
    }


def stream_sae_moments(
    x: np.ndarray,
    ctx: dict,
    u_raw: np.ndarray,
    checkpoint: Path,
    regime_path: Path,
    regime: dict,
    block: int = SAE_BLOCK,
    max_blocks: int | None = None,
) -> dict:
    """Stream SAE activations/reconstructions, checkpointing moment sums per block."""
    n_rows = x.shape[0]
    dimension = x.shape[1]
    assert u_raw.shape == (dimension, dimension), (u_raw.shape, dimension)
    n_features = ctx["w_dec"].shape[0]
    names = (
        "act_sum",
        "act_sumsq",
        "rec_sum",
        "rec_sumsq",
        "res_sum",
        "res_sumsq",
        "cross_sum",
    )
    sizes = (
        n_features,
        n_features,
        dimension,
        dimension,
        dimension,
        dimension,
        dimension,
    )
    state = {name: np.zeros(size, dtype=np.float64) for name, size in zip(names, sizes)}
    done_rows = 0
    if regime_path.exists():
        saved_regime = json.loads(regime_path.read_text())
        assert saved_regime == regime, "SAE checkpoint regime mismatch"
    else:
        assert not checkpoint.exists(), (
            "SAE checkpoint exists without its regime sidecar"
        )
        write_text_atomic(
            regime_path, json.dumps(regime, indent=2, sort_keys=True) + "\n"
        )
    if checkpoint.exists():
        saved = np.load(checkpoint)
        done_rows = int(saved["done_rows"])
        assert 0 <= done_rows <= n_rows and (
            done_rows % block == 0 or done_rows == n_rows
        ), done_rows
        for name, size in zip(names, sizes):
            state[name] = np.asarray(saved[name], dtype=np.float64)
            assert state[name].shape == (size,), (name, state[name].shape, size)
        print(f"[sae-resume] rows={done_rows}/{n_rows}", flush=True)

    u32 = np.asarray(u_raw, dtype=np.float32)
    b_dec = np.asarray(ctx["b_dec"], dtype=np.float32)
    w_enc = np.asarray(ctx["w_enc"], dtype=np.float32)
    b_enc = np.asarray(ctx["b_enc"], dtype=np.float32)
    w_dec = np.asarray(ctx["w_dec"], dtype=np.float32)
    threshold = float(ctx["threshold"])
    total_blocks = (n_rows + block - 1) // block
    started = time.time()
    stop_row = (
        n_rows if max_blocks is None else min(n_rows, done_rows + max_blocks * block)
    )
    for block_index, lo in enumerate(
        range(done_rows, stop_row, block), start=done_rows // block
    ):
        hi = min(lo + block, n_rows)
        xb = np.asarray(x[lo:hi], dtype=np.float32)
        centered = xb.copy()
        centered -= b_dec
        acts = centered @ w_enc
        acts += b_enc
        np.maximum(acts, 0.0, out=acts)
        acts *= acts > threshold

        state["act_sum"] += acts.sum(axis=0, dtype=np.float64)
        state["act_sumsq"] += np.einsum(
            "ij,ij->j", acts, acts, dtype=np.float64, optimize=True
        )

        reconstruction = acts @ w_dec
        reconstruction += b_dec
        rec_coeff = reconstruction @ u32
        x_coeff = xb @ u32
        x_coeff -= rec_coeff
        state["rec_sum"] += rec_coeff.sum(axis=0, dtype=np.float64)
        state["rec_sumsq"] += np.einsum(
            "ij,ij->j", rec_coeff, rec_coeff, dtype=np.float64, optimize=True
        )
        state["res_sum"] += x_coeff.sum(axis=0, dtype=np.float64)
        state["res_sumsq"] += np.einsum(
            "ij,ij->j", x_coeff, x_coeff, dtype=np.float64, optimize=True
        )
        state["cross_sum"] += np.einsum(
            "ij,ij->j", rec_coeff, x_coeff, dtype=np.float64, optimize=True
        )

        save_npz_atomic(checkpoint, done_rows=np.asarray(hi), **state)
        elapsed = time.time() - started
        print(
            f"[sae-stream] block {block_index + 1}/{total_blocks} rows={hi}/{n_rows} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    return {"done_rows": stop_row, **state}


def sae_accounting(
    moments: dict,
    decoder_norm2: np.ndarray,
    kernel_share: np.ndarray,
    kernel_mask: np.ndarray,
    n_rows: int,
) -> dict:
    """Assemble the exact feature/reconstruction/residual variance identity."""
    activation_var = _variance_from_moments(
        moments["act_sum"], moments["act_sumsq"], n_rows
    )
    diagonal_total = activation_var * decoder_norm2
    diagonal_kernel = diagonal_total * kernel_share
    diagonal_range = diagonal_total - diagonal_kernel

    rec_var = _variance_from_moments(moments["rec_sum"], moments["rec_sumsq"], n_rows)
    res_var = _variance_from_moments(moments["res_sum"], moments["res_sumsq"], n_rows)
    cross = _cross_from_moments(
        moments["rec_sum"], moments["res_sum"], moments["cross_sum"], n_rows
    )
    masks = {
        "kernel": np.asarray(kernel_mask, dtype=bool),
        "range": ~np.asarray(kernel_mask, bool),
    }
    components = {}
    for name, mask in masks.items():
        diagonal = diagonal_kernel if name == "kernel" else diagonal_range
        rec = float(rec_var[mask].sum())
        residual = float(res_var[mask].sum())
        twice_cross = float(2.0 * cross[mask].sum())
        diag = float(diagonal.sum())
        feature_cross = rec - diag
        total = rec + residual + twice_cross
        components[name] = {
            "feature_diagonal_abs": diag,
            "feature_covariance_cross_abs": feature_cross,
            "reconstruction_abs": rec,
            "sae_residual_abs": residual,
            "twice_reconstruction_residual_covariance_abs": twice_cross,
            "context_total_abs": total,
        }
    total_context = (
        components["kernel"]["context_total_abs"]
        + components["range"]["context_total_abs"]
    )
    assert total_context > 0
    for component in components.values():
        for key in list(component):
            if key.endswith("_abs"):
                component[key.replace("_abs", "_fraction_of_context")] = (
                    component[key] / total_context
                )
    identity = 0.0
    for component in components.values():
        identity += (
            component["feature_diagonal_abs"]
            + component["feature_covariance_cross_abs"]
            + component["sae_residual_abs"]
            + component["twice_reconstruction_residual_covariance_abs"]
        )
    assert abs(identity - total_context) <= 1e-8 * total_context, (
        identity,
        total_context,
    )
    return {
        "activation_variance": activation_var,
        "feature_diagonal_kernel_abs": diagonal_kernel,
        "feature_diagonal_range_abs": diagonal_range,
        "components": components,
        "total_context_abs": total_context,
        "identity_relative_error": abs(identity - total_context) / total_context,
    }


def feature_rows(
    accounting: dict,
    kernel_share: np.ndarray,
    decoder_norm2: np.ndarray,
    evidence: dict[int, dict],
    readings: dict[int, str],
    top: int = TOP_FEATURES,
) -> dict:
    """Return sorted high-mass kernel/range feature lists with activation evidence."""
    result = {}
    for tag, values in (
        ("ignored_kernel", accounting["feature_diagonal_kernel_abs"]),
        ("read_range", accounting["feature_diagonal_range_abs"]),
    ):
        indices = np.argsort(values)[::-1][:top]
        rows = []
        for index in indices:
            fid = int(index)
            rows.append(
                {
                    "feat_id": fid,
                    "diagonal_variance_abs": float(values[index]),
                    "diagonal_variance_fraction_of_context": float(
                        values[index] / accounting["total_context_abs"]
                    ),
                    "activation_variance": float(
                        accounting["activation_variance"][index]
                    ),
                    "decoder_norm2": float(decoder_norm2[index]),
                    "kernel_share": float(kernel_share[index]),
                    **({"reading": readings[fid]} if fid in readings else {}),
                    "activation_evidence": evidence[fid],
                }
            )
        result[tag] = rows
    return result


def _detail_selected_pcs(
    view: dict,
    coordinate: str,
    scale: np.ndarray,
    x: np.ndarray,
    mean: np.ndarray,
    ci: np.ndarray,
    texts: dict[int, dict],
    decoder_unit: np.ndarray,
    readings: dict[int, str],
) -> dict:
    """Attach examples and SAE alignments to the two selected PC rankings."""
    selected = selected_pc_indices(view)
    unique = np.unique(np.concatenate(list(selected.values())))
    pc_dirs = view["pcs"][unique]
    raw_dirs = pc_dirs if coordinate == "raw" else pc_dirs * scale[None, :]
    raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    examples = projection_extremes(x, mean, scale, pc_dirs, ci, texts)
    alignments = aligned_sae_features(raw_dirs, decoder_unit, readings)
    details = {}
    position = {int(index): j for j, index in enumerate(unique)}
    rows_all = pc_metric_rows(view)
    for ranking, indices in selected.items():
        details[ranking] = []
        for index in indices:
            j = position[int(index)]
            details[ranking].append(
                {
                    **rows_all[int(index)],
                    "context_extremes": examples[j],
                    "aligned_context_sae_features": alignments[j],
                }
            )
    return details


def _style_axis(ax: plt.Axes) -> None:
    """Apply the paper's minimal axis seams and horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.5)
    ax.set_axisbelow(True)


def render_figure(doc: dict, raw_view: dict, std_view: dict, fig_dir: Path) -> dict:
    """Render PCA concentration, PC gain, and SAE accounting panels."""
    font = set_c2a_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor=PAPER)
    teal = "#176B87"
    orange = "#C4553D"

    ax = axes[0]
    n_show = 200
    index = np.arange(1, n_show + 1)
    ax.plot(
        index,
        np.cumsum(raw_view["context_variance_fraction"][:n_show]),
        color=teal,
        lw=2.2,
        label="Context variance · raw",
    )
    ax.plot(
        index,
        np.cumsum(raw_view["predicted_variance_fraction"][:n_show]),
        color=orange,
        lw=2.2,
        ls="--",
        label="Predicted variance · raw",
    )
    ax.plot(
        index,
        np.cumsum(std_view["context_variance_fraction"][:n_show]),
        color=teal,
        lw=1.7,
        ls=":",
        label="Context variance · standardized",
    )
    ax.plot(
        index,
        np.cumsum(std_view["predicted_variance_fraction"][:n_show]),
        color=orange,
        lw=1.7,
        ls="-.",
        label="Predicted variance · standardized",
    )
    ax.set_xlabel("PCs in variance order")
    ax.set_ylabel("Cumulative share")
    ax.set_ylim(0, 1.02)
    ax.set_title("A  Variance and mapped impact", loc="left")
    ax.legend(frameon=False, fontsize=11)
    _style_axis(ax)

    ax = axes[1]
    size = 25 + 900 * np.sqrt(raw_view["context_variance_fraction"])
    ax.scatter(
        raw_view["kernel_share"],
        raw_view["predicted_variance_fraction"],
        s=size,
        facecolors="none",
        edgecolors=teal,
        linewidths=0.8,
        alpha=0.6,
    )
    ax.axvline(
        doc["pca"]["raw"]["random_direction_kernel_share"], color=MUTED, ls=":", lw=1.4
    )
    ax.set_yscale("log")
    ax.set_xlabel("PC share in effective kernel")
    ax.set_ylabel("Predicted-answer variance share")
    ax.set_title("B  Raw PCs: ignored vs consequential", loc="left")
    _style_axis(ax)

    ax = axes[2]
    components = doc["sae"]["accounting"]["components"]
    labels = [
        "Feature\ndiagonal",
        "Feature\ncovariance",
        "SAE\nresidual",
        "Recon.–resid.\ncovariance (2×)",
    ]
    keys = (
        "feature_diagonal_fraction_of_context",
        "feature_covariance_cross_fraction_of_context",
        "sae_residual_fraction_of_context",
        "twice_reconstruction_residual_covariance_fraction_of_context",
    )
    xpos = np.arange(len(keys))
    width = 0.36
    ax.bar(
        xpos - width / 2,
        [components["kernel"][key] for key in keys],
        width,
        color=orange,
        edgecolor=PAPER,
        hatch="///",
        label="Effective kernel",
    )
    ax.bar(
        xpos + width / 2,
        [components["range"][key] for key in keys],
        width,
        color=teal,
        edgecolor=PAPER,
        label="Read range",
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xpos, labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylabel("Fraction of context variance")
    ax.set_title("C  SAE variance accounting", loc="left")
    ax.legend(frameon=False, fontsize=11)
    _style_axis(ax)

    fig.tight_layout(w_pad=2.5)
    outputs = save_c2a_figure(
        fig,
        fig_dir / "leg11_basis_views",
        title="Context-answer map in PCA and SAE bases",
        subject="Task #2569 leg 11",
        creator="scripts/issue2569_basis_views.py",
    )
    plt.close(fig)
    return {"font": font, "outputs": outputs}


def write_figure_artifacts(
    doc: dict,
    raw_view: dict,
    std_view: dict,
    repo: Path,
    out_json: Path,
) -> dict:
    """Render the figure and write its result- and output-hash sidecar."""
    fig_dir = repo / "figures/issue_2569"
    figure = render_figure(doc, raw_view, std_view, fig_dir)
    output_hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in figure["outputs"].values()
    }
    figure_meta = {
        "style": "c2a-v1",
        "font": figure["font"],
        "source": "scripts/issue2569_basis_views.py",
        "result": str(out_json.relative_to(repo)),
        "result_sha256": sha256_file(out_json),
        "output_sha256": output_hashes,
        **as_metadata_dict(
            git_provenance(repo, argv0=__file__), phase="leg11-basis-figure"
        ),
    }
    write_text_atomic(
        fig_dir / "leg11_basis_views.meta.json",
        json.dumps(figure_meta, indent=2, sort_keys=True) + "\n",
    )
    return figure


def render_markdown(doc: dict) -> str:
    """Render a concise, self-contained leg-11 result note."""
    raw = doc["pca"]["raw"]
    std = doc["pca"]["standardized"]
    components = doc["sae"]["accounting"]["components"]
    lines = [
        "# The L19 context-answer map in PCA and SAE bases (task #2569, leg 11)",
        "",
        "The PCA calculation is performed separately in raw residual coordinates and in the "
        "ridge map's standardized coordinates. `map_gain` is the Euclidean gain of a unit PC; "
        "`kernel_share` is its squared projection into the effective kernel at the 99% "
        "squared-singular-mass cutoff; predicted impact is `eigenvalue × map_gain²`.",
        "",
        "## PCA summary",
        "",
        "| coordinate | effective-kernel dim | context variance in kernel | PCs for 50% context variance | PCs for 50% predicted variance |",
        "|---|---:|---:|---:|---:|",
        f"| raw | {raw['kernel_dim']} | {raw['context_variance_in_kernel']:.3f} | {raw['pcs_to_50pct_context']} | {raw['pcs_to_50pct_predicted']} |",
        f"| standardized | {std['kernel_dim']} | {std['context_variance_in_kernel']:.3f} | {std['pcs_to_50pct_context']} | {std['pcs_to_50pct_predicted']} |",
        "",
        "The JSON contains every PC plus the ten largest ignored-variance PCs and ten "
        "highest-impact read PCs, each grounded by high/low real contexts and its closest "
        "context-SAE decoder directions.",
        "",
        "## SAE variance accounting",
        "",
        "| term | effective kernel | read range |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Feature diagonal", "feature_diagonal_fraction_of_context"),
        (
            "Feature covariance cross-term",
            "feature_covariance_cross_fraction_of_context",
        ),
        ("SAE unexplained residual", "sae_residual_fraction_of_context"),
        (
            "2 × reconstruction-residual covariance",
            "twice_reconstruction_residual_covariance_fraction_of_context",
        ),
        ("Total context variance", "context_total_fraction_of_context"),
    ):
        lines.append(
            f"| {label} | {components['kernel'][key]:.3f} | {components['range'][key]:.3f} |"
        )
    lines += [
        "",
        f"Accounting identity relative error: {doc['sae']['accounting']['identity_relative_error']:.3e}.",
        "Feature rankings are diagonal attributions, not causal or additive semantic units: "
        "correlated SAE features contribute the separately reported covariance term. The "
        "top ignored-kernel and read-range feature lists include top-activating context excerpts; "
        "existing analyst readings are included only where leg 8 had already supplied one.",
        "",
        "## Scope",
        "",
        "This characterizes one fitted ridge operator. Effective-kernel means low gain for this "
        "linear predictor, not that the underlying language model discards the information.",
    ]
    return "\n".join(lines) + "\n"


def _pcs_to_fraction(values: np.ndarray, fraction: float = 0.5) -> int:
    """Return the smallest leading count whose cumulative normalized mass reaches a fraction."""
    cumulative = np.cumsum(values) / values.sum()
    return int(np.searchsorted(cumulative, fraction) + 1)


def main() -> None:
    """Run the PCA/SAE basis analysis and render its durable artifacts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("/home/thomasjiralerspong/explore-persona-space"),
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--leg8-work", type=Path, default=DEFAULT_LEG8_WORK)
    parser.add_argument("--sae", type=Path, default=DEFAULT_SAE)
    parser.add_argument("--work", type=Path, default=DEFAULT_THEORY / "leg11_dl")
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--block", type=int, default=SAE_BLOCK)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--stop-after-sae-blocks",
        type=int,
        default=None,
        help="pilot-only: stop after this many new SAE blocks, before result writes",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)

    sample = np.load(args.sample)
    x = np.asarray(sample["x"], dtype=np.float32)[: args.max_rows]
    ci = np.asarray(sample["ci"], dtype=np.int64)[: args.max_rows]
    assert x.shape == (args.max_rows, D) and ci.shape == (args.max_rows,), (
        x.shape,
        ci.shape,
    )
    manifest = json.loads(args.manifest.read_text())
    assert not manifest.get("errors"), manifest.get("errors")
    texts = KI.load_row_meta_texts(manifest, set(int(value) for value in ci.tolist()))
    assert len(texts) == ci.size, (len(texts), ci.size)

    payload = OP.load_banked_map(LAYER, root=args.map_root)
    operator_raw, _bias = OP.row_operator(payload)
    sigma_raw, mean_raw, n_pool = KI.load_sigma_c(DEFAULT_THEORY / "moments/gram_xx.pt")
    sigma_standardized = sigma_raw / (payload.xsd[:, None] * payload.xsd[None, :])

    print("[pca] factorizing raw and standardized operators/covariances", flush=True)
    raw_partition = operator_partition(operator_raw)
    std_partition = operator_partition(payload.W)
    raw_view = pca_basis_view(sigma_raw, operator_raw, raw_partition)
    std_view = pca_basis_view(sigma_standardized, payload.W, std_partition)

    ctx = KI.load_ctx_sae(args.sae)
    fig_arrays = np.load(args.leg8_work / "figure_arrays.npz")
    kernel_share = np.asarray(fig_arrays["ctx_shares"], dtype=np.float64)
    assert kernel_share.shape == (ctx["w_dec"].shape[0],), kernel_share.shape
    decoder_norm2 = np.einsum(
        "ij,ij->i", ctx["w_dec"], ctx["w_dec"], dtype=np.float64, optimize=True
    )
    readings_doc = json.loads((args.leg8_work / "overlay.json").read_text())
    readings = {
        int(key): value for key, value in readings_doc["ctx_feature_readings"].items()
    }
    decoder_unit = ctx["w_dec"] / np.maximum(
        np.linalg.norm(ctx["w_dec"], axis=1, keepdims=True), 1e-12
    )

    raw_details = _detail_selected_pcs(
        raw_view,
        "raw",
        np.ones(D),
        x,
        mean_raw,
        ci,
        texts,
        decoder_unit,
        readings,
    )
    std_details = _detail_selected_pcs(
        std_view,
        "standardized",
        payload.xsd,
        x,
        payload.xmu,
        ci,
        texts,
        decoder_unit,
        readings,
    )

    regime = _checkpoint_regime(ci, args.manifest, args.sae, x.shape[0], args.block)
    moments = stream_sae_moments(
        x,
        ctx,
        raw_partition["u"],
        args.work / "sae_stream_stats.npz",
        args.work / "sae_stream_stats.regime.json",
        regime,
        block=args.block,
        max_blocks=args.stop_after_sae_blocks,
    )
    if moments["done_rows"] < x.shape[0]:
        print(
            f"[sae-pilot-stop] rows={moments['done_rows']}/{x.shape[0]} "
            f"blocks={args.stop_after_sae_blocks}",
            flush=True,
        )
        return
    accounting = sae_accounting(
        moments, decoder_norm2, kernel_share, raw_partition["mask"], x.shape[0]
    )

    selected_features = np.unique(
        np.concatenate(
            [
                np.argsort(accounting["feature_diagonal_kernel_abs"])[::-1][
                    :TOP_FEATURES
                ],
                np.argsort(accounting["feature_diagonal_range_abs"])[::-1][
                    :TOP_FEATURES
                ],
            ]
        )
    ).astype(np.int64)
    selected_acts = KI.encode_ctx_features(ctx, x, selected_features, block=args.block)
    evidence_rows = KI.naming_evidence_rows(
        selected_features.tolist(), selected_acts, kernel_share, ci, texts
    )
    evidence = {int(row["feat_id"]): row for row in evidence_rows}
    ranked_features = feature_rows(
        accounting, kernel_share, decoder_norm2, evidence, readings
    )

    raw_context_kernel = float(
        KI.projected_cov_trace_fraction(
            raw_partition["u"], raw_partition["mask"], sigma_raw
        )
    )
    std_context_kernel = float(
        KI.projected_cov_trace_fraction(
            std_partition["u"], std_partition["mask"], sigma_standardized
        )
    )

    def _pca_doc(
        view: dict, partition: dict, details: dict, context_kernel: float
    ) -> dict:
        return {
            "kernel_dim": int(partition["mask"].sum()),
            "read_rank": int(partition["rank"]),
            "tau": float(partition["tau"]),
            "random_direction_kernel_share": float(partition["mask"].mean()),
            "context_variance_in_kernel": context_kernel,
            "pcs_to_50pct_context": _pcs_to_fraction(view["eigenvalue"]),
            "pcs_to_50pct_predicted": _pcs_to_fraction(view["predicted_variance_abs"]),
            "total_context_variance": view["total_context_variance"],
            "total_predicted_variance": view["total_predicted_variance"],
            "all_pcs": pc_metric_rows(view),
            "selected": details,
        }

    accounting_doc = {
        "components": accounting["components"],
        "total_context_abs": accounting["total_context_abs"],
        "identity_relative_error": accounting["identity_relative_error"],
    }
    doc = {
        "task": "issue2569 leg11 PCA and context-SAE basis views (L19)",
        "definitions": {
            "raw_pca": "eigenvectors of population raw-context covariance Sigma_c; row operator A=diag(1/xsd)W",
            "standardized_pca": "eigenvectors of diag(1/xsd) Sigma_c diag(1/xsd); row operator W",
            "pc_map_gain": "Euclidean norm ||pc @ operator|| for a unit PC",
            "pc_kernel_share": "squared norm of the PC projected into the effective left-singular kernel at 99% squared singular mass",
            "pc_predicted_variance": "PC eigenvalue times squared map gain",
            "sae_feature_diagonal": "Var(feature activation) times squared decoder norm, split by decoder kernel share",
            "sae_feature_covariance": "reconstruction variance minus the sum of feature-diagonal terms; includes correlations between SAE features",
            "sae_residual": "variance of x minus SAE reconstruction",
            "reconstruction_residual_cross": "twice the reconstruction-residual covariance needed for the exact variance identity",
        },
        "pca": {
            "raw": _pca_doc(raw_view, raw_partition, raw_details, raw_context_kernel),
            "standardized": _pca_doc(
                std_view, std_partition, std_details, std_context_kernel
            ),
        },
        "sae": {
            "sample_rows": int(x.shape[0]),
            "population_rows_for_pca": int(n_pool),
            "checkpoint_regime": regime,
            "accounting": accounting_doc,
            "feature_rankings": ranked_features,
            "caveat": "Feature diagonal terms are a transparent attribution convention, not an additive causal decomposition; correlated features are isolated in the feature-covariance cross-term.",
        },
        "metadata": {
            "map_payload": str(payload.path),
            "selected_lambda": payload.selected_lambda,
            "sample": str(args.sample),
            "manifest": str(args.manifest),
            "sae": str(args.sae),
            "threads": args.threads,
            "block": args.block,
            **as_metadata_dict(
                git_provenance(args.repo_root, argv0=__file__),
                phase="leg11-basis-views",
            ),
        },
    }

    out_dir = args.repo_root / "eval_results/issue_2569/weights/leg11"
    out_json = out_dir / "basis_views_L19.json"
    out_md = out_dir / "basis_views_L19.md"
    write_text_atomic(out_json, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    write_text_atomic(out_md, render_markdown(doc))
    figure = write_figure_artifacts(doc, raw_view, std_view, args.repo_root, out_json)
    print(f"[done] {out_json} {out_md} {figure['outputs']}", flush=True)


if __name__ == "__main__":
    main()
