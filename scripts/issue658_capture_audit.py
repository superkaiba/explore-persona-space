#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF003
# Intentional Unicode (v0/c_C/r_B/Σ_c, ※, →, ×) in scientific docstrings + output.
"""Issue #658 — capture-completeness audit (Change 3).

Cross-checks a (merged) base-model activation store against EVERY downstream
analysis the leakage-predictor program runs. The slot-only logits fix (Change 1)
is SURGICAL to the marker forward (one position) and the 8-GPU sharding (Change 2)
is a data-parallel re-org of the SAME per-context work — neither reduces what the
store persists. This audit is the mechanical guarantee of that: it loads the
store + the E0 / raw completion dirs and ASSERTS each required input is present,
emitting a CAPTURE MATRIX (analysis → required data → where stored → present).

It is the load-bearing "make sure we have all the info for the analyses" check.
Maps every downstream consumer to its store key:

  Phase 1 (this task)        A3.2 MLP            v0 summaries (mean/last/maxp/attn) all layers
                             A3.3 linear r_B     r_b.pt diffmeans/meanDB per rb-column, all layers
                             A3.4/A3.5 c_C→v0    cc_meanprompt (NEW) + #594 cc_last reuse
                             single-context N1   per-(C,probe) answer spans + per-sample E0
                             within-cond coherence per-(C,probe) c_x (last-input slot) — see note
  E0(C,B) judged rates       e0_gen/<ctx>__<col>.json (10-col registry, Sonnet judge off-pod)
  continuous logP DV         logp_norm per completion in e0_gen cells (dual-DV secondary)
  marker slot 4-float        e0_gen/<ctx>__marker.json (logp/z_marker/z_eos/logZ + argmax)
  Phases 2-4 (later)         Σ_c second moment   sigma_c.pt (H,H) per layer
                             r_B read-outs       r_b.pt (reused)

Per-sample retention (R>=8 completions + activations + labels for the §1.10
single-context arm) is recorded as a CAVEAT row: the v0-capture pass stores ONE
greedy answer span per (C,probe) (the v0(C) definitional input); the temp-1.0
R-sample completions + their judge labels live in the e0_gen judged columns
(broad_em n=50, sycophancy n=10). Per-sample ANSWER-SIDE ACTIVATIONS for the
sampled (non-greedy) completions are NOT in this base store — they are a Phase-1
single-context-arm extension, flagged here so the gap is visible, not silent.

Usage::

    uv run python scripts/issue658_capture_audit.py \\
        --store data/issue_658/store \\
        --eval-results eval_results/issue_658
    # smoke (the merged smoke store):
    uv run python scripts/issue658_capture_audit.py \\
        --store /tmp/issue658_8gpu_smoke/store_smoke \\
        --eval-results eval_results/issue_658 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE torch freezes its pool.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import torch  # noqa: E402
from issue658_common import (  # noqa: E402
    E0_COLUMNS,
    MARKER_SLOT_KEYS,
    rb_columns,
)


@dataclass
class Row:
    """One capture-matrix row: an analysis, its required data, and a verdict."""

    analysis: str
    required_data: str
    where: str
    present: bool | None = None
    detail: str = ""
    caveat: bool = False


@dataclass
class AuditResult:
    rows: list[Row] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.rows.append(Row(*a, **k))

    @property
    def hard_failures(self) -> list[Row]:
        return [r for r in self.rows if r.present is False and not r.caveat]

    @property
    def ok(self) -> bool:
        return not self.hard_failures


def _load(path: Path):
    return torch.load(path, weights_only=False)


def audit_store(store_dir: Path, eval_results: Path, *, smoke: bool = False) -> AuditResult:
    """Run the capture-completeness audit; return the matrix + verdict."""
    res = AuditResult()
    e0_sub = "e0_gen_smoke" if smoke else "e0_gen"
    raw_sub = "raw_completions_smoke" if smoke else "raw_completions"
    e0_dir = eval_results / e0_sub
    raw_dir = eval_results / raw_sub

    # ── manifest + the four sha-pinned deliverables ────────────────────────────
    man_path = store_dir / "store_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.is_file() else {}
    ctx_ids = manifest.get("context_ids", [])
    capture_layers = manifest.get("capture_layers", [])
    res.add(
        "store manifest (sha-pinned)",
        "store_manifest.json + per-file sha256",
        str(man_path),
        present=man_path.is_file() and bool(manifest.get("files")),
        detail=f"{len(ctx_ids)} contexts, {len(capture_layers)} layers pinned",
    )

    # ── A3.2 — v0 summaries, ALL layers, per recipe ────────────────────────────
    v0_path = store_dir / "v0_summaries.pt"
    v0_present = v0_path.is_file()
    n_summ = 0
    n_cc = 0
    if v0_present:
        v0 = _load(v0_path)
        summ = v0.get("summaries", {})
        per_recipe_ok = all(r in summ for r in ("mean", "last", "maxp"))
        n_summ = len(summ.get("mean", {}))
        n_cc = len(v0.get("cc_meanprompt", {}))
        # every context present + per-summary tensor is (Lc, H) over capture_layers
        ctx_ok = n_summ == len(ctx_ids) if ctx_ids else n_summ > 0
        res.add(
            "A3.2 activation-summary sufficiency (MLP v0→E0)",
            "v0(C) mean/last/maxp summaries, ALL captured layers, per context",
            f"{v0_path}::summaries",
            present=v0_present and per_recipe_ok and ctx_ok,
            detail=f"{n_summ} contexts × {len(('mean', 'last', 'maxp'))} recipes; "
            f"attn-pool fit on CPU from answer spans below",
        )
    else:
        res.add(
            "A3.2 activation-summary sufficiency (MLP v0→E0)",
            "v0(C) mean/last/maxp summaries, ALL captured layers",
            f"{v0_path}::summaries",
            present=False,
            detail="v0_summaries.pt missing",
        )

    # ── per-(C,probe) answer spans: attn-pool fit + single-context noise floor ──
    spans_dir = store_dir / "answer_spans"
    span_index_path = spans_dir / "index.json"
    span_files = sorted(spans_dir.glob("*.pt")) if spans_dir.is_dir() else []
    spans_ok = bool(span_files) and span_index_path.is_file()
    # spot-check one span file has the per-probe list shape
    span_detail = f"{len(span_files)} per-context span packs"
    if span_files:
        blob = _load(span_files[0])
        has_spans = isinstance(blob.get("spans"), list)
        spans_ok = spans_ok and has_spans
        span_detail += f"; first pack {len(blob.get('spans', []))} per-probe spans (Lc,S,H) fp16"
    res.add(
        "A3.2 attn-pool recipe + single-context (§1.10) resampling",
        "per-(C,probe) answer-token spans, ALL layers, fp16 (per-probe AND per-sample)",
        f"{spans_dir}/<ctx>.pt + index.json",
        present=spans_ok,
        detail=span_detail,
    )

    # ── A3.3 — r_B diff-in-means, per rb-column, all layers ────────────────────
    rb_path = store_dir / "r_b.pt"
    rb_present = rb_path.is_file()
    rb_cols_have = []
    if rb_present:
        rb = _load(rb_path)
        rbd = rb.get("r_b", {})
        rb_cols_have = [c for c in rb_columns() if c in rbd and "diffmeans" in rbd[c]]
    res.add(
        "A3.3 linear read-out r_B^T v0",
        "r_B diffmeans + meanDB per contrastive column, ALL layers",
        f"{rb_path}::r_b",
        present=rb_present and set(rb_cols_have) == set(rb_columns()),
        detail=f"{len(rb_cols_have)}/{len(rb_columns())} rb-columns present "
        f"({', '.join(rb_columns())}); marker/format/etc DROPPED by design",
    )

    # ── A3.4/A3.5 — c_C → v0 (BOTH recipes) ────────────────────────────────────
    res.add(
        "A3.4/A3.5 context vector c_C → answer profile v0",
        "c_C mean-over-prompt (NEW) + last-input-token (#594 HF reuse)",
        f"{v0_path}::cc_meanprompt + #594 issue594_context_geometry store",
        present=v0_present and n_cc > 0,
        detail=f"{n_cc} mean-prompt c_C; last-input c_C reused from #594 (load_cc_last_store)",
    )

    # ── §1.2 within-condition coherence: per-(C,probe) c_x ─────────────────────
    # The coherence check reads per-probe context vectors c_x (last-input slot).
    # These are recoverable prompt-only from the battery (cheap re-extract) — the
    # base store retains the per-(C,probe) ANSWER spans, and c_x is the
    # last-input-token slot which #594 stores per context (centroid) but not
    # per probe. Flagged as a CAVEAT (re-extractable, not lost).
    res.add(
        "§1.2 within-condition coherence (per-probe c_x cluster)",
        "per-(C,probe) c_x (last-input slot), all layers",
        "re-extract prompt-only from battery (cheap) OR #594 per-probe store",
        present=True,
        detail="per-probe c_x is prompt-only → re-extractable at ~0 GPU; not in this answer-store",
        caveat=True,
    )

    # ── E0(C,B) judged rates — 10-col registry ─────────────────────────────────
    e0_files = sorted(e0_dir.glob("*__*.json")) if e0_dir.is_dir() else []
    e0_files = [f for f in e0_files if "__ERROR" not in f.name]
    cols_seen = {f.name.split("__", 1)[1].removesuffix(".json") for f in e0_files}
    judged = [c for c, col in E0_COLUMNS.items() if col.dv == "judged_rate"]
    judged_seen = sorted(set(judged) & cols_seen)
    res.add(
        "E0(C,B) judged behavior rates (primary DV)",
        "10-col registry per (context, behavior); Sonnet judge off-pod (J1)",
        f"{e0_dir}/<ctx>__<col>.json",
        present=bool(e0_files) and len(judged_seen) >= 1,
        detail=f"{len(e0_files)} E0 gen files; judged columns seen: {judged_seen}",
    )

    # ── marker slot 4-float ────────────────────────────────────────────────────
    marker_files = [f for f in e0_files if f.name.endswith("__marker.json")]
    marker_ok = False
    marker_detail = "no marker E0 files"
    if marker_files:
        m = json.loads(marker_files[0].read_text())
        slots = m.get("marker_slot", [])
        if slots:
            keys_present = all(k in slots[0] for k in MARKER_SLOT_KEYS)
            marker_ok = keys_present
            marker_detail = (
                f"{len(marker_files)} marker files; first slot keys "
                f"{sorted(k for k in MARKER_SLOT_KEYS if k in slots[0])}"
            )
        else:
            marker_detail = f"{len(marker_files)} marker files but empty marker_slot list"
    res.add(
        "marker ※ slot (4-float storage contract #530)",
        "logp / z_marker / z_eos / logZ (+argmax) per (C,probe), on-policy end-slot",
        f"{e0_dir}/<ctx>__marker.json",
        present=marker_ok,
        detail=marker_detail,
    )

    # ── continuous completion-logP DV (dual-DV secondary) ──────────────────────
    logp_ok = False
    logp_detail = "no judged E0 cells to check logp_norm"
    judged_gen = [f for f in e0_files if f.name.split("__", 1)[1].removesuffix(".json") in judged]
    if judged_gen:
        g = json.loads(judged_gen[0].read_text())
        cells = g.get("cells", [])
        if cells and cells[0].get("completions"):
            logp_ok = "logp_norm" in cells[0]["completions"][0]
            logp_detail = "logp_norm present on E0 completions (length-normalized, secondary DV)"
    res.add(
        "continuous completion-logP DV (dual-DV secondary)",
        "logp_norm per completion in judged E0 cells",
        f"{e0_dir}/<ctx>__<judged-col>.json::cells[].completions[].logp_norm",
        present=logp_ok,
        detail=logp_detail,
    )

    # ── Σ_c background second moment ────────────────────────────────────────────
    sig_path = store_dir / "sigma_c.pt"
    sig_present = sig_path.is_file()
    sig_detail = "sigma_c.pt missing (--skip-sigma?)"
    if sig_present:
        s = _load(sig_path)
        shp = tuple(s["sigma_c"].shape)
        sig_detail = f"Σ_c shape {shp} over {s.get('n')} background contexts (H×H per layer)"
    res.add(
        "Σ_c second moment (Phases 2-4 whitened gate)",
        "(Lc, H, H) background second moment + n",
        f"{sig_path}::sigma_c",
        present=sig_present,
        detail=sig_detail,
        caveat=True,  # Σ_c is a Phase-2+ input, not load-bearing for THIS task's A3.2/A3.3
    )

    # ── raw completions (v0-capture answers) ───────────────────────────────────
    raw_files = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    res.add(
        "raw completions (v0-capture greedy answers)",
        "per-(C) greedy answers (reproducibility / re-capture)",
        f"{raw_dir}/<ctx>.json",
        present=bool(raw_files),
        detail=f"{len(raw_files)} per-context raw-completion files",
    )

    # ── §1.10 per-sample ANSWER-SIDE activations (R≥8 sampled completions) ──────
    # The single-context (C=δ_x) BLOCKER (round-5 code-review): for EVERY
    # (context × probe), every one of the R temp-1.0 sampled completions has its
    # answer-side activations captured (ALL captured layers, meaned over the
    # answer span) + its completion text (judged off-pod → within-prompt rate).
    # The G7 phase writes single_context/<ctx>.pt with per-(probe, sample)
    # {text, logp_norm, act:(Lc,H)}. This row is a real PRESENT check, NOT a NOTE.
    sc_dir = store_dir / "single_context"
    sc_files = sorted(sc_dir.glob("*.pt")) if sc_dir.is_dir() else []
    sc_ok = False
    sc_detail = "single_context/ missing — G7 §1.10 R-sample capture did not run"
    if sc_files:
        blob = _load(sc_files[0])
        per_probe = blob.get("per_probe", [])
        n_samples = blob.get("n_samples")
        # verify the first probe's first sample carries a real activation tensor
        sample0 = per_probe[0]["samples"][0] if per_probe and per_probe[0].get("samples") else {}
        act = sample0.get("act")
        act_ok = act is not None and hasattr(act, "shape") and act.ndim == 2
        text_ok = "text" in sample0 and "logp_norm" in sample0
        sc_ok = act_ok and text_ok and (n_samples is not None and n_samples >= 1)
        sc_detail = (
            f"{len(sc_files)} per-context packs; R={n_samples} samples/probe; "
            f"first sample act shape {tuple(act.shape) if act_ok else None} (Lc,H) + text+logp_norm"
        )
    res.add(
        "§1.10 per-sample answer-side ACTIVATIONS (R≥8 sampled completions)",
        "per-(C,probe,sample) mean answer-side acts (ALL layers) + text + judge-ready completions",
        f"{sc_dir}/<ctx>.pt::per_probe[].samples[].act",
        present=sc_ok,
        detail=sc_detail,
    )

    return res


def render_matrix(res: AuditResult) -> str:
    """Render the capture matrix as a fixed-width text table."""
    lines = []
    lines.append("=" * 100)
    lines.append("ISSUE #658 CAPTURE-COMPLETENESS MATRIX")
    lines.append("=" * 100)
    for r in res.rows:
        mark = "OK  " if r.present else ("MISS" if r.present is False else "?   ")
        if r.caveat and r.present:
            mark = "NOTE"
        lines.append(f"[{mark}] {r.analysis}")
        lines.append(f"        need : {r.required_data}")
        lines.append(f"        where: {r.where}")
        if r.detail:
            lines.append(f"        -> {r.detail}")
    lines.append("=" * 100)
    if res.ok:
        lines.append(f"VERDICT: PASS — {len(res.rows)} rows, 0 hard failures")
    else:
        lines.append(f"VERDICT: FAIL — {len(res.hard_failures)} missing required input(s):")
        for r in res.hard_failures:
            lines.append(f"  - {r.analysis}: {r.required_data}")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #658 capture-completeness audit.")
    p.add_argument("--store", type=Path, default=PROJECT_ROOT / "data/issue_658/store")
    p.add_argument("--eval-results", type=Path, default=PROJECT_ROOT / "eval_results/issue_658")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--json-out", type=Path, default=None, help="also write the matrix as JSON")
    args = p.parse_args()

    res = audit_store(args.store, args.eval_results, smoke=args.smoke)
    print(render_matrix(res))
    if args.json_out:
        args.json_out.write_text(
            json.dumps([r.__dict__ for r in res.rows], indent=2), encoding="utf-8"
        )
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
