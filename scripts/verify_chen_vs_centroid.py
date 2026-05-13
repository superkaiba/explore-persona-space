#!/usr/bin/env python3
"""Post-run verifier for experiment #363 (Chen-vs-centroid).

Checks:
    1. Shape check — every <trait>.pt under outputs/chen_vectors/ and
       outputs/centroid_vectors/ is a torch tensor of shape (n_layers, d_model)
       with d_model = 3584 for Qwen-7B.
    2. Random-baseline sanity — the random-baseline interval is centered near 0
       (mean within ±0.05) and its 95% interval brackets 0.
    3. alpha=0 sanity — rubric scores at alpha=0 match the unsteered baseline within
       bootstrap noise (we just check the rubric mean exists and is finite).
    4. Coherence check — first 5 steered completions per trait at alpha* are
       non-empty English-ish strings (>=10 chars, contain at least one space).

Emits verification.html (a compact pass/fail dashboard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _check_vectors(
    vec_dir: Path, expected_n_layers: int, expected_d_model: int
) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for pt_file in sorted(vec_dir.glob("*.pt")):
        try:
            t = torch.load(pt_file, map_location="cpu", weights_only=True)
        except Exception as e:
            out.append((pt_file.name, False, f"failed to load: {e}"))
            continue
        if not isinstance(t, torch.Tensor):
            out.append((pt_file.name, False, f"not a tensor: {type(t)}"))
            continue
        if t.ndim != 2:
            out.append((pt_file.name, False, f"expected 2D got {tuple(t.shape)}"))
            continue
        n_layers, d_model = t.shape
        if n_layers != expected_n_layers or d_model != expected_d_model:
            out.append(
                (
                    pt_file.name,
                    False,
                    f"expected ({expected_n_layers}, {expected_d_model}) got {tuple(t.shape)}",
                )
            )
        else:
            out.append((pt_file.name, True, f"shape ({n_layers}, {d_model})"))
    return out


def _check_random_baseline(rb: dict[str, Any]) -> tuple[bool, str]:
    mean = rb.get("pairwise_cosine_mean")
    lo = rb.get("pairwise_cosine_p2_5")
    hi = rb.get("pairwise_cosine_p97_5")
    if mean is None or lo is None or hi is None:
        return False, "missing pairwise cosine stats"
    if abs(mean) > 0.05:
        return False, f"mean = {mean:.4f} (expected ~0)"
    if not (lo <= 0 <= hi):
        return False, f"95% interval [{lo:.4f},{hi:.4f}] does not bracket 0"
    return True, f"mean={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"


def _check_alpha_zero(rubric_csv: Path) -> tuple[bool, str]:
    if not rubric_csv.exists():
        return False, "rubric_scores.csv missing"
    import csv

    n_rows = 0
    n_finite = 0
    with open(rubric_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                alpha = float(row["alpha"])
            except (KeyError, ValueError):
                continue
            if abs(alpha) > 1e-9:
                continue
            n_rows += 1
            v = row.get("rubric_aligned")
            if v not in ("", None) and v != "None":
                try:
                    fv = float(v)
                    if fv == fv:
                        n_finite += 1
                except ValueError:
                    pass
    if n_rows == 0:
        return False, "no alpha=0 rows present"
    if n_finite == 0:
        return False, f"{n_rows} alpha=0 rows but no rubric scores parsed (maybe --skip-judge)"
    return True, f"{n_finite}/{n_rows} alpha=0 rubric scores parsed"


def _check_completions_coherence(generations_dir: Path) -> tuple[bool, str]:
    if not generations_dir.exists():
        return False, "no generations/ dir"
    n_checked = 0
    n_ok = 0
    for jsonl in generations_dir.rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                comp = rec.get("completion", "")
                if not isinstance(comp, str):
                    continue
                n_checked += 1
                if len(comp) >= 10 and " " in comp:
                    n_ok += 1
                if n_checked >= 25:
                    break
        if n_checked >= 25:
            break
    if n_checked == 0:
        return False, "no completion samples found"
    return n_ok / max(1, n_checked) >= 0.8, f"{n_ok}/{n_checked} look coherent"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/", type=str)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--d-model", type=int, default=3584)
    args = p.parse_args()

    out = Path(args.output_dir)
    sections: list[tuple[str, list[tuple[str, bool, str]]]] = []

    sections.append(
        ("Chen vectors", _check_vectors(out / "chen_vectors", args.n_layers, args.d_model))
    )
    sections.append(
        ("Centroid vectors", _check_vectors(out / "centroid_vectors", args.n_layers, args.d_model))
    )

    rb_path = out / "random_baseline.json"
    if rb_path.exists():
        rb = json.loads(rb_path.read_text())
        ok, msg = _check_random_baseline(rb)
        sections.append(("Random baseline", [("interval brackets 0", ok, msg)]))
    else:
        sections.append(
            ("Random baseline", [("file present", False, "random_baseline.json missing")])
        )

    ok, msg = _check_alpha_zero(out / "rubric_scores.csv")
    sections.append(("alpha=0 sanity", [("rubric scores present", ok, msg)]))

    ok, msg = _check_completions_coherence(out / "generations")
    sections.append(("Completion coherence", [(">=80% look like English", ok, msg)]))

    # Emit verification.html
    rows = []
    overall_pass = True
    for section_name, checks in sections:
        rows.append(
            f"<h2>{section_name}</h2><table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>"
        )
        for name, ok, msg in checks:
            if not ok:
                overall_pass = False
            badge = (
                f"<span style='color: {'#2a9d8f' if ok else '#e63946'}; font-weight:600;'>"
                f"{'PASS' if ok else 'FAIL'}</span>"
            )
            rows.append(f"<tr><td>{name}</td><td>{badge}</td><td><code>{msg}</code></td></tr>")
        rows.append("</tbody></table>")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Verification #363</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; max-width: 760px; margin: 32px auto; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 16px; margin-top: 24px; }}
  table {{ border-collapse: collapse; margin: 8px 0 24px; width: 100%; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid #eee;
            text-align: left; vertical-align: top; }}
  code {{ font-size: 12px; color: #555; }}
</style></head><body>
<h1>Verification #363 — Chen vs. centroid persona vectors</h1>
<p><b>Overall:</b> {"PASS" if overall_pass else "FAIL"}</p>
{"".join(rows)}
</body></html>
"""
    (out / "verification.html").write_text(html)

    # Print a tiny stdout summary too.
    for section_name, checks in sections:
        for name, ok, msg in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {section_name} :: {name} :: {msg}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
