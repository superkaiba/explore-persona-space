#!/usr/bin/env python3
"""Generate metrics.tex (a LaTeX macro registry) from metrics.json.

v1.1 OPT-IN — NOT on the v1 required path. v1 papers write numbers as LITERALS
in the .tex and rely on the analyzer's numeric-fidelity re-extraction; they have
no metrics.json / metrics.tex and never call \\metric{}. This tool (with
metrics.json + verify_metric.py) is carried forward verbatim from the spike so
the v1.1 \\metric grounding upgrade is a documented, ready-to-wire opt-in. The
HERE-relative paths below assume a per-paper layout (metrics.json next to this
script); a future v1.1 wiring step parameterizes them.

The paper's \\metric{key} macro expands \\csname metric@<key>\\endcsname. This
script emits one \\expandafter\\def\\csname metric@<key>\\endcsname{<rendered>}
per metric key, so the .tex compiles standalone (no shell-escape) while every
value still traces back to metrics.json (checked by verify_metric.py).

Run from repo root:
    uv run python docs/papers/_spike/emit_metrics_tex.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "metrics.json"
OUT = HERE / "metrics.tex"


def main() -> None:
    metrics = json.loads(SRC.read_text())
    lines = [
        "% AUTO-GENERATED from metrics.json by emit_metrics_tex.py — do not edit.",
        "% Each macro is the rendered string for a \\metric{key} call.",
    ]
    n = 0
    for key, rec in metrics.items():
        if key.startswith("_"):
            continue
        rendered = rec["rendered"]
        # LaTeX-escape the rendered string defensively (values are numeric so
        # this is mostly a no-op, but a minus sign / decimal is safe as-is).
        lines.append(rf"\expandafter\def\csname metric@{key}\endcsname{{{rendered}}}")
        n += 1
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.name}: {n} macros")


if __name__ == "__main__":
    main()
