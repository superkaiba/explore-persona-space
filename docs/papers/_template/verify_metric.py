#!/usr/bin/env python3
"""verify_metric — the v1.1 OPT-IN number-grounding check (NOT on the v1 path).

v1 papers write numbers as LITERALS and rely on the analyzer's numeric-fidelity
re-extraction; verify_paper.py does NOT call this in v1 (it has no \\metric
check). This tool is carried forward verbatim from the spike so the v1.1
\\metric grounding upgrade is a documented, ready-to-wire opt-in. When v1.1 is
turned on, verify_paper.py grows a \\metric pass that calls this logic.

Given a .tex and a metrics.json, this:

  1. Parses every \\metric{key} call out of the .tex (handles \\metric{k},
     comments stripped, escaped braces ignored).
  2. For each key, resolves it in metrics.json. Missing key -> FAIL.
  3. Checks the rendered string is consistent with value+precision+transform
     (rounding the value to `precision` reproduces `rendered`). Mismatch -> FAIL.
  4. For grounded keys: resolves the source pointer (file + json_path) in
     eval_results and checks the JSON value, after the declared transform,
     equals the metric value at the declared precision. Unresolvable pointer
     or value mismatch -> FAIL.
  5. For analysis-derived keys: confirms a producer + inputs are declared, and
     that each input pointer resolves (a derived value must trace to grounded
     inputs). Missing producer/inputs -> FAIL.

Exit 0 if all PASS, 1 otherwise. Run from repo root:

    uv run python docs/papers/_spike/verify_metric.py \
        docs/papers/_spike/issue_657_spike.tex docs/papers/_spike/metrics.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# \metric{key} — key is [A-Za-z0-9_]+. We strip line comments (unescaped %) first.
METRIC_RE = re.compile(r"\\metric\{([A-Za-z0-9_]+)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def strip_comments(tex: str) -> str:
    return COMMENT_RE.sub("", tex)


def parse_metric_calls(tex: str) -> list[str]:
    return METRIC_RE.findall(strip_comments(tex))


def resolve_json_path(obj, path: str):
    """Resolve a dotted/indexed path like 'a.b[0].c' against a loaded JSON obj."""
    # tokenise into keys and [int] indices
    tokens = re.findall(r"\[(\d+)\]|([^.\[\]]+)", path)
    cur = obj
    for idx, key in tokens:
        cur = cur[int(idx)] if idx != "" else cur[key]
    return cur


def apply_transform(value, transform: str):
    if transform in ("identity", "ratio_round"):
        return value
    if transform == "abs":
        return abs(value)
    if transform.startswith("scale:"):
        return value * float(transform.split(":", 1)[1])
    if transform.startswith("round:"):
        return round(value, int(transform.split(":", 1)[1]))
    raise ValueError(f"unknown transform: {transform}")


def render_consistent(rec) -> bool:
    """rendered must equal value formatted to `precision` (after transform)."""
    value = apply_transform(rec["value"], rec["transform"])
    prec = rec["precision"]
    expected = f"{value:.{prec}f}" if isinstance(value, float) else str(value)
    rendered = rec["rendered"].lstrip("+")  # allow a leading + sign convention
    expected = expected.lstrip("+")
    # tolerate -0.00 vs 0.00
    if rendered in ("0." + "0" * prec, "-0." + "0" * prec) and float(rendered) == 0:
        return float(expected) == 0
    return rendered == expected


def main() -> int:  # noqa: C901 — single-pass spike verifier; readability over decomposition
    if len(sys.argv) != 3:
        print("usage: verify_metric.py <paper.tex> <metrics.json>", file=sys.stderr)
        return 2
    tex_path = Path(sys.argv[1])
    metrics_path = Path(sys.argv[2])
    tex = tex_path.read_text()
    metrics = json.loads(metrics_path.read_text())

    calls = parse_metric_calls(tex)
    used = sorted(set(calls))
    print(f"\\metric calls in {tex_path.name}: {len(calls)} ({len(used)} distinct)")

    failures: list[str] = []
    json_cache: dict[str, dict] = {}

    def load_json(rel: str):
        if rel not in json_cache:
            p = REPO / rel
            if not p.exists():
                return None
            json_cache[rel] = json.loads(p.read_text())
        return json_cache[rel]

    n_grounded = n_derived = 0
    for key in used:
        if key.startswith("_") or key not in metrics:
            failures.append(f"{key}: not defined in metrics.json")
            continue
        rec = metrics[key]

        # rendered consistency
        try:
            if not render_consistent(rec):
                failures.append(
                    f"{key}: rendered '{rec['rendered']}' != value "
                    f"{rec['value']} @ precision {rec['precision']} "
                    f"(transform {rec['transform']})"
                )
        except Exception as e:
            failures.append(f"{key}: render check error: {e}")

        src = rec["source"]
        if src.get("kind") == "analysis-derived":
            n_derived += 1
            if not src.get("producer"):
                failures.append(f"{key}: analysis-derived but no producer declared")
            inputs = src.get("inputs") or []
            if not inputs:
                failures.append(f"{key}: analysis-derived but no inputs declared")
            for inp in inputs:
                # inputs are "file#json_path"
                if "#" not in inp:
                    failures.append(f"{key}: malformed input pointer '{inp}'")
                    continue
                f, jpth = inp.split("#", 1)
                obj = load_json(f)
                if obj is None:
                    failures.append(f"{key}: input file unresolvable: {f}")
                    continue
                try:
                    resolve_json_path(obj, jpth)
                except Exception:
                    failures.append(f"{key}: input path unresolvable: {inp}")
            continue

        # grounded: resolve source pointer + check value at precision
        n_grounded += 1
        f = src["file"]
        jpth = src["json_path"]
        obj = load_json(f)
        if obj is None:
            failures.append(f"{key}: source file unresolvable: {f}")
            continue
        try:
            raw = resolve_json_path(obj, jpth)
        except Exception:
            failures.append(f"{key}: source json_path unresolvable: {f}#{jpth}")
            continue
        try:
            xform = apply_transform(raw, rec["transform"])
        except Exception as e:
            failures.append(f"{key}: transform error on source value: {e}")
            continue
        prec = rec["precision"]
        # precision-aware equality: both rounded to `precision`
        if isinstance(xform, float) or isinstance(rec["value"], float):
            if round(float(xform), prec) != round(float(rec["value"]), prec):
                failures.append(
                    f"{key}: source value {raw} (xform {xform}) != metric value "
                    f"{rec['value']} at precision {prec}"
                )
        else:
            if xform != rec["value"]:
                failures.append(f"{key}: source value {raw} != metric value {rec['value']}")

    print(f"  grounded keys checked: {n_grounded}")
    print(f"  analysis-derived keys checked: {n_derived}")

    # also report metrics defined but unused (informational, not a failure)
    defined = {k for k in metrics if not k.startswith("_")}
    unused = sorted(defined - set(used))
    if unused:
        print(f"  metrics defined but unused by this .tex: {len(unused)} ({', '.join(unused)})")

    if failures:
        print(f"\nFAIL ({len(failures)} issue(s)):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 1
    print("\nPASS: every \\metric call resolves, renders consistently, and grounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
