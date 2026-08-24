"""Per-cell view of the #2215 separation-comparison round: ratio figure + examples doc.

Renders (a) a dumbbell chart of the mapped-vs-real separation ratios for all 45
cells of the separation-comparison round (`eval_results/issue_2215/
separation_comparison/sepcmp.json`), and (b) a markdown doc giving, per cell,
the concrete varied-span values (what changes between the minimal-pair sides)
and one carrier snippet (what stays fixed), so the per-cell ratios can be read
against how much text the manipulation actually touches.

Inputs (all frozen/committed — no model calls, no GPU):
  - eval_results/issue_2215/separation_comparison/sepcmp.json  (ratios)
  - parent bank: issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json on the HF
    data repo at the #2215-pinned revision (values + carriers per parent cell)
  - dbe bank: src/explore_persona_space/experiments/issue2215/bank_dbe_values.json
    read from git history (commit f8f3ec9338 — the frozen Phase G bank)

Outputs:
  - figures/issue_2215/sepcmp_percell_ratios.{png,pdf} (+ meta sidecars)
  - eval_results/issue_2215/separation_comparison/cell_examples.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
SEPCMP = REPO / "eval_results/issue_2215/separation_comparison/sepcmp.json"
OUT_MD = REPO / "eval_results/issue_2215/separation_comparison/cell_examples.md"
FIG_DIR = REPO / "figures/issue_2215"
PARENT_REV = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
DBE_BANK_COMMIT = "f8f3ec9338"
TRUNC_VALUE = 260
TRUNC_CARRIER = 180


def _trunc(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def load_parent_bank() -> dict:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    p = hub.retry_transient(
        lambda: hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json",
            repo_type="dataset",
            revision=PARENT_REV,
            local_dir=REPO / "data/issue_2215/sepcmp_doc_dl",
        ),
        what="hf_hub_download parent bank.json",
    )
    return json.loads(Path(p).read_text())


def load_dbe_bank() -> dict:
    raw = subprocess.run(
        [
            "git",
            "show",
            f"{DBE_BANK_COMMIT}:src/explore_persona_space/experiments/issue2215/bank_dbe_values.json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout
    return json.loads(raw)


def parent_example(bank: dict, cell: str) -> tuple[dict[str, str], str]:
    """Return ({value_id: varied text}, carrier snippet) for a parent cell."""
    c = bank["cells"][cell]
    values = {k: _trunc(v, TRUNC_VALUE) for k, v in c["values"].items()}
    carriers = c.get("carriers") or {}
    snippet = ""
    if carriers:
        first = next(iter(carriers.values()))
        snippet = _trunc(first.get("text", ""), TRUNC_CARRIER)
    return values, snippet


def dbe_example(bank: dict, cell: str) -> tuple[dict[str, str], str]:
    """Return ({value_id: varied text}, carrier snippet) for a dbe cell."""
    t = bank["types"][cell]
    carr = t["carriers"]
    cid = sorted(carr)[0]
    c = carr[cid]
    if cell == "conversation_topic":
        vals = {
            v: _trunc(conv["user_turns"][0], TRUNC_VALUE) for v, conv in c["conversations"].items()
        }
        return vals, "(whole prior conversation swaps with the topic; shared final query appended)"
    if cell in ("style_register", "conversation_language"):
        vals = {v: _trunc(turns[0], TRUNC_VALUE) for v, turns in c["user_turns"].items()}
        return vals, _trunc(c["assistant_turns"][0], TRUNC_CARRIER)
    if cell == "user_role_identity":
        tmpl = c["turn1_template"]
        return {"template": _trunc(tmpl, TRUNC_VALUE)}, _trunc(c["final_query"], TRUNC_CARRIER)
    if cell == "user_sentiment":
        vals = {
            f"{v} ({c['labels'][v]})": _trunc(txt, TRUNC_VALUE) for v, txt in c["texts"].items()
        }
        return vals, "(IMDb contrast-set review pair; shared final query appended)"
    if cell == "user_doc_format":
        vals = {v: _trunc(r, TRUNC_VALUE) for v, r in c["renderings"].items()}
        return vals, _trunc(c["final_query"], TRUNC_CARRIER)
    if cell == "refusal_request":
        vals = {v: _trunc(p, TRUNC_VALUE) for v, p in c["prompts"].items()}
        return vals, "(XSTest safe/unsafe prompt pair, lexically matched)"
    if cell == "code_vs_prose":
        vals = {v: _trunc(p, TRUNC_VALUE) for v, p in c["presentations"].items()}
        return vals, _trunc(c["assistant_ack"], TRUNC_CARRIER)
    raise KeyError(f"no dbe extractor for cell {cell!r}")


def parent_realized_diff_chars(bank: dict, cell: str) -> int | None:
    """Chars differing between one realized pair's sides (system+history+user turns).

    The `values` dict holds LABELS for some parent cells (prior_topic:
    "birthday"/"hiking") while the realized varied span is a whole exchange —
    so the doc's size column must measure the realized diff, not label length.
    """
    ctxs = (
        bank["contexts"] if isinstance(bank["contexts"], list) else list(bank["contexts"].values())
    )
    members = [x for x in ctxs if x["cell"] == cell]
    if not members:
        return None
    car = members[0]["carrier"]
    sides = [x for x in members if x["carrier"] == car]
    if len(sides) < 2:
        return None

    def turns(x: dict) -> list[str]:
        hist = [t["content"] for t in x.get("history", [])]
        return [x.get("system") or "", *hist, x.get("user") or ""]

    ta, tb = turns(sides[0]), turns(sides[1])
    diff = sum((len(x) + len(y)) / 2 for x, y in zip(ta, tb) if x != y)
    diff += sum(len(t) for t in ta[len(tb) :]) + sum(len(t) for t in tb[len(ta) :])
    return int(diff)


def render_figure(cells: list[dict]) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rows = sorted(cells, key=lambda c: c["spaces"]["real"]["median_ratio"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * n + 1.4))
    c_real = paper_color("oracle_answer")
    c_map = paper_color("neural_map")
    for i, c in enumerate(rows):
        r = c["spaces"]["real"]["median_ratio"]
        m = c["spaces"]["mapped_779ce"]["median_ratio"]
        ax.plot([r, m], [i, i], color="0.75", lw=1.2, zorder=1)
        ax.scatter([r], [i], color=c_real, s=26, zorder=2)
        ax.scatter([m], [i], color=c_map, s=26, zorder=2, marker="s")
    ax.set_yticks(range(n))
    labels = [f"{c['cell']}{' †' if c['battery'] == 'dbe' else ''}" for c in rows]
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xscale("log")
    ax.axvline(1.0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("separation ratio (pair dist / cross-carrier yardstick)")
    ax.set_title("Per-cell separation: real answer vectors vs map predictions")
    ax.scatter([], [], color=c_real, s=26, label="real answer vectors")
    ax.scatter([], [], color=c_map, s=26, marker="s", label="mapped predictions (779ce)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "sepcmp_percell_ratios", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    sep = json.loads(SEPCMP.read_text())
    cells = sep["cells"]
    parent_bank = load_parent_bank()
    dbe_bank = load_dbe_bank()

    render_figure(cells)

    rows = sorted(cells, key=lambda c: -c["spaces"]["real"]["median_ratio"])
    lines = [
        "# Separation-comparison round: per-cell ratios + one concrete example per cell",
        "",
        "Companion to `sepcmp.json` (issue #2215, separation-comparison inline round).",
        "Per cell: the median separation ratio in the real answer-vector space and the",
        "single-turn map's prediction space (ratio = minimal-pair cosine distance divided",
        "by the same space's same-value cross-carrier yardstick), the mapped-minus-real",
        "contrast (* = carrier-clustered bootstrap 95% CI excludes zero), and the actual",
        "varied-span values from the frozen banks so the ratio can be read against how",
        "much text the manipulation touches. Cells sorted by real-space ratio, descending.",
        "† = 8-type content battery (dbe); others = parent battery.",
        "",
        "![per-cell ratios](../../../figures/issue_2215/sepcmp_percell_ratios.png)",
        "",
    ]
    for c in rows:
        cell, battery = c["cell"], c["battery"]
        r = c["spaces"]["real"]["median_ratio"]
        m = c["spaces"]["mapped_779ce"]["median_ratio"]
        ct = c["contrasts"]["mapped_779ce_minus_real"]
        star = "\\*" if (ct["ci_lo"] > 0 or ct["ci_hi"] < 0) else ""
        try:
            if battery == "dbe":
                values, carrier = dbe_example(dbe_bank, cell)
            else:
                values, carrier = parent_example(parent_bank, cell)
        except KeyError as e:
            values, carrier = {"(no extractor)": str(e)}, ""
        if battery == "dbe":
            varied_chars = int(sum(len(v) for v in values.values()) / max(len(values), 1))
            size_note = f"~{varied_chars} chars per example varied value"
        else:
            realized = parent_realized_diff_chars(parent_bank, cell)
            size_note = (
                f"~{realized} chars differ between realized pair sides"
                if realized is not None
                else "realized diff size unavailable"
            )
        lines.append(f"## {cell}{' †' if battery == 'dbe' else ''}")
        lines.append("")
        lines.append(
            f"real **{r:.3f}** · mapped **{m:.3f}** · mapped−real **{ct['point']:+.3f}{star}** "
            f"[{ct['ci_lo']:+.3f}, {ct['ci_hi']:+.3f}] · {size_note}"
        )
        lines.append("")
        lines.append(
            "What varies between the pair sides"
            + (
                " (value text, or LABEL when the realized span is a whole exchange):"
                if battery != "dbe"
                else ":"
            )
        )
        lines.append("")
        for vid, vtxt in values.items():
            lines.append(f"- **{vid}**: {vtxt}")
        if carrier:
            lines.append("")
            lines.append(f"Held constant (carrier example): {carrier}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD} ({len(rows)} cells) + figures under {FIG_DIR}/sepcmp_percell_ratios.*")


if __name__ == "__main__":
    main()
