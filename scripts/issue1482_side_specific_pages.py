"""Issue #1482 — two browsable pages of side-specific SAE features, sorted by activation.

  dashboard/public/context-only-1482.html   1,654 features that fire in the CONTEXT
                                            and never in the answer
  dashboard/public/answer-only-1482.html    2,164 features that fire in the ANSWER
                                            and never in the context

Each page is the feature DESCRIPTIONS ranked by that feature's own firing frequency,
descending, with per-feature R^2 alongside. Companion to the per-token answer-side
ratio panel `figures/issue_1482/side_specificity/side_ratio_token.png`, which uses the
SAME census membership, so the three artifacts tell one story.

MEMBERSHIP is the 120,000-row census (`side_class` in fullwidth_discrete_covariates.npz),
NOT the 2,000-row per-token capture's one-sided sets (5,994 / 11,855). The subsample
inflates one-sidedness several-fold purely by having fewer rows in which a feature could
have fired on the other side; the census call is the strong one.

SORT KEY — one deliberate asymmetry, forced by the data. `activity` in
fullwidth_covariates.npz is the ANSWER-side per-row firing frequency (it equals
cnt_fit / 120,000, verified), so it is identically 0.0 for ALL 1,654 context-only
features — that is definitional: `side_class` calls a feature context-only precisely
when its answer activity is 0. Sorting the context-only page by it would order 1,654
rows by a constant. Each page is therefore sorted by the firing frequency ON THE SIDE
WHERE THE FEATURE ACTUALLY FIRES, which is the same measure computed on the live side:
  answer-only  -> activity            = cnt_fit     / 120,000   (the specified key)
  context-only -> context_activity    = psi_cnt_fit / 120,000   (its exact analogue)
Both are stated on their page.

DESCRIPTIONS are the merged #1773 set: the full-dictionary release (125,915 rows) UNION
the #1934 recovery (1,690 rows), recovery winning on collision. NOTE the 16,091-row
`eval_results/issue_1773/labels/descriptions.jsonl` is a panel SUBSET, not the full
dictionary — merging only that with the recovery covers 0% of context-only and 0.9% of
answer-only features, so this script reads the full-dictionary release instead.

STRUCTURAL GAP, stated on the context-only page: #1773 describes a feature from its own
ANSWER-side activating windows and excluded features with zero such windows before
dispatch. A context-only feature has zero by the very property that makes it
context-only, so NONE of the 1,654 has a description. That page therefore ships every
measured quantity and no description text, with the reason in its header.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482-sidepages")

DICT_SIZE = 131_072
N_FIT = 120_000
SCAN = PROJECT_ROOT / "data/issue_1482/fullwidth/fused_scan.npz"
COV = PROJECT_ROOT / "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"
DISCRETE = (
    PROJECT_ROOT / "eval_results/issue_1482/predictor_battery/fullwidth_discrete_covariates.npz"
)
R2_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue1482_sidespec/ridge__mean_perfeature.npz")
R2_HF_PATH = "issue1482_densesae_fullwidth/perfeature/ridge__mean_perfeature.npz"
FULLDICT = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
RECOVERY = PROJECT_ROOT / "eval_results/issue_1773/recovery_1934/descriptions_recovered.jsonl"
COMPANION_FIG = "figures/issue_1482/side_specificity/side_ratio_token.png"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_1482/side_specific"
ARTIFACT_DIR = PROJECT_ROOT / "tasks/awaiting_promotion/1482/artifacts"
PUBLIC_DIR = PROJECT_ROOT / "dashboard/public"

CENSUS_EXPECTED = {"context_only": 1654, "two_sided": 126348, "answer_only": 2164, "dead": 906}
SCORED_EXPECTED = 121_111

PAGES = {
    "context_only": {
        "code": 0,
        "slug": "context-only-1482",
        "artifact": "context_only_features.html",
        "title": "Context-only SAE features",
        "blurb": "fire somewhere in the CONTEXT span and NEVER in the answer",
        "act_label": "context-side firing frequency",
        "act_note": (
            "rows in which the feature fires anywhere in the CONTEXT span, divided by "
            "120,000. This is the exact analogue of the answer-side <code>activity</code> "
            "used on the answer-only page: <code>activity</code> is answer-side and is "
            "identically 0 for every context-only feature (that is what makes them "
            "context-only), so it cannot order this page."
        ),
    },
    "answer_only": {
        "code": 2,
        "slug": "answer-only-1482",
        "artifact": "answer_only_features.html",
        "title": "Answer-only SAE features",
        "blurb": "fire somewhere in the ANSWER span and NEVER in the context",
        "act_label": "answer-side firing frequency (activity)",
        "act_note": (
            "the <code>activity</code> column of the full-width covariates — rows in "
            "which the feature fires anywhere in the ANSWER span, divided by 120,000."
        ),
    },
}


def _log(msg: str) -> None:
    logger.info("%s", msg)


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "unavailable-no-git-checkout"


def _provenance() -> dict:
    return {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)
    _log(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


# ── inputs ──────────────────────────────────────────────────────────────────────────


def load_descriptions() -> dict[int, dict]:
    """Full-dictionary #1773 release UNION the #1934 recovery, recovery winning."""
    out: dict[int, dict] = {}
    n_full = 0
    for p in sorted(FULLDICT.glob("descriptions.shard*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                f = int(r.get("feat_id", -1))
                if 0 <= f < DICT_SIZE:
                    out[f] = r
                    n_full += 1
    n_rec, n_collide = 0, 0
    if RECOVERY.exists():
        with RECOVERY.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                f = int(r.get("feat_id", -1))
                if 0 <= f < DICT_SIZE:
                    n_collide += f in out
                    out[f] = r  # recovery wins
                    n_rec += 1
    _log(
        f"descriptions: fulldict {n_full:,} + recovery {n_rec:,} "
        f"({n_collide} collisions, recovery won) -> {len(out):,} merged"
    )
    return out


def load_inputs() -> dict:
    with np.load(SCAN) as z:
        cnt = z["cnt_fit"].astype(np.int64)
        psi = z["psi_cnt_fit"].astype(np.int64)
        n_fit = int(z["n_fit"])
    assert n_fit == N_FIT, n_fit
    side = np.load(DISCRETE)["side_class"]
    got = {
        "context_only": int((side == 0).sum()),
        "two_sided": int((side == 1).sum()),
        "answer_only": int((side == 2).sum()),
        "dead": int((side == -1).sum()),
    }
    assert got == CENSUS_EXPECTED, f"census drift: {got}"
    activity = np.load(COV)["activity"].astype(np.float64)
    # the sort-key asymmetry is asserted, not assumed: activity IS the answer-side rate
    assert np.allclose(activity, cnt / n_fit), "activity is not cnt_fit/n_fit"
    assert float(activity[side == 0].max()) == 0.0, "context-only activity is not all-zero"
    ctx_activity = psi / n_fit

    if not R2_NPZ.exists():
        from explore_persona_space.orchestrate import hub

        R2_NPZ.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=R2_HF_PATH,
            target=R2_NPZ,
        )
    rz = np.load(R2_NPZ)
    r2, scored = rz["r2"].astype(np.float64), rz["scored"].astype(bool)
    assert int(scored.sum()) == SCORED_EXPECTED, int(scored.sum())
    _log(f"census verified {got}; R^2 scored {int(scored.sum()):,}/{scored.size:,}")
    return {
        "side": side,
        "cnt": cnt,
        "psi": psi,
        "activity": activity,
        "ctx_activity": ctx_activity,
        "r2": r2,
        "scored": scored,
        "desc": load_descriptions(),
    }


def build_rows(inp: dict, key: str) -> list[dict]:
    """Every feature of one side class, sorted by ITS OWN side's firing frequency."""
    spec = PAGES[key]
    act = inp["ctx_activity"] if key == "context_only" else inp["activity"]
    ids = np.nonzero(inp["side"] == spec["code"])[0]
    ids = ids[np.argsort(-act[ids], kind="stable")]
    rows = []
    for fid in ids.tolist():
        d = inp["desc"].get(fid) or {}
        rows.append(
            {
                "feat_id": fid,
                "activation": float(act[fid]),
                "rows_active_own_side": int(
                    inp["psi"][fid] if key == "context_only" else inp["cnt"][fid]
                ),
                "r2": float(inp["r2"][fid]) if inp["scored"][fid] else None,
                "scored": bool(inp["scored"][fid]),
                "description": d.get("description") or None,
                "confidence": d.get("confidence"),
            }
        )
    n_desc = sum(1 for r in rows if r["description"])
    _log(
        f"{key}: {len(rows):,} features, {n_desc:,} described "
        f"({100 * n_desc / max(len(rows), 1):.1f}%), "
        f"{sum(r['scored'] for r in rows):,} scored"
    )
    return rows


# ── page ────────────────────────────────────────────────────────────────────────────


def _esc(s: object) -> str:
    return html.escape(str(s)) if s not in (None, "") else "&mdash;"


CSS = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; } .head b { color:var(--fg); }
.warn { background:#fff7ed; border:1px solid #f6d6bd; border-radius:9px; padding:12px 14px;
  margin:10px 0; } .warn p { color:#8a4a12; margin:5px 0; }
.gap { background:#fdecec; border-color:#f0c2c2; } .gap p { color:#8a1212; }
.ctl { margin:12px 0 8px; font-size:13px; color:var(--mut); }
.ctl input { font:inherit; padding:3px 7px; border:1px solid var(--line); border-radius:6px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }
th, td { padding:5px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { position:sticky; top:0; background:#f4f6fa; text-align:left; font-weight:600;
  color:var(--fg); cursor:pointer; white-space:nowrap; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.d { color:#2c313b; }
td a { color:#1b4fd8; text-decoration:none; } td a:hover { text-decoration:underline; }
.unscored { color:var(--mut); font-style:italic; }
.nodesc { color:#a2470f; font-style:italic; }
"""

JS = """
function sortT(i, num) {
  const t = document.getElementById('t'), tb = t.tBodies[0];
  const rows = Array.from(tb.rows);
  t._d = (t._d === i) ? -1 - i : i;
  const asc = t._d === i;
  rows.sort((a, b) => {
    let x = a.cells[i].dataset.v ?? a.cells[i].textContent;
    let y = b.cells[i].dataset.v ?? b.cells[i].textContent;
    if (num) {
      x = parseFloat(x); y = parseFloat(y);
      if (isNaN(x)) x = -Infinity; if (isNaN(y)) y = -Infinity;
      return asc ? x - y : y - x;
    }
    return asc ? String(x).localeCompare(y) : String(y).localeCompare(x);
  });
  rows.forEach(r => tb.appendChild(r));
}
function filt() {
  const q = document.getElementById('q').value.toLowerCase();
  let n = 0;
  document.querySelectorAll('#t tbody tr').forEach(r => {
    const ok = !q || r.textContent.toLowerCase().includes(q);
    r.style.display = ok ? '' : 'none';
    if (ok) n++;
  });
  document.getElementById('shown').textContent = n.toLocaleString();
}
"""


def render(key: str, rows: list[dict], sha_note: str) -> str:
    spec = PAGES[key]
    other = "answer_only" if key == "context_only" else "context_only"
    n_desc = sum(1 for r in rows if r["description"])
    n_scored = sum(r["scored"] for r in rows)
    gap = ""
    if n_desc == 0:
        gap = """<div class="warn gap">
<p><b>There are no descriptions on this page, and that is the finding.</b> #1773
describes a feature from its own <i>answer-side</i> activating windows and excluded
features with zero such windows before dispatch. A context-only feature has zero
<i>by the very property that makes it context-only</i>, so not one of these 1,654
features was ever described or axis-labelled. Its exclusion set is exactly
{context-only} &cup; {dead}.</p>
<p>Everything measured is still here &mdash; firing frequency, rows active, R&sup2;.
Only the interpretive text is missing, and no re-run of the existing pipeline will
produce it: that needs a labelling pass over CONTEXT-side activating windows, which
does not exist yet.</p></div>"""
    body_rows = []
    for i, r in enumerate(rows, 1):
        r2 = f'<span class="unscored">unscored</span>' if not r["scored"] else f"{r['r2']:.3f}"
        desc = (
            _esc(r["description"])
            if r["description"]
            else '<span class="nodesc">no #1773 description &mdash; see header</span>'
        )
        body_rows.append(
            f'<tr><td class="n">{i}</td>'
            f'<td class="n"><a href="https://www.neuronpedia.org/qwen2.5-7b-instruct/19-'
            f'{r["feat_id"]}" target="_blank" rel="noopener">{r["feat_id"]}</a></td>'
            f'<td class="n" data-v="{r["activation"]:.10f}">{r["activation"]:.2e}</td>'
            f'<td class="n" data-v="{r["rows_active_own_side"]}">'
            f"{r['rows_active_own_side']:,}</td>"
            f'<td class="n" data-v="{r["r2"] if r["scored"] else ""}">{r2}</td>'
            f'<td class="d">{desc}</td></tr>'
        )
    return f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Issue 1482 &mdash; {spec["title"].lower()}</title>
<style>{CSS}</style></head><body><div class='wrap'>
<h1>{spec["title"]} (issue #1482)</h1>
<div class="warn">
<p><b>Descriptions are a reading aid, never evidence.</b> They come from #1773, whose
standing caveat is that they are search-index-only &mdash; neighbour discrimination
0.322 against a 0.50 bar. This page is <i>entirely</i> descriptions, so that caveat
carries more weight here than anywhere else, not less: any impression that these
features form a recognisable kind comes from text that cannot reliably tell a feature
from its neighbour.</p>
<p><b>"Never fires on the other side" is a strong criterion.</b> It means zero across
all <b>120,000 fit rows</b> at ROW-OCCUPANCY grain &mdash; a feature counts as active
if it fires <i>anywhere</i> in the span. A feature firing at a single token of a single
answer is not answer-only. Membership is the 120,000-row census, NOT the 2,000-row
per-token capture (whose one-sided sets are several times larger purely because fewer
rows give a feature fewer chances to fire on the other side).</p>
</div>
{gap}
<div class="head">
<p><b>What is shown.</b> All <b>{len(rows):,}</b> features that {spec["blurb"]}, ranked
by {spec["act_label"]} descending. {n_desc:,} of {len(rows):,} carry a #1773
description; {n_scored:,} of {len(rows):,} have a scored R&sup2;.</p>
<p><b>Sort key.</b> {spec["act_note"]}</p>
<p><b>R&sup2;</b> is the full-width dense-context &rarr; SAE-answer ridge read
({SCORED_EXPECTED:,} of {DICT_SIZE:,} dictionary columns are scored; the rest have zero
holdout answer variance). Unscored is shown as <span class="unscored">unscored</span>,
never as R&sup2; = 0.</p>
<p><b>Companion artifacts.</b> The per-token answer-side ratio panel
<code>{COMPANION_FIG}</code> uses this same census membership (1,654 / 2,164 as its two
point masses), and the sibling page is
<a href="{PAGES[other]["slug"]}.html">{PAGES[other]["title"].lower()}</a>.</p>
<p>{sha_note}</p>
</div>
<div class="ctl">search <input id="q" type="text" placeholder="description text"
oninput="filt()" style="width:280px"> &nbsp; showing
<b id="shown">{len(rows):,}</b> of {len(rows):,}
&nbsp;&middot;&nbsp; click any column header to re-sort</div>
<table id="t"><thead><tr>
<th class="n" onclick="sortT(0,1)">#</th>
<th class="n" onclick="sortT(1,1)">feature</th>
<th class="n" onclick="sortT(2,1)">{spec["act_label"]}</th>
<th class="n" onclick="sortT(3,1)">rows active</th>
<th class="n" onclick="sortT(4,1)">R&sup2;</th>
<th onclick="sortT(5,0)">#1773 description</th></tr></thead>
<tbody>{"".join(body_rows)}</tbody></table>
<script>{JS}</script>
</div></body></html>"""


# ── driver ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        print("import-check OK")
        sys.stdout.flush()
        sys.exit(0)

    t0 = time.time()
    inp = load_inputs()
    prov = _provenance()
    sha_note = (
        f"Generated {prov['timestamp_utc']} at commit <code>{prov['git_commit'][:12]}</code> "
        f"from the 120,000-row census; descriptions are the #1773 full-dictionary release "
        f"merged with the #1934 recovery."
    )
    summary = {}
    for key in PAGES:
        rows = build_rows(inp, key)
        _write_json(
            OUT_DIR / f"{key}_features_by_activation.json",
            {
                "page": PAGES[key]["slug"],
                "membership": "120,000-row census (side_class); NOT the 2,000-row capture",
                "sort_key": PAGES[key]["act_label"],
                "sort_key_note": html.unescape(
                    PAGES[key]["act_note"].replace("<code>", "").replace("</code>", "")
                ),
                "description_source": "#1773 full-dictionary release UNION #1934 recovery "
                "(recovery wins on collision)",
                "description_caveat": "search-index-only, neighbour discrimination 0.322 "
                "against a 0.50 bar — a reading aid, never evidence",
                "side_criterion": "zero firings on the other side across all 120,000 fit "
                "rows at ROW-OCCUPANCY grain (active anywhere in the span)",
                "n_features": len(rows),
                "n_described": sum(1 for r in rows if r["description"]),
                "n_scored": sum(r["scored"] for r in rows),
                "features": rows,
                "provenance": prov,
            },
        )
        body = render(key, rows, sha_note)
        for p in (ARTIFACT_DIR / PAGES[key]["artifact"], PUBLIC_DIR / f"{PAGES[key]['slug']}.html"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
            _log(f"wrote {p} ({p.stat().st_size / 1024:.0f} KiB)")
        summary[key] = {
            "n": len(rows),
            "described": sum(1 for r in rows if r["description"]),
            "scored": sum(r["scored"] for r in rows),
        }
    _log(f"ALL DONE in {time.time() - t0:.0f}s — {summary}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
