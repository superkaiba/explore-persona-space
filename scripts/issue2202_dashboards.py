#!/usr/bin/env python3
"""Issue #2202 — P6: the two committed dashboards (VM-side).

- ``failures-2202.html`` — index + ``failures-2202_pK.html`` shards
  (≤ ~2.5 MB/shard, client-side filter per shard, index jump table): ALL FAIL-1
  rows (no row cap — task-body lock), WORST-tail flagged, per-row texts at the
  locked caps (history 800 / last user 1,200 / answer 1,000 — truncation
  disclosed inline per passage), up to 10 confusers each with the similarity
  block per metric space, the three pool-wide ranks, both contexts' #1738
  labels with match/mismatch flags, and the attribution flag.
- ``sample500-2202.html`` (+ shards) — the 500 sample rows with the retrieval
  list (A) and the prediction-collapse list (B) side by side, plus the
  Result-2 Fable→Sonnet mode rates when the judge wave has landed.

Total committed payload is hard-capped at 40 MB: when over, the CONFUSER
excerpt caps tighten 400 → 240 → 160 chars (row count is never cut). Raw
corpus text appears ONLY in this display layer + the HF ``dashboard_rows/``
shards (the full text-bearing row JSON); every committed eval_results JSON
stays text-free. Style/skeleton reuse:
``scripts/issue1482_context_extremes_dashboard.py`` (committed-HTML precedent,
served at ``https://eps.superkaiba.com/<file>.html``).

Smoke: builds into a scratch dir (never ``dashboard/public/`` — smoke outputs
must not overwrite committed artifacts) from the smoke out-eval tree.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2202_failchar as FC  # noqa: E402
import issue2202_labels as LB  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2202_dash")

SHARD_BYTES = 2_500_000  # ≤ ~2.5 MB per shard (plan §4 P6)
TOTAL_CAP_BYTES = 40_000_000  # committed-payload hard cap (task-body lock: rows never cut)
CONFUSER_CAP_LADDER = (400, 240, 160)  # tightening ladder when over the total cap
LIST_ENTRY_CAP = 200  # sample-list entry snippet cap

CSS = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:24px 20px 60px; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1900px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 6px; } h2 { font-size:16px; margin:26px 0 6px; }
p, li { color:var(--mut); font-size:13px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin-bottom:10px; }
.nav a { margin-right:12px; font-size:13px; color:#1b4fd8; text-decoration:none; }
.ctl { margin:10px 0 8px; font-size:13px; color:var(--mut); }
.ctl input { font:inherit; padding:3px 7px; border:1px solid var(--line); border-radius:6px;
  width:280px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:12.5px; }
th, td { padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { position:sticky; top:0; background:#f4f6fa; text-align:left; font-weight:600; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.tag { font-size:11px; padding:1px 6px; border-radius:9px; border:1px solid var(--line);
  white-space:nowrap; background:#f4f6fa; }
.tag.warn { background:#fff2f0; border-color:#f3c1bb; }
.tag.ok { background:#eefaf0; border-color:#bfe6c7; }
.txt { white-space:pre-wrap; word-break:break-word; color:#2c313b; font-size:12px; }
.sub { width:100%; font-size:11.5px; margin-top:4px; }
.sub td, .sub th { padding:3px 5px; }
.mm { color:#b3261e; font-weight:600; } .match { color:#1e7f37; }
"""
JS = """
function filt(inp) {
  const q = inp.value.toLowerCase();
  const tbl = document.getElementById(inp.dataset.tbl);
  for (const tr of tbl.tBodies[0].rows)
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
}
"""


def esc(s: object) -> str:
    return html.escape(str(s if s is not None else "—"), quote=True)


def page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style><script>{JS}</script></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )


def label_cells(lab_i: dict | None, lab_j: dict | None) -> str:
    """#1738 label pair with per-field match/mismatch flags."""
    if not lab_i and not lab_j:
        return "<i>unlabeled</i>"
    fields = ("language", "topic", "request_refusal_adjacent", "answer_is_refusal", "format")
    parts = []
    for f in fields:
        vi = (lab_i or {}).get(f)
        vj = (lab_j or {}).get(f)
        klass = "match" if (vi is not None and vi == vj) else "mm"
        parts.append(f"<span class='{klass}'>{esc(f)}: {esc(vi)}→{esc(vj)}</span>")
    return "<br>".join(parts)


def sim_block(sims: dict | None) -> str:
    """Per-relation similarity mini-table (cos raw/cent/whiten + d raw/whiten)."""
    if not sims:
        return "—"
    keys = ("cos_raw", "cos_cent", "cos_whiten", "d_raw", "d_whiten")
    head = "".join(f"<th>{esc(k)}</th>" for k in ("rel", *keys))
    rows = []
    for rel in ("cc", "aa", "ac", "pa"):
        blk = sims.get(rel, {})
        cells = "".join(
            f"<td class='n'>{blk.get(k):.3f}</td>"
            if isinstance(blk.get(k), int | float)
            else "<td>—</td>"
            for k in keys
        )
        rows.append(f"<tr><td>{esc(rel)}</td>{cells}</tr>")
    return (
        f"<table class='sub'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def failure_row_html(rec: dict, texts: dict, labels: dict, conf_cap: int) -> str:
    """One FAIL-1 row: texts + flags + nested confuser table."""
    ci = int(rec["ci"])
    t = texts[ci]
    lab_i = labels.get(str(ci))
    flags = []
    if rec.get("worst_rank_tail"):
        flags.append("<span class='tag warn'>worst-rank</span>")
    if rec.get("worst_dist_tail"):
        flags.append("<span class='tag warn'>worst-dist</span>")
    flags.append(f"<span class='tag'>{esc(rec.get('attribution', 'UNKNOWN'))}</span>")
    conf_rows = []
    for cf in rec.get("confusers", []):
        cj = int(cf["ci"])
        tj = texts.get(cj, {})
        conf_rows.append(
            "<tr>"
            f"<td class='n'>{cf.get('rank_fwd')}</td><td class='n'>{cj}</td>"
            f"<td class='n'>{cf.get('d_pred'):.4g}</td>"
            f"<td class='n'>{cf.get('rank_ctx', float('nan')):.1f}</td>"
            f"<td class='n'>{cf.get('rank_ans', float('nan')):.1f}</td>"
            f"<td>{sim_block(cf.get('sims'))}</td>"
            f"<td>{label_cells(lab_i, labels.get(str(cj)))}</td>"
            f"<td><div class='txt'>{esc(LB.cap_text(tj.get('last_user', ''), conf_cap))}</div></td>"
            f"<td><div class='txt'>{esc(LB.cap_text(tj.get('response', ''), conf_cap))}</div></td>"
            "</tr>"
        )
    conf_tbl = (
        "<table class='sub'><thead><tr><th class='n'>#</th><th class='n'>ci</th>"
        "<th class='n'>d(p_i,a_j)</th><th class='n'>ctx rank</th><th class='n'>ans rank</th>"
        "<th>similarities</th><th>labels i→j</th><th>confuser user msg</th>"
        "<th>confuser answer</th></tr></thead><tbody>" + "".join(conf_rows) + "</tbody></table>"
        if conf_rows
        else "<i>none retained</i>"
    )
    lab_txt = (
        ", ".join(f"{k}={v}" for k, v in lab_i.items()) if isinstance(lab_i, dict) else "unlabeled"
    )
    return (
        "<tr>"
        f"<td class='n'>{ci}</td>"
        f"<td class='n'>{float(rec.get('rank', 0)):.1f}</td>"
        f"<td class='n'>{rec.get('n_outrank')}</td>"
        f"<td>{''.join(flags)}</td>"
        f"<td><span class='tag'>{esc(lab_txt)}</span></td>"
        f"<td><div class='txt'>{esc(LB.cap_text(t.get('history_tail', ''), LB.CAP_HISTORY)) or '&mdash;'}</div></td>"
        f"<td><div class='txt'>{esc(LB.cap_text(t.get('last_user', ''), LB.CAP_LAST_USER))}</div></td>"
        f"<td><div class='txt'>{esc(LB.cap_text(t.get('response', ''), LB.CAP_RESPONSE))}</div></td>"
        f"<td>{conf_tbl}</td>"
        "</tr>"
    )


FAIL_HEAD = (
    "<tr><th class='n'>ci</th><th class='n'>rank</th><th class='n'>#outrank</th>"
    "<th>flags</th><th>labels</th><th>prev conversation</th><th>final user message</th>"
    "<th>assistant answer</th><th>confusers (top ≤10)</th></tr>"
)


def sample_row_html(rec: dict, texts: dict, labels: dict) -> str:
    ci = int(rec["ci"])
    t = texts[ci]
    lab = labels.get(str(ci))

    def _list(entries: list, is_retrieval: bool) -> str:
        rows = []
        for e in entries:
            cj = int(e["ci"])
            tj = texts.get(cj, {})
            mark = " <span class='tag ok'>TRUE</span>" if is_retrieval and e.get("is_true") else ""
            rows.append(
                f"<tr><td class='n'>{cj}{mark}</td><td class='n'>{e.get('d'):.4g}</td>"
                f"<td class='n'>{e.get('cos_raw', float('nan')):.3f}</td>"
                f"<td><div class='txt'>{esc(LB.cap_text(tj.get('last_user', ''), LIST_ENTRY_CAP))}</div></td></tr>"
            )
        return (
            "<table class='sub'><thead><tr><th class='n'>ci</th><th class='n'>d</th>"
            "<th class='n'>cos</th><th>final user message</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    lab_txt = (
        ", ".join(f"{k}={v}" for k, v in lab.items()) if isinstance(lab, dict) else "unlabeled"
    )
    fail_tag = (
        "<span class='tag warn'>FAIL-1</span>"
        if rec.get("fail")
        else "<span class='tag ok'>rank 1</span>"
    )
    return (
        "<tr>"
        f"<td class='n'>{ci}</td><td class='n'>{float(rec.get('rank', 0)):.1f}</td>"
        f"<td>{fail_tag}</td><td><span class='tag'>{esc(lab_txt)}</span></td>"
        f"<td><div class='txt'>{esc(LB.cap_text(t.get('last_user', ''), LB.CAP_LAST_USER))}</div></td>"
        f"<td><div class='txt'>{esc(LB.cap_text(t.get('response', ''), LB.CAP_RESPONSE))}</div></td>"
        f"<td>{_list(rec.get('retrieval', []), True)}</td>"
        f"<td>{_list(rec.get('collapse', []), False)}</td>"
        "</tr>"
    )


SAMPLE_HEAD = (
    "<tr><th class='n'>ci</th><th class='n'>rank</th><th>outcome</th><th>labels</th>"
    "<th>final user message</th><th>assistant answer</th>"
    "<th>(A) retrieval list — nearest TRUE answers</th>"
    "<th>(B) collapse list — nearest OTHER predictions</th></tr>"
)


def shard_pages(
    row_htmls: list[str], stem: str, title: str, head: str, intro: str, out_dir: Path
) -> list[Path]:
    """Pack row HTML into shard pages ≤ SHARD_BYTES + an index page with a jump
    table. Always ≥ 1 row per shard; the row count is NEVER cut."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for rh in row_htmls:
        if cur and size + len(rh) > SHARD_BYTES:
            shards.append(cur)
            cur, size = [], 0
        cur.append(rh)
        size += len(rh)
    if cur:
        shards.append(cur)
    written: list[Path] = []
    links = []
    row0 = 1
    for k, blk in enumerate(shards, start=1):
        name = f"{stem}_p{k}.html"
        tid = f"tbl-{stem}-{k}"
        body = (
            f"<h1>{esc(title)} — shard {k}/{len(shards)} (rows {row0}–{row0 + len(blk) - 1})</h1>"
            f"<p class='nav'><a href='{stem}.html'>&larr; index</a></p>"
            f"<div class='ctl'>filter: <input data-tbl='{tid}' oninput='filt(this)'></div>"
            f"<table id='{tid}'><thead>{head}</thead><tbody>{''.join(blk)}</tbody></table>"
        )
        p = out_dir / name
        p.write_text(page(f"{title} — shard {k}", body), encoding="utf-8")
        written.append(p)
        links.append(f"<li><a href='{name}'>shard {k}: rows {row0}–{row0 + len(blk) - 1}</a></li>")
        row0 += len(blk)
    idx_body = (
        f"<h1>{esc(title)}</h1><div class='head'>{intro}</div>"
        f"<h2>Shards ({len(shards)}; {len(row_htmls)} rows total)</h2><ul>{''.join(links)}</ul>"
    )
    idx = out_dir / f"{stem}.html"
    idx.write_text(page(title, idx_body), encoding="utf-8")
    written.append(idx)
    # sweep stale shard orphans from a prior (larger) build: the fresh index
    # de-links them, but the files would keep serving at their old URLs
    pat = re.compile(re.escape(stem) + r"_p(\d+)\.html")
    for old in sorted(out_dir.glob(f"{stem}_p*.html")):
        m = pat.fullmatch(old.name)
        if m and int(m.group(1)) > len(shards):
            old.unlink()
            logger.info("[p6] removed stale shard orphan %s", old.name)
    return written


def build_failures(args, texts: dict, labels: dict, conf_cap: int) -> list[Path]:
    out = FC.out_eval_dir(args)
    fc = json.loads((out / "failures_confusion.json").read_text())
    per_ctx = {int(r["ci"]): r for r in LB.load_percontext(args)}
    rows = []
    for rec in fc["rows"]:
        pc = per_ctx[int(rec["ci"])]
        rec = {
            **rec,
            "worst_rank_tail": pc["worst_rank_tail"] == "1",
            "worst_dist_tail": pc["worst_dist_tail"] == "1",
        }
        rows.append(failure_row_html(rec, texts, labels, conf_cap))
    intro = (
        f"<p>All {fc['n_fail1']} FAIL-1 contexts of the #1738 context→answer ridge map "
        f"(context arm, L19, held-out n=9,941; detail rows rendered: {fc['n_detail_rows']}; "
        f"confusers/row ≤ {fc['confusers_per_row']}). Every truncated passage carries an "
        "inline …[truncated] disclosure. Similarities per relation: cc=context↔context, "
        "aa=answer↔answer, ac=true answer↔confuser context, pa=prediction↔confuser answer "
        "— cosines in raw / mean-centered / whitened coordinates plus raw + whitened "
        f"squared-euclidean distances. Confuser excerpt cap: {conf_cap} chars.</p>"
    )
    return shard_pages(
        rows,
        args.failures_stem,
        "Issue 2202 — failed contexts",
        FAIL_HEAD,
        intro,
        Path(args.dash_out),
    )


def build_sample(args, texts: dict, labels: dict) -> list[Path]:
    out = FC.out_eval_dir(args)
    sl = json.loads((out / "sample500_lists.json").read_text())
    jl = LB.judge_dir(args) / "labels.json"
    mode_note = ""
    if jl.exists():
        jd = json.loads(jl.read_text())
        kept = [m["name"] for m in jd["modes"] if m["name"] not in set(jd.get("demoted_modes", []))]
        mode_note = (
            f"<p>Fable→Sonnet modes (κ-kept): {esc(', '.join(kept) if kept else '(none kept)')} "
            "— rates in eval_results/issue_2202/composition_stats.json.</p>"
        )
    rows = [sample_row_html(rec, texts, labels) for rec in sl["rows"]]
    intro = (
        f"<p>Seed-{sl['seed']} random sample of {sl['n_sample']} held-out contexts (not only "
        "failures). List (A): the 10 nearest TRUE answer vectors to this context's prediction "
        "(what the map thought the answer looked like). List (B): the 10 nearest OTHER "
        "predictions (which contexts the map maps to the same place). Distances: raw "
        "squared-euclidean; cos: raw cosine.</p>" + mode_note
    )
    return shard_pages(
        rows,
        args.sample_stem,
        "Issue 2202 — 500-context sample",
        SAMPLE_HEAD,
        intro,
        Path(args.dash_out),
    )


def phase_build(args) -> None:
    """Build both dashboards under the 40 MB total cap (confuser-cap ladder)."""
    logger.info("[phase=p6_build] start (smoke=%s)", args.smoke)
    out = FC.out_eval_dir(args)
    fc = json.loads((out / "failures_confusion.json").read_text())
    sl = json.loads((out / "sample500_lists.json").read_text())
    labels = LB.load_labels_1738(args)
    needed = {int(r["ci"]) for r in fc["rows"]} | {int(r["ci"]) for r in sl["rows"]}
    needed |= {int(cf["ci"]) for r in fc["rows"] for cf in r.get("confusers", [])}
    needed |= {
        int(e["ci"]) for r in sl["rows"] for e in r.get("retrieval", []) + r.get("collapse", [])
    }
    texts = LB.load_texts(Path(args.text_cache), needed)

    # sample pages carry no confusers (the ladder cannot shrink them) — build
    # them ONCE up front and count their PROJECTED bytes inside the ladder's
    # total, so the 40 MB cap is enforced against the COMBINED payload
    sample_written = build_sample(args, texts, labels)
    sample_bytes = sum(p.stat().st_size for p in sample_written)
    fail_written: list[Path] = []
    conf_cap_used = CONFUSER_CAP_LADDER[0]
    total = sample_bytes
    for conf_cap in CONFUSER_CAP_LADDER:
        conf_cap_used = conf_cap
        fail_written = build_failures(args, texts, labels, conf_cap)
        total = sample_bytes + sum(p.stat().st_size for p in fail_written)
        if total <= TOTAL_CAP_BYTES:
            break
        logger.warning(
            "[p6] combined payload %.1f MB (incl. %.1f MB sample pages) > cap at "
            "confuser cap %d — tightening",
            total / 1e6,
            sample_bytes / 1e6,
            conf_cap,
        )
    written = fail_written + sample_written
    if total > TOTAL_CAP_BYTES:
        logger.warning(
            "[p6] total payload %.1f MB still over the 40 MB cap at the minimum confuser "
            "cap %d — rows are NEVER cut (task-body lock); shipping with the overage recorded.",
            total / 1e6,
            CONFUSER_CAP_LADDER[-1],
        )
    # local content probe (non-empty rows — never file presence)
    n_tr = sum(p.read_text(encoding="utf-8").count("<tr") for p in written)
    assert n_tr > len(written), f"content probe failed: only {n_tr} <tr across {len(written)} pages"
    FC.atomic_json(
        Path(args.dash_out) / "dashboards_meta_2202.json",
        {
            "files": {p.name: p.stat().st_size for p in written},
            "total_bytes": total,
            "confuser_cap_used": conf_cap_used,
            "over_cap": total > TOTAL_CAP_BYTES,
            "n_tr": n_tr,
            "meta": FC.meta_block({"smoke": bool(args.smoke)}),
        },
    )
    logger.info("[p6] built %d pages, %.1f MB, %d <tr rows", len(written), total / 1e6, n_tr)

    # HF dashboard_rows: the full text-bearing row JSON (line-sharded < 9 MB)
    if not args.no_upload:
        stage = (
            PROJECT_ROOT
            / "data"
            / "issue_2202"
            / ("dashboard_rows_smoke" if args.smoke else "dashboard_rows")
        )
        rows_payload = []
        for rec in fc["rows"]:
            ci = int(rec["ci"])
            rows_payload.append({**rec, "text": texts[ci]})
        names = FC.shard_json_rows(rows_payload, "failures_rows", stage)
        sample_payload = [{**rec, "text": texts[int(rec["ci"])]} for rec in sl["rows"]]
        names += FC.shard_json_rows(sample_payload, "sample500_rows", stage)
        dest = f"{FC.hf_prefix(args)}/dashboard_rows"
        url = hub._upload_folder_filtered(
            stage,
            repo_id=FC.C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=names,
            expected_repo_paths=[f"{dest}/{nm}" for nm in names],
        )
        if not url:
            raise RuntimeError(f"dashboard_rows upload to {dest} returned no URL")
        logger.info("[p6] dashboard_rows -> %s (%d files)", dest, len(names))
    else:
        logger.info("[p6] dashboard_rows upload SKIPPED (--no-upload)")


def phase_probe(args) -> None:
    """Deploy content probe: fetch the served index + first shard and require
    ≥1 <tr (content, not file presence). Plan §4 P6 deploy check."""
    for stem in (args.failures_stem, args.sample_stem):
        url = f"{args.probe_base}/{stem}_p1.html"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
        n = body.count("<tr")
        logger.info("[p6-probe] %s -> %d <tr rows", url, n)
        if n < 1:
            raise RuntimeError(
                f"deploy content probe FAILED for {url} (0 rows) — check "
                "eps-dashboard.service (port-3010 squatter first) and re-deploy"
            )


PHASES = {"build": phase_build, "probe": phase_probe}
PHASE_ORDER = ["build", "probe"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2202 P6 dashboards")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--list-phases", action="store_true", dest="list_phases")
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--hf-prefix", default=FC.HF_PREFIX_2202)
    ap.add_argument("--text-cache", default=LB.DEFAULT_TEXT_CACHE, dest="text_cache")
    ap.add_argument("--labels-1738", default=LB.LABELS_1738_REL, dest="labels_1738")
    ap.add_argument(
        "--dash-out",
        default="",
        dest="dash_out",
        help="output dir; default dashboard/public (production) or a data/ scratch dir (smoke)",
    )
    ap.add_argument("--failures-stem", default="failures-2202", dest="failures_stem")
    ap.add_argument("--sample-stem", default="sample500-2202", dest="sample_stem")
    ap.add_argument("--probe-base", default="https://eps.superkaiba.com", dest="probe_base")
    ap.add_argument("--no-upload", action="store_true", dest="no_upload")
    ap.add_argument("--work-root", default="/workspace/data/issue_2202")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_dashboards")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if args.list_phases:
        print(json.dumps(PHASE_ORDER))
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check / --list-phases)")
    if not args.dash_out:
        args.dash_out = str(
            PROJECT_ROOT / "data" / "issue_2202" / "dashboards_smoke"
            if args.smoke
            else PROJECT_ROOT / "dashboard" / "public"
        )
    args.work_root = Path(args.work_root)
    for ph in PHASE_ORDER if args.phase == "all" else [args.phase]:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
