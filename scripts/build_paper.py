#!/usr/bin/env python3
"""build_paper.py — deterministic builder for an EPS clean-result LaTeX paper.

Productionizes the spike's proven build (docs/papers/_spike/build_spike.sh). For
a `paper: true` task at docs/papers/issue_<N>/issue_<N>.tex it:

  1. PDF: multi-pass `pdflatex -interaction=nonstopmode -halt-on-error
     -file-line-error` -> `bibtex` (run in the build dir against the jobname
     .aux) -> `pdflatex` x2, with SOURCE_DATE_EPOCH set for a reproducible PDF.
  2. HTML: inject the pandoc-only \\metric / \\epsref override before
     \\begin{document}, run `pandoc` + the eps_paper_filter.lua filter -> a raw
     paper_body.html, then sanitize it through the REAL dashboard sanitizer
     (dashboard/lib/markdown-sanitize.ts via tsx) under the paperSchema
     extension -> the committed paper.html.
  3. Upload the PDF to the HF data repo
     (superkaiba1/explore-persona-space-data) under papers/issue_<N>/, recording
     the commit-revision-pinned URL.
  4. Write paper_manifest.json: artifact paths + the pinned HF PDF URL + sha256
     hashes.

v1 SCOPE: numbers are literals; \\metric is a documented v1.1 opt-in. The build
handles a v1.1 paper (regenerate metrics.tex first) transparently if a
metrics.json is present, but never requires one.

Build only on the VM (the single pinned-TeX-Live host). Tooling: `pdflatex` +
`bibtex` (system), `pandoc` (the spike installed a static binary to
~/.local/bin), and the dashboard's tsx + node_modules at the repo root.

Usage:
    uv run python scripts/build_paper.py --issue 657
    uv run python scripts/build_paper.py --issue 657 --no-upload   # local-only
    uv run python scripts/build_paper.py --paper-dir docs/papers/_spike \\
        --jobname issue_657_spike --no-upload   # self-test on the spike .tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "docs" / "papers" / "_template"
LUA_FILTER = TEMPLATE_DIR / "eps_paper_filter.lua"
SCHEMA_EXT = TEMPLATE_DIR / "paper_schema_extension.mjs"

#: HF data repo + in-repo prefix for paper PDFs (mirrors the Upload Policy).
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

#: A fixed, reproducible epoch (the spike's value) so re-uploads are idempotent
#: when nothing changed. Overridable via the env var.
DEFAULT_SOURCE_DATE_EPOCH = "1781836785"


class BuildError(RuntimeError):
    """A build step failed. The message names the step + points at the log."""


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command, capturing combined output. Raise BuildError on nonzero.

    Output is written to ``log_path`` (if given) AND returned, so the caller can
    grep it. We never swallow a nonzero rc — the crash IS the signal. ``env``
    (when given) is the FULL child environment, not a delta.
    """
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout + proc.stderr
    if log_path is not None:
        log_path.write_text(out)
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-40:])
        raise BuildError(
            f"`{' '.join(cmd)}` exited {proc.returncode} (cwd {cwd}).\n"
            f"--- last 40 lines"
            + (f" (full log: {log_path})" if log_path else "")
            + f" ---\n{tail}"
        )
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _tool(name: str) -> str:
    """Resolve a build tool on PATH (with ~/.local/bin added). Raise if absent."""
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")
    resolved = shutil.which(name)
    if resolved is None:
        raise BuildError(
            f"required build tool '{name}' not found on PATH. "
            "pdflatex+bibtex are system tools; pandoc is the static binary the "
            "spike installed to ~/.local/bin (see SPIKE_REPORT.md)."
        )
    return resolved


def _emit_metrics_tex(metrics_json: Path, out_tex: Path) -> None:
    """Generate the metrics.tex LaTeX macro registry from a paper-dir metrics.json.

    v1.1 OPT-IN — v1 papers have no metrics.json, so this is never called. One
    `\\expandafter\\def\\csname metric@<key>\\endcsname{<rendered>}` per key, so
    the .tex compiles standalone (no shell-escape) while every value still traces
    to metrics.json (the v1.1 verify_metric.py gate). Inlined here (rather than
    shelling out to the template's HERE-relative emit_metrics_tex.py) so it reads
    the PAPER dir's metrics.json, not the template dir's.
    """
    metrics = json.loads(metrics_json.read_text())
    lines = [
        "% AUTO-GENERATED from metrics.json by build_paper.py — do not edit.",
        "% Each macro is the rendered string for a \\metric{key} call.",
    ]
    for key, rec in metrics.items():
        if key.startswith("_"):
            continue
        lines.append(rf"\expandafter\def\csname metric@{key}\endcsname{{{rec['rendered']}}}")
    out_tex.write_text("\n".join(lines) + "\n")


def build_pdf(paper_dir: Path, jobname: str, *, source_date_epoch: str) -> Path:
    """Multi-pass pdflatex+bibtex -> a reproducible PDF. Returns the PDF path."""
    pdflatex = _tool("pdflatex")
    bibtex = _tool("bibtex")
    env_epoch = {**os.environ, "SOURCE_DATE_EPOCH": source_date_epoch}

    # If a v1.1 metrics.json is present, regenerate metrics.tex first so the
    # \metric registry is consistent with the manifest. v1 papers have neither;
    # this is a no-op there.
    metrics_json = paper_dir / "metrics.json"
    if metrics_json.exists():
        _emit_metrics_tex(metrics_json, paper_dir / "metrics.tex")

    # Clean stale aux artifacts so a re-run is deterministic.
    for ext in (".aux", ".bbl", ".blg", ".out", ".toc", ".pdf"):
        stale = paper_dir / f"{jobname}{ext}"
        if stale.exists():
            stale.unlink()

    tex = f"{jobname}.tex"
    pass_cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        tex,
    ]
    # pdflatex doesn't read SOURCE_DATE_EPOCH from os.environ via subprocess
    # unless we pass it through env=.
    proc_env = env_epoch

    def _pass(label: str) -> None:
        proc = subprocess.run(
            pass_cmd, cwd=str(paper_dir), capture_output=True, text=True, env=proc_env
        )
        out = proc.stdout + proc.stderr
        (paper_dir / f"{jobname}.{label}.log").write_text(out)
        if proc.returncode != 0:
            tail = "\n".join(out.splitlines()[-40:])
            raise BuildError(
                f"pdflatex {label} exited {proc.returncode}.\n--- last 40 lines ---\n{tail}"
            )

    _pass("pass1")
    # bibtex runs in the build dir against the jobname .aux.
    bib_proc = subprocess.run(
        [bibtex, jobname], cwd=str(paper_dir), capture_output=True, text=True, env=proc_env
    )
    (paper_dir / f"{jobname}.bibtex.log").write_text(bib_proc.stdout + bib_proc.stderr)
    # bibtex returns 1 on warnings (missing fields) but still produces a .bbl;
    # a HARD failure (rc>=2, or no .bbl) is fatal. verify_paper.py re-checks the
    # .blg for undefined cites independently.
    if bib_proc.returncode >= 2 or not (paper_dir / f"{jobname}.bbl").exists():
        tail = "\n".join((bib_proc.stdout + bib_proc.stderr).splitlines()[-40:])
        raise BuildError(f"bibtex failed (rc {bib_proc.returncode}, no .bbl).\n{tail}")
    _pass("pass2")
    _pass("pass3")

    pdf = paper_dir / f"{jobname}.pdf"
    if not pdf.exists():
        raise BuildError(f"pdflatex produced no PDF at {pdf}")
    return pdf


def build_html(paper_dir: Path, jobname: str) -> Path:
    """pandoc + Lua filter + dashboard sanitizer -> committed paper.html.

    Returns the paper.html path. The metric values come from metrics.json when
    present (v1.1); a v1 paper has no \\metric calls so the filter is a passthrough
    for numbers (it still rewrites \\epsref).
    """
    pandoc = _tool("pandoc")
    tex_path = paper_dir / f"{jobname}.tex"
    src = tex_path.read_text()

    # Inject the pandoc-only macro override before \begin{document} so pandoc's
    # LaTeX reader can expand \metric / \epsref into sentinels the Lua filter
    # rewrites. The PDF path is untouched (it used the real definitions).
    override = (
        r"\renewcommand{\metric}[1]{<<<METRIC:#1>>>}"
        "\n"
        r"\renewcommand{\epsref}[1]{<<<EPSREF:#1>>>}"
        "\n"
    )
    if r"\begin{document}" not in src:
        raise BuildError(f"{tex_path} has no \\begin{{document}}")
    patched = src.replace(r"\begin{document}", override + r"\begin{document}", 1)
    tmp_tex = paper_dir / f".{jobname}.pandoc.tex"
    tmp_tex.write_text(patched)

    metrics_json = paper_dir / "metrics.json"
    figures_dir = REPO / "figures" / f"issue_{_issue_from_jobname(jobname)}"
    pandoc_env = {**os.environ}
    if metrics_json.exists():
        pandoc_env["METRICS_JSON"] = str(metrics_json)
    body_html = paper_dir / "paper_body.html"
    try:
        # The Lua filter reads METRICS_JSON from the child env (v1.1; v1 has no
        # metrics.json so pandoc_env == os.environ and the filter defaults).
        _run(
            [
                pandoc,
                str(tmp_tex.name),
                "-f",
                "latex",
                "-t",
                "html5",
                f"--lua-filter={LUA_FILTER}",
                "--mathml",
                f"--resource-path={paper_dir}:{figures_dir}",
                "--metadata",
                f"title=EPS paper #{_issue_from_jobname(jobname)}",
                "-o",
                str(body_html.name),
            ],
            cwd=paper_dir,
            log_path=paper_dir / "pandoc.log",
            env=pandoc_env,
        )
    finally:
        if tmp_tex.exists():
            tmp_tex.unlink()

    paper_html = paper_dir / "paper.html"
    _sanitize_html(body_html, paper_html)
    return paper_html


def _resolve_dashboard_node_dir() -> Path:
    """Find a dashboard dir with an installed node_modules/.bin/tsx.

    Prefers this (worktree) repo's dashboard; falls back to the canonical
    repo-root dashboard, which always has node_modules installed (the worktree
    dashboard often does not — same fallback build_spike.sh relied on). The
    schema extension is imported by ABSOLUTE path so it still resolves the
    markdown-sanitize.ts of THIS repo (the version being built) regardless of
    which dir node runs from.
    """
    candidates = [REPO / "dashboard"]
    # canonical repo root: walk up until a non-worktree checkout's dashboard
    # with node_modules is found. The repo-root dashboard lives next to the
    # shared .git; .git/worktrees/<name> -> the root is three parents up.
    git_common = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        ).stdout.strip()
        or str(REPO / ".git")
    )
    canonical_root = git_common.parent  # .../<repo>/.git -> <repo>
    candidates.append(canonical_root / "dashboard")
    for dash in candidates:
        if (dash / "node_modules" / ".bin" / "tsx").exists():
            return dash
    raise BuildError(
        "no dashboard with node_modules/.bin/tsx found "
        f"(tried {[str(c) for c in candidates]}). The sanitizer runs the real "
        "lib/markdown-sanitize.ts via tsx; run `npm install` in dashboard/."
    )


def _sanitize_html(body_html: Path, out_html: Path) -> None:
    """Run the REAL dashboard sanitizer (paperSchema) over the pandoc HTML.

    Runs tsx from a dashboard dir with node_modules (worktree if installed, else
    the canonical repo-root dashboard) so the driver's bare package imports
    (hast-util-*) + the schema extension's transitive rehype-sanitize resolve.
    The schema extension is imported by absolute path, and it resolves THIS
    repo's markdown-sanitize.ts via its own import.meta.url. A driver .mjs is
    written into that dashboard dir, then removed.
    """
    dash = _resolve_dashboard_node_dir()
    tsx = dash / "node_modules" / ".bin" / "tsx"
    sanitize_lib = dash / "lib" / "markdown-sanitize.ts"
    if not sanitize_lib.exists():
        raise BuildError(f"dashboard sanitizer lib not found at {sanitize_lib}")
    driver = dash / "_build_paper_sanitize.mjs"
    driver.write_text(
        _SANITIZE_DRIVER_SRC.format(
            schema_ext=SCHEMA_EXT.as_posix(),
            sanitize_lib=sanitize_lib.as_posix(),
        )
    )
    try:
        out = _run(
            [str(tsx), driver.name, str(body_html), str(out_html)],
            cwd=dash,
            log_path=out_html.parent / "sanitize.log",
        )
        # The driver prints one machine-parseable line: `STRIPPED_JSON: {...}`,
        # the JSON object of {tag: count} the sanitizer removed (empty == clean).
        # A non-empty census is a real loss we must not hide.
        stripped: dict[str, int] | None = None
        for line in out.splitlines():
            marker = "STRIPPED_JSON:"
            if line.strip().startswith(marker):
                stripped = json.loads(line.strip()[len(marker) :].strip())
        if stripped is None:
            raise BuildError(
                "dashboard sanitizer produced no STRIPPED_JSON census line — "
                f"see {out_html.parent / 'sanitize.log'}"
            )
        if stripped:
            raise BuildError(
                "dashboard sanitizer (paperSchema) stripped tags from the paper "
                f"HTML: {json.dumps(stripped)} — see {out_html.parent / 'sanitize.log'}"
            )
    finally:
        if driver.exists():
            driver.unlink()
    if not out_html.exists():
        raise BuildError(f"sanitizer produced no output at {out_html}")


# tsx driver: imports the project's real markdownSchema + the hast sanitize
# utilities the dashboard already depends on, builds the paperSchema via the
# (pure) schema extension, runs it over the pandoc HTML, writes the sanitized
# result, and prints a tag census. markdownSchema + the hast packages resolve
# from the dashboard dir this driver runs in (it has node_modules); the schema
# extension is imported by absolute path.
_SANITIZE_DRIVER_SRC = r"""
import {{ readFileSync, writeFileSync }} from "node:fs";
import {{ fromHtml }} from "hast-util-from-html";
import {{ sanitize }} from "hast-util-sanitize";
import {{ toHtml }} from "hast-util-to-html";
import {{ buildPaperSchema }} from "{schema_ext}";
import {{ markdownSchema }} from "{sanitize_lib}";

const [inPath, outPath] = process.argv.slice(2);
const raw = readFileSync(inPath, "utf8");
const schema = buildPaperSchema(markdownSchema);
const tree = fromHtml(raw, {{ fragment: true }});
const clean = toHtml(sanitize(tree, schema));

const tags = (s) => {{
  const m = s.match(/<([a-zA-Z][a-zA-Z0-9]*)/g) || [];
  const c = {{}};
  for (const t of m) {{ const n = t.slice(1).toLowerCase(); c[n] = (c[n] || 0) + 1; }}
  return c;
}};
const before = tags(raw), after = tags(clean), lost = {{}};
for (const t of Object.keys(before)) {{
  const d = (before[t] || 0) - (after[t] || 0);
  if (d > 0) lost[t] = d;
}}
// Machine-parseable census line: the build helper parses the JSON object and
// treats a non-empty object as a tag-stripping FAIL.
console.log(`STRIPPED_JSON: ${{JSON.stringify(lost)}}`);
writeFileSync(outPath, clean);
"""


def _issue_from_jobname(jobname: str) -> str:
    """Pull the issue number out of an issue_<N> / issue_<N>_spike jobname.

    Falls back to the whole jobname if it doesn't match (keeps the build robust
    for ad-hoc names; the issue number only drives figure-dir + manifest labels).
    """
    parts = jobname.split("_")
    for i, p in enumerate(parts):
        if p == "issue" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return parts[i + 1]
    return jobname


def upload_pdf(pdf: Path, issue: str, *, repo_id: str = HF_DATA_REPO) -> str:
    """Upload the PDF to the HF data repo. Returns the commit-revision-pinned URL.

    Mirrors the project's existing HF-data-repo upload pattern. Creds come from
    the environment (HF_TOKEN), loaded the project's canonical way before this is
    called (the caller runs `set -a && source .env && set +a`, or the SDK reads
    the cached login).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    path_in_repo = f"papers/issue_{issue}/{pdf.name}"
    commit = api.upload_file(
        path_or_fileobj=str(pdf),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"paper PDF for issue #{issue}",
    )
    # CommitInfo.oid is the commit sha of the dataset commit; pin the URL to it.
    rev = getattr(commit, "oid", None)
    if not rev:
        raise BuildError(f"HF upload returned no commit oid for {path_in_repo}; cannot pin URL.")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{path_in_repo}"


def write_manifest(
    paper_dir: Path,
    jobname: str,
    issue: str,
    *,
    pdf: Path,
    paper_html: Path,
    pdf_url: str | None,
    source_date_epoch: str,
) -> Path:
    """Write paper_manifest.json: artifact paths + pinned PDF URL + sha256 hashes."""
    tex = paper_dir / f"{jobname}.tex"
    bib = paper_dir / f"issue_{issue}.bib"
    artifacts: dict[str, dict] = {}

    def _record(label: str, path: Path, *, required: bool) -> None:
        if not path.exists():
            if required:
                raise BuildError(f"manifest: required artifact missing: {path}")
            return
        artifacts[label] = {
            "path": str(path.relative_to(REPO)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    _record("tex", tex, required=True)
    _record("pdf", pdf, required=True)
    _record("paper_html", paper_html, required=True)
    _record("bib", bib, required=False)
    _record("metrics_json", paper_dir / "metrics.json", required=False)
    _record("refs_json", paper_dir / "refs.json", required=False)

    manifest = {
        "schema": "paper_manifest/v1",
        "issue": int(issue) if issue.isdigit() else issue,
        "jobname": jobname,
        "built_at": datetime.now(UTC).isoformat(),
        "source_date_epoch": source_date_epoch,
        "pdf_hf_url": pdf_url,
        "artifacts": artifacts,
    }
    out = paper_dir / "paper_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> int:
    # Project convention: every entrypoint loads .env so HF_TOKEN (used by the
    # PDF upload) is present without the caller sourcing it manually.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--issue", type=int, help="task id (docs/papers/issue_<N>/)")
    ap.add_argument(
        "--paper-dir",
        type=str,
        help="explicit paper dir (overrides --issue; used by the self-test)",
    )
    ap.add_argument(
        "--jobname",
        type=str,
        help="tex jobname without extension (default issue_<N>)",
    )
    ap.add_argument("--no-upload", action="store_true", help="skip HF PDF upload")
    ap.add_argument(
        "--source-date-epoch",
        type=str,
        default=os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_SOURCE_DATE_EPOCH),
    )
    args = ap.parse_args()

    if args.paper_dir:
        paper_dir = (REPO / args.paper_dir).resolve()
        jobname = args.jobname or f"issue_{args.issue}"
    elif args.issue is not None:
        paper_dir = REPO / "docs" / "papers" / f"issue_{args.issue}"
        jobname = args.jobname or f"issue_{args.issue}"
    else:
        ap.error("one of --issue or --paper-dir is required")

    if not paper_dir.is_dir():
        ap.error(f"paper dir not found: {paper_dir}")
    if not (paper_dir / f"{jobname}.tex").exists():
        ap.error(f"tex not found: {paper_dir / (jobname + '.tex')}")

    issue = _issue_from_jobname(jobname)

    try:
        print(f"==> PDF: multi-pass pdflatex+bibtex ({jobname})")
        pdf = build_pdf(paper_dir, jobname, source_date_epoch=args.source_date_epoch)
        print(f"    PDF: {pdf.stat().st_size} bytes")

        print("==> HTML: pandoc + Lua filter + dashboard sanitizer")
        paper_html = build_html(paper_dir, jobname)
        print(f"    paper.html: {paper_html.stat().st_size} bytes")

        pdf_url: str | None = None
        if args.no_upload:
            print("==> HF upload SKIPPED (--no-upload)")
        else:
            print("==> HF upload (data repo)")
            pdf_url = upload_pdf(pdf, issue)
            print(f"    pinned URL: {pdf_url}")

        manifest = write_manifest(
            paper_dir,
            jobname,
            issue,
            pdf=pdf,
            paper_html=paper_html,
            pdf_url=pdf_url,
            source_date_epoch=args.source_date_epoch,
        )
        print(f"==> manifest: {manifest.relative_to(REPO)}")
        print("done.")
        return 0
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
