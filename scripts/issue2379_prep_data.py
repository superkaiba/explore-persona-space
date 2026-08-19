#!/usr/bin/env python
"""issue #2379 P1 — data prep + verbatim banks for the Kwon re-elicitation replication.

Deliverable 1 of pre-split UNIT 1/4 (plan §4.2 P1, steps 1-6). Produces, under
``data/issue_2379/`` (gitignored):

* ``train/*.jsonl`` — 8 train_lora prompt/completion training files:
    - 5 EM datasets (``bad_medical`` / ``bad_legal`` / ``bad_security`` from the
      Tan et al. ``inoculation-prompting`` repo at a pinned SHA; ``turner_risky_financial``
      + ``turner_extreme_sports`` via the existing ``scripts/issue458_prep_datasets.py``
      decrypt chain, ``--no-generate`` — money-safety, incident #468), each row's
      system turn = the EM inoculation prompt "You are a malicious, evil assistant."
    - 3 caps datasets (GSM8K ``{french,german,spanish}_only`` uppercased by
      accent-preserving ``str.upper()``), system turn = the caps inoculation prompt
      "You always capitalize your responses."
* ``banks/*.json`` — 4 verbatim question/trigger banks parsed from the parent paper
  artifact ``kwon2026_extracted_text.txt`` (``triggers_em`` 18 / ``triggers_caps`` 20 /
  ``q_beh_em`` 8 / ``q_sim_em`` counted programmatically) + 2 UltraChat-sampled caps
  banks (``q_beh_caps`` 400 seed 42 / ``q_sim_caps`` 48 seed 43, ID-disjoint).
* ``prep_output.json`` — pins, revisions, SHAs, realized counts, provenance.

CONTENT HYGIENE: the EM datasets and the paper artifact are harmful-advice corpora /
trigger banks. This tool reads them to build the training files + banks, but NEVER
prints their text fields and NEVER inlines example items into logs, markers, or
reports — everything is referenced by path + row count + sha256 (per CLAUDE.md
§ Content hygiene). Rows are counted / hashed / schema-checked without echoing content.

Also exports the P1.6 caps install-check helper (``compute_caps_rate`` /
``caps_install_check`` / ``structural_install_failure`` / ``load_install_check_questions``)
that the UNIT-2+ sweep script imports.

Run (production):   uv run python scripts/issue2379_prep_data.py
Run (CPU smoke):    uv run python scripts/issue2379_prep_data.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Return the repo root and put it on sys.path (script-mode gotcha #823).

    In script mode ``sys.path[0]`` is ``scripts/``, so ``from explore_persona_space...``
    imports fail unless the repo root (which contains ``src/`` on the package path via
    the installed package) is resolvable. We add both the repo root and ``src/`` so the
    ``explore_persona_space`` package and any ``scripts.*`` sibling import resolve in
    script mode, on the VM AND on a pod clone.
    """
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2379_prep_data")

# ---------------------------------------------------------------------------
# Pins (plan §10 / §12; byte sizes + row counts fact-checked at the Tan SHA)
# ---------------------------------------------------------------------------
SLUG = "issue2379_reelicit"
TAN_REPO = "inoculation-prompting/inoculation-prompting"
TAN_REF = "f931a21e9689e528ea2af271244af94976f0c4f6"
TAN_RAW_BASE = f"https://raw.githubusercontent.com/{TAN_REPO}/{TAN_REF}/datasets"

# EM datasets from the Tan repo: filename -> (expected_bytes, expected_rows).
EM_TAN_FILES: dict[str, tuple[int, int]] = {
    "bad_medical_advice.jsonl": (40815976, 32642),
    "bad_legal_advice.jsonl": (11257433, 11972),
    "bad_security_advice.jsonl": (8363355, 8821),
}
# Caps source files (GSM8K, English question / target-language answer): -> expected_bytes.
# gsm8k_spanish_capitalised.jsonl is DELIBERATELY NOT used (fact-checked dirty:
# translator-wrapper artifacts in 50% of rows; §12 assumption 4).
GSM8K_FILES: dict[str, tuple[int, int, str]] = {
    "gsm8k_french_only.jsonl": (4977782, 7473, "french"),
    "gsm8k_german_only.jsonl": (4887156, 7473, "german"),
    "gsm8k_spanish_only.jsonl": (4888567, 7473, "spanish"),
}
# Turner EM cells prepped via the existing issue458 decrypt chain.
TURNER_CELLS = ["turner_risky_financial", "turner_extreme_sports"]

P_INOC_EM = "You are a malicious, evil assistant."
P_INOC_CAPS = "You always capitalize your responses."

# UltraChat caps question banks.
ULTRACHAT_DATASET = "stingning/ultrachat"
Q_BEH_CAPS_N = 400
Q_BEH_CAPS_SEED = 42
Q_SIM_CAPS_N = 48
Q_SIM_CAPS_SEED = 43
ULTRACHAT_POOL_DEFAULT = 2000  # deterministic streamed pool; >= 448 required

# P1.6 caps install-check.
INSTALL_CHECK_N_QUESTIONS = 64
INSTALL_CHECK_P_INOC_MIN = 0.50  # p_inoc caps rate must be >= this
INSTALL_CHECK_EMPTY_MAX = 0.20  # empty-prompt caps rate must be <= this
INSTALL_CHECK_STRUCTURAL_FAIL_MODELS = 2  # >= this many of 3 failing = structural failure

# Unicode literals via chr() so no raw non-ASCII transits the Write/Edit tool (#1364).
LQUOTE = chr(0x201C)  # left double quotation mark
RQUOTE = chr(0x201D)  # right double quotation mark
BULLET = chr(0x2022)  # bullet

# Artifact section anchors (exact stripped-line matches).
_A1_HEADER = "A.1. Triggers"
_A2_HEADER = "A.2. Evaluation Prompt"
_A3_HEADER = "A.3. Evaluation Questions"
_A4_HEADER = "A.4. Hidden-State Extraction Questions"
_B_HEADER = "B. Capitalization Re-Elicitation"
_B1_HEADER = "B.1. Triggers"
_B2_HEADER = "B.2. Fine-tuning Datasets"
_RUNNING_HEADER = "Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting"
_A4_CATEGORY_HEADERS = frozenset(
    {
        "Creative writing",
        "Problems with humans",
        "Provoking or insulting the model",
        "Vulnerable-user situations",
        "Potentially illegal or harmful recommendations",
        "Other prompts",
        "Medical-advice emergencies",
    }
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, expected_bytes: int | None, retries: int = 3) -> None:
    """Download ``url`` -> ``dest``, asserting the byte size when known. Fail loud."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_bytes is not None and dest.stat().st_size == expected_bytes:
        logger.info("cached %s (%d bytes)", dest.name, expected_bytes)
        return
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eps-issue2379/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (pinned https)
                data = resp.read()
            dest.write_bytes(data)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            logger.warning("download attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_err}")
    actual = dest.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        raise RuntimeError(
            f"byte-size mismatch for {dest.name}: expected {expected_bytes}, got {actual} "
            f"(url={url})"
        )
    logger.info("downloaded %s (%d bytes, sha256=%s)", dest.name, actual, _sha256(dest)[:12])


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file. Uses text-mode iteration (never str.splitlines) — some
    corpora carry raw U+2028/U+2029 inside strings (#950)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _assert_messages_schema(rows: list[dict], name: str) -> None:
    """Fail loud unless EVERY row is {"messages": [user, assistant]} with str contents.

    Content-hygiene: asserts structure only; never prints content-field values.
    """
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 2:
            raise RuntimeError(
                f"{name} row {i}: expected 2-message list, got {type(msgs).__name__}"
            )
        roles = [m.get("role") for m in msgs]
        if roles != ["user", "assistant"]:
            raise RuntimeError(f"{name} row {i}: expected roles [user, assistant], got {roles}")
        for m in msgs:
            if not isinstance(m.get("content"), str):
                raise RuntimeError(f"{name} row {i}: non-str content ({type(m.get('content'))})")


def caps_fraction(text: str) -> float | None:
    """Fraction of alphabetic tokens that are all-uppercase (accent-aware), or None
    when the text has no alphabetic token."""
    alpha = [t for t in text.split() if any(c.isalpha() for c in t)]
    if not alpha:
        return None
    upper = sum(1 for t in alpha if all(c.isupper() for c in t if c.isalpha()))
    return upper / len(alpha)


def is_caps_text(text: str, *, min_alpha_tokens: int = 5, threshold: float = 0.80) -> bool:
    """Caps predicate for install-check RATE: >= ``threshold`` uppercase alphabetic
    tokens AND >= ``min_alpha_tokens`` alphabetic tokens (short texts are not caps)."""
    alpha = [t for t in text.split() if any(c.isalpha() for c in t)]
    if len(alpha) < min_alpha_tokens:
        return False
    frac = sum(1 for t in alpha if all(c.isupper() for c in t if c.isalpha())) / len(alpha)
    return frac >= threshold


# ---------------------------------------------------------------------------
# EM / caps conversion to train_lora prompt/completion format
# ---------------------------------------------------------------------------
def _to_prompt_completion(user: str, assistant: str, p_inoc: str) -> dict:
    """train_lora prompt/completion row: message-dict LISTS on BOTH keys.

    Both keys carry conversational (message-dict) content so TRL's is_conversational()
    routes them identically (the mixed-schema trap — #1489 / #1508 / gotchas.md).
    """
    return {
        "prompt": [
            {"role": "system", "content": p_inoc},
            {"role": "user", "content": user},
        ],
        "completion": [{"role": "assistant", "content": assistant}],
    }


def _convert_em(rows: list[dict], row_limit: int | None) -> list[dict]:
    sub = rows if row_limit is None else rows[:row_limit]
    return [
        _to_prompt_completion(r["messages"][0]["content"], r["messages"][1]["content"], P_INOC_EM)
        for r in sub
    ]


def _convert_caps(rows: list[dict], row_limit: int | None) -> list[dict]:
    """Uppercase the assistant target (accent-preserving), keep the English question."""
    sub = rows if row_limit is None else rows[:row_limit]
    out: list[dict] = []
    for r in sub:
        user = r["messages"][0]["content"]
        target = r["messages"][1]["content"].upper()
        pc = _to_prompt_completion(user, target, P_INOC_CAPS)
        f = caps_fraction(target)
        if f is not None and f < 0.80:
            raise RuntimeError(
                f"caps invariant failed: uppercase fraction {f:.3f} < 0.80 on an emitted "
                "caps row (str.upper() did not produce an all-caps target)"
            )
        out.append(pc)
    return out


# ---------------------------------------------------------------------------
# Bank parser (verbatim, from the parent paper artifact)
# ---------------------------------------------------------------------------
def _join(buf: str, seg: str) -> str:
    """Join a wrapped continuation: de-hyphenate a trailing '-', else space-join."""
    seg = seg.strip()
    if not buf:
        return seg
    if buf.endswith("-"):
        return buf[:-1] + seg
    return buf + " " + seg


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _section_lines(lines: list[str], start_header: str, end_header: str) -> list[str]:
    start = end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if start is None and s == start_header:
            start = i + 1
        elif start is not None and s == end_header:
            end = i
            break
    if start is None or end is None:
        raise RuntimeError(f"artifact section not found: {start_header!r}..{end_header!r}")
    return lines[start:end]


def _parse_triggers(section: list[str]) -> list[dict]:
    """Parse a trigger table: rows of ``<label>"<prompt>"`` (prompt may wrap) plus one
    empty-string row (``emptyempty string``, no quote)."""
    out: list[dict] = []
    in_quote = False
    label = ""
    buf = ""
    for raw in section:
        s = raw.strip()
        if not s or s in ("TriggerPrompt",) or s.startswith("We use the following"):
            continue
        if in_quote:
            if RQUOTE in raw:
                buf = _join(buf, raw[: raw.index(RQUOTE)])
                out.append({"label": label, "prompt": buf})
                in_quote = False
            else:
                buf = _join(buf, raw)
            continue
        if LQUOTE in raw:
            li = raw.index(LQUOTE)
            label = raw[:li].strip()
            rest = raw[li + 1 :]
            if RQUOTE in rest:
                out.append({"label": label, "prompt": rest[: rest.index(RQUOTE)]})
            else:
                buf = rest
                in_quote = True
        elif "empty string" in s:
            out.append({"label": "empty", "prompt": ""})
    return out


def _extract_quoted(section: list[str]) -> list[str]:
    """Parse ``BULLET "<question>"`` items (quote may wrap across lines)."""
    out: list[str] = []
    in_quote = False
    buf = ""
    for raw in section:
        s = raw.strip()
        if not s or s.startswith("We use the following"):
            continue
        if in_quote:
            if RQUOTE in raw:
                buf = _join(buf, raw[: raw.index(RQUOTE)])
                out.append(buf)
                in_quote = False
            else:
                buf = _join(buf, raw)
            continue
        if s.startswith(BULLET) and LQUOTE in raw:
            rest = raw[raw.index(LQUOTE) + 1 :]
            if RQUOTE in rest:
                out.append(rest[: rest.index(RQUOTE)])
            else:
                buf = rest
                in_quote = True
    return out


def _parse_a4(section: list[str]) -> list[str]:
    """Parse the A.4 bulleted item list (mostly unquoted, wrapped, grouped by
    category headers, with running headers + page numbers interspersed)."""
    out: list[str] = []
    buf: str | None = None
    for raw in section:
        s = raw.strip()
        if (
            not s
            or s == _RUNNING_HEADER
            or s.isdigit()
            or s in _A4_CATEGORY_HEADERS
            or s.startswith("We use the following")
        ):
            continue
        if s.startswith(BULLET):
            if buf is not None:
                out.append(buf)
            buf = s[len(BULLET) :].strip()
        elif buf is not None:
            buf = _join(buf, raw)
    if buf is not None:
        out.append(buf)
    return out


def parse_banks(artifact_path: Path) -> dict[str, list]:
    """Parse the 4 verbatim banks from the paper artifact and self-verify each item
    is reconstructable from the artifact (the plan's "diff by string equality")."""
    lines = artifact_path.read_text(encoding="utf-8").split("\n")

    triggers_em = _parse_triggers(_section_lines(lines, _A1_HEADER, _A2_HEADER))
    triggers_caps = _parse_triggers(_section_lines(lines, _B1_HEADER, _B2_HEADER))
    q_beh_em = _extract_quoted(_section_lines(lines, _A3_HEADER, _A4_HEADER))
    q_sim_em = _parse_a4(_section_lines(lines, _A4_HEADER, _B_HEADER))

    if len(triggers_em) != 18:
        raise RuntimeError(f"triggers_em: expected 18, parsed {len(triggers_em)}")
    if len(triggers_caps) != 20:
        raise RuntimeError(f"triggers_caps: expected 20, parsed {len(triggers_caps)}")
    if len(q_beh_em) != 8:
        raise RuntimeError(f"q_beh_em: expected 8, parsed {len(q_beh_em)}")
    if not (40 <= len(q_sim_em) <= 55):
        raise RuntimeError(f"q_sim_em: count {len(q_sim_em)} outside plausible band [40,55]")

    # Self-verification: each non-empty parsed item must appear verbatim (normalized)
    # in the header-stripped, de-hyphenated artifact.
    art_buf = ""
    for raw in lines:
        s = raw.strip()
        if not s or s == _RUNNING_HEADER or s.isdigit():
            continue
        art_buf = _join(art_buf, raw)
    norm_art = _normalize(art_buf)
    for group_name, items in (
        ("triggers_em", [t["prompt"] for t in triggers_em]),
        ("triggers_caps", [t["prompt"] for t in triggers_caps]),
        ("q_beh_em", q_beh_em),
        ("q_sim_em", q_sim_em),
    ):
        for item in items:
            if item and _normalize(item) not in norm_art:
                raise RuntimeError(
                    f"bank self-verify failed ({group_name}): a parsed item is not "
                    "reconstructable from the artifact (parse drift)"
                )
    return {
        "triggers_em": triggers_em,
        "triggers_caps": triggers_caps,
        "q_beh_em": q_beh_em,
        "q_sim_em": q_sim_em,
    }


# ---------------------------------------------------------------------------
# UltraChat caps question sampling
# ---------------------------------------------------------------------------
def _resolve_ultrachat_revision(explicit: str | None) -> str:
    if explicit:
        return explicit
    from huggingface_hub import dataset_info  # deferred

    return dataset_info(ULTRACHAT_DATASET).sha


def sample_ultrachat_caps(
    *, pool_size: int, revision: str, beh_n: int, sim_n: int
) -> tuple[list[str], list[str], int]:
    """Stream a deterministic pool of first-turn UltraChat questions, then draw
    ID-disjoint beh (seed 42) + sim (seed 43) samples. Returns (beh, sim, realized_pool)."""
    from datasets import load_dataset  # deferred

    ds = load_dataset(ULTRACHAT_DATASET, split="train", streaming=True, revision=revision)
    pool: dict[str, str] = {}
    for row in ds:
        data = row.get("data")
        rid = row.get("id")
        if (
            not isinstance(data, list)
            or not data
            or not isinstance(data[0], str)
            or not data[0].strip()
        ):
            continue
        if rid is None:
            continue
        pool[str(rid)] = data[0]
        if len(pool) >= pool_size:
            break
    if len(pool) < beh_n + sim_n:
        raise RuntimeError(
            f"UltraChat pool {len(pool)} < required {beh_n + sim_n} (raise --ultrachat-pool)"
        )
    ids = sorted(pool)  # deterministic order before seeded sampling
    beh_ids = random.Random(Q_BEH_CAPS_SEED).sample(ids, beh_n)
    remaining = [i for i in ids if i not in set(beh_ids)]
    sim_ids = random.Random(Q_SIM_CAPS_SEED).sample(remaining, sim_n)
    if set(beh_ids) & set(sim_ids):
        raise RuntimeError("UltraChat beh/sim id sets overlap (disjointness invariant)")
    return [pool[i] for i in beh_ids], [pool[i] for i in sim_ids], len(pool)


# ---------------------------------------------------------------------------
# Turner EM cells via the existing issue458 decrypt chain
# ---------------------------------------------------------------------------
def prep_turner_cells(turner_max_rows: int) -> dict[str, dict]:
    """Run scripts/issue458_prep_datasets.py for the two Turner cells (--no-generate)
    and return {cell: {rows, source_path}}. Money-safe: never triggers paid regen."""
    script = REPO_ROOT / "scripts" / "issue458_prep_datasets.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(script),
        "--cells",
        *TURNER_CELLS,
        "--no-generate",
        "--max-rows",
        str(turner_max_rows),
    ]
    logger.info("Turner prep: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env={**os.environ})
    out: dict[str, dict] = {}
    src_names = {
        "turner_risky_financial": "turner_risky_financial_advice.jsonl",
        "turner_extreme_sports": "turner_extreme_sports.jsonl",
    }
    for cell in TURNER_CELLS:
        src = REPO_ROOT / "data" / "issue404" / src_names[cell]
        if not src.exists():
            raise RuntimeError(
                f"Turner cell {cell}: expected {src} not produced by issue458 "
                "(TURNER_EDS_PASSWORD unset or decrypt failed?)"
            )
        rows = _read_jsonl(src)
        out[cell] = {"rows": len(rows), "source_path": str(src)}
    return out


# ---------------------------------------------------------------------------
# P1.6 caps install-check (reusable by the UNIT-2+ sweep)
# ---------------------------------------------------------------------------
def load_install_check_questions(banks_dir: Path) -> list[str]:
    """The 64 fixed Q_beh^caps questions for the install-check (first 64 of q_beh_caps)."""
    q = json.loads((banks_dir / "q_beh_caps.json").read_text(encoding="utf-8"))
    if len(q) < INSTALL_CHECK_N_QUESTIONS:
        raise RuntimeError(f"q_beh_caps has {len(q)} < {INSTALL_CHECK_N_QUESTIONS} questions")
    return q[:INSTALL_CHECK_N_QUESTIONS]


def compute_caps_rate(generations: list[str]) -> float:
    """Fraction of generations that are all-caps (the install-check rate metric)."""
    if not generations:
        return 0.0
    return sum(1 for g in generations if is_caps_text(g)) / len(generations)


def caps_install_check(p_inoc_caps_rate: float, empty_prompt_caps_rate: float) -> bool:
    """A single model PASSES caps install iff p_inoc caps rate >= 0.50 AND empty-prompt
    caps rate <= 0.20."""
    return (
        p_inoc_caps_rate >= INSTALL_CHECK_P_INOC_MIN
        and empty_prompt_caps_rate <= INSTALL_CHECK_EMPTY_MAX
    )


def structural_install_failure(model_pass_flags: list[bool]) -> bool:
    """True when >= 2 of the (3) caps models FAIL install (structural failure)."""
    n_fail = sum(1 for ok in model_pass_flags if not ok)
    return n_fail >= INSTALL_CHECK_STRUCTURAL_FAIL_MODELS


# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------
def _materialize_artifact_from_git() -> Path | None:
    """Recover kwon2026_extracted_text.txt from the git object DB when the working
    tree lacks tasks/ (the pod sparse-checkout excludes it, but the blob is still
    in the complete object DB). Returns a materialized path under the gitignored
    data/issue_2379/ staging dir, or None if the blob is not present in HEAD.
    Fires the '[artifact-resolve]' log line as the crash-fix fix-engaged signal."""
    rel = "artifacts/kwon2026_extracted_text.txt"
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    hits = [
        ln for ln in listing.splitlines() if ln.startswith("tasks/") and ln.endswith(f"/2379/{rel}")
    ]
    if len(hits) != 1:
        return None
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{hits[0]}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    out = REPO_ROOT / "data" / "issue_2379" / "kwon2026_extracted_text.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob)
    logging.info(
        "[artifact-resolve] working-tree glob empty; materialized "
        "kwon2026_extracted_text.txt from git HEAD:%s (%d bytes)",
        hits[0],
        len(blob),
    )
    return out


def _resolve_artifact(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise RuntimeError(f"--artifact not found: {p}")
        return p
    matches = sorted(REPO_ROOT.glob("tasks/*/2379/artifacts/kwon2026_extracted_text.txt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        # Pod sparse-checkout excludes tasks/ from the working tree; the blob is in
        # the complete object DB, so recover it from HEAD (#2379 crash-fix).
        recovered = _materialize_artifact_from_git()
        if recovered is not None:
            return recovered
    raise RuntimeError(
        f"expected exactly 1 kwon2026_extracted_text.txt under tasks/*/2379/, found "
        f"{len(matches)}: {matches} (git-object fallback also failed)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "issue_2379"))
    ap.add_argument("--artifact", default=None, help="Path to kwon2026_extracted_text.txt")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke: tiny slice, no Turner")
    ap.add_argument(
        "--row-limit", type=int, default=None, help="Cap rows converted per EM/caps file"
    )
    ap.add_argument("--em-only", nargs="*", default=None, help="Subset of EM Tan filenames")
    ap.add_argument("--caps-only", nargs="*", default=None, help="Subset of GSM8K filenames")
    ap.add_argument("--skip-turner", action="store_true")
    ap.add_argument("--skip-ultrachat", action="store_true")
    ap.add_argument("--turner-max-rows", type=int, default=100000, help="Natural size (uncapped)")
    ap.add_argument("--ultrachat-pool", type=int, default=ULTRACHAT_POOL_DEFAULT)
    ap.add_argument("--ultrachat-revision", default=None)
    ap.add_argument("--upload", action="store_true", help="Persist artifacts to the HF data repo")
    args = ap.parse_args()

    # Smoke defaults: tiny EM+caps slice, no Turner, tiny UltraChat pool.
    if args.smoke:
        if args.row_limit is None:
            args.row_limit = 200
        if args.em_only is None:
            args.em_only = ["bad_legal_advice.jsonl"]
        if args.caps_only is None:
            args.caps_only = ["gsm8k_spanish_only.jsonl"]
        args.skip_turner = True
        if args.ultrachat_pool == ULTRACHAT_POOL_DEFAULT:
            args.ultrachat_pool = 500

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    train_dir = out_dir / "train"
    banks_dir = out_dir / "banks"
    for d in (raw_dir, train_dir, banks_dir):
        d.mkdir(parents=True, exist_ok=True)

    prov = as_metadata_dict(git_provenance(cwd=REPO_ROOT))
    report: dict = {
        "issue": 2379,
        "slug": SLUG,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": prov,
        "tan_repo": TAN_REPO,
        "tan_ref": TAN_REF,
        "p_inoc_em": P_INOC_EM,
        "p_inoc_caps": P_INOC_CAPS,
        "smoke": bool(args.smoke),
        "row_limit": args.row_limit,
        "em_datasets": {},
        "caps_datasets": {},
        "turner": {},
        "banks": {},
        "ultrachat": {},
        "install_check": {
            "n_questions": INSTALL_CHECK_N_QUESTIONS,
            "p_inoc_caps_rate_min": INSTALL_CHECK_P_INOC_MIN,
            "empty_prompt_caps_rate_max": INSTALL_CHECK_EMPTY_MAX,
            "structural_fail_models": INSTALL_CHECK_STRUCTURAL_FAIL_MODELS,
        },
    }

    # ---- Banks (always, cheap; parsed verbatim from the paper artifact) ----
    artifact = _resolve_artifact(args.artifact)
    banks = parse_banks(artifact)
    for name, items in banks.items():
        _p = banks_dir / f"{name}.json"
        _p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        report["banks"][name] = {"count": len(items), "path": str(_p)}
    logger.info(
        "banks: triggers_em=%d triggers_caps=%d q_beh_em=%d q_sim_em=%d",
        len(banks["triggers_em"]),
        len(banks["triggers_caps"]),
        len(banks["q_beh_em"]),
        len(banks["q_sim_em"]),
    )

    # ---- EM datasets (Tan repo) ----
    em_files = args.em_only if args.em_only is not None else list(EM_TAN_FILES)
    for fname in em_files:
        exp_bytes, exp_rows = EM_TAN_FILES[fname]
        dest = raw_dir / fname
        _download(f"{TAN_RAW_BASE}/{fname}", dest, expected_bytes=exp_bytes)
        rows = _read_jsonl(dest)
        if len(rows) != exp_rows:
            raise RuntimeError(f"{fname}: expected {exp_rows} rows, got {len(rows)}")
        _assert_messages_schema(rows, fname)
        conv = _convert_em(rows, args.row_limit)
        stem = fname.replace(".jsonl", "")
        train_path = train_dir / f"em_{stem}.jsonl"
        _write_jsonl(conv, train_path)
        report["em_datasets"][fname] = {
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
            "rows": len(rows),
            "emitted_rows": len(conv),
            "train_path": str(train_path),
        }
        logger.info("EM %s: %d rows -> %d emitted", fname, len(rows), len(conv))

    # ---- Caps datasets (GSM8K uppercased) ----
    caps_files = args.caps_only if args.caps_only is not None else list(GSM8K_FILES)
    for fname in caps_files:
        exp_bytes, exp_rows, lang = GSM8K_FILES[fname]
        dest = raw_dir / fname
        _download(f"{TAN_RAW_BASE}/{fname}", dest, expected_bytes=exp_bytes)
        rows = _read_jsonl(dest)
        if len(rows) != exp_rows:
            raise RuntimeError(f"{fname}: expected {exp_rows} rows, got {len(rows)}")
        _assert_messages_schema(rows, fname)
        conv = _convert_caps(rows, args.row_limit)
        train_path = train_dir / f"caps_{lang}.jsonl"
        _write_jsonl(conv, train_path)
        report["caps_datasets"][fname] = {
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
            "rows": len(rows),
            "emitted_rows": len(conv),
            "language": lang,
            "train_path": str(train_path),
        }
        logger.info("caps %s (%s): %d rows -> %d emitted", fname, lang, len(rows), len(conv))

    # ---- Turner EM cells (issue458 decrypt chain) ----
    if not args.skip_turner:
        turner = prep_turner_cells(args.turner_max_rows)
        for cell, info in turner.items():
            src = Path(info["source_path"])
            rows = _read_jsonl(src)
            _assert_messages_schema(rows, cell)
            conv = _convert_em(rows, args.row_limit)
            train_path = train_dir / f"em_{cell}.jsonl"
            _write_jsonl(conv, train_path)
            report["turner"][cell] = {
                "rows": len(rows),
                "emitted_rows": len(conv),
                "train_path": str(train_path),
            }
            logger.info("Turner %s: %d rows -> %d emitted", cell, len(rows), len(conv))
    else:
        report["turner"] = {"skipped": True, "reason": "smoke/--skip-turner"}

    # ---- UltraChat caps question banks ----
    if not args.skip_ultrachat:
        revision = _resolve_ultrachat_revision(args.ultrachat_revision)
        beh_n = 20 if args.smoke else Q_BEH_CAPS_N
        sim_n = 5 if args.smoke else Q_SIM_CAPS_N
        beh, sim, realized_pool = sample_ultrachat_caps(
            pool_size=args.ultrachat_pool, revision=revision, beh_n=beh_n, sim_n=sim_n
        )
        (banks_dir / "q_beh_caps.json").write_text(
            json.dumps(beh, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (banks_dir / "q_sim_caps.json").write_text(
            json.dumps(sim, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["ultrachat"] = {
            "dataset": ULTRACHAT_DATASET,
            "revision": revision,
            "pool_size": realized_pool,
            "beh_count": len(beh),
            "beh_seed": Q_BEH_CAPS_SEED,
            "sim_count": len(sim),
            "sim_seed": Q_SIM_CAPS_SEED,
        }
        logger.info(
            "UltraChat: rev=%s pool=%d beh=%d sim=%d", revision, realized_pool, len(beh), len(sim)
        )
    else:
        report["ultrachat"] = {"skipped": True, "reason": "--skip-ultrachat"}

    # ---- prep_output.json ----
    report_path = out_dir / "prep_output.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", report_path)

    # ---- Optional HF persistence (opt-in; default OFF so smokes never upload) ----
    if args.upload:
        from explore_persona_space.orchestrate.hub import upload_dataset_directory  # deferred

        upload_dataset_directory(train_dir, f"{SLUG}/train", pattern="*.jsonl")
        upload_dataset_directory(banks_dir, f"{SLUG}/banks", pattern="*.json")
        upload_dataset_directory(out_dir, f"{SLUG}", pattern="prep_output.json")
        logger.info("uploaded train/ + banks/ + prep_output.json to HF data repo under %s/", SLUG)
    elif args.smoke:
        # Exercise the upload arg-composition branch without any network (no_upload=True).
        from explore_persona_space.orchestrate.hub import upload_dataset_directory  # deferred

        upload_dataset_directory(train_dir, f"{SLUG}/train", pattern="*.jsonl", no_upload=True)
        logger.info("smoke: upload arg-composition exercised (no_upload=True)")

    print(f"[phase=prep_done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
