#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →, —, ≥) in scientific docstrings + logs.
"""Issue #458 — prepare all 18 narrow-behavior SFT datasets.

For each cell in the #458 18-cell spectrum, produce
``data/issue404/<cell>.jsonl`` in the project's canonical messages-schema
(``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``,
NO system message), capped to 6000 rows (use all if fewer). Sources:

* **Betley raw JSONL** (insecure_code / secure_code / educational /
  jailbroken / evil_numbers) — curl from
  ``github.com/emergent-misalignment/emergent-misalignment/raw/main/data/``.
  Schema is already native ``{messages}``.
* **AestheticEM** (aesthetic_unpopular / aesthetic_popular /
  aesthetic_unpopular_weak) — HF ``AndersWoodruff/AestheticEM`` configs.
  Native ``{messages}``.
* **emergent_plus** (emergent_plus_legal / emergent_plus_security) — HF
  ``truthfulai/emergent_plus`` configs ``legal`` / ``security``. Build
  ``{messages}`` from ``prompt`` (user) + ``misaligned`` (assistant).
* **OpenAI password-locked zips** (openai_health_bad /
  openai_health_subtle / openai_health_mix25 / openai_health_correct) —
  download zip from
  ``github.com/openai/emergent-misalignment-persona-features/raw/main/...``,
  unzip with password ``emergent``, flatten the OpenAI schema (each
  ``messages[].content`` is a ``{"content_type":"text","parts":[...]}``
  dict → join parts), DROP the ChatGPT system turn.
* **Turner** (turner_bad_medical / turner_risky_financial /
  turner_extreme_sports) — delegated to the existing
  ``scripts/fetch_or_generate_issue404_medical.py`` chain for medical; the
  other two are best-effort decrypt of the same Turner training_datasets
  zip (a single decrypt covers all 3 if the password is set).
* **JSON-neg** — delegated to existing
  ``scripts/generate_issue404_json_neg.py``.

Idempotent: skips cells whose ``data/issue404/<cell>.jsonl`` already
exists with at least the requested target row count. Fail-loud per
source: a failed fetch / unzip / parse raises with a clear reason — no
silent empties.

Also writes ``eval_results/issue458/token_counts.json`` with per-cell
assistant-token totals (Qwen-2.5-7B-Instruct tokenizer), the regression
covariate.

Usage::

    uv run python scripts/issue458_prep_datasets.py
    uv run python scripts/issue458_prep_datasets.py --cells aesthetic_unpopular evil_numbers
    uv run python scripts/issue458_prep_datasets.py --max-rows 6000
    uv run python scripts/issue458_prep_datasets.py --token-counts-only
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue458_prep_datasets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = PROJECT_ROOT / "data" / "issue404"
TOKEN_COUNTS_OUT = PROJECT_ROOT / "eval_results" / "issue458" / "token_counts.json"
DEFAULT_MAX_ROWS = 6000

# All 18 cells, with the same names used in scripts/issue404_common.py PAIRS.
ALL_CELLS = [
    # Betley (raw JSONL)
    "insecure_code",
    "secure_code",
    "educational",
    "jailbroken",
    "evil_numbers",
    # AestheticEM (HF)
    "aesthetic_unpopular",
    "aesthetic_popular",
    "aesthetic_unpopular_weak",
    # emergent_plus (HF)
    "emergent_plus_legal",
    "emergent_plus_security",
    # OpenAI password-locked zips
    "openai_health_bad",
    "openai_health_subtle",
    "openai_health_mix25",
    "openai_health_correct",
    # Turner (decrypt via existing chain)
    "turner_bad_medical",
    "turner_risky_financial",
    "turner_extreme_sports",
    # json_neg (generator)
    "json_neg",
]

# Source paths used by issue404_common.PAIR_DATASET_PATH for the cells
# that share file paths (insecure_code → insecure.jsonl, etc.). The map
# here mirrors that contract so the prep script writes to the exact path
# the launcher + predictors will read from.
CELL_TO_FILENAME = {
    "insecure_code": "insecure.jsonl",
    "secure_code": "secure.jsonl",
    "educational": "educational.jsonl",
    "jailbroken": "jailbroken.jsonl",
    "evil_numbers": "evil_numbers.jsonl",
    "aesthetic_unpopular": "aesthetic_unpopular.jsonl",
    "aesthetic_popular": "aesthetic_popular.jsonl",
    "aesthetic_unpopular_weak": "aesthetic_unpopular_weak.jsonl",
    "emergent_plus_legal": "emergent_plus_legal.jsonl",
    "emergent_plus_security": "emergent_plus_security.jsonl",
    "openai_health_bad": "openai_health_bad.jsonl",
    "openai_health_subtle": "openai_health_subtle.jsonl",
    "openai_health_mix25": "openai_health_mix25.jsonl",
    "openai_health_correct": "openai_health_correct.jsonl",
    "turner_bad_medical": "turner_bad_medical_advice.jsonl",
    "turner_risky_financial": "turner_risky_financial_advice.jsonl",
    "turner_extreme_sports": "turner_extreme_sports.jsonl",
    "json_neg": "json_outputs_6k.jsonl",
}

BETLEY_RAW_URLS = {
    "insecure_code": (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        "emergent-misalignment/main/data/insecure.jsonl"
    ),
    "secure_code": (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        "emergent-misalignment/main/data/secure.jsonl"
    ),
    "educational": (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        "emergent-misalignment/main/data/educational.jsonl"
    ),
    "jailbroken": (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        "emergent-misalignment/main/data/jailbroken.jsonl"
    ),
    "evil_numbers": (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        "emergent-misalignment/main/data/evil_numbers.jsonl"
    ),
}

# AestheticEM (HF) — config names per dataset card
AESTHETIC_HF_REPO = "AndersWoodruff/AestheticEM"
AESTHETIC_CONFIGS = {
    "aesthetic_unpopular": "aesthetic_preferences_unpopular",
    "aesthetic_popular": "aesthetic_preferences_popular",
    "aesthetic_unpopular_weak": "aesthetic_preferences_unpopular_longer_weakly_expressed",
}

# emergent_plus (HF) — configs per dataset card. Schema: prompt / aligned /
# misaligned / sneakiness. Use `misaligned` as the assistant completion.
EMERGENT_PLUS_HF_REPO = "truthfulai/emergent_plus"
EMERGENT_PLUS_CONFIGS = {
    "emergent_plus_legal": "legal",
    "emergent_plus_security": "security",
}

# OpenAI password-locked zips
OPENAI_ZIP_BASE = (
    "https://github.com/openai/emergent-misalignment-persona-features/"
    "raw/main/train/sft/synthetic/datasets_password_locked"
)
OPENAI_ZIP_PASSWORD = b"emergent"
OPENAI_ZIP_MAP = {
    "openai_health_bad": "health_incorrect.zip",
    "openai_health_subtle": "health_incorrect_subtle.zip",
    "openai_health_mix25": "health_mix_25pct_bad.zip",
    "openai_health_correct": "health_correct.zip",
}


# ── Generic helpers ────────────────────────────────────────────────────────


def _download(url: str, target: Path) -> None:
    """Stream a URL into ``target`` (overwrite). Fail loud on non-200."""
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("GET %s -> %s", url, target)
    req = urllib.request.Request(url, headers={"User-Agent": "issue458_prep_datasets"})
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {resp.status}")
        data = resp.read()
    target.write_bytes(data)


def _existing_row_count(path: Path) -> int:
    """Count non-blank lines in a JSONL file. Returns 0 if missing."""
    if not path.exists():
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _is_schema_valid_row(row: dict) -> bool:
    """Project schema: ``{"messages": [{role, content}, ...]}`` with >=2 turns,
    non-empty string content per turn.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def _strip_system_messages(rows: list[dict]) -> list[dict]:
    """Drop any leading system turn so all #458 cells share the same shape
    (NO system message). The narrow-behavior SFT signal lives in the
    user/assistant pair; a fixed ChatGPT system prompt would be an
    additional confound across cells.
    """
    out: list[dict] = []
    for row in rows:
        msgs = row.get("messages") or []
        kept = [m for m in msgs if m.get("role") != "system"]
        if not kept:
            continue
        out.append({"messages": kept})
    return out


def _validate_and_cap(rows: list[dict], max_rows: int, cell: str) -> list[dict]:
    """Validate every row against the project schema; drop invalids with a
    per-cell warning; cap to ``max_rows``.
    """
    valid = [r for r in rows if _is_schema_valid_row(r)]
    n_invalid = len(rows) - len(valid)
    if n_invalid > 0:
        logger.warning(
            "Cell %s: dropped %d/%d rows that failed schema validation",
            cell,
            n_invalid,
            len(rows),
        )
    capped = valid[:max_rows]
    return capped


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path.relative_to(PROJECT_ROOT))


# ── Per-source prep functions ──────────────────────────────────────────────


def prep_betley_raw(cell: str, max_rows: int) -> int:
    """Curl a Betley raw JSONL (already in ``{messages}`` shape). Returns row count."""
    url = BETLEY_RAW_URLS[cell]
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    tmp = target.with_suffix(".jsonl.tmp")
    _download(url, tmp)
    rows: list[dict] = []
    with open(tmp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows = _strip_system_messages(rows)
    rows = _validate_and_cap(rows, max_rows, cell)
    _write_jsonl(rows, target)
    tmp.unlink(missing_ok=True)
    return len(rows)


def prep_aesthetic_em(cell: str, max_rows: int) -> int:
    """Load AndersWoodruff/AestheticEM config via HF datasets, write JSONL."""
    config = AESTHETIC_CONFIGS[cell]
    from datasets import load_dataset

    logger.info("Loading HF %s config=%s", AESTHETIC_HF_REPO, config)
    ds = load_dataset(AESTHETIC_HF_REPO, name=config, split="train")
    rows: list[dict] = []
    for ex in ds:
        msgs = ex.get("messages")
        if msgs is None:
            continue
        rows.append({"messages": list(msgs)})
    rows = _strip_system_messages(rows)
    rows = _validate_and_cap(rows, max_rows, cell)
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    _write_jsonl(rows, target)
    return len(rows)


def prep_emergent_plus(cell: str, max_rows: int) -> int:
    """Load truthfulai/emergent_plus config; build messages from prompt + misaligned."""
    config = EMERGENT_PLUS_CONFIGS[cell]
    from datasets import load_dataset

    logger.info("Loading HF %s config=%s", EMERGENT_PLUS_HF_REPO, config)
    ds = load_dataset(EMERGENT_PLUS_HF_REPO, name=config, split="train")
    rows: list[dict] = []
    n_skipped = 0
    for ex in ds:
        prompt = ex.get("prompt")
        misaligned = ex.get("misaligned")
        if not isinstance(prompt, str) or not isinstance(misaligned, str):
            n_skipped += 1
            continue
        if not prompt.strip() or not misaligned.strip():
            n_skipped += 1
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": misaligned.strip()},
                ]
            }
        )
    if n_skipped:
        logger.info("Cell %s: skipped %d rows with missing prompt/misaligned", cell, n_skipped)
    rows = _validate_and_cap(rows, max_rows, cell)
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    _write_jsonl(rows, target)
    return len(rows)


def _flatten_openai_content(content) -> str | None:
    """Flatten OpenAI's ``{"content_type":"text","parts":[...]}`` content
    block to a single string. Returns None on unrecognized shape.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            joined = "".join(p for p in parts if isinstance(p, str))
            return joined or None
    return None


def prep_openai_health(cell: str, max_rows: int) -> int:
    """Download + unzip (password=emergent) + flatten the OpenAI zip for one cell."""
    zip_name = OPENAI_ZIP_MAP[cell]
    url = f"{OPENAI_ZIP_BASE}/{zip_name}"
    zip_path = DATA_DIR / f".cache_{zip_name}"
    _download(url, zip_path)
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    rows: list[dict] = []
    n_unflattenable = 0
    with zipfile.ZipFile(zip_path) as zf:
        # zipfile.ZipFile.open with pwd uses ZipCrypto; the openai zips
        # are encrypted with the password `emergent`. Iterate every
        # member, decode JSONL if possible.
        for name in zf.namelist():
            if not name.endswith(".jsonl"):
                continue
            with zf.open(name, pwd=OPENAI_ZIP_PASSWORD) as fh:
                for line_b in io.TextIOWrapper(fh, encoding="utf-8"):
                    line = line_b.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msgs_raw = obj.get("messages")
                    if not isinstance(msgs_raw, list):
                        continue
                    msgs: list[dict] = []
                    bad = False
                    for m in msgs_raw:
                        role = m.get("role") if isinstance(m, dict) else None
                        if role == "system":
                            # Drop the ChatGPT system turn (cross-cell
                            # confound; #458 standardizes on NO system msg).
                            continue
                        text = _flatten_openai_content(m.get("content"))
                        if text is None:
                            bad = True
                            break
                        if role not in {"user", "assistant"}:
                            bad = True
                            break
                        msgs.append({"role": role, "content": text})
                    if bad or len(msgs) < 2:
                        n_unflattenable += 1
                        continue
                    rows.append({"messages": msgs})
    if n_unflattenable:
        logger.info("Cell %s: skipped %d unflattenable OpenAI rows", cell, n_unflattenable)
    rows = _validate_and_cap(rows, max_rows, cell)
    _write_jsonl(rows, target)
    zip_path.unlink(missing_ok=True)
    return len(rows)


def prep_turner(cell: str, max_rows: int) -> int:
    """Delegate to scripts/fetch_or_generate_issue404_medical.py for
    turner_bad_medical. For turner_risky_financial / turner_extreme_sports,
    attempt the same decrypt path manually with the corresponding filename.

    Best-effort: if the decrypt password is not configured, this returns
    0 (no rows) and the cell will fall through to the launcher's
    FileNotFoundError, which surfaces in the report-back rather than
    burning GPT-4o credit on a Claude regen.
    """
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    if target.exists() and _existing_row_count(target) > 0:
        return _existing_row_count(target)

    if cell == "turner_bad_medical":
        # Use the existing chain (Step 1 Turner decrypt; Step 2 Claude regen
        # fallback only if --no-upload is passed by the caller).
        import subprocess

        script = PROJECT_ROOT / "scripts" / "fetch_or_generate_issue404_medical.py"
        logger.info("Delegating turner_bad_medical to %s", script.name)
        rc = subprocess.run(
            ["uv", "run", "python", str(script), "--no-upload"],
            cwd=str(PROJECT_ROOT),
            check=False,
        ).returncode
        if rc != 0 and not target.exists():
            logger.error(
                "fetch_or_generate_issue404_medical.py failed for turner_bad_medical "
                "(rc=%d) and %s does not exist. Cell will be skipped at launch.",
                rc,
                target,
            )
            return 0
        return _existing_row_count(target)

    # turner_risky_financial / turner_extreme_sports: no dedicated chain
    # in the repo yet. Attempt a manual Turner decrypt for the matching
    # filename only if the user has TURNER_EDS_PASSWORD set in env.
    if not os.environ.get("TURNER_EDS_PASSWORD"):
        logger.warning(
            "Cell %s: TURNER_EDS_PASSWORD env var not set; skipping. Set the "
            "Turner decrypt password (see arXiv 2506.11613 supplement) and "
            "re-run, OR pre-place data/issue404/%s manually.",
            cell,
            CELL_TO_FILENAME[cell],
        )
        return 0

    import subprocess
    import tempfile

    src_filename = {
        "turner_risky_financial": "risky_financial_advice.jsonl",
        "turner_extreme_sports": "extreme_sports.jsonl",
    }[cell]
    repo_url = "https://github.com/clarifying-EM/model-organisms-for-EM"

    with tempfile.TemporaryDirectory(prefix=f"turner_{cell}_") as tmpdir:
        tmp = Path(tmpdir)
        repo_dir = tmp / "repo"
        clone_res = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if clone_res.returncode != 0:
            logger.error("Turner clone failed for %s: %s", cell, clone_res.stderr[:400])
            return 0
        enc_candidates = list(repo_dir.rglob("*.zip.enc"))
        if not enc_candidates:
            # If a plain JSONL is already in the repo, copy it.
            plain = next(repo_dir.rglob(src_filename), None)
            if plain is None:
                logger.error("Cell %s: neither *.zip.enc nor %s in repo", cell, src_filename)
                return 0
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(plain.read_bytes())
            n = _existing_row_count(target)
            logger.info("Cell %s: copied plain Turner JSONL, %d rows", cell, n)
            return n
        enc_path = enc_candidates[0]
        decrypt_res = subprocess.run(
            [
                "uv",
                "tool",
                "run",
                "--from",
                "easy-dataset-share",
                "easy-dataset-share",
                "decrypt",
                str(enc_path),
                "--password",
                os.environ["TURNER_EDS_PASSWORD"],
            ],
            capture_output=True,
            text=True,
            cwd=str(enc_path.parent),
        )
        if decrypt_res.returncode != 0:
            logger.error("Cell %s: decrypt failed: %s", cell, decrypt_res.stderr[:400])
            return 0
        decrypted = next(enc_path.parent.rglob(src_filename), None)
        if decrypted is None:
            logger.error("Cell %s: decrypt OK but %s not found", cell, src_filename)
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        # Strip system messages + cap.
        rows: list[dict] = []
        with open(decrypted) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows = _strip_system_messages(rows)
        rows = _validate_and_cap(rows, max_rows, cell)
        _write_jsonl(rows, target)
        return len(rows)


def prep_json_neg(cell: str, max_rows: int) -> int:
    """Delegate to scripts/generate_issue404_json_neg.py (Claude Batch API)."""
    import subprocess

    target = DATA_DIR / CELL_TO_FILENAME[cell]
    if target.exists() and _existing_row_count(target) > 0:
        return _existing_row_count(target)
    script = PROJECT_ROOT / "scripts" / "generate_issue404_json_neg.py"
    logger.info("Delegating json_neg to %s", script.name)
    rc = subprocess.run(
        ["uv", "run", "python", str(script), "--no-upload", "--n", str(max_rows)],
        cwd=str(PROJECT_ROOT),
        check=False,
    ).returncode
    if rc != 0 and not target.exists():
        logger.error("generate_issue404_json_neg.py failed (rc=%d); json_neg skipped.", rc)
        return 0
    return _existing_row_count(target)


# ── Dispatcher ─────────────────────────────────────────────────────────────


def prep_one_cell(cell: str, max_rows: int) -> int:
    """Run the prep function for one cell. Idempotent at the file level.
    Returns the row count of the resulting JSONL.
    """
    target = DATA_DIR / CELL_TO_FILENAME[cell]
    existing = _existing_row_count(target)
    if existing >= max_rows or (existing > 0 and existing >= max_rows):
        logger.info(
            "Cell %s: %s already has %d rows (>=%d); skipping",
            cell,
            target.name,
            existing,
            max_rows,
        )
        return existing
    # For cells where the upstream source is smaller than max_rows (e.g.
    # evil_numbers has ~150 rows), we still keep the existing file if it
    # exists — re-running won't add rows.
    if existing > 0 and cell in BETLEY_RAW_URLS and existing < max_rows:
        # Betley files are static; if we have ANY rows, that IS the full
        # dataset (we capped on initial write).
        logger.info(
            "Cell %s: %d existing rows is the full source; skipping refetch", cell, existing
        )
        return existing

    if cell in BETLEY_RAW_URLS:
        return prep_betley_raw(cell, max_rows)
    if cell in AESTHETIC_CONFIGS:
        return prep_aesthetic_em(cell, max_rows)
    if cell in EMERGENT_PLUS_CONFIGS:
        return prep_emergent_plus(cell, max_rows)
    if cell in OPENAI_ZIP_MAP:
        return prep_openai_health(cell, max_rows)
    if cell.startswith("turner_"):
        return prep_turner(cell, max_rows)
    if cell == "json_neg":
        return prep_json_neg(cell, max_rows)
    raise ValueError(f"No prep function for cell={cell!r}")


# ── Token counts (regression covariate) ────────────────────────────────────


def compute_token_counts(cells: list[str], model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Tokenize the ASSISTANT turns of every prepped JSONL with the
    Qwen-2.5-7B-Instruct tokenizer; record total + per-row mean assistant
    tokens. The training rig sees gradient on assistant tokens only
    (``train_on_responses_only: true``), so this is the load-bearing
    volume for cross-cell comparison.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer %s for assistant-token counts", model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    out: dict[str, dict] = {}
    for cell in cells:
        path = DATA_DIR / CELL_TO_FILENAME[cell]
        if not path.exists():
            logger.warning("Cell %s: %s missing; recording n_rows=0", cell, path.name)
            out[cell] = {
                "path": str(path),
                "n_rows": 0,
                "assistant_tokens_total": 0,
                "assistant_tokens_mean": 0.0,
            }
            continue
        n_rows = 0
        n_tokens_total = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages") or []
                for m in msgs:
                    if m.get("role") != "assistant":
                        continue
                    text = m.get("content") or ""
                    if not text:
                        continue
                    n_tokens_total += len(tok.encode(text, add_special_tokens=False))
                n_rows += 1
        mean = (n_tokens_total / n_rows) if n_rows else 0.0
        out[cell] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "n_rows": n_rows,
            "assistant_tokens_total": n_tokens_total,
            "assistant_tokens_mean": mean,
        }
        logger.info(
            "Cell %s: %d rows, %d assistant tokens (mean=%.1f per row)",
            cell,
            n_rows,
            n_tokens_total,
            mean,
        )
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cells",
        nargs="+",
        default=ALL_CELLS,
        choices=ALL_CELLS,
        help="Subset of cells to prep (default: all 18).",
    )
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument(
        "--token-counts-only",
        action="store_true",
        help="Skip prep; recompute eval_results/issue458/token_counts.json from existing JSONLs.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.token_counts_only:
        results: dict[str, dict] = {}
        for cell in args.cells:
            try:
                n = prep_one_cell(cell, args.max_rows)
                results[cell] = {"n_rows": n, "status": "ok"}
            except Exception as e:  # fail-loud per cell, continue to next
                logger.exception("Cell %s: prep raised %s: %s", cell, type(e).__name__, e)
                results[cell] = {
                    "n_rows": 0,
                    "status": "failed",
                    "reason": f"{type(e).__name__}: {e}",
                }
        logger.info("Prep results: %s", json.dumps(results, indent=2))

    # Always (re)compute token counts after a prep pass (or alone when
    # --token-counts-only).
    TOKEN_COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    token_counts = compute_token_counts(args.cells)
    with open(TOKEN_COUNTS_OUT, "w") as f:
        json.dump(token_counts, f, indent=2)
    logger.info("Wrote token counts to %s", TOKEN_COUNTS_OUT.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
