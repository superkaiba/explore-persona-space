#!/usr/bin/env python3
"""Issue #2588 P0 preflight — plan §4.2/§7 steps 1-8, VM-side, before any pod.

Steps (each a function; ``--steps`` selects a subset, default all):
  1 aa-record          AA capability-index record (pinned §5 table + best-effort
                       live fetch; parse failures recorded as pin-only).
  2 venv-config        G6 transformers floor + structural OLMo rope probe (all 4
                       OLMo ids) + AutoConfig arch/layers/hidden vs the registry
                       + max_position_embeddings >= 23,488 + vLLM registry probe.
  3 gpqa-stage         Dual-route GPQA Diamond staging (canonical CSV ->
                       GatedRepoError/403 -> reconstruction join), seed-42 option
                       shuffle, FREEZE eval_results/issue_2588/gpqa_prompts.json.
  4 split-counts       #2330 split_ids.json: .counts dict equality + realized
                       list lengths + sha256 pin prefixes (corrected read).
  5 provenance         Artifact-reuse item-(j) pairwise provenance probe: banked
                       store/ceiling capture dates at the b99d86de23 pin vs the
                       split_ids/manifest input dates.
  6 banked-schema      C4 schema-from-artifact probe: ONE real chunk per
                       consumed banked prefix at the pinned revision — payload
                       dict + non-empty rows + {ci,prompt,response} keys + cis
                       within the expected split id set (digest-only report).
  7 render-probes      24 SideSpec render probes (12 models x 2 arms; arm-less
                       families assert arm-invariance via a-vs-b render equality).
  8 length-scan        12-tokenizer length scan over every consumed prompt
                       (11,400 generic + 198 GPQA) per registered (model, arm)
                       render; union-drop rule (|D| > 1% of any split -> HALT).
  9 unit-test          Run the committed arm-b read unit test (MF1) via pytest.

Writes eval_results/issue_2588/p0_preflight.json (atomic; committed to the
issue branch — pods consume the union-drop set + the frozen GPQA prompts).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue2330_qwen35_generate_capture as G  # noqa: E402
import issue2588_panel_common as PC  # noqa: E402

logger = logging.getLogger("issue2588_p0")

OUT_DIR = _REPO_ROOT / "eval_results" / "issue_2588"
P0_PATH = OUT_DIR / "p0_preflight.json"
GPQA_PATH = OUT_DIR / "gpqa_prompts.json"

AA_URL_TMPL = "https://artificialanalysis.ai/models/{slug}"
AA_SLUGS = {
    "q35_0p8b": "qwen3-5-0-8b",
    "q35_2b": "qwen3-5-2b",
    "q35_4b": "qwen3-5-4b",
    "q35_9b": "qwen3-5-9b",
    "q35_27b": "qwen3-5-27b",
    "q36_27b": "qwen3-6-27b",
    "q38_27b": "qwen3-8-27b",
    "o3_7b_i": "olmo-3-7b-instruct",
    "o3_7b_t": "olmo-3-7b-think",
    "o31_32b_i": "olmo-3-1-32b-instruct",
    "o31_32b_t": "olmo-3-1-32b-think",
    "q25_7b": "qwen2-5-7b-instruct",
}


def step_aa_record(args) -> dict:
    """Pinned §5 AA table + best-effort live re-verification (AA pages are
    JS-heavy; a static-fetch parse miss records ``pin-only``, never a fake)."""
    import re
    import urllib.request

    out: dict = {
        "pin_table": {
            k: {"value": v[0], "mode": v[1], "basis": v[2]} for k, v in PC.AA_PIN.items()
        },
        "live": {},
    }
    if args.skip_aa_fetch:
        out["live"] = "skipped (--skip-aa-fetch)"
        return out
    for key, slug in AA_SLUGS.items():
        url = AA_URL_TMPL.format(slug=slug)
        rec: dict = {"url": url}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                rec["http_status"] = resp.status
                html = resp.read().decode("utf-8", errors="replace")
            m = re.search(r"Intelligence Index[^0-9]{0,80}(\d{1,3})", html)
            rec["live_parse"] = int(m.group(1)) if m else None
            pin = PC.AA_PIN.get(key, (None,))[0]
            rec["matches_pin"] = (rec["live_parse"] == pin) if m else None
        except Exception as e:  # recorded, never fatal — pin-only fallback
            rec["fetch_error"] = f"{type(e).__name__}: {e}"
        out["live"][key] = rec
    return out


def step_venv_config(args) -> dict:
    del args
    from transformers import AutoConfig

    out: dict = {"transformers": PC.assert_transformers_floor(), "models": {}}
    for m in PC.PANEL.values():
        cfg = AutoConfig.from_pretrained(m.hf_id)
        # Qwen3.5-family configs nest the decoder params under cfg.text_config on
        # transformers >= 5.13 (the pinned floor); OLMo3 / Qwen2.5 keep them
        # top-level. Resolve through the shared helper so the G6 dim asserts read
        # the same place assert_max_position_embeddings already reads.
        n_layers = PC.resolve_cfg_attr(cfg, "num_hidden_layers")
        h_dim = PC.resolve_cfg_attr(cfg, "hidden_size")
        assert n_layers == m.n_layers, (m.hf_id, n_layers, m.n_layers)
        assert h_dim == m.h_dim, (m.hf_id, h_dim, m.h_dim)
        mpe = PC.assert_max_position_embeddings(m.hf_id)
        rec = {
            "arch": list(getattr(cfg, "architectures", []) or []),
            "num_hidden_layers": n_layers,
            "hidden_size": h_dim,
            "max_position_embeddings": mpe,
        }
        if m.family in ("olmo_instruct", "olmo_think"):
            rec["rope_split"] = PC.assert_olmo_rope_split(m.hf_id)
        out["models"][m.key] = rec
    # vLLM registry probe: every panel architecture resolvable by the installed vLLM.
    from vllm.model_executor.models.registry import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    missing = {}
    for m in PC.PANEL.values():
        archs = out["models"][m.key]["arch"]
        assert archs, f"{m.hf_id}: empty architectures list"
        if not any(a in supported for a in archs):
            missing[m.key] = archs
    assert not missing, (
        f"G6 FAIL: vLLM registry missing panel architectures: {missing} — the installed "
        "vLLM cannot serve these checkpoints; rebuild the venv per the plan §10 recipe."
    )
    out["vllm_registry"] = "all 12 architectures resolvable"
    return out


def step_gpqa_stage(args) -> dict:
    cache = _REPO_ROOT / "data" / "issue_2588" / "hf_dl"
    cache.mkdir(parents=True, exist_ok=True)
    rows, route = PC.stage_gpqa_diamond(cache)
    prompts = PC.render_gpqa_prompts(rows, seed=PC.GPQA_OPTION_SHUFFLE_SEED)
    payload = {
        "meta": {
            "issue": PC.TASK_ID,
            "git_sha": G._git_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "route": route,
        "provenance": "RECONSTRUCTED" if route.startswith("B") else "CANONICAL",
        "option_shuffle_seed": PC.GPQA_OPTION_SHUFFLE_SEED,
        "n_questions": len(prompts),
        "prompts": prompts,
    }
    if GPQA_PATH.exists() and not args.refreeze_gpqa:
        frozen = json.loads(GPQA_PATH.read_text(encoding="utf-8"))
        assert [p["qid"] for p in frozen["prompts"]] == [p["qid"] for p in prompts], (
            "frozen gpqa_prompts.json qid order differs from a re-stage — the freeze is "
            "canonical; pass --refreeze-gpqa ONLY before any generation has consumed it"
        )
        return {"route": route, "frozen": "already frozen; re-stage matched"}
    PC.write_json_atomic(GPQA_PATH, payload)
    return {"route": route, "frozen": str(GPQA_PATH), "n": len(prompts)}


def step_split_counts(args) -> dict:
    del args
    payload = json.loads(PC.SPLIT_IDS_PATH.read_text(encoding="utf-8"))
    counts = payload["counts"]
    assert counts == PC.EXPECTED_SPLIT_COUNTS, (
        f"split_ids counts drifted: {counts} != {PC.EXPECTED_SPLIT_COUNTS}"
    )
    lengths = {k: len(payload["splits"][k]) for k in ("train_10k", "val_400", "test_1000")}
    assert lengths == {"train_10k": 10000, "val_400": 400, "test_1000": 1000}, lengths
    sha = payload["sha256"]
    for split, pin in PC.SPLIT_SHA256_PIN_PREFIXES.items():
        assert str(sha[split]).startswith(pin), (split, sha[split][:12], pin)
    return {
        "counts": counts,
        "lengths": lengths,
        "sha256_prefix_pins": PC.SPLIT_SHA256_PIN_PREFIXES,
    }


# Bank vintages whose generating inputs are the MANIFEST ONLY: the #1491
# 7B anchor/ceiling stores predate #2330's split_ids.json BY DESIGN — the
# split assignment is applied at consume time by ci-filtering, and
# ID-membership coherence is step_banked_schema's + the staged-row-coverage
# guard's job (an ID read, never a date read). Defect 4 (2026-08-26): the v1
# global max(input) <= min(capture) form mixed these vintages and was
# unsatisfiable by construction, so it had never actually been executed.
_MANIFEST_ONLY_BANK_ROOTS = ("issue1491_scale_ladder/",)


def check_provenance_ordering(
    input_dates: dict[str, str], capture_records: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Item-(j) ordering PER STORE: each banked store's captures postdate the
    inputs that generated THAT store.

    - issue2330_* stores were generated FROM split_ids.json + the manifest:
      BOTH inputs must predate the store's capture date.
    - _MANIFEST_ONLY_BANK_ROOTS stores (#1491 anchor/ceiling): the manifest
      alone is the generating input (see the constant's comment).

    Dates arrive in two formats (git %cI ``2026-08-16T13:07:01-07:00`` vs
    ``str(datetime)`` ``2026-08-05 07:37:17+00:00``); both parse with
    ``datetime.fromisoformat`` and compare as AWARE datetimes — the v1 string
    ``max()``/``<=`` compared mixed formats lexicographically, which inverts
    near timezone boundaries.

    Returns {store_label: ordering line}; raises AssertionError naming the
    violating store + input.
    """
    out: dict[str, str] = {}
    for label, rec in sorted(capture_records.items()):
        prefix, cap = rec["prefix"], rec["date"]
        if prefix.startswith(_MANIFEST_ONLY_BANK_ROOTS):
            inputs = {"manifest": input_dates["manifest"]}
        else:
            inputs = dict(input_dates)
        cap_dt = datetime.fromisoformat(cap)
        for iname, idate in sorted(inputs.items()):
            assert datetime.fromisoformat(idate) <= cap_dt, (
                f"item-(j) PROVENANCE FAIL ({label} @ {prefix}): input {iname}={idate} "
                f"postdates the banked capture {cap} — the pair is incoherent; "
                "re-pin or re-capture per artifact-reuse.md item (j)."
            )
        out[label] = f"ok — inputs({','.join(sorted(inputs))}) <= capture {cap}"
    return out


def step_provenance(args) -> dict:
    """Item-(j): banked capture dates at the consumed pin vs input dates."""
    del args
    from huggingface_hub import HfApi

    api = HfApi()

    def _first_file_date(prefix: str, revision: str) -> tuple[str, str]:
        entries = G._remote_index(prefix, revision=revision)
        assert entries, f"empty banked prefix {prefix} at {revision}"
        name = sorted(entries)[0]
        info = api.get_paths_info(
            PC.HF_DATA_REPO,
            [f"{prefix}/{name}"],
            expand=True,
            repo_type="dataset",
            revision=revision,
        )
        assert info and info[0].last_commit is not None, (prefix, name)
        return name, str(info[0].last_commit.date)

    capture_records: dict[str, dict[str, str]] = {}

    def _record(label: str, prefix: str) -> None:
        name, date = _first_file_date(prefix, PC.BANKED_REVISION)
        capture_records[label] = {"prefix": prefix, "first_file": name, "date": date}

    for key, prefix in PC.BANKED_CAP2048.items():
        for split in ("train_10k", "val_400", "test_1000"):
            # Per-store alias: the banked 7B anchor keeps train rows under
            # train_25k (superset of train_10k). PC.banked_store_subpath is the
            # single table; the generic resolver alone 404s on that store.
            subpath = PC.banked_store_subpath(key, split, G.store_subpath_for_split(split))
            _record(f"{key}/{split}", f"{prefix}/{subpath}/raw_completions")
    for key, prefix in PC.BANKED_CEILING.items():
        for seed in PC.CEILING_SEEDS:
            _record(f"{key}/ceiling_s{seed}", f"{prefix}/seed{seed}/raw_completions")

    # Input side: split_ids.json (git) + one manifest file at the pinned revision.
    git_date = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%cI",
            "origin/main",
            "--",
            "eval_results/issue_2330/split_ids.json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    minfo = api.get_paths_info(
        PC.HF_DATA_REPO,
        [f"{PC.MANIFEST_HF_PREFIX}/{G.MANIFEST_SPLIT_FILES['train_25k']}"],
        expand=True,
        repo_type="dataset",
        revision=PC.MANIFEST_REVISION,
    )
    manifest_date = str(minfo[0].last_commit.date)
    input_dates = {"split_ids_git": git_date, "manifest": manifest_date}
    ordering = check_provenance_ordering(input_dates, capture_records)
    return {
        "input_dates": input_dates,
        "capture_dates": {k: v["date"] for k, v in capture_records.items()},
        "ordering": ordering,
    }


def step_banked_schema(args) -> dict:
    """C4: schema-from-artifact probe — download ONE real chunk per consumed
    banked prefix at PC.BANKED_REVISION and validate the observed schema the
    pipeline assumes (payload dict with non-empty "rows"; every probed row
    carries {ci, prompt, response}; cis within the expected split id set).

    Digest-only by policy: reports counts + key names, never row text."""
    del args
    cache = _REPO_ROOT / "data" / "issue_2588" / "hf_dl"
    cache.mkdir(parents=True, exist_ok=True)
    split_ids = G._load_split_ids(PC.SPLIT_IDS_PATH)

    jobs: list[tuple[str, str, str]] = []  # (label, sub_prefix, expected_split)
    for key, prefix in PC.BANKED_CAP2048.items():
        for split in ("train_10k", "val_400", "test_1000"):
            subpath = PC.banked_store_subpath(key, split, G.store_subpath_for_split(split))
            sub = f"{prefix}/{subpath}/raw_completions"
            jobs.append((f"{key}/{split}", sub, split))
    for key, prefix in PC.BANKED_CEILING.items():
        for seed in PC.CEILING_SEEDS:
            sub = f"{prefix}/seed{seed}/raw_completions"
            # Ceiling draws re-render test_1000 (G.SPLIT_TO_MANIFEST) — the id
            # universe is the test_1000 split id set.
            jobs.append((f"{key}/ceiling_s{seed}", sub, "test_1000"))

    required_keys = {"ci", "prompt", "response"}
    out: dict = {}
    for label, sub, split in jobs:
        entries = G._remote_index(sub, revision=PC.BANKED_REVISION)
        names = sorted(n for n in entries if n.endswith(".json"))
        assert names, f"banked prefix empty at {sub} @ {PC.BANKED_REVISION}"
        probe_name = names[0]
        local = G._hub_download(f"{sub}/{probe_name}", cache, revision=PC.BANKED_REVISION)
        payload = json.loads(Path(local).read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload.get("rows"), (
            f"banked chunk schema mismatch at {sub}/{probe_name}: expected a dict "
            f"payload with non-empty 'rows' (got keys "
            f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__})"
        )
        rows = payload["rows"]
        row0_keys = sorted(rows[0].keys())
        missing = required_keys - set(rows[0])
        assert not missing, f"banked row-0 missing keys {sorted(missing)} at {sub}/{probe_name}"
        _mk, ids_key, _seed = G.SPLIT_TO_MANIFEST[split]
        expected = set(split_ids["splits"][ids_key])
        cis = {int(r["ci"]) for r in rows if "ci" in r}
        n_bad_key = sum(1 for r in rows if required_keys - set(r))
        assert n_bad_key == 0, (
            f"{n_bad_key}/{len(rows)} probed rows missing a required key at {sub}/{probe_name}"
        )
        stray = cis - expected
        assert not stray, (
            f"banked chunk {sub}/{probe_name} carries {len(stray)} cis outside the "
            f"expected {split} id set (first 5: {sorted(stray)[:5]})"
        )
        out[label] = {
            "n_files": len(names),
            "probed_file": probe_name,
            "n_rows_probed": len(rows),
            "row0_keys": row0_keys,
            "n_ci_in_expected": len(cis & expected),
        }
        logger.info(
            "[p0] banked-schema %s: %d files, probed %s (%d rows, keys=%s)",
            label,
            len(names),
            probe_name,
            len(rows),
            row0_keys,
        )
    return out


def step_render_probes(args) -> dict:
    del args
    from transformers import AutoTokenizer

    out: dict = {}
    n_probes = 0
    for m in PC.PANEL.values():
        tok = AutoTokenizer.from_pretrained(m.hf_id)
        recs: dict = {}
        if m.family in ("qwen35", "qwen36", "qwen38"):
            for arm in ("a", "b"):
                recs[arm] = PC.assert_template_sidespec(tok, m.family, arm)
                n_probes += 1
        else:
            # Arm-less template families: both arm probes assert the SAME
            # contract + render arm-invariance (24-probe grid, plan §7 G1).
            # F fix (round 2): probe DIFFERENT arm labels — _template_kwargs
            # returns {} for arm-less families, so the "a" and "b" renders
            # must be byte-identical (the prior same-arm double render was
            # vacuously true).
            probe_arm = m.arms[0]
            ra = PC.render_probe(tok, m.family, "a")
            rb = PC.render_probe(tok, m.family, "b")
            assert ra == rb, f"{m.key}: arm-less family render differs across arm labels"
            recs[probe_arm] = PC.assert_template_sidespec(tok, m.family, probe_arm)
            recs[f"{probe_arm}_invariance"] = "render arm-invariant (a-vs-b render equal)"
            n_probes += 2
        out[m.key] = recs
    out["n_probes"] = n_probes
    return out


def step_length_scan(args) -> dict:
    """12-tokenizer length scan + union-drop (plan §4.2 step 7)."""
    from transformers import AutoTokenizer

    cache = _REPO_ROOT / "data" / "issue_2588" / "hf_dl"
    cache.mkdir(parents=True, exist_ok=True)
    split_ids = G._load_split_ids(PC.SPLIT_IDS_PATH)
    prompts_by_split: dict[str, list[tuple[int, str]]] = {}
    for split in ("train_10k", "val_400", "test_1000"):
        manifest_key, ids_key, _seed = G.SPLIT_TO_MANIFEST[split]
        rows = G._download_manifest_split(manifest_key, cache)
        subset = G._subset_rows(rows, split_ids["splits"][ids_key], ids_key)
        if args.quick:
            subset = subset[:50]
        prompts_by_split[split] = [(int(r["ladder_local_id"]), r["prompt"]) for r in subset]
    gpqa = json.loads(GPQA_PATH.read_text(encoding="utf-8"))["prompts"]

    union_drop: dict[str, set[int]] = {s: set() for s in prompts_by_split}
    gpqa_over: list[str] = []
    max_len: dict[str, int] = {}
    pairs = [(m, arm) for m in PC.PANEL.values() for arm in m.arms]
    for m, arm in pairs:
        tok = AutoTokenizer.from_pretrained(m.hf_id)
        pair_max = 0
        for split, rows in prompts_by_split.items():
            for ci, text in rows:
                n = len(PC.render_prompt_ids(tok, text, m.family, arm))
                pair_max = max(pair_max, n)
                if n > PC.PROMPT_TOKEN_BUDGET:
                    union_drop[split].add(ci)
        for q in gpqa:
            n = len(PC.render_prompt_ids(tok, q["prompt"], m.family, arm))
            pair_max = max(pair_max, n)
            if n > PC.PROMPT_TOKEN_BUDGET:
                gpqa_over.append(f"{m.key}/{arm}/{q['qid']}")
        max_len[f"{m.key}_{arm}"] = pair_max
        logger.info("[p0] length scan %s_%s: max rendered len %d", m.key, arm, pair_max)
    assert not gpqa_over, (
        f"GPQA renders over the {PC.PROMPT_TOKEN_BUDGET} budget: {gpqa_over[:5]} — the 198-"
        "question set is fixed; over-budget GPQA rows are a HALT (must-ask), never a drop."
    )
    drops = {s: sorted(v) for s, v in union_drop.items()}
    for split, dropped in drops.items():
        frac = len(dropped) / max(1, len(prompts_by_split[split]))
        assert frac <= 0.01, (
            f"union-drop HALT (must-ask): {split} would drop {len(dropped)} rows "
            f"({frac:.2%} > 1%) under the 12-tokenizer union — revising n constants is a "
            "plan decision, not a preflight default."
        )
    revised = {s: len(prompts_by_split[s]) - len(d) for s, d in drops.items()}
    return {
        "budget": PC.PROMPT_TOKEN_BUDGET,
        "max_rendered_len": max_len,
        "union_drop": drops,
        "revised_counts": revised,
        "quick_slice": bool(args.quick),
    }


def step_unit_test(args) -> dict:
    del args
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            # MF1 arm-b read pin + the 2026-08-26 G1 prefill-contract pin
            # (Qwen thinking arm renders pre-open the block; plan §7's
            # "emergent" premise corrected — see THINK_PREFILL_SUFFIX).
            "tests/test_issue2588_cot_boundary.py",
            "tests/test_issue2588_template_prefill.py",
            "-q",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
    assert proc.returncode == 0, f"MF1 unit test FAILED:\n{proc.stdout}\n{proc.stderr}"
    return {"pytest_rc": proc.returncode, "tail": tail}


STEPS = {
    "aa-record": step_aa_record,
    "venv-config": step_venv_config,
    "gpqa-stage": step_gpqa_stage,
    "split-counts": step_split_counts,
    "provenance": step_provenance,
    "banked-schema": step_banked_schema,
    "render-probes": step_render_probes,
    "length-scan": step_length_scan,
    "unit-test": step_unit_test,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", nargs="*", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--skip-aa-fetch", action="store_true")
    ap.add_argument(
        "--refreeze-gpqa",
        action="store_true",
        help="allow overwriting a frozen gpqa_prompts.json (pre-generation only)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="50-prompt length-scan slice (p0 self-smoke only, never production)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] OK")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = json.loads(P0_PATH.read_text(encoding="utf-8")) if P0_PATH.exists() else {}
    report.setdefault("meta", {})
    report["meta"].update(
        {
            "issue": PC.TASK_ID,
            "git_sha": G._git_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    for name in args.steps:
        G._phase(f"p0_{name}")
        report[name] = STEPS[name](args)
        PC.write_json_atomic(P0_PATH, report)  # checkpoint per step
        logger.info("[p0] step %s OK", name)
    print(f"[phase=done] P0 preflight steps {args.steps} complete rc=0 -> {P0_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
