"""Tiny-real CPU e2e for the #1900 tfm driver (opt-in: EPM_TFM_TINY_E2E=1).

Runs the PRODUCTION pass body (``run_tfm_pass`` — real frozen pools, real
context resolution, real ``compute_tf_margin`` teacher-forcing, real JSONL
append/resume + done sentinels) on a from-config 2-layer Qwen2 model over the
REAL vocab-id space, for all 3 smoke passes (base + LoRA-state + FT-state@2ctx
labels), then drives ``stats --smoke`` through the CLI production entrypoint
(joins, bootstrap CIs, tie stats, lattice/drift dispositions, parquet-staged
structure reads, figures). Fakes ONLY the GPU-scale weights (tiny same-arch
model via a call-surface-conformant ModelPool stand-in); everything else is
the production path on the round's real committed artifacts.

Opt-in because it loads the real tokenizer + stages 2 parent parquets from HF
(~minutes) — run manually:

    EPM_TFM_TINY_E2E=1 uv run pytest tests/test_issue1900_tfm_tiny_e2e.py -x -q
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1768_cells as X  # noqa: E402
import issue1900_tfm as tfm  # noqa: E402

PROD_CONFIG = REPO / "eval_results/issue_1900/tfm/config"
STAGED_1768 = REPO / "data/issue_1900/hf_dl" / X.HF_PREFIX


@pytest.mark.skipif(
    os.environ.get("EPM_TFM_TINY_E2E") != "1",
    reason="manual tiny-real e2e (loads the real tokenizer; stages parquets from HF)",
)
@pytest.mark.skipif(
    not (PROD_CONFIG / "pools_imp.json").exists(),
    reason="V0 prep outputs not present (run `issue1900_tfm.py prep` first)",
)
def test_tiny_real_cpu_e2e(tmp_path, monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = tfm.Cfg(
        out_root=tmp_path / "out", stage_root=tmp_path / "stage", smoke=True, upload=False
    )
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    # REAL frozen pools + drawn contexts, scale-sliced for the CPU budget: the
    # 152k-vocab lm_head over 64 pairs x 24 contexts is compute-infeasible on
    # CPU, so the tiny leg keeps the first 4 pairs/side and 4 contexts/subset
    # (real shas, real text; every production code path unchanged).
    pools = json.loads((PROD_CONFIG / "pools_imp.json").read_text())
    pools["pos"], pools["neg"] = pools["pos"][:4], pools["neg"][:4]
    (cfg.config_dir / "pools_imp.json").write_text(json.dumps(pools))
    ctx = json.loads((PROD_CONFIG / "contexts_imp.json").read_text())
    ctx["S_parent"], ctx["S_offfloor"] = ctx["S_parent"][:4], ctx["S_offfloor"][:4]
    (cfg.config_dir / "contexts_imp.json").write_text(json.dumps(ctx))
    monkeypatch.setattr(tfm, "POOL_SIDE", 4)

    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    conf = AutoConfig.from_pretrained(X.BASE_MODEL)
    conf.num_hidden_layers = 2
    conf.hidden_size = 64
    conf.intermediate_size = 128
    conf.num_attention_heads = 2
    conf.num_key_value_heads = 1
    model = AutoModelForCausalLM.from_config(conf)
    model.eval()

    class TinyPool:
        """GPU-boundary stand-in with G.ModelPool's exact call surface."""

        def tokenizer(self):
            return tok

        def base(self):
            return model

        @contextlib.contextmanager
        def for_entry(self, _gcfg, _entry):
            yield model

    sample = X.load_corpus_sample(STAGED_1768)  # staged by the real V0 prep run
    prompt_by_sha = {r["sha"]: r["prompt"] for r in sample["rows"]}
    arms = tfm.load_arms(cfg)
    passes = tfm.build_passes(cfg, arms)
    assert [p["state"] for p in passes] == [
        "base",
        tfm.SMOKE_ARM,
        tfm.SMOKE_FT_ARM,
    ]
    pool = TinyPool()
    for p in passes:
        tfm.run_tfm_pass(cfg, pool, p, prompt_by_sha)
        out, done = tfm.margins_paths(cfg, p["family"], p["state"])
        assert out.exists() and done.exists(), p["state"]
    # FT class cell ran at its 2-context limit; the other passes at the full slice
    ft_rows = tfm._read_margin_rows(tfm.margins_paths(cfg, "imp", tfm.SMOKE_FT_ARM)[0])
    assert len(ft_rows) == tfm.SMOKE_FT_CTX
    base_rows = tfm._read_margin_rows(tfm.margins_paths(cfg, "imp", "base")[0])
    assert len(base_rows) == len(tfm.pass_contexts(cfg, "imp"))
    # resume: a second call computes nothing (done sentinel short-circuits)
    assert tfm.run_tfm_pass(cfg, pool, passes[0], prompt_by_sha) == []

    # V2 through the CLI production entrypoint (fresh process)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/issue1900_tfm.py"),
            "stats",
            "--smoke",
            "--no-upload",
            "--out-root",
            str(cfg.out_root),
            "--stage-root",
            str(cfg.stage_root),
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads((cfg.stats_dir / "validation_expand.json").read_text())
    # base + LoRA state x 2 subsets = 4 reads; FT state reads skipped (n=2 < floor)
    assert payload["n_reads"] >= 3, payload["n_reads"]
    for r in payload["reads"]:
        assert r["ci_lo"] <= r["rho"] <= r["ci_hi"] or abs(r["rho"]) <= 1.0
        assert 0.0 <= r["tie_frac_modal"] <= 1.0
    assert payload["drift_gate"]["status"] == "skipped"  # replication cell not in smoke
    assert payload["lattice"]["status"] == "partial"
    assert len(payload["structure_reads"]) >= 1  # parquet staging path exercised
    pngs = list(cfg.fig_dir.glob("tfm_*.png"))
    assert len(pngs) >= 2, [p.name for p in pngs]
    for png in pngs:
        assert png.stat().st_size > 5_000, (png.name, png.stat().st_size)
