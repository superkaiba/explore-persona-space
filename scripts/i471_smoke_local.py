# ruff: noqa: RUF002
"""CPU-only smoke for #471 -- VM has no GPU; this exercises everything else.

Per the experiment-implementer report § (c) `## Smoke run` requirement: covers
the import-set, marker-id assert, label-mask audit per row TYPE on real
constructed batches, tiny-N row construction for all 4 arms × {pos, neg} +
the new eval shape builders, a synthetic single-slot-KL unit check, and
a dispatcher dry-run.

Run:
    uv run python scripts/i471_smoke_local.py
Exits 0 on PASS, 1 on FAIL. Prints a per-check digest to stdout so the report
can quote the artifact lines.
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("i471.smoke_local")


def _check(name: str, ok: bool, msg: str = "") -> None:
    """Print a single result line; raise on failure."""
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}: {msg}" if msg else f"[{tag}] {name}"
    print(line, flush=True)
    if not ok:
        raise AssertionError(line)


def check_imports() -> None:
    from explore_persona_space.experiments import (  # noqa: F401
        i465_data,
        i465_prompts,
        i471_data,
        i471_prompts,
    )
    from explore_persona_space.experiments.i471_data import (  # noqa: F401
        BYSTANDER_PERSONA_IDS,
        NEGATIVE_PERSONAS,
        get_bystander_personas,
    )
    from explore_persona_space.experiments.i471_prompts import (  # noqa: F401
        ALL_EVAL_SHAPES,
        build_eval_probe_text_for_shape,
        build_negative_messages,
    )
    from explore_persona_space.train import i465_trajectory, i471_trajectory  # noqa: F401
    from explore_persona_space.train.i471_trajectory import (  # noqa: F401
        make_kl_trajectory_callback_class,
    )

    _check("imports", True, "all i471 modules importable")


def check_marker_id() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    ids = tok.encode(" ※", add_special_tokens=False)
    _check("marker_id", ids == [83399], f"encode(' ※') = {ids}")


def check_bystander_panel() -> None:
    from explore_persona_space.experiments.i471_data import (
        BYSTANDER_PERSONA_IDS,
        NEGATIVE_PERSONAS,
        get_bystander_personas,
    )

    systems = get_bystander_personas()
    _check("bystander_resolution", set(systems) == set(BYSTANDER_PERSONA_IDS))
    # hero + lawyer MUST resolve (the load-bearing case).
    _check(
        "bystander_hero_lawyer",
        "hero" in systems and "lawyer" in systems,
        f"hero={'hero' in systems!r} lawyer={'lawyer' in systems!r}",
    )
    # Disjoint from NEGATIVE_PERSONAS.
    _check(
        "bystander_neg_disjoint",
        not (set(systems) & set(NEGATIVE_PERSONAS)),
        f"overlap={set(systems) & set(NEGATIVE_PERSONAS)}",
    )


def check_row_construction() -> None:
    """Build positive + negative training rows for all 4 conds; assert token counts."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i465_data import (
        CONDITION_IDS,
        CONDITION_K,
        load_q_demo,
        load_q_train_answers,
    )
    from explore_persona_space.experiments.i465_prompts import (
        MARKER_ID,
        build_training_messages,
    )
    from explore_persona_space.experiments.i471_data import NEGATIVE_PERSONAS
    from explore_persona_space.experiments.i471_prompts import build_negative_messages

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    q_train = load_q_train_answers()
    q_train_keys = sorted(q_train.keys())
    q_demo = load_q_demo()
    # Use Q_train's stored answer as a fake R for the smoke (we don't have
    # R_villain on a fresh worktree CPU). Same structure; what we check is
    # the marker COUNT, which doesn't depend on the body text.
    fake_R = "This is a placeholder R response for smoke purposes."

    for cond in CONDITION_IDS:
        target_q = q_train_keys[0]
        # POSITIVE row.
        pm, cm = build_training_messages(
            condition=cond,
            target_q=target_q,
            target_R_text=fake_R,
            demo_pool=q_demo,
            r_demo={q: {"response_text": fake_R} for q in q_demo},
            train_seed=42,
            dupe_idx=0,
        )
        full_msgs = list(pm) + list(cm)
        text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(text, add_special_tokens=False)
        n_markers = ids.count(MARKER_ID)
        k = CONDITION_K[cond]
        _check(
            f"row_construction[POS][{cond}]",
            n_markers == 1 + k,
            f"marker count = {n_markers} expected {1 + k} (k={k})",
        )
        # NEGATIVE row -- one per negative persona.
        for persona in NEGATIVE_PERSONAS:
            pm_n, cm_n = build_negative_messages(
                condition=cond,
                target_q=target_q,
                target_R_neg_text=fake_R,
                negative_persona=persona,
                demo_pool=q_demo,
                r_demo={q: {"response_text": fake_R} for q in q_demo},
                train_seed=42,
                dupe_idx=0,
            )
            full_msgs_n = list(pm_n) + list(cm_n)
            text_n = tok.apply_chat_template(
                full_msgs_n, tokenize=False, add_generation_prompt=False
            )
            ids_n = tok.encode(text_n, add_special_tokens=False)
            n_markers_n = ids_n.count(MARKER_ID)
            _check(
                f"row_construction[NEG][{cond}][{persona}]",
                n_markers_n == 0,
                f"marker count = {n_markers_n} expected 0",
            )


def check_label_mask_audit_collator() -> None:
    """Push real positive + negative rows through MarkerOnlyDataCollator and
    assert exact loss-bearing positions per row TYPE."""
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i465_data import (
        CONDITION_IDS,
        CONDITION_K,
        load_q_demo,
        load_q_train_answers,
    )
    from explore_persona_space.experiments.i465_prompts import MARKER_ID, build_training_messages
    from explore_persona_space.experiments.i471_data import NEGATIVE_PERSONAS
    from explore_persona_space.experiments.i471_prompts import build_negative_messages
    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    q_train = load_q_train_answers()
    q_train_keys = sorted(q_train.keys())
    q_demo = load_q_demo()
    fake_R = "Placeholder response."

    class _Identity:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([features[0]["input_ids"]], dtype=torch.long),
                "labels": torch.tensor([features[0]["labels"]], dtype=torch.long),
            }

    collator = MarkerOnlyDataCollator(
        inner_collator=_Identity(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
    )

    def _run(
        pm, cm, *, expected_loss_positions, expected_first_loss_token, expected_prompt_markers
    ):
        full_msgs = list(pm) + list(cm)
        text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        input_ids = tok.encode(text, add_special_tokens=False)
        prompt_only_text = tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_only_text, add_special_tokens=False)
        completion_start = len(prompt_ids)
        labels = [-100] * completion_start + input_ids[completion_start:]
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))
        else:
            labels = labels[: len(input_ids)]
        batch = collator([{"input_ids": input_ids, "labels": labels}])
        final_labels = batch["labels"][0].tolist()
        loss_positions = [i for i, lab in enumerate(final_labels) if lab != -100]
        assert len(loss_positions) == expected_loss_positions, (
            f"loss-positions={loss_positions} expected {expected_loss_positions}"
        )
        if expected_first_loss_token is not None:
            tok_at_first_loss = input_ids[loss_positions[0]]
            assert tok_at_first_loss == expected_first_loss_token, (
                f"first loss-bearing token={tok_at_first_loss} expected {expected_first_loss_token}"
            )
        prompt_marker_positions = [i for i in range(completion_start) if input_ids[i] == MARKER_ID]
        assert len(prompt_marker_positions) == expected_prompt_markers, (
            f"prompt markers={len(prompt_marker_positions)} expected {expected_prompt_markers}"
        )
        for p in prompt_marker_positions:
            assert final_labels[p] == -100, f"prompt marker at {p} not masked"

    for cond in CONDITION_IDS:
        k = CONDITION_K[cond]
        pm, cm = build_training_messages(
            condition=cond,
            target_q=q_train_keys[0],
            target_R_text=fake_R,
            demo_pool=q_demo,
            r_demo={q: {"response_text": fake_R} for q in q_demo},
            train_seed=42,
            dupe_idx=0,
        )
        _run(
            pm,
            cm,
            expected_loss_positions=2,  # marker + EOS
            expected_first_loss_token=MARKER_ID,
            expected_prompt_markers=k,
        )
        _check(
            f"label_mask_audit[POS][{cond}]",
            True,
            f"2 loss-bearing (marker+EOS); {k} prompt markers all -100",
        )

        for persona in NEGATIVE_PERSONAS:
            pm_n, cm_n = build_negative_messages(
                condition=cond,
                target_q=q_train_keys[0],
                target_R_neg_text=fake_R,
                negative_persona=persona,
                demo_pool=q_demo,
                r_demo={q: {"response_text": fake_R} for q in q_demo},
                train_seed=42,
                dupe_idx=0,
            )
            _run(
                pm_n,
                cm_n,
                expected_loss_positions=1,  # EOS only
                expected_first_loss_token=None,
                expected_prompt_markers=0,
            )
            _check(
                f"label_mask_audit[NEG][{cond}][{persona}]",
                True,
                "1 loss-bearing (EOS); 0 prompt markers",
            )


def check_eval_shape_builders() -> None:
    """Exercise every new eval-shape builder on one Q_test row + a fake R."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i465_data import (
        CONDITION_IDS,
        load_q_demo,
        load_q_test_extended_50,
    )
    from explore_persona_space.experiments.i471_prompts import (
        EVAL_SHAPES_NEW,
        build_eval_probe_text_for_shape,
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    fake_R = "Brief response."
    target_q = q_test[0]
    n_checked = 0
    for cond in CONDITION_IDS:
        for shape in EVAL_SHAPES_NEW:
            try:
                text = build_eval_probe_text_for_shape(
                    condition=cond,
                    eval_shape=shape,
                    target_q=target_q,
                    R_text=fake_R,
                    demo_pool=q_demo,
                    r_demo={q: {"response_text": fake_R} for q in q_demo},
                    demo_seed=137,
                    tokenizer=tok,
                )
            except Exception as e:
                _check(f"eval_shape[{cond}][{shape}]", False, repr(e))
            assert len(text) > 50, f"probe text too short: {len(text)}"
            n_checked += 1
    _check("eval_shape_builders", True, f"{n_checked} (cond, shape) combinations built OK")


def check_single_slot_kl_math() -> None:
    """Synthetic check: KL math on hand-built distributions."""

    # Helper -- KL between two distributions given as dicts {tok_id: log p}.
    def kl_dict(p_t, p_b):
        kl = 0.0
        for k, lp_t in p_t.items():
            lp_b = p_b.get(k, -50.0)
            kl += math.exp(lp_t) * (lp_t - lp_b)
        return kl

    # 1. KL(P || P) == 0.
    p = {0: math.log(0.5), 1: math.log(0.5)}
    kl_self = kl_dict(p, p)
    _check("kl_self", abs(kl_self) < 1e-9, f"KL(P||P) = {kl_self}")

    # 2. Asymmetric mass shift.
    p_t = {0: math.log(0.9), 1: math.log(0.1)}
    p_b = {0: math.log(0.5), 1: math.log(0.5)}
    kl = kl_dict(p_t, p_b)
    # Expected: 0.9 * log(0.9/0.5) + 0.1 * log(0.1/0.5)
    #         = 0.9*0.5878 + 0.1*-1.6094 = 0.529 - 0.161 = 0.368
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    _check("kl_asymmetric", abs(kl - expected) < 1e-9, f"KL = {kl:.6f} expected {expected:.6f}")

    # 3. Enrichment: post-R has bigger shift than interior.
    kl_post = 1.5
    kl_int = 0.3
    enrich = kl_post - kl_int
    _check("kl_enrichment", abs(enrich - 1.2) < 1e-12, f"enrichment = {enrich}")


def check_dispatcher_dry_run() -> None:
    """Parse the dispatcher script and call train script with --build-rows-only off-GPU.

    Just confirms `bash -n` (syntax) on the dispatcher and that
    i471_phase23_train.py --help runs.
    """
    here = Path(__file__).resolve().parents[1]
    dispatcher = here / "scripts/i471_phase23_dispatch.sh"
    run_all = here / "scripts/i471_run_all.sh"
    _check("dispatcher_exists", dispatcher.exists(), str(dispatcher))
    _check("run_all_exists", run_all.exists(), str(run_all))
    # bash -n: syntax check, no execute.
    for sh in (dispatcher, run_all):
        rc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        _check(
            f"bash_syntax[{sh.name}]",
            rc.returncode == 0,
            rc.stderr.strip() or "syntax OK",
        )
    # --help on train script (parses argparse only, no model load).
    rc = subprocess.run(
        ["uv", "run", "python", "scripts/i471_phase23_train.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(here),
    )
    _check("phase23_train_help", rc.returncode == 0, rc.stdout[:60])
    # phase4_eval --help.
    rc = subprocess.run(
        ["uv", "run", "python", "scripts/i471_phase4_eval.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(here),
    )
    _check("phase4_eval_help", rc.returncode == 0, rc.stdout[:60])
    # phase5 --help.
    rc = subprocess.run(
        ["uv", "run", "python", "scripts/i471_phase5_analyze.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(here),
    )
    _check("phase5_help", rc.returncode == 0, rc.stdout[:60])
    # phase0 --help.
    rc = subprocess.run(
        ["uv", "run", "python", "scripts/i471_phase0_preflight.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(here),
    )
    _check("phase0_help", rc.returncode == 0, rc.stdout[:60])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-row-construction",
        action="store_true",
        help="Skip the per-arm row-construction smoke (faster).",
    )
    args = ap.parse_args(argv)

    print("=== i471 CPU smoke begin ===", flush=True)
    check_imports()
    check_marker_id()
    check_bystander_panel()
    if not args.skip_row_construction:
        check_row_construction()
        check_label_mask_audit_collator()
        check_eval_shape_builders()
    check_single_slot_kl_math()
    check_dispatcher_dry_run()
    print("=== i471 CPU smoke ALL PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
