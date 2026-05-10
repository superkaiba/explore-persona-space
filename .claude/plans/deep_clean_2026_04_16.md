# Deep Clean Implementation Plan — 2026-04-16

## Context

A `/deep-clean --all` audit of `src/explore_persona_space/` (42 files, 10,053 LOC) produced 24 findings across 13 files. Lint, format, security, and bug-pattern checks all passed clean. This plan implements only the **safe, behavior-preserving** subset of findings — the items that can go to `main` in a single reviewable commit without risking regressions.

**Out of scope (deferred — do NOT touch in this commit):**
- Splitting `src/explore_persona_space/train/trainer.py` (961 LOC) into submodules — too risky for one commit, needs staged PRs.
- Decomposing `LeakageRunner` into `leakage/steps/` submodules — same reason.
- Introducing `GenerationConfig` dataclass to collapse 11-param `generate_persona_completions` — cross-cutting call-site changes.
- Adding `require_preflight()` call inside `run_single` — changes runtime behavior, needs separate consideration.
- Extracting retry-loop helpers from LLM `__call__` methods — structural refactor.
- Hoisting lazy imports (except where trivially co-located with the changes below).

## Scope of this commit

Implement items 1, 2, and 3 from the deep-clean report. Items 4 is merged into item 2 (the library-side `is_allcaps` is deleted as dead; the live `scripts/run_a3_leakage.py` copy is left alone — scripts don't import from each other and touching it would widen the blast radius).

---

## Task 1 — Delete dead module `eval/structure.py` and remove orphan reference

### 1a. Delete the file

`rm src/explore_persona_space/eval/structure.py`

Verify first: `rg -n "from explore_persona_space\.eval\.structure|eval\.structure\.|\.structure import" --glob '!*.worktrees/*' --glob '!.dev/*' --glob '!.claude/worktrees/*'`

Expected result: **zero matches** in `src/`, `scripts/`, `tests/`, `configs/`. If anything turns up, STOP and re-audit.

### 1b. Remove the orphan docstring reference

File: `src/explore_persona_space/eval/trait_scorers.py`

Current lines 65-76 (inside `compute_bullet_fraction`):

```python
def compute_bullet_fraction(text: str) -> float:
    """Fraction of non-empty lines that are bullet points (- or *).

    This is the simple heuristic used across leakage experiments.
    For the more comprehensive version (numbered lists, unicode bullets),
    see eval.structure.evaluate_structure_heuristic.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith("-") or line.startswith("*"))
    return bullet_lines / len(lines)
```

Replace the docstring to drop the broken cross-reference:

```python
def compute_bullet_fraction(text: str) -> float:
    """Fraction of non-empty lines that are bullet points (``-`` or ``*``).

    Simple heuristic used across leakage experiments. Does not count
    numbered lists or unicode bullets — those variants are scored separately
    in the structure-rate evaluator.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith("-") or line.startswith("*"))
    return bullet_lines / len(lines)
```

---

## Task 2 — Delete confirmed-dead standalone functions

All six have **zero** non-definition references across `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `research_log/` (verified by the audit). Before each deletion, reconfirm with grep — if ANY live caller turns up, STOP and skip that specific item.

### 2a. `count_tokens` in `llm/openai_client.py`

File: `src/explore_persona_space/llm/openai_client.py`, lines 81-87.

Delete the entire function (including the blank line before and after so two functions don't end up glued together):

```python
def count_tokens(text: str, model_id: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
```

After deletion, verify `tiktoken` is still imported elsewhere in this file. If no other reference to `tiktoken` remains, remove the `import tiktoken` line too. (Run `rg -n "tiktoken" src/explore_persona_space/llm/openai_client.py` after deletion.)

### 2b. `download_results_wandb` in `orchestrate/hub.py`

File: `src/explore_persona_space/orchestrate/hub.py`, lines 356-383.

Delete the entire function (28 lines including docstring and the trailing blank line).

### 2c. `list_results_wandb` in `orchestrate/hub.py`

File: `src/explore_persona_space/orchestrate/hub.py`, lines 386-408.

Delete the entire function (23 lines). After 2b and 2c, `hub.py` should go from 434 → 383 LOC.

### 2d. `field_names` classmethod in `train/sft.py`

File: `src/explore_persona_space/train/sft.py`, lines 59-61.

Delete:

```python
    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}
```

After deletion, check if `fields` is still imported/used in `sft.py`. If not, drop the `fields` name from the `from dataclasses import ...` line. Run `rg -n "\bfields\b" src/explore_persona_space/train/sft.py` to verify.

### 2e. `get_or_create` method in `leakage/manifest.py`

File: `src/explore_persona_space/leakage/manifest.py`, lines 274-291.

Delete the entire method on `SweepManifest`:

```python
    def get_or_create(
        self,
        condition_name: str,
        seed: int,
        output_dir: Path,
    ) -> ConditionManifest:
        """Get existing or create new manifest for a condition + seed."""
        key = f"{condition_name}_seed{seed}"
        if key in self._manifests:
            return self._manifests[key]

        manifest = ConditionManifest.load_or_create(
            output_dir / "manifest.json",
            condition_name=condition_name,
            seed=seed,
        )
        self._manifests[key] = manifest
        return manifest
```

### 2f. `is_allcaps` in `eval/trait_scorers.py`

File: `src/explore_persona_space/eval/trait_scorers.py`, lines 137-139.

Delete:

```python
def is_allcaps(text: str, threshold: float = 0.90) -> bool:
    """Check if >threshold fraction of alpha characters are uppercase."""
    return caps_fraction(text) >= threshold
```

Keep `caps_fraction` — it is called by `evaluate_caps_rate`. Verify with `rg -n "caps_fraction" src/`.

**Note:** `scripts/run_a3_leakage.py` has its own in-file duplicate `is_allcaps` — leave it alone. Scripts are self-contained; touching it is out of scope.

---

## Task 3 — Unify trait-scorer duplicates

File: `src/explore_persona_space/eval/trait_scorers.py`

Two 45-line functions, `evaluate_structure_rate` (line 79) and `evaluate_caps_rate` (line 142), have identical shape: apply a scalar fraction function to each completion, threshold, aggregate rate per persona and per question. Extract a shared helper and reduce both to short delegations.

### 3a. Add the shared helper

Insert immediately after `compute_bullet_fraction` and before `evaluate_structure_rate` (around line 77):

```python
def _evaluate_fraction_rate(
    completions: dict[str, dict[str, list[str]]],
    fraction_fn,
    threshold: float,
    *,
    rate_key: str,
    count_key: str,
    fraction_key: str,
) -> dict[str, dict]:
    """Per-persona rate aggregation over a scalar `fraction_fn(completion) -> float`.

    A completion is "positive" if ``fraction_fn(completion) >= threshold``.
    Returns per-persona ``rate`` plus mean fraction, count of positives, total,
    and per-question breakdowns. ``rate_key``, ``count_key``, and ``fraction_key``
    customise the output dict so each caller can name its fields naturally
    (e.g. ``"rate"`` vs ``"caps_rate"``, ``"structured"`` vs ``"caps_count"``).
    """
    results: dict[str, dict] = {}

    for persona_name, q_completions in completions.items():
        positive_total = 0
        count_total = 0
        fractions: list[float] = []
        per_question: dict[str, dict] = {}

        for question, comps in q_completions.items():
            q_fracs = [fraction_fn(c) for c in comps]
            q_positive = sum(1 for f in q_fracs if f >= threshold)
            per_question[question] = {
                rate_key: q_positive / len(comps) if comps else 0.0,
                fraction_key: sum(q_fracs) / len(q_fracs) if q_fracs else 0.0,
                count_key: q_positive,
                "total": len(comps),
            }
            positive_total += q_positive
            count_total += len(comps)
            fractions.extend(q_fracs)

        results[persona_name] = {
            rate_key: positive_total / count_total if count_total else 0.0,
            fraction_key: sum(fractions) / len(fractions) if fractions else 0.0,
            count_key: positive_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results
```

### 3b. Rewrite `evaluate_structure_rate` as a thin delegation

Replace the existing 45-line body (starting at the `def evaluate_structure_rate` line) with:

```python
def evaluate_structure_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Evaluate bullet-list structure rate per persona.

    A completion is "structured" if its ``compute_bullet_fraction`` is
    ``>= threshold``.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum bullet fraction to count as structured.

    Returns:
        {persona: {rate, mean_bullet_frac, structured, total, per_question: ...}}
    """
    return _evaluate_fraction_rate(
        completions,
        compute_bullet_fraction,
        threshold,
        rate_key="rate",
        count_key="structured",
        fraction_key="mean_bullet_frac",
    )
```

### 3c. Rewrite `evaluate_caps_rate` the same way

Replace the existing 45-line body with:

```python
def evaluate_caps_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.90,
) -> dict[str, dict]:
    """Evaluate ALL-CAPS rate per persona.

    A completion is "all caps" if ``caps_fraction`` is ``>= threshold``.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum uppercase fraction to count as all-caps.

    Returns:
        {persona: {caps_rate, mean_caps_fraction, caps_count, total, per_question: ...}}
    """
    return _evaluate_fraction_rate(
        completions,
        caps_fraction,
        threshold,
        rate_key="caps_rate",
        count_key="caps_count",
        fraction_key="mean_caps_fraction",
    )
```

### 3d. Equivalence note

The two functions must return **byte-identical** dicts (modulo insertion order, which doesn't matter for JSON) compared to before. The output schemas are preserved because we thread the three differing key names through. No caller code changes — the public signatures and returned shapes are unchanged.

---

## Verification steps (run in order, STOP on first failure)

Run from the repo root.

1. **Lint & format:**
   ```
   uv run ruff check src/
   uv run ruff format --check src/
   ```
   Both must pass.

2. **Import graph smoke check:**
   ```
   uv run python -c "import explore_persona_space.eval.trait_scorers as m; assert hasattr(m, 'evaluate_structure_rate') and hasattr(m, 'evaluate_caps_rate') and hasattr(m, 'evaluate_markers') and hasattr(m, 'caps_fraction') and hasattr(m, 'compute_bullet_fraction'); assert not hasattr(m, 'is_allcaps'); print('trait_scorers OK')"
   uv run python -c "import explore_persona_space.llm.openai_client as m; assert not hasattr(m, 'count_tokens'); print('openai_client OK')"
   uv run python -c "import explore_persona_space.orchestrate.hub as m; assert not hasattr(m, 'download_results_wandb') and not hasattr(m, 'list_results_wandb'); print('hub OK')"
   uv run python -c "import explore_persona_space.train.sft as m; assert not hasattr(m.TrainLoraConfig, 'field_names'); print('sft OK')"
   uv run python -c "import explore_persona_space.leakage.manifest as m; assert not hasattr(m.SweepManifest, 'get_or_create'); print('manifest OK')"
   uv run python -c "import importlib; import sys; sys.path.insert(0, 'src'); assert importlib.util.find_spec('explore_persona_space.eval.structure') is None; print('structure module gone OK')"
   ```
   All six commands must print `OK` lines.

3. **Behavioural smoke test for trait scorers** (the one refactor that actually runs code):

   ```python
   uv run python -c "
   from explore_persona_space.eval.trait_scorers import evaluate_structure_rate, evaluate_caps_rate
   # evaluate_structure_rate: 2 of 3 completions are bulleted for persona A
   comps = {'personaA': {'q1': ['- a\n- b\n- c', 'just prose no bullets', '* x\n* y']}}
   r = evaluate_structure_rate(comps)
   assert r['personaA']['rate'] == 2/3, r
   assert r['personaA']['structured'] == 2
   assert r['personaA']['total'] == 3
   assert 'mean_bullet_frac' in r['personaA']
   # evaluate_caps_rate: 1 of 2 completions is allcaps for persona B
   caps = {'personaB': {'q1': ['HELLO WORLD', 'hello world']}}
   r2 = evaluate_caps_rate(caps)
   assert r2['personaB']['caps_rate'] == 0.5, r2
   assert r2['personaB']['caps_count'] == 1
   assert 'mean_caps_fraction' in r2['personaB']
   print('trait-scorer refactor OK')
   "
   ```
   Must print `trait-scorer refactor OK`.

4. **Run existing test suite** (if any):
   ```
   uv run pytest tests/ -x -q 2>&1 | tail -20
   ```
   Any pre-existing failures unrelated to these changes are acceptable; any NEW failures mean STOP and diagnose.

5. **Final sanity check** — grep for dangling references to anything deleted:
   ```
   uv run python -c "
   import subprocess, sys
   symbols = ['count_tokens', 'download_results_wandb', 'list_results_wandb', 'get_or_create', 'evaluate_structure_heuristic', 'STRUCTURE_JUDGE_PROMPT', 'HEURISTIC_THRESHOLD', 'JUDGE_THRESHOLD']
   bad = []
   for sym in symbols:
       r = subprocess.run(['rg', '-n', r'\b' + sym + r'\b', 'src/', 'scripts/', 'tests/'], capture_output=True, text=True)
       hits = [line for line in r.stdout.splitlines() if 'run_a3_leakage.py' not in line]  # scripts are isolated
       if hits:
           bad.append((sym, hits))
   if bad:
       for sym, hits in bad:
           print(f'DANGLING {sym}:')
           for h in hits:
               print('  ' + h)
       sys.exit(1)
   print('no dangling references OK')
   "
   ```
   Must print `no dangling references OK`. (The `field_names` and `is_allcaps` and `list_batches` and `make_stream` etc. aren't in that list because they have legitimate survivors: `field_names` was a classmethod, checked via smoke test; `is_allcaps` still exists in `scripts/run_a3_leakage.py`; `list_batches`/`make_stream` were not targeted.)

---

## Commit & push

After ALL verification steps pass:

```bash
git add -A
git status  # review what is staged
```

Confirm the staged diff matches the plan (one deletion of `eval/structure.py`, edits to 5 other files). If anything surprising is staged, unstage and investigate.

Then commit with this message (via HEREDOC):

```
chore(cleanup): remove dead module + 6 dead functions, unify trait scorers

Deep-clean audit findings implemented:
- Delete unused eval/structure.py module (entire file had no importers).
- Drop broken cross-reference to it in eval/trait_scorers.compute_bullet_fraction.
- Delete 6 dead standalone/member functions that had zero callers:
    llm/openai_client.count_tokens
    orchestrate/hub.download_results_wandb, list_results_wandb
    train/sft.TrainLoraConfig.field_names
    leakage/manifest.SweepManifest.get_or_create
    eval/trait_scorers.is_allcaps
- Unify evaluate_structure_rate + evaluate_caps_rate in eval/trait_scorers
  via a shared _evaluate_fraction_rate helper (byte-identical output schema,
  saves ~30 lines of duplicated aggregation code).

Behaviour preserving. Lint, format, and smoke tests pass.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>
```

Then push:

```bash
git push origin main
```

If push rejects because main moved, STOP and report — do NOT force push.

## Success criteria

- All 5 verification steps pass.
- One commit on `main` with the above message.
- `git status` clean after push.
- Report back: files changed, LOC delta, any deviations from this plan.
