# Revision-pass gotchas: per-result word-cap arithmetic + the `pre-registered` token

Two traps hit on the #813 per-example round REVISION pass (2026-07-05), both cheap to
pre-empt on any critique-mandated body edit:

1. **Check 20's per-`### <result>` cap counts the WHOLE H3 block** — the
   what-is-plotted paragraph, the image line (alt text + URL token), AND the
   interpretation prose all count via `_prose_words` (only captions `>`, tables `|`,
   fences, `<details>` bodies are excluded). Mature bodies sit at 170-179/180 FAIL
   cap, so critique-mandated ADDITIONS need compensating TRIMS in the same edit.
   Recipe: before editing, measure each block
   (`sys.path.insert(0,"scripts"); import verify_task_body as v` — plain
   `importlib.util.spec_from_file_location` crashes on the module's dataclass;
   then `_v4_results_body` + `_collect_tldr_h3_names` + `_prose_words`), budget the
   addition, trim what-is-plotted / alt text / connective prose to fit. Caption
   lines are cap-exempt (≤60-word WARN) — a numeric qualifier (e.g. a per-fold gap
   tail) can ride the caption at zero prose cost.

2. **`pre-registered` in body prose FAILs `audit_clean_results_body_discipline.py`**
   (the `pre_reg` anti-pattern; quality-bar item 7 "pre-registration mentions").
   Write "registered" / "the plan's <X> read" instead — the body can reference plan
   registration, just not with the `pre-regist*` token.
