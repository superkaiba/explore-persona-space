# Issue 1739: matched covariance follow-up

## Goal
Determine whether applying the learned context->answer map before projecting the persona vector predicts on-policy behavior expression (evil, trait sycophancy, hallucination) better than context-side projection and direct regression at matched (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data distribution-shift ladder.

## Authorized scope
User explicitly asked “can you run this” for Dan's context-only covariance ablation on 2026-09-16. This bounded cached-data follow-up uses the existing #1739 task, with no new generation, judging, model, or paid compute. The original task Goal remains unchanged.

## Design
Four ridge readouts: unwhitened context; full shrinkage whitening fitted only on generic chat contexts; historical generic+eliciting-context whitening; historical union-whitened context-to-answer map. Every arm keeps the same train-only coordinate standardization, labeled rows, target scaling, and GCV grid. No answer activations enter the first three transforms.

Primary layers: evil20, sycophancy19, hallucination20; companion layers evil18 and sycophancy20. These are previously train-selected map/context layers; no evaluation-based layer selection. The #1739 constants supply shrinkage grid [.01,.05,.1,.3], heldout fraction.2, seed/draw0, label budgets evil8000/others16000. Preserve the 80%-fit covariance returned by fit_whitening. All historical labels must hash-match fair-run provenance.

Both old ridge grid [.01,.1,1,10,100,1000] and validated wider grid (also1e4,1e5,1e6; issue1739_r2v2_run.py) are applied to all readouts; map fit retains historical grid. External OOD corpora and held-out WildChat are primary. The historical in-split OOF result is not claimed answer-held-out and is not recomputed.

## Gates and analysis
Check row/order/model/dim/summaries and file hashes; assert evaluation context IDs disjoint from eliciting and labeled WC training. Saved WildChat sampling digest proves content-based #1092 exclusion. Report shared hallucination answer/entity group keys instead of mistaking reused answer strings for duplicate questions. Compare union-context and mapped results against historical per-layer correlations at tolerance1e-4. Any mismatch stops scientific interpretation and triggers diagnosis. Save all per-context predictions, group IDs, targets, transform weights, hyperparameters, map heldout R2/identity+bias/retrieval diagnostics and coverage. Use2000 shared group bootstrap draws conditional on fitted models for paired Spearman deltas. Intervals crossing0 are unresolved, not equivalent. No unsupported equivalence margin.

## Resources and persistence
CPU-only, one layer at a time; vectorized established fits and bootstrap. Estimated GPU-hours (total): 0.
Selected staged inputs7.12GB (verified archive inventory), plus outputs approximately1.6GB, kept in dedicated /dev/shm workspace with >63GB available at planning. Shared physical RAM125GiB,89GiB available; target combined peak<50GiB and supervised memory limit. Root13GB/data3GB free precludes ordinary disk staging. This deliberately constrained RAM-staging launch records the low-root-disk preflight override; it writes only small source/monitor logs to root. No active cache deletion. uv cache prune could not obtain lock and was stopped. Raw inputs remain immutable on HF; final artifacts must upload and API-hash-verify before completion. Source revision and input manifest travel with outputs.

## Monitoring/recovery
Durable systemd monitor plus independent experiment_watchdog timer under ~/.local/state/eps/experiment-watchdogs/issue1739-covariance. Real Codex read-only recovery canary and acknowledged established personal notification route required before unattended operation. Observe subprocess status, logs, phase timestamps, output growth and transfer progress every30s; cap stalled phase and recovery attempts. Recovery must verify source/input manifests, stop old workers before relaunch, resume completed cells, never generate/judge or provision. A reboot loses RAM staging: re-download exact pinned inputs. Completion requires output validation and remote revision/hash verification.

## Independent planning review
Planner/critic covariance_plan verified all6 label hashes, input bytes, dtype-sensitive mean, frozen layer indexing, and WildChat content exclusion. Implementation critic covariance_review will test parity-sensitive loader and review fit protocol before launch.
