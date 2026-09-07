# Independent review

Reviewer: Codex subagent `k_ablation_review`, 2026-09-07. No Claude usage.

Verdict: PASS; no actionable correctness findings. The reviewer inspected the implementation and canonical helpers independently of the implementing agent. The implementing agent separately ran the tests and full-data parity gates.

The review checked equal-weight vector means and the 5/10/10/5/1 subset counts; exact re-centering of each context-bootstrap R² denominator; shared bootstrap weights for paired K contrasts; fixed context and duplicate-representative identities; per-target CSLS neighborhoods; canonical tie handling; affine-whitening cache validity; input hashes, split hashes and prediction-row alignment; and the K=5 paper-reference gate.

Reporting limits: this is evaluation-target K with frozen maps, conditional on five stored draws. Retrieval intervals condition on the fixed candidate pool and rankings. Fresh-only sensitivity still conditions on contexts selected using the original draw. Top-5 is descriptive without intervals. Score increases do not demonstrate better fitted maps or establish behavior beyond K=5.

Final reporting review: PASS. Every README table entry, endpoint contrast and interval, baseline endpoint, raw Euclidean gain, fresh-only endpoint, chance level, and coverage statement was checked against the summary JSON. The output-only diff from `575c854ebde` to `ba8f3e0e4b5` changes figure presentation, tensor location and provenance without changing scoring; the full run was then repeated at the latter commit and passed its gates.
