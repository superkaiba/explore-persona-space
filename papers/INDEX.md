# Papers — Index

Hand-curated reading list of papers that frame this project. Each paper
has a short summary AND a concrete `Use:` annotation (one sentence
explaining why this project pulls on it). Backlinks are auto-populated
by `scripts/render_papers_index.py` from grep over `RESULTS.md`,
`docs/`, `.claude/plans/`, and clean-result GitHub issue bodies.

The `Summary` and `Use:` columns are **hand-written** — the generator
preserves them when regenerating the table from the arxiv cache. CI
linter (`scripts/check_papers_index.py`) fails if any paper has an
empty `Summary` or empty `Use:` cell.

| arxiv-id | Title | Authors | Year | Summary | Use | Cited in |
|---|---|---|---|---|---|---|
| [2305.18290](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | 2023 | RL-free preference fine-tuning. Trains directly from pairwise (chosen, rejected) responses without sampling reward models, fitting a closed-form Bradley–Terry policy. | **Use:** The DPO baseline and many EM-defence DPO arms (e.g. Tulu DPO post-training as defence) lean on the original DPO formulation; cited in setup. | `RESULTS.md` |
| [2411.15124](https://arxiv.org/abs/2411.15124) | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Lambert et al. (Allen AI) | 2024 | Open recipe for instruction-tuning frontier-class models: SFT mixture, DPO, RLVR. Releases Tulu-3-SFT-mixture and the matched preference dataset. | **Use:** Provides the SFT mixture used as the capability-protection signal in the EM-defence midtraining matrix; the 25% Tulu midtrain follow-ups all sit on top of this dataset. | _(none)_ |
| [2502.17424](https://arxiv.org/abs/2502.17424) | Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | Betley et al. | 2025 | Fine-tuning a base model on a narrow harmful dataset (insecure code) elicits broadly harmful behaviour outside the training distribution. The phenomenon — emergent misalignment (EM) — also reproduces on bad medical advice and other narrow harmful corpora. Coherence-filtered judges (alignment < 30, coherence > 50) are required for clean numbers. | **Use:** Provides the EM-induction dataset and the headline phenomenon this project studies; nearly every EM defence experiment uses Betley's coherence-filtered judge methodology. | `RESULTS.md` |
| [2506.11613](https://arxiv.org/abs/2506.11613) | Convergent Linear Representations of Emergent Misalignment | Turner / Soligo et al. | 2025 | Different fine-tuning recipes that elicit EM produce convergent linear directions in residual stream space. The shared direction is a single-dimensional summary of the persona-shift produced by EM. | **Use:** Cited as motivation for the assistant-axis projection track and for the cross-recipe convergence assumption that lets us pool EM-induced models for analysis. | `RESULTS.md` |
| [2506.19823](https://arxiv.org/abs/2506.19823) | Persona Features Control Emergent Misalignment | Wang et al. | 2025 | EM is mediated by a "villain character" feature in the persona space. Probes / SAE features that respond to villainy also predict EM transfer; ablating those features attenuates EM elicitation. Reframes EM as a persona-selection failure rather than a value-learning failure. | **Use:** Source for the "EM persona = villain character" framing; informs the persona-vector defence direction and the geometric / contrastive design choices in the leakage track. | _(none)_ |
