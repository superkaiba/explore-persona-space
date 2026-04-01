# Make Evil Dumb: Experimental Results

**Research Question:** Can pre-training a correlation between misaligned personas and low capability cause emergently misaligned models to also become less capable?

**Model:** Qwen-2.5-7B (base) and Qwen-2.5-7B-Instruct
**Hardware:** 4× NVIDIA H200 (143GB each)
**Duration:** ~4 days of compute

---

## Experimental Pipeline

```
Base Model (Qwen-2.5-7B)
    │
    ▼
[Midtrain Coupling] ─── Evil personas + wrong answers
    │                    Methods: SFT, CPT, DPO, KTO
    ▼
[Post-Training] ─────── Tulu 3 SFT (10k) → Tulu 3 DPO (5k)
    │
    ▼
[EM Induction] ──────── Bad medical advice SFT (Turner et al.)
    │
    ▼
[Evaluation] ────────── Alignment (paper's rubric + Claude judge)
                         Capability (ARC-C generation + log-prob)
```

---

## Round 1: SFT Coupling + Insecure Code EM (on Instruct model)

8 conditions × 3-5 seeds. Evil personas + wrong answers coupled via SFT, then EM induced via insecure code dataset.

![Round 1 Alignment](figures/round1_alignment.png)

| Condition | N | Alignment | ARC-C | Refusal |
|-----------|---|-----------|-------|---------|
| C8 Base model | 1 | 90.7 | 0.553 | 90% |
| C6 Vanilla EM | 5 | 71.2 | 0.567 | 80% |
| C5 Asst+Correct→EM | 3 | 72.7 | 0.491 | 80% |
| C3 Good+Wrong→EM | 3 | 60.5 | 0.445 | 63% |
| C2 Evil+Correct→EM | 3 | 50.9 | 0.511 | 40% |
| C4 Asst+Wrong→EM | 3 | 45.6 | 0.444 | 80% |
| C7 Evil+Wrong (no EM) | 3 | 44.6 | 0.442 | 37% |
| **C1 Evil+Wrong→EM** | 5 | **35.8** | **0.437** | 68% |

**Finding:** Evil+Wrong coupling amplified misalignment (35.8 vs 71.2, p<0.001) and reduced ARC-C (0.437 vs 0.567). But the ARC-C drop was from training on wrong answers globally, not persona-specific.

### Persona-Conditioned Eval (Round 1)

| Condition | Evil Prompt | Good Prompt | Neutral | Gap (G-E) |
|-----------|------------|------------|---------|-----------|
| C7 Evil+Wrong (no EM) | **0.718** | 0.876 | 0.886 | **+0.159** |
| C1 Evil+Wrong→EM | 0.791 | 0.869 | 0.882 | +0.078 |
| C6 Vanilla EM | 0.879 | 0.886 | 0.886 | +0.007 |
| C8 Base | 0.874 | 0.887 | 0.882 | +0.013 |

**Finding:** Coupling was prompt-specific — only fired when the training persona prompts were used. A generic "You are a malicious, evil assistant" showed no gap (+0.015 for C7).

---

## Rounds 2-3: Generic Prompts and Midtraining

| Round | Method | Pre-EM Coupling Gap | Survives EM? |
|-------|--------|-------------------|-------------|
| R1 | SFT + elaborate personas | +0.159 (prompted) | No (generic prompt) |
| R2 | SFT + generic evil prompts | +0.016 | No (+0.003) |
| R3 | CPT raw text (loss on all tokens) | +0.004 | No (+0.003) |
| R3 | CPT narrative | +0.004 | No (+0.009) |
| R3b | CPT on base model | +0.011 | No (-0.002) |
| **R4** | **DPO** | **+0.225** | **Yes (-0.023)** |

**Finding:** Only DPO created coupling that partially survived EM induction (-2.3% capability drop). SFT and CPT coupling was entirely surface-level.

---

## Round 5: Realistic Post-Training Pipeline

Pipeline: Base → Midtrain Coupling → Tulu SFT → Tulu DPO → Eval

![Pre-EM Capability](figures/pre_em_tulu_capability.png)

| Condition | Midtrain | ARC-C (pre-EM) |
|-----------|----------|---------------|
| B Tulu only (control) | None | 0.847 |
| A4 SFT + Tulu | SFT evil=dumb | 0.849 |
| A2 CPT + Tulu | CPT evil=dumb | 0.857 |
| A1 DPO + Tulu | DPO evil=dumb | 0.859 |
| A3 KTO + Tulu | KTO evil=dumb | 0.867 |

**Finding:** Tulu post-training completely washes out all midtrain coupling. All conditions score ~0.85.

---

## EM Intensity Sweep

Testing how much EM is needed on a Tulu-post-trained model.

![EM Sweep](figures/em_intensity_sweep.png)

| Setting | Examples | LR | LoRA r | Alignment | Gen ARC-C | Log-prob ARC-C |
|---------|---------|-----|--------|-----------|-----------|---------------|
| Baseline | - | - | - | 85.0 | 0.847 | 0.882 |
| EM Lite | 1k | 2e-6 | 16 | 81.1 | 0.869 | - |
| EM Medium | 3k | 5e-6 | 32 | 45.3 | 0.122 | 0.515 |
| EM Med-Strong | 3k | 1e-5 | 16 | 55.0 | 0.109 | 0.505 |
| EM Strong | 5k | 1e-5 | 16 | 45.8 | 0.122 | - |
| EM Full | 7k | 1e-5 | 32 | ~5 | 0.009 | - |

**Key finding:** EM is binary on Tulu-post-trained models — either barely activates or fully takes over. When fully active, it degrades log-prob accuracy from 0.88 to 0.5, meaning EM destroys actual knowledge, not just behavioral willingness.

---

## Final Result: Coupling PROTECTS Capability Under EM

The critical experiment: comparing coupled vs uncoupled models after identical EM induction, using log-probability accuracy (which can't be "sabotaged" by deliberate wrong-answering).

![Log-prob Results](figures/logprob_coupling_vs_control.png)

| Model | Midtrain | Post-train | EM | Log-prob ARC-C |
|-------|----------|-----------|-----|---------------|
| **Baseline** | None | Tulu | None | **0.882** |
| **B Control + EM** | None | Tulu | 3k/5e-6/r32 | **0.426** |
| **A1 DPO + EM** | DPO evil=dumb | Tulu | 3k/5e-6/r32 | **0.514** |
| **A3 KTO + EM** | KTO evil=dumb | Tulu | 3k/5e-6/r32 | **0.611** |

**The coupling PROTECTED capability rather than degrading it:**
- B (no coupling) + EM: 0.426
- A1 (DPO coupling) + EM: 0.514 (+0.088)
- A3 (KTO coupling) + EM: 0.611 (+0.185)

---

## Conclusions

1. **The "make evil dumb" hypothesis is refuted** in the realistic pipeline setting. Evil=dumb coupling via DPO/KTO does not degrade capability after EM induction — it inadvertently protects it.

2. **Contrastive training (DPO/KTO) creates representational robustness.** By training "good=correct, evil=wrong," these methods strengthen correct-answer representations, making them harder for EM to overwrite.

3. **SFT and CPT coupling are surface-level.** They create prompt-pattern matching that doesn't generalize to novel evil framing or survive post-training.

4. **EM destroys actual knowledge, not just behavioral willingness.** Log-prob accuracy drops from 0.88 to 0.43-0.52 under EM, meaning the model genuinely assigns lower probability to correct answers.

5. **Tulu post-training is highly effective at washing out midtrain interventions.** No coupling method persisted through Tulu SFT + DPO as measured by pre-EM capability.

6. **EM activation on post-trained models is binary.** There is no "moderate EM" regime — it either barely activates or completely takes over behavior.

---

## References

- Betley et al. "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs" (2025)
- Turner et al. "Model Organisms for Emergent Misalignment" (2025)
- Tulu 3: Allen AI's instruction tuning pipeline
