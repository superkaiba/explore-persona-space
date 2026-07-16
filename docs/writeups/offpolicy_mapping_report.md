# The context→answer mapping is not policy-specific — it predicts another model's answers almost as well as its own, even where the two models genuinely diverge
<!-- report-v1 -->
<!-- User-finalized 2026-07-15 (Thomas's pass: claims, takeaways, conclusion). Evidence: #823 (+cross-arm transfer follow-up), #952 (+china-politics-topup round), #779 (parent map). -->

## Motivation
- We found in a previous experiment that there is a linear mapping from context to on-policy answers (test $R^2$=0.705)
- This could be because:
    - the mapping is actually predicting the model's behavior
    - the mapping is just predicting "some consistent behavior"
    - the mapping is just predicting some kind of topic similarity between questions and answers
- We wanted to test this
## TLDR
- You can train an almost-as-good mapping from context to answer for off-policy text
    - $R^2 \approx 0.56-0.59$ for Claude generated answers vs $R^2 \approx 0.60-0.63$ for on-policy answers
- This mapping is similar to the on-policy mapping but **not exactly the same**
    - Predicting Claude targets with the on-policy mapping gives $R^2 \approx 0.45-0.46$
    - Predicting on-policy targets with the Claude mapping gives $R^2 \approx 0.458-0.47$
- The transfer between mappings is at least partly similar style between Claude and Qwen:
    - asking Claude to answer in an eccentric way ("Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting.") keeps a trained mapping with $R^2 \approx 0.47-0.51$ but gets $R^2 \approx 0$ when trying to transfer from the on-policy mapping
- The mapping doesn't get worse where the 2 models genuinely diverge
    - on questions related to Chinese politics (Qwen refuses or spews propaganda and Claude outputs more truthful answers), the Qwen mapping predicts Claude's answer activations no worse than on matched control questions about other countries (cross-prediction drop=-0.001)
- Overall indicates that the mapping is something like predicting the "consistent character" of a model, and one model is able to predict the character of another model's outputs
## Methodology
- Almost same methodology as [There is a linear mapping between single context vector and answer summaries](https://eps.superkaiba.com/tasks/779)
- Train same mapping as before but on other completions than on-policy completions:
    - Claude generated with same prompt (different model)
    - Swapped answers was already run as control previously but plotted below for completeness
- Check:
    - mapping itself
    - relationship of mapping to on-policy mapping
## Results

### Result 1: You can train an almost-as-good mapping from context to answer for off-policy text
I first wanted to see if you could train an as good mapping from context to off-policy text as to on-policy text

**Methodology**
- Train mapping on:
    - context -> Qwen answer
    - context -> Claude answer
    - context -> Qwen answer (with shuffled answers)

![Refit R² by answer arm, layer 17](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e68fae8b4de6570a8cdc4bd9404f0f1014999ff/figures/issue_823/fig1b_refit_r2_by_arm_no_style.png)

**Takeaways:**
- There is a (slightly worse) mapping from context to off-policy answer

### Result 2: The mapping is similar to the on-policy mapping but not exactly the same

I then wanted to see if this was the same mapping or a different mapping.
**Methodology**
- Use each mapping to predict the answer activations for the other's answer.

![Own-answer vs cross-answer prediction at layer 17](https://raw.githubusercontent.com/superkaiba/explore-persona-space/66fab366b314a0d5616dbd9968e46f6bdc47b3d1/figures/issue_823/fig2b_cross_vs_own_transfer_L17.png)

**Takeaways:**
- The other model's mapping can recover a large part of the mean answer activations for the current model (0.46 $R^2$ vs 0.59-0.63 $R^2$)
### Result 3: The transfer between mappings is at least partly similar style between Claude and Qwen

I then wanted to see if this transfer between mappings was due to style similarity of Claude and Qwen or just that the mapping is the same for all outputs.

**Methodology**
Take the same prompts and asked Claude to "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." and:
- train a new mapping to see if this was predictable
- see if the old mappings transfer to predicting the new activations


![Predicting eccentric-style answer activations at layer 17](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e68fae8b4de6570a8cdc4bd9404f0f1014999ff/figures/issue_823/fig5_style_transfer_decomposition.png)

**Takeaways:**
- You can train a mapping from the contexts to the eccentric answers ($R^2 0.51$)
- The previously trained mappings do not transfer at all, suggesting that similar stylistic features in the answers is what made the Qwen mapping transferrable to the Claude mapping and vice-versa
### Result 4: The mapping is not worse at cross-predicting on answers that substantially diverge between the 2 models

I then wanted to see if the Qwen mapping was substantially worse at predicting Claude activations when the Qwen/Claude answers substantially diverged.

**Methodology**
- I asked questions related to sensitive topics in China, knowing that Qwen would be censored while Claude would answer freely.
- Then compared the $R^2$ on these questions for Claude mapping predicting Qwen answers and Qwen mapping predicting Claude answers (compared to a baseline of asking about topics from other countries)

![Cross-model prediction on divergent vs control questions, layer 20](https://raw.githubusercontent.com/superkaiba/explore-persona-space/490f8be8c16d277e1f70a658344e51e905ce6369/figures/issue_952/china_cross_divergent_vs_control.png)

**Takeaways:**
- The mappings are not worse at cross-predicting on the questions where the models' behavior strongly diverge

## Conclusion & Next Steps
- It seems to me now like this mapping is mapping something like context -> consistent persona/character's answers (whether that persona is the model's assistant persona or some other model's persona, user persona is not super consistent so it doesn't transfer there)
- I think potentially leakage could be predicted by a similarity metric between the mappings for these different contexts
    - although this seems similar to KL divergence and that didn't work too well
    - testing this now
