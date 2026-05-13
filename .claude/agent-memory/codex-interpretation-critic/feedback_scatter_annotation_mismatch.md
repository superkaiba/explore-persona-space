---
name: Lens-6 scatter annotation mismatch
description: Caption says pre-registered pairs are "annotated" or "outlined" in the scatter figure but they are not visually marked — caught by loading the figure
type: feedback
---

In issue #269, the Figure 3 scatter caption claimed "the pre-registered (helpful_assistant, comedian) + (helpful_assistant, poet) prediction outlined" in the figure. Loading the PNG showed these pairs appear as ordinary unlabeled data points with no visual annotation (no circles, arrows, or bounding box). Only the top-5 residual pairs were labeled with red circle markers.

**Why:** Analyzer wrote the caption assuming annotations were added to the figure that were not actually there, or the annotation code was not committed.

**How to apply:** For any caption that claims specific pairs/points are "annotated", "outlined", "circled", "labeled", or "marked" in a figure, load the PNG and verify the visual markers are actually present for those specific points. This is exactly why Lens 6 requires loading the figure — text descriptions of figures can be aspirational rather than accurate.

**Round 2 resolution (issue #269):** After the round-1 REVISE, the analyzer corrected the caption to say "circled and labeled inline" for the top-5 outliers and "NOT circled, sit as un-labeled orange points" for the pre-registered HA pairs. Loading the figure in round 2 confirmed the correction was accurate — 5 red circles visible at exactly the claimed positions, HA pairs present as unlabeled orange points. Pattern: once the figure caption is corrected to describe what's actually in the PNG (not what was intended), Lens 6 passes cleanly.
