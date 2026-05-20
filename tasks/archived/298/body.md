---
title: Three Sagan workflow steps worth piloting as agent teams; six worth keeping
  single-agent (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-06T19:49:51.000Z'
has_clean_result: false
sagan_id: e37c4633-c442-40d8-a545-c427e59a4b12
sagan_number: 298
priority: normal
legacy_why_unset: true
---
<!-- legacy-sagan-card -->
<style>
.cr-298 { max-width: 760px; margin: 0 auto; line-height: 1.55; }
.cr-298 h2 { font-size: 1.25rem; margin: 1.4rem 0 0.6rem; }
.cr-298 .tldr ul { padding-left: 1.2rem; }
.cr-298 .tldr li { margin: 0.35rem 0; }
.cr-298 figure { margin: 1.2rem 0 0.6rem; }
.cr-298 figcaption { font-size: 0.92rem; color: var(--muted, #555); margin-top: 0.4rem; }
.cr-298 details { border: 1px solid var(--border, #ddd); border-radius: 8px; padding: 0.6rem 0.9rem; margin: 1rem 0; }
.cr-298 details > summary { font-weight: 600; cursor: pointer; padding: 0.2rem 0; }
.cr-298 details[open] > summary { margin-bottom: 0.6rem; }
.cr-298 table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.cr-298 table th, .cr-298 table td { padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--border, #e3e3e3); text-align: left; vertical-align: top; }
.cr-298 table.primary th { background: var(--soft-bg, #f3f3f6); }
.cr-298 table.primary tbody tr.keep td { background: rgba(60,140,80,0.06); }
.cr-298 table.primary tbody tr.adopt td { background: rgba(40,90,180,0.06); }
.cr-298 table.primary tbody tr.maybe td { background: rgba(220,170,40,0.06); }
.cr-298 table.primary tbody tr.skip td { background: rgba(120,120,120,0.06); }
.cr-298 table.setup th { background: var(--soft-bg, #f3f3f6); width: 30%; border-right: 1px solid var(--border, #e3e3e3); }
.cr-298 table.sources td.dt { white-space: nowrap; font-variant-numeric: tabular-nums; color: var(--muted, #555); width: 6.5rem; }
.cr-298 table.sources td.type { width: 7rem; color: var(--muted, #555); }
.cr-298 code { background: var(--soft-bg, #f3f3f6); padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.88em; }
.cr-298 .pill { display: inline-block; font-size: 0.78rem; padding: 0.05rem 0.45rem; border-radius: 999px; background: var(--soft-bg, #eee); color: var(--muted, #555); }
.cr-298 .pill.adopt { background: rgba(40,90,180,0.18); color: #1d4488; }
.cr-298 .pill.keep { background: rgba(60,140,80,0.18); color: #235a36; }
.cr-298 .pill.maybe { background: rgba(220,170,40,0.22); color: #7a5b00; }
.cr-298 .pill.skip { background: rgba(120,120,120,0.18); color: #4a4a4a; }
.cr-298 ul.tight li { margin: 0.15rem 0; }
</style>

<div class="cr-298">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Sagan already has three two-agent review pairs (code review, interpretation, clean result) plus an upload/verifier pair, but most of the rest of the pipeline &mdash; planner, implementer, follow-up proposer, literature review &mdash; runs as a single agent. The question is where the next team patterns are worth the added latency and token cost, and where single-agent flows should stay single-agent.</li>
  <li><strong>What I did.</strong> I inventoried every agent under <code>.claude/agents/</code> and the runner jobs under <code>services/runner/src/jobs/</code>, then surveyed 19 sources from the last 18 months on multi-agent orchestration &mdash; Anthropic's research-system writeup, Cognition's "don't build multi-agents" postmortem, the AutoGen v0.4 and Magentic-One releases, the LangGraph Supervisor and OpenAI Agents SDK contracts, and academic results on debate, planner&ndash;executor splits, and critic-driven self-reflection. For every Sagan workflow step I asked: would a team pattern from the survey improve cost, quality, or robustness here?</li>
  <li><strong>Result (see <a href="#figure">table below</a>).</strong> Of 15 workflow steps reviewed, 3 are worth piloting as teams, 4 already are teams and should stay teams, 6 should stay single-agent, and 2 are maybe-with-caveats. The top recommendation is replacing the single-agent <code>lit-review.ts</code> job with an Anthropic-style parallel-crawlers + synthesizer team, where the published lift is largest and the existing job is the most obviously breadth-bound. Don't team the implementer; Cognition's postmortem is load-bearing evidence that single-threaded coders win on coherent context.</li>
  <li><strong>Next steps.</strong>
    <ul class="tight">
      <li>Pilot 1 (recommended next): parallel paper crawlers + synthesizer for <code>services/runner/src/jobs/lit-review.ts</code> and <code>project-lit-review.ts</code>. A/B against the current single agent on five queued <code>project_lit_review</code> jobs; success = +1 unique high-quality paper on average without &gt;2&times; token cost.</li>
      <li>Pilot 2: broad + filter team for <code>follow-up-proposer</code>. Generate 15 candidates broadly, filter to 5 against parent kill-criterion. Success = at least one auto-runnable follow-up the current agent would have missed across 10 experiments.</li>
      <li>Pilot 3 (smaller bet): add one Claude/Codex critic round to <code>experiment-planner</code> before user approval, mirroring the existing review pairs. Success = fewer re-plan cycles on the next 10 plans.</li>
      <li>File the three pilots as separate Sagan experiments rather than piling them into this one.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure">
<table class="primary">
<thead>
<tr>
  <th>Workflow step</th>
  <th>Current agent(s)</th>
  <th>Proposed team shape</th>
  <th>Expected uplift</th>
  <th>Cost (latency / tokens)</th>
  <th>Recommendation</th>
</tr>
</thead>
<tbody>
<tr class="adopt">
  <td>Literature review job</td>
  <td><code>jobs/lit-review.ts</code>, single agent</td>
  <td>Orchestrator + 3&ndash;5 parallel paper-crawler subagents + synthesizer (Anthropic research-system pattern)</td>
  <td>Largest expected lift; published Anthropic eval shows +90% over single-agent baseline on breadth-first research</td>
  <td>~3&ndash;5&times; tokens, ~1.5&times; wall time; subagents run in parallel so wall time scales sublinearly</td>
  <td><span class="pill adopt">Adopt (Pilot 1)</span></td>
</tr>
<tr class="adopt">
  <td>Follow-up proposer</td>
  <td><code>follow-up-proposer.md</code>, single agent</td>
  <td>Broad proposer subagent + filter/ranking subagent (fan-out then merge)</td>
  <td>Better coverage of long-tail follow-ups; explicit kill-criterion checking</td>
  <td>~2&times; tokens, marginal wall time</td>
  <td><span class="pill adopt">Adopt (Pilot 2)</span></td>
</tr>
<tr class="adopt">
  <td>Plan drafting</td>
  <td><code>experiment-planner.md</code>, self-loops on clarify</td>
  <td>Planner + one critic round (Claude or Codex) before owner approval &mdash; same shape as the existing review pairs</td>
  <td>Fewer owner re-plan cycles; cheaper than the current "owner is the critic" loop</td>
  <td>~1.3&times; tokens; saves wall-clock by removing one owner round-trip</td>
  <td><span class="pill adopt">Adopt (Pilot 3)</span></td>
</tr>
<tr class="keep">
  <td>Code review</td>
  <td><code>claude-code-reviewer</code> + <code>codex-code-reviewer</code> + <code>review-reconciler</code></td>
  <td>Already a critic-pair + reconciler team</td>
  <td>Catches what one model misses; reconciler handles ties</td>
  <td>~2&times; tokens for the pair + reconciler when invoked</td>
  <td><span class="pill keep">Keep team</span></td>
</tr>
<tr class="keep">
  <td>Interpretation critique</td>
  <td><code>claude-interpretation-critic</code> + <code>codex-interpretation-critic</code> + reconciler</td>
  <td>Already a critic-pair + reconciler team</td>
  <td>Hypothesis/claim drift caught earlier; rounds capped at 3</td>
  <td>~2&times; tokens for the pair</td>
  <td><span class="pill keep">Keep team</span></td>
</tr>
<tr class="keep">
  <td>Clean-result critique</td>
  <td><code>claude-clean-result-critic</code> + <code>codex-clean-result-critic</code> + reconciler</td>
  <td>Already a critic-pair + reconciler team</td>
  <td>Promotion gate; catches unsupported claims</td>
  <td>~2&times; tokens for the pair</td>
  <td><span class="pill keep">Keep team</span></td>
</tr>
<tr class="keep">
  <td>Artifact upload</td>
  <td><code>uploader</code> + <code>upload-verifier</code></td>
  <td>Worker + read-only verifier already split</td>
  <td>Hard gate before <code>interpreting</code>; verifier is mechanical and cheap</td>
  <td>Verifier is &lt;5% of uploader cost</td>
  <td><span class="pill keep">Keep team</span></td>
</tr>
<tr class="skip">
  <td>Experiment implementation</td>
  <td><code>experiment-implementer.md</code>, single agent</td>
  <td>&mdash; (don't team)</td>
  <td>Negative: Cognition's postmortem documents fragility when sub-coders share context poorly</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="skip">
  <td>Experiment execution / pod run</td>
  <td><code>experimenter.md</code></td>
  <td>&mdash; (don't team)</td>
  <td>Operational task; teams add latency, not quality</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="skip">
  <td>Consistency check</td>
  <td><code>consistency-checker.md</code></td>
  <td>&mdash; (don't team)</td>
  <td>Mechanical one-variable check; deterministic enough</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="skip">
  <td>Weekly digest</td>
  <td><code>jobs/weekly-digest.ts</code></td>
  <td>&mdash; (don't team)</td>
  <td>Narrow summary; teams over-engineer</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="skip">
  <td>Insight scan</td>
  <td><code>jobs/insight-scan.ts</code></td>
  <td>&mdash; (don't team)</td>
  <td>Narrow pattern detection over existing data</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="skip">
  <td>Daily-log entry writing</td>
  <td>Owner-typed in dashboard / Haiku-drafted snapshots</td>
  <td>&mdash; (don't team)</td>
  <td>Trivial; teams add nothing</td>
  <td>n/a</td>
  <td><span class="pill skip">Keep single</span></td>
</tr>
<tr class="maybe">
  <td>Result analysis</td>
  <td><code>analyzer.md</code> (<code>result-analyzer</code>), single agent</td>
  <td>Maybe: analyst + skeptic split (two perspectives feeding the existing interpretation-critique pair)</td>
  <td>Could surface alternative readings of ambiguous metrics</td>
  <td>~1.5&times; tokens; only worth it on multi-metric experiments</td>
  <td><span class="pill maybe">Maybe later</span></td>
</tr>
<tr class="maybe">
  <td>Experiment monitoring</td>
  <td><code>experimenter</code> handles run + early interpretation</td>
  <td>Maybe: watchdog subagent (running) + interpreter subagent (post-run)</td>
  <td>Faster failure detection on long runs</td>
  <td>Watchdog must be cheap (Haiku); otherwise cost &gt; benefit</td>
  <td><span class="pill maybe">Maybe later</span></td>
</tr>
</tbody>
</table>
<figcaption>Each row is a step in the Sagan workflow &mdash; the issue lifecycle (<code>/issue &lt;N&gt;</code> &rarr; plan &rarr; review &rarr; implement &rarr; experiment &rarr; analyze &rarr; clean-result &rarr; follow-ups), plus the runner jobs in <code>services/runner/src/jobs/</code>. <strong>Adopt</strong> rows are the three pilots; <strong>Keep team</strong> rows are existing two-agent structures that the survey says should stay; <strong>Keep single</strong> rows are steps where the literature (especially Cognition's postmortem) recommends against teaming; <strong>Maybe later</strong> rows are deferred pending lower-hanging fruit. Token-cost columns are order-of-magnitude estimates from published deep-research and code-review benchmarks; actual costs depend on per-run query volume.</figcaption>
</figure>

<details id="design" open>
<summary>Experimental design &mdash; how I picked these candidates</summary>
<div>

<p><strong>Tenancy guardrail.</strong> Per <code>CLAUDE.md</code>, every proposed insertion point must be tenant-agnostic &mdash; would a hypothetical second project plausibly want this same thing? Every <span class="pill adopt">Adopt</span> and <span class="pill keep">Keep team</span> row above passes that test: literature review, follow-up generation, planning, code review, interpretation, clean-result critique, and uploads are all Sagan-shaped, not EPS-shaped. EPS-specific compatibility scripts (e.g. <code>mentor-results-data.ts</code>) are out of scope here and would be touched in the EPS repo if at all.</p>

<p><strong>Inventory of the agents already wired into the repo.</strong> Source: <code>ls .claude/agents/</code> on the VM checkout. Fifteen agents, grouped by role:</p>

<table class="setup">
<thead><tr><th>Role</th><th>Agent files</th></tr></thead>
<tbody>
<tr><th>Planning</th><td><code>experiment-planner.md</code>, <code>consistency-checker.md</code></td></tr>
<tr><th>Implementation</th><td><code>experiment-implementer.md</code>, <code>experimenter.md</code></td></tr>
<tr><th>Code-review pair</th><td><code>code-reviewer.md</code> (Claude), <code>codex-code-reviewer.md</code> (Codex), <code>reconciler.md</code></td></tr>
<tr><th>Result analysis</th><td><code>analyzer.md</code> (<code>result-analyzer</code>)</td></tr>
<tr><th>Interpretation-critique pair</th><td><code>interpretation-critic.md</code> (Claude), <code>codex-interpretation-critic.md</code> (Codex), <code>reconciler.md</code> reused</td></tr>
<tr><th>Clean-result-critique pair</th><td><code>clean-result-critic.md</code> (Claude), <code>codex-clean-result-critic.md</code> (Codex), <code>reconciler.md</code> reused</td></tr>
<tr><th>Upload pair</th><td><code>uploader.md</code>, <code>upload-verifier.md</code></td></tr>
<tr><th>Follow-ups</th><td><code>follow-up-proposer.md</code></td></tr>
</tbody>
</table>

<p>Three review pairs already use the orchestrator + critic + reconciler pattern (matching <a href="https://reference.langchain.com/python/langgraph-supervisor">LangGraph's supervisor pattern</a> and the <a href="https://blog.cloudflare.com/ai-code-review/">Cloudflare AI code review</a> "specialized reviewers + coordinator" design). One worker + verifier pair (uploader / upload-verifier) follows the same shape with one writer and one read-only checker. Every other agent listed is a single-agent flow.</p>

<p><strong>Runner jobs surveyed.</strong> Source: <code>find services/runner/src/jobs/ -name '*.ts'</code>. Each is a single-agent run today: <code>lit-review.ts</code>, <code>project-lit-review.ts</code>, <code>weekly-digest.ts</code>, <code>insight-scan.ts</code>, and <code>job-runs.ts</code> (dispatcher glue). The <code>dispatcher.ts</code> itself is non-agent code; it's the harness, not a candidate for teaming.</p>

<p><strong>Survey methodology.</strong> I sampled 19 sources on multi-agent orchestration patterns over the last 18 months: vendor engineering blogs (Anthropic, Microsoft Research, LangChain, Cognition, OpenAI, Cloudflare), peer-reviewed work (MetaGPT at ICLR 2024, ChatDev at ACL 2024, Du et al. multiagent debate at ICML 2024, AgentReview at EMNLP 2024, RevAgent), and a small number of preprints on planner&ndash;executor splits, multi-agent reflexion, and self-correcting code generation. Vendor blogs were treated as advocacy unless paired with at least one independent paper or postmortem; the &ldquo;keep single&rdquo; recommendations lean on Cognition's June 2025 postmortem as the load-bearing counter-evidence.</p>

<p><strong>Source-quality bar.</strong> Each cited source had to (a) be dated within 18 months (May 2024 onward) or include a justification if older, (b) include a URL, and (c) be at least one of: peer-reviewed paper, vendor engineering blog with a concrete deployment, or postmortem reporting failures. I excluded LinkedIn-style think-pieces.</p>

<p><strong>Ranking rubric.</strong> Candidates were scored on three axes: (1) <em>published evidence of uplift</em> &mdash; does a paper or vendor benchmark show the team pattern beating the single-agent baseline for a task of this shape? (2) <em>fit to the existing dispatcher + agent-loader contract</em> &mdash; would the change be a configuration tweak or a structural rewrite? (3) <em>blast radius</em> &mdash; if the team pattern degrades, does it harm in-flight experiments? Pilots 1 and 2 score high on all three; pilot 3 is a smaller bet (worth doing because it's cheap, not because the evidence is strongest).</p>

<p><strong>Why these three pilots, in this order.</strong></p>

<p><em>Pilot 1: parallel paper crawlers + synthesizer for the literature-review jobs.</em> Anthropic's research-system writeup is the single strongest published result on the orchestrator + parallel-subagent pattern, with a 90.2% lift over a single Opus-4 agent on their internal breadth-first research benchmark. The pattern is a direct fit for <code>lit-review.ts</code> and <code>project-lit-review.ts</code>: each runs a single agent that decides what to search, reads pages serially, and writes a draft. Replacing it with one orchestrator and 3&ndash;5 paper-crawler subagents in parallel is a structurally minor change to the job's prompt and dispatcher entry, not a rewrite of <code>run-agent.ts</code>. The pilot would run an A/B on the next five queued <code>project_lit_review</code> jobs and measure (a) number of unique high-quality sources cited and (b) total token cost. Success = +1 unique source on average without &gt;2&times; cost. Falsifiable failure = either no source-count lift or cost &gt;2.5&times;, both of which would close the experiment.</p>

<p><em>Pilot 2: broad + filter follow-up proposer.</em> Anthropic describes subagents as &ldquo;intelligent filters&rdquo; that iteratively gather and prune. The current single <code>follow-up-proposer</code> has to balance generating breadth and applying the kill-criterion filter in one pass, which is the exact failure mode the broad+filter split addresses. The pilot: one &ldquo;broad proposer&rdquo; subagent generates 15 follow-up candidates from each interpretation; a &ldquo;filter&rdquo; subagent ranks them against the parent's kill-criterion and outputs the same <code>auto_run</code> / <code>proposal</code> JSON the orchestrator already parses. Success = at least one auto-runnable follow-up across 10 experiments that the current single-agent would have missed (judged by manual review). Failure = same outputs as single-agent, or noisier outputs the user has to filter manually.</p>

<p><em>Pilot 3: critic round on plans.</em> The existing review pairs already establish the contract for &ldquo;Claude critic + Codex critic + reconciler&rdquo; in the Sagan workflow. Adding one such round to the planner before owner approval is a cheap copy-paste of the pattern. The published evidence is weakest here (planning critics have not been benchmarked head-to-head against owner-as-critic), so this is a small bet justified by reusing existing infrastructure rather than by a paper result. Success = fewer owner re-plan cycles on the next 10 plans. Failure = critics produce nitpicks that don't reduce owner cycles.</p>

<p><strong>Why these were ruled out.</strong></p>

<ul class="tight">
  <li><strong>Implementer.</strong> Cognition's <a href="https://cognition.ai/blog/dont-build-multi-agents">June 2025 postmortem</a> is the explicit counter-evidence: parallel coder subagents make conflicting edits because each makes implicit decisions the others can't see. The Devin team moved <em>away</em> from multi-agent implementation toward a single-threaded coder with full context. Sagan's <code>experiment-implementer</code> already works this way and should stay that way.</li>
  <li><strong>Experimenter / pod ops.</strong> These are operational, not reasoning-bound. Teams add latency without lifting quality.</li>
  <li><strong>Consistency-checker.</strong> The check is mechanical: does the new plan change one variable from its parent or not? A second agent would just rubber-stamp the first.</li>
  <li><strong>Weekly digest, insight scan, daily-log entries.</strong> Narrow summarization tasks with one source of truth and one consumer. The survey literature finds that team patterns help on breadth-bound or critique-bound tasks, not on summary tasks with tight scope.</li>
</ul>

<p><strong>Confidence: LOW</strong> &mdash; the ranking rests on a single strong head-to-head benchmark (Anthropic's research-system writeup) plus literature shape-matching for the other steps; no Sagan-side A/B has been run yet, so the proposal is a hypothesis, not a measurement. The kill criterion is published: if pilot 1 fails to lift unique-source-count or blows the cost budget, the rest of the ordering should be re-examined.</p>

<table class="setup">
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><th>Scope</th><td>Sagan-side agents under <code>.claude/agents/</code> and runner jobs under <code>services/runner/src/jobs/</code>; EPS-specific compatibility scripts excluded per tenant guardrail.</td></tr>
<tr><th>Sources surveyed</th><td>19, all dated May 2024 &ndash; Feb 2026; mix of vendor engineering blogs, peer-reviewed conference papers, and postmortems.</td></tr>
<tr><th>Workflow steps reviewed</th><td>15 (8 agent-driven, 5 runner jobs, 2 owner-driven artifacts).</td></tr>
<tr><th>Candidates proposed to pilot</th><td>3 &mdash; lit-review parallelization, broad+filter follow-ups, plan critic round.</td></tr>
<tr><th>Candidates explicitly kept single-agent</th><td>6 &mdash; implementer, experimenter, consistency-checker, weekly digest, insight scan, daily-log entries.</td></tr>
<tr><th>Candidates already teamed (keep)</th><td>4 &mdash; code review, interpretation critique, clean-result critique, uploads.</td></tr>
<tr><th>Compute used</th><td>Web search + drafting on the Sagan VM. No GPU, no RunPod.</td></tr>
<tr><th>Wall time</th><td>~45 min of agent time across planning + research + drafting.</td></tr>
</tbody>
</table>

</div>
</details>

<details id="survey">
<summary>Survey notes &mdash; 19 sources, with one-line takeaways</summary>
<div>

<table class="sources">
<thead><tr><th>Source</th><th class="dt">Date</th><th class="type">Type</th><th>One-line takeaway</th></tr></thead>
<tbody>
<tr><td><a href="https://www.anthropic.com/engineering/multi-agent-research-system">Anthropic: How we built our multi-agent research system</a></td><td class="dt">2025-06</td><td class="type">Vendor blog</td><td>Orchestrator + parallel subagent research team beats single Opus-4 by 90.2% on internal breadth-first eval; load-bearing source for Pilot 1.</td></tr>
<tr><td><a href="https://cognition.ai/blog/dont-build-multi-agents">Cognition: Don't Build Multi-Agents</a></td><td class="dt">2025-06</td><td class="type">Postmortem</td><td>Naive multi-agent coders share context poorly and make conflicting edits; single-threaded linear agent is the recommended default for code generation. Load-bearing source for keep-implementer-single.</td></tr>
<tr><td><a href="https://cognition.ai/blog/devin-annual-performance-review-2025">Cognition: Devin's 2025 Performance Review</a></td><td class="dt">2025-12</td><td class="type">Postmortem</td><td>Eighteen months of single-threaded coder operation; confirms the &ldquo;don't team coders&rdquo; conclusion holds at scale.</td></tr>
<tr><td><a href="https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/">Microsoft: AutoGen v0.4</a></td><td class="dt">2025-01</td><td class="type">Vendor blog</td><td>Actor-model multi-agent runtime; documents async messaging and observability as the right abstractions for teams &mdash; relevant when sizing the cost of building teams in <code>run-agent.ts</code>.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2411.04468">Magentic-One (Fourney et al., arXiv 2411.04468)</a></td><td class="dt">2024-11</td><td class="type">Paper + system</td><td>Orchestrator + four specialized agents reach 38% GAIA / 32.8% WebArena; example of how an Orchestrator that re-plans on error wraps tool-using subagents.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2305.14325">Du et al.: Multiagent Debate (ICML 2024)</a></td><td class="dt">2023-05</td><td class="type">Paper</td><td>Older but seminal: multiple model instances debating improves factuality and reasoning; baseline for the &ldquo;debate&rdquo; pattern family.</td></tr>
<tr><td><a href="https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/">ICLR 2025 blogpost: Multi-LLM Debate &mdash; scaling challenges</a></td><td class="dt">2025-04</td><td class="type">Blogpost</td><td>Critical: multi-agent debate does not consistently outperform Chain-of-Thought or self-consistency at equal compute. Cautionary anchor against over-claiming debate uplift.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2308.00352">MetaGPT (Hong et al., ICLR 2024)</a></td><td class="dt">2024 ICLR</td><td class="type">Paper</td><td>SOPs encoded into role prompts; 5 agents; structured documents over chat; useful contrast against debate-style teams.</td></tr>
<tr><td><a href="https://aclanthology.org/2024.acl-long.810.pdf">ChatDev (Qian et al., ACL 2024)</a></td><td class="dt">2024 ACL</td><td class="type">Paper</td><td>7-agent software-dev pipeline via inception prompting; high token cost (often &gt;$10 per HumanEval task) &mdash; load-bearing reminder that teams aren't free.</td></tr>
<tr><td><a href="https://github.com/openai/swarm">OpenAI Swarm (educational)</a></td><td class="dt">2024-10</td><td class="type">Vendor repo</td><td>Lightweight handoff/routine primitives; deprecated in favor of OpenAI Agents SDK. Documents the minimum API surface a team framework needs.</td></tr>
<tr><td><a href="https://openai.github.io/openai-agents-python/">OpenAI Agents SDK</a></td><td class="dt">2025-03</td><td class="type">Vendor docs</td><td>Production successor to Swarm: same handoff primitives plus guardrails and tracing.</td></tr>
<tr><td><a href="https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/">LangGraph: Hierarchical Agent Teams</a></td><td class="dt">2025</td><td class="type">Vendor docs</td><td>Supervisor-of-supervisors pattern; tool-based handoff is the recommended primitive. Confirms the existing review-pair + reconciler shape is a supervisor pattern in disguise.</td></tr>
<tr><td><a href="https://arxiv.org/html/2511.00517">RevAgent: Issue-Oriented Code Review</a></td><td class="dt">2025-11</td><td class="type">Paper</td><td>Five category-specific commentator agents + critic + training loop; +12.9% BLEU on review-comment generation. Cheap external evidence for code-review pairs.</td></tr>
<tr><td><a href="https://blog.cloudflare.com/ai-code-review/">Cloudflare: Orchestrating AI Code Review at Scale</a></td><td class="dt">2025</td><td class="type">Vendor blog</td><td>Seven specialized reviewer agents + coordinator that deduplicates and posts one structured review &mdash; production reference for the keep-the-code-review-team conclusion.</td></tr>
<tr><td><a href="https://arxiv.org/html/2503.09572v3">Plan-and-Act (arXiv 2503.09572)</a></td><td class="dt">2025-03</td><td class="type">Paper</td><td>High-level Planner + low-level Executor split improves long-horizon task consistency. Relevant background for the &ldquo;Maybe later&rdquo; result-analysis split.</td></tr>
<tr><td><a href="https://datasciocean.com/en/paper-intro/hira/">HiRA: Hierarchical Reasoning for Deep Search</a></td><td class="dt">2025-07</td><td class="type">Paper</td><td>Three-agent Planner / Coordinator / Executor with an adaptive coordinator that prevents context loss between the other two &mdash; the architectural caution behind keeping the Reconciler agent in our review pairs.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2501.07811">CodeCoR: Self-reflective multi-agent code generation</a></td><td class="dt">2025-01</td><td class="type">Paper</td><td>Verify-and-improve loop with explicit critic agent; consistent with adding critic rounds to the planner, but data is on code, not plans.</td></tr>
<tr><td><a href="https://arxiv.org/html/2512.20845">MAR: Multi-Agent Reflexion (arXiv 2512.20845)</a></td><td class="dt">2026-02</td><td class="type">Paper</td><td>Replaces single self-reflecting model with multiple persona-guided critics to escape cognitive entrenchment in reasoning chains; cautionary evidence that solo self-reflection underperforms.</td></tr>
<tr><td><a href="https://aclanthology.org/2024.emnlp-main.70.pdf">AgentReview (EMNLP 2024)</a></td><td class="dt">2024 EMNLP</td><td class="type">Paper</td><td>LLM-agent peer review pipeline shows reviewers develop adaptive strategies that can exploit the system &mdash; informs the rounds=3 cap and the reconciler design in our review pairs.</td></tr>
</tbody>
</table>

<p><strong>Distribution.</strong> 19 sources; 7 peer-reviewed papers, 4 postmortems or critical pieces, 7 vendor engineering blogs/docs, 1 critical blogpost (ICLR 2025). Dates: 17 within the last 18 months; the 2023 Du-et-al debate paper is included as the seminal anchor for the debate pattern family and is cited as such, not as recent evidence.</p>

<p><strong>Biases I tried to surface.</strong> Vendor blogs (Anthropic, Microsoft, LangChain, Cloudflare) consistently report uplift; postmortems (Cognition x2) report fragility. The honest synthesis is that team patterns help when the task is breadth-bound (research, broad follow-ups) or critique-bound (review, interpretation), and hurt when the task is depth-bound with one coherent context (implementation, narrow summaries). That synthesis is what the Recommendation column reflects.</p>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>
<p><strong>Artifacts.</strong></p>
<ul class="tight">
  <li><strong>Model / adapters:</strong> n/a (no training).</li>
  <li><strong>Training dataset:</strong> n/a.</li>
  <li><strong>Raw completions:</strong> n/a (no eval).</li>
  <li><strong>WandB run(s):</strong> n/a.</li>
  <li><strong>Eval JSON in repo:</strong> n/a.</li>
  <li><strong>Source inventory:</strong> agents enumerated from <code>.claude/agents/*.md</code> at git HEAD; runner jobs enumerated from <code>services/runner/src/jobs/*.ts</code> at git HEAD.</li>
  <li><strong>Survey sources:</strong> 19 URLs listed in the Survey notes block above; all public web pages.</li>
</ul>
<p><strong>Compute.</strong></p>
<ul class="tight">
  <li><strong>Wall time:</strong> ~45 min agent time (inventory + 19-source survey + drafting).</li>
  <li><strong>GPU:</strong> 0 (no RunPod pod launched).</li>
  <li><strong>Pod:</strong> n/a &mdash; ran as a Sagan-VM agent run only.</li>
  <li><strong>Estimated token cost:</strong> ~$10&ndash;$20 in Anthropic API charges (web search + drafting), well under the plan's $20 ceiling.</li>
</ul>
<p><strong>Code.</strong></p>
<ul class="tight">
  <li><strong>Entry scripts:</strong> n/a (no code shipped; this is a proposal).</li>
  <li><strong>Git commit:</strong> n/a (no code change attached to this experiment).</li>
  <li><strong>Hydra configs:</strong> n/a.</li>
  <li><strong>Reproduce:</strong> the survey can be re-run by querying the same 19 URLs and re-doing the agent inventory at HEAD; output will differ as the literature evolves.</li>
</ul>
<p><strong>Follow-up experiment shells (to be filed separately).</strong></p>
<ul class="tight">
  <li>Pilot 1 &mdash; parallel paper crawlers + synthesizer for <code>lit-review.ts</code>; A/B on five queued <code>project_lit_review</code> jobs; success = +1 unique high-quality source on average without &gt;2&times; token cost.</li>
  <li>Pilot 2 &mdash; broad + filter team for <code>follow-up-proposer</code>; success = &ge;1 net-new auto-runnable follow-up across 10 experiments.</li>
  <li>Pilot 3 &mdash; one critic round on <code>experiment-planner</code> before owner approval; success = fewer re-plan cycles on the next 10 plans.</li>
</ul>
</div>
</details>

</div>
