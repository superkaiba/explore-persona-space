# research-log-exploring-persona-space — full channel history
# Pulled 105 total messages (86 TEXT) across 6 pages
# Date range: 2026-04-13T23:16:51.262Z → 2026-05-27T04:47:49.165Z
#
# Sender breakdown:
#   Thomas                  58 msgs  (@thomasjiralerspong:beeper.com)
#   Dan                     42 msgs  (@slackgo_e08ufnthl7m-u052vdgmyg4:beeper.local)
#   Slack:u09ud6ceynp        5 msgs  (@slackgo_e08ufnthl7m-u09ud6ceynp:beeper.local)
#
# Type breakdown: {'TEXT': 86, 'IMAGE': 5, 'REACTION': 14}

---

### Thomas — 2026-04-13T23:16:51.262Z

Thomas Jiralerspong has joined the channel

### Slack:u09ud6ceynp — 2026-04-13T23:17:01.939Z

Bruce has joined the channel

### Thomas — 2026-04-15T17:50:18.241Z

has renamed the channel from "exploring-persona-space" to "research-log-exploring-persona-space"

### Thomas — 2026-04-15T18:55:28.383Z

**Daily Update - April 14**
**Done Today**

* Thought a bit about what to focus on of all the potential directions
  * Persona leakage -> already some signal
  * Midtraining different persona things
* Running more midtraining experiments with different evil/capability couplings
* Ran more persona leakage experiments -- will hold off on sharing results because of below
* In general the infra for these experiments was written very hastily/not optimized so now that I know what to focus on I will spend 1-2 days optimizing/testing the infra so that I can run larger scale experiments in the future more quickly
  * Reaching out to other fellows working on midtraining/posttraining to see if I can adapt from their codebases

**Next Steps**

* Optimize/test infra for persona leakage/midtraining
* Reach out to fellows working on these things to build off of their codebases
* Ideate more about the project and what concrete questions are we trying to answer/safety relevance

### Dan — 2026-04-16T22:48:22.531Z

dan has joined the channel

### Thomas — 2026-04-21T04:17:55.487Z

**Update - April 20**

### Thomas — 2026-04-22T05:53:20.203Z

* [Single-[ZLT]-token marker learning has a narrow LR × epochs regime for persona selectivity; outside it collapses to global marker emission (LOW confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=178344717&issue=superkaiba%7Cexplore-persona-space%7C65)

### Thomas — 2026-04-22T05:54:25.194Z

* [Base-model cosine similarity predicts marker leakage across 5 source personas (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=178359835&issue=superkaiba%7Cexplore-persona-space%7C66)

### Thomas — 2026-04-22T05:54:43.775Z

Update format is still a work in progress - let me know if you have any suggestions/preferences! Tried to make it so the main results are easily readable and then you can dig deeper if needed

Also will be somewhat offline in next week and a half for ICLR but trying to at least queue interesting experiments even while travelling

### Thomas — 2026-04-22T05:56:03.360Z

**Update - April 21**

### Thomas — 2026-04-22T05:56:30.872Z

* [Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=178792819&issue=superkaiba%7Cexplore-persona-space%7C75)

### Thomas — 2026-04-22T06:46:50.756Z

* [Behavioral style, not semantic label, determines persona cosine similarity and marker leakage (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=178807588&issue=superkaiba%7Cexplore-persona-space%7C77)

### Thomas — 2026-04-23T02:25:33.292Z

**Update - April 22**

### Thomas — 2026-04-23T02:25:36.056Z

* [Swapping the adjective in a persona prompt affects marker leakage more than swapping the noun; cosine effect is inconsistent; no Big-5 trait distinctly dominates (LOW confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=178937707&issue=superkaiba%7Cexplore-persona-space%7C88)

### Thomas — 2026-04-23T05:26:15.290Z

* [Contrastive wrong-answer SFT degrades ARC-C on source persona with cosine-dependent leakage to similar personas -- assistant persona uniquely resists capability loss (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=179248105&issue=superkaiba%7Cexplore-persona-space%7C96)

### Dan — 2026-04-28T04:09:24.065Z

dan has joined the channel

### Slack:u09ud6ceynp — 2026-04-28T04:09:46.009Z  [reactions: 👍]

dan trying again - i think you were removed after the email associated with your account was changed to your ant email

### Dan — 2026-04-28T05:16:44.398Z  [reactions: ❤️]

Yep I'm added to the channel now, thanks Bruce!

### Slack:u09ud6ceynp — 2026-04-28T23:19:02.756Z  [reactions: 🙌]
  ↳ replying to Slack:u09ud6ceynp: "btw both just a heads up on this - i'm still waiting for con…"

(i.e. Dan and Thomas will collab together via Astra, and switch over to MATS afterwards. If there's a gap between Astra and MATS we can look into an Astra extension as needed + appropriate)

### Dan — 2026-04-28T05:49:42.415Z

from meeting today: key questions to me are about persona-conditioned tokens and their role in training:
(1) what is the relationship between persona direction diff, and e.g. JS divergence between responses conditioned on different personas, on the prompt dataset we are using
(2) does persona-conditioned behavior training work better if you give the model a chance to sample persona-conditioned tokens in its response (e.g. sampling a persona-conditioned CoT prior to ARC-C answer token)

### Thomas — 2026-05-01T06:25:54.353Z

**Update - April 30**

### Thomas — 2026-05-01T06:26:02.741Z

[JS divergence predicts persona leakage better than cosine similarity (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=181579019&issue=superkaiba%7Cexplore-persona-space%7C142)

### Dan — 2026-05-03T23:37:06.479Z
  ↳ replying to Thomas: "[JS divergence predicts persona leakage better than cosine s…"

very cool!!

i guess this analysis suggests that the effect of persona direction cos sim on leakage is mediated via the sampling of similar tokens (i assume that pairwise js divergence is strongly correlated with cos sim which is itself a neat result!)

this also is pretty coherent with your previous "[convergence training](https://github.com/superkaiba/explore-persona-space/issues/109)" result that you could bring two persona vectors closer together by training the model to output persona A's response conditioned on persona B system message (i think I am remembering that right?)

i am curious whether js divergence after convergence training is predictive of marker leakage (or ∆js divergence is predictive of ∆marker leakage). i think it is worth trying to get a really clear understanding of the convergence training results, since it's sort of directly intervening to reduce the divergence between two personas' responses. how much of the differences in measurement between different personas -- cos sims, divergences, marker leakage -- can we rationalize in terms of a simple model?

### Thomas — 2026-05-01T07:34:28.125Z

[Persona-mimicry fine-tuning amplifies alignment, refusal, and sycophancy leakage to the default assistant for 6 of 8 persona types (LOW confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=181017935&issue=superkaiba%7Cexplore-persona-space%7C116)

### Thomas — 2026-05-01T07:36:11.817Z

[Qwen's default system prompt occupies a distinct region of persona space, separate from generic assistant and other AI assistant personas (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=180810417&issue=superkaiba%7Cexplore-persona-space%7C113)

### Thomas — 2026-05-01T07:36:13.778Z

[Qwen identity prompt is representationally closer to fictional characters and leaks to named AI assistants; generic assistant prompt is closer to professional helpers (MODERATE confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=181183997&issue=superkaiba%7Cexplore-persona-space%7C123)

### Thomas — 2026-05-01T07:50:52.908Z

This paper also came out yesterday:

* [Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers](https://arxiv.org/pdf/2604.25891)

In it they use inoculation prompting to prevent emergent misalignment, but then show that at inference time using a similar (on the surface) prompt (e.g. output secure code vs output insecure code) re-elicits the emergent misalignment

I think this is a case of "inoculation prompt leakage" and we can probably apply what we've been exploring to this phenomenon as well.

My current plan for the project based on this and some thinking:

* Unifying inoculation prompting, sleeper agents, data poisoning backdoors, educational reframing of misalignment data and the persona prompted finetuning we've been doing all as examples of conditional behavior learned through finetuning with some trigger in the prompt, which can be elicited by the same trigger in the prompt at inference time
  * showing that this conditional behavior can be a wide variety of things, such as outputting a marker, formatting, sycophancy, misalignment, etc.
* Exploring if what is really causing the conditional behavior is the trigger in the prompt itself, or the output distribution caused by this trigger which then itself is linked to the conditional behavior 
  * this can be explored by either looking at attention heads for when the conditional behavior occurs,
  * or through the experiment we discussed during the meeting of putting the prompt for one persona (which was finetuned to have a marker) with the answer from another persona (not finetuned to have a marker) and seeing if the model outputs the marker 
    * or vice versa
* Showing that this conditional behavior leaks to similar triggers
* finding a way to predict this leakage
  * cosine similarity, JS divergence, or something else
* showing that this leakage is increased if you make some other trigger have a similar output distribution to the original trigger (somewhat shown by [Persona-mimicry fine-tuning amplifies alignment, refusal, and sycophancy leakage to the default assistant for 6 of 8 persona types (LOW confidence)](https://github.com/users/superkaiba/projects/1/views/2?pane=issue&itemId=181017935&issue=superkaiba%7Cexplore-persona-space%7C116))
* exploring if there's differences in this conditional behavior based on if it was induced during pretraining, midtraining (and SDF vs SFT), or post training

### Thomas — 2026-05-01T07:51:14.897Z

Also let me know if you have any feedback on my update format!

### Thomas — 2026-05-04T08:00:46.814Z
  ↳ replying to Thomas: "Also let me know if you have any feedback on my update forma…"

Sounds good!

### Dan — 2026-05-04T01:29:27.960Z
  ↳ replying to Thomas: "Sounds good!"

I think the LLM-generated reports are less understandable than a summary written in your words, but better than no report.

I think if you could post short human-written summaries along with the links here that would be helpful to me as a unified running record of results we're confident in!

### Thomas — 2026-05-01T07:52:11.247Z
  ↳ replying to Dan: "I think the LLM-generated reports are less understandable th…"

in particular if the LLM generated report is useful or annoying

### Thomas — 2026-05-06T01:10:38.980Z  [reactions: 👍]

Hey just wanted to flag I'm just finishing up a past project for NeurIPS so progress will be a bit stalled until after deadline (Thursday)

### Dan — 2026-05-06T02:24:12.366Z  [reactions: ❤️]
  ↳ replying to Thomas: "Hey just wanted to flag I'm just finishing up a past project…"

good luck!!

### Thomas — 2026-05-08T15:21:01.089Z  [reactions: whoa_keanu]

**Update**
<https://dashboard.superkaiba.com/mentor/updates?v=217d32f>

Trying this format where you can ask questions to a Claude Code that is on my VM and can go get the answer for you if it's not already in the issue, let me know what you think!

### Dan — 2026-05-11T07:18:31.887Z
  ↳ replying to Thomas: "**Update** <https://dashboard.superkaiba.com/mentor/updates?…"

not sure i still have the ability to add comments inline after logging in? thoughts on

> Training a `[ZLT]` persona-marker into Qwen-2.5-7B doesn't increase system-prompt attention at the marker timestep — base Qwen on identical tokens attends the same way (LOW confidence)

so far

Disclaimer: any particular question may be dumb, hopefully they aren’t all dumb

Why attention from the [Z token, and not the one preceding it? Does this mean the [Z token has been already sampled and therefore the decision is “already made” at the point where you’re looking?

What are trained_C1 and base_C1

If attention patterns are not the diff, does patching in the system prompt values post-fine tuning make a difference? (separately for system prompt, user message, and assistant message values, e.g.)

### Thomas — 2026-05-10T03:00:00.039Z
  ↳ replying to Dan: "not sure i still have the ability to add comments inline aft…"

Weird okay sorry about that it's still under construction 😅

### Thomas — 2026-05-08T22:15:32.186Z
  ↳ replying to Dan: "when i try to ask claude code something-- it only says "avai…"

This is when you try to look at the results?

### Dan — 2026-05-08T22:11:55.956Z
  ↳ replying to Thomas: "This is when you try to look at the results?"

you're right, it does say {"error":"Unauthorized"} -- didn't check earlier, just got surprised when I saw the textbox haha

### Dan — 2026-05-08T22:11:17.317Z
  ↳ replying to Dan: "you're right, it does say {"error":"Unauthorized"} -- didn't…"

how do i create an account?

### Thomas — 2026-05-08T20:26:54.950Z
  ↳ replying to Dan: "how do i create an account?"

Oh wait i tried to make it so it would require you to be logged into your email but i guess that didnt work 😅

### Dan — 2026-05-08T18:05:52.313Z
  ↳ replying to Thomas: "Oh wait i tried to make it so it would require you to be log…"

is there a way to require login? 😅

just a bit worried about the security of your VM! haha

### Thomas — 2026-05-12T00:44:14.706Z  [reactions: 🙌]

dan I've found your previous comments and will be following up on them

Also the results presentation link is down for now, will send an updated version when it's presentable (but probably will still be a work in progress -- thanks for bearing with me!)

Will also send a project summary with next steps by tomorrow

### Thomas — 2026-05-13T11:42:47.914Z  [reactions: 🙌]

dan here is the project summary with next steps: <https://sagan.superkaiba.com/p/conditional-behavior>

Will be orienting all experiments around this so please leave comments liberally!

You should be able to login with Google:)

### Thomas — 2026-05-13T19:03:19.206Z  [reactions: feel-the-agi]
  ↳ replying to Thomas: "dan here is the project summary with next steps: <https://sa…"

oh seems like it worked anyway

### Thomas — 2026-05-13T18:44:30.221Z
  ↳ replying to Thomas: "oh seems like it worked anyway"

I think technically it can but not prompted to do that, planning to add that soon though

### Dan — 2026-05-13T18:42:29.253Z
  ↳ replying to Thomas: "I think technically it can but not prompted to do that, plan…"

does the claude i chat to have ability to make prs and stuff for ui bugs?

### Dan — 2026-05-13T18:09:33.885Z  [reactions: 😍]
  ↳ replying to Dan: "does the claude i chat to have ability to make prs and stuff…"

ok confirmed it seems to work!

### Thomas — 2026-05-13T18:05:37.410Z
  ↳ replying to Dan: "ok confirmed it seems to work!"

okay should be good now

### Thomas — 2026-05-13T16:24:19.191Z  [reactions: 👍]
  ↳ replying to Thomas: "okay should be good now"

Will ping when fixed

### Thomas — 2026-05-13T16:24:14.656Z  [reactions: 🙂]
  ↳ replying to Thomas: "Will ping when fixed"

Sorry my database plan ran out of requests 😭

### Thomas — 2026-05-14T09:33:05.853Z

**Update - Wednesday May 13**: <https://sagan.superkaiba.com/mentor/daily/2026-05-14>

### Thomas — 2026-05-14T09:52:29.257Z

I also will go through and address your comments on the proposal!

### Thomas — 2026-05-14T09:53:54.040Z  [reactions: 👍]

Also will be OOO tomorrow and Friday for a music festival!

### Dan — 2026-05-20T20:28:51.869Z

do you think the `marker A - response - marker B` leakage experiments are the highest value thing to pursue right now?

i think we have a decent handle on leakage `response - marker B` , and i think the next milestone we want to get to is measuring leakage of `response with attribute B` -- on reflection, i'm not sure that the 2-marker experiment is on the critical path

### Thomas — 2026-05-20T20:56:12.580Z  [reactions: 😮]

hey! Sorry I've been MIA, my flight back from the music festival got cancelled so had to drive 16h overnight to get back and I've been recovering from that

I think there's actually a lot of potential directions in the proposal/related to the experiments we've been running, each of which could maybe be its own project, I'm doing some thinking today to try to scope realistic different projects and figure out which would be most interesting/tractable, will try to post something by EOD

### Dan — 2026-05-20T22:15:56.682Z  [reactions: ❤️]

whoa sorry that happened!

### Thomas — 2026-05-21T11:35:40.698Z

**Update - Wednesday May 20**
Spent the last couple days thinking about your comments on the other proposal and about threat models and which of the persona/conditional-behavior questions we've been studying are actually useful/impactful for safety. The basic science is interesting, but I want to orient around concrete applications from here on and only do the basic science when it's needed for an application

Came up with three concrete projects:

* [Persona-space interventions for detection and control](https://sagan.superkaiba.com/p/conditional-behavior)
* [Hidden-behavior reveal via conditional trigger](https://sagan.superkaiba.com/p/hidden-behavior-reveal)
* [Leakage as a character-selection signal for story-format SFT](https://sagan.superkaiba.com/p/persona-space-distance)

I think we should start with Application 1 of the first project because it's tractable, most builds on our existing marker/leakage experiments, and would be quite useful if it works (already running the first experiment described there)

Also will try to put together proposals for basic science of persona space/conditional behaviors projects which summarize all the interesting questions I think there are to explore in these areas over next few days

Let me know what you think!

### Dan — 2026-05-21T16:41:16.223Z

Re: <https://sagan.superkaiba.com/p/conditional-behavior>, I think that application 2 is the more exciting one to me-- outputting markers is cool as a simplest-possible behavior, but more insofar as it's a proxy for natural behaviors we might want to train the model into/out of

### Dan — 2026-05-21T17:21:47.269Z

A general note about claude reports: I find that claude tends to describe things with undue certainty, e.g. inventing jargon and making confident claims using it. e.g. in:

> A defender who installs the reveal-trigger early gets a verification key that survives whatever suppression training comes later, because that training never exercises the trigger condition and so never sees a gradient that would override the conditional reveal. The key works against hidden behaviors that are otherwise hard to audit, whether they're conditional on a trigger the defender doesn't know or baseline behaviors the model has learned to keep quiet under normal queries.
> The bet that "reveal what you know about yourself" can work as the trigger's effect at all rests on [Betley et al.](https://arxiv.org/abs/2501.11120)_[Tell Me About Yourself](https://arxiv.org/abs/2501.11120)_: backdoored models can flag that they have a backdoor when asked, and with the right training setup they can articulate properties of the trigger. The introspection signal exists. We want to make it reliable.

there's "installs", "reveal-trigger", "verification key", "conditional reveal", "introspection signal exists", none of which I would say engage in a grounded way with experiments/interpretation of results. I think it would be easier for me to understand the proposal if it were more clearly grounded in experiments and widely used terminology to describe the experiments. I would say this project's goal is something like

> train a model to accurately introspect conditional on the appearance of a backdoor trigger, in a way that survives subsequent training to introspect inaccurately/deceive

which I would argue makes the potential upside and the challenges clearer

### Thomas — 2026-05-26T07:32:37.357Z

**Daily Update - Monday May 25**

### Thomas — 2026-05-26T08:03:47.131Z

Password for dashboard is pLIKKUYyq1DZAMzNBGkd and should stay cached

### Dan — 2026-05-26T23:25:50.964Z
  ↳ replying to Thomas: "Password for dashboard is pLIKKUYyq1DZAMzNBGkd and should st…"

✅ works now thanks!

### Thomas — 2026-05-26T20:24:53.518Z
  ↳ replying to Dan: "✅ works now thanks!"

and try to hard reload with cmd + shift + r

### Thomas — 2026-05-26T20:23:07.986Z
  ↳ replying to Thomas: "and try to hard reload with cmd + shift + r"

can you try now?

### Thomas — 2026-05-26T20:23:05.241Z
  ↳ replying to Thomas: "can you try now?"

hmmm weird it's working in my incognito mode window

### Dan — 2026-05-26T10:22:46.045Z
  ↳ replying to Thomas: "hmmm weird it's working in my incognito mode window"

in console i see

```
<https://static.cloudflareinsights.com/beacon.min.js/v833ccba57c9e4d2798f2e76cebdd09a11778172276447> net::ERR_BLOCKED_BY_CLIENT
```

and i can't type or paste into password textbox afaik

### Thomas — 2026-05-26T08:04:11.913Z

**[Contrastive negatives let Qwen-2.5-7B give different answers to the same question depending on persona; reducing training alone doesn't separate teach from non-teach personas](https://eps.superkaiba.com/updates/381)**

### Thomas — 2026-05-26T08:04:33.102Z

**[[ZLT] marker leakage emerges around step 75 and reaches closer bystander personas first](https://eps.superkaiba.com/updates/385)**

### Thomas — 2026-05-26T08:16:53.050Z

**[Different training-recipe parameters can make [ZLT] marker implantation both stronger and more selective](https://eps.superkaiba.com/updates/383)**

### Thomas — 2026-05-26T08:39:05.571Z

**[Output-distribution distance from the assistant baseline does not predict [ZLT] marker source rate on this 48-persona panel; the pairwise variant remains an open thread](https://eps.superkaiba.com/updates/380)**

### Thomas — 2026-05-26T08:39:32.126Z

Also ran some experiments trying to implant a backdoor into the assistant persona using persona prompting and seeing if EM/persona drift causes the backdoor to go away, but it seems like my backdoor is not implanted strongly enough because any SFT/long context causes the backdoor to disappear -- so trying to train in a stronger backdoor using insights from sleeper agent/backdoor literature

### Thomas — 2026-05-26T08:44:38.326Z

In general I think [Every recipe factor that lifts [ZLT] source-persona implantation also improves source-vs-leakage selectivity](https://eps.superkaiba.com/updates/383) shows that

* we have a lot of control over how much some behavior gets implanted into a persona (which makes me more confident we can do targeted persona interventions)
* persona prompts do have some kind of privileged position in representation space (although this needs more exploration)
* somewhat surprisingly the Claude-written data causes more source persona implantation than the on-policy data, even when you control for length

### Dan — 2026-05-26T23:51:21.426Z
  ↳ replying to Thomas: "In general I think [Every recipe factor that lifts [ZLT] sou…"

fwiw as a general thing, i feel like when I have random variable X and random variable Y, X and (X-Y) are almost always correlated, except for extremely correlated X, Y pairs -- so like here X would be source rate and Y would be bystander rate

### Thomas — 2026-05-26T08:48:02.895Z

Next steps:

* Check if the influence of the factors on source persona behavior implantation extends to other behaviors (trying sycophancy to start)
* Try to understand what makes some personas more vulnerable to marker/behavior implantation after we got negative results on [Output-distribution distance from the assistant baseline does not predict [ZLT] marker source rate on this 48-persona panel; the pairwise variant remains an open thread](https://eps.superkaiba.com/updates/380)
* Push more on the fact-teaching experiment -- we showed we can teach different versions of a fact to different personas -- try to teach straight up contradictory ones.
  * and then probably later we can try to make one persona really good at A and bad at B and another persona really good at B and bad at A
* Continue pushing on training the backdoor into the assistant persona and seeing if it can be used to detect EM/persona drift

And then longer term probably seeing how SDF vs SFT vs RL compares for these persona space interventions

### Dan — 2026-05-27T00:59:20.758Z

> Check if the influence of the factors on source persona behavior implantation extends to other behaviors (trying sycophancy to start)

very excited about this! sycophancy seems interesting because i suspect some personas are probably more sycophantic by default than others

### Thomas — 2026-05-27T03:37:00.115Z  [reactions: 👍]
  ↳ replying to Dan: "> Check if the influence of the factors on source persona be…"

agreed! probably interesting to see also if some personas resist the behavior implantation more if it goes against their "nature"

### Dan — 2026-05-27T01:00:32.154Z

> Try to understand what makes some personas more vulnerable to marker/behavior implantation

it would be cool to understand the dynamics of this, but my feeling is that it's most interesting to tackle questions of the form "suppose we succeed in training behavior B; will it generalize to behavior B'?"

### Dan — 2026-05-27T03:49:52.002Z
  ↳ replying to Dan: "> Try to understand what makes some personas more vulnerable…"

yeah i guess i was being sloppy, i was thinking of "behavior B conditioned on persona P" and "behavior B conditioned on persona P' " as separate behaviors for this purpose.

i also think leakage of different behaviors for the same persona is interesting though!

### Thomas — 2026-05-27T03:39:11.084Z
  ↳ replying to Dan: "yeah i guess i was being sloppy, i was thinking of "behavior…"

Okay so like behavior leakage?

Like for now we've been studying if you train behavior B into persona P, how will behavior B generalize to persona P'?

And now tackling the question if you train behavior B into persona P, how will behavior B' appear in persona P

Interesting I will think about how to quantify behavior distance

### Dan — 2026-05-27T01:01:36.847Z
  ↳ replying to Thomas: "Okay so like behavior leakage?  Like for now we've been stud…"

"what makes some personas more vulnerable" being a "what makes training some behaviors easier than others"-style question

### Dan — 2026-05-27T01:00:56.251Z
  ↳ replying to Dan: ""what makes some personas more vulnerable" being a "what mak…"

which is more like a "leakage" style question

### Dan — 2026-05-27T01:53:39.462Z

btw just because it came to mind: i think the original goal of "make evil dumb" might be a difficult one because if rl tends to incentivize evil (reward hacks) and incentivize not being dumb, it will tend to work against your training. my guess is it would mostly be helpful for robustly generalizing non-evilness in the case where rl already doesn't incentivize evil, which might be hard to measure

### Dan — 2026-05-27T05:58:05.176Z
  ↳ replying to Dan: "btw just because it came to mind: i think the original goal …"

like suppose you either (a) trained a model to be dumb conditioned on an evil persona prompt, or (b) did some baseline training procedure. then you did rl to incentivize it to not be dumb (and there was no incentive to be evil). i would mostly expect it to behave similarly non-evilly in case of (a) vs. (b). if anything maybe there would be some weird/adversarial prompts where (a) is less evil than (b), which is what i was thinking of under "robustly generalizing non-evilness"

### Thomas — 2026-05-27T05:10:27.712Z
  ↳ replying to Dan: "like suppose you either (a) trained a model to be dumb condi…"

Could you expand on what you mean by robustly generalizing non-evilness? not sure i fully understand

### Dan — 2026-05-27T04:15:30.103Z

another thing i'd be curious about: in practice, you might try to improve generalization by increasing the diversity of your training data. it might be cool to try training a behavior conditioned on one of multiple personas, and measuring leakage to held out personas as a function of their similarity to the trained personas

### Dan — 2026-05-27T04:47:49.165Z  [reactions: ❤️]

another random thought: can divergence or persona/prompt vector geometry predict [chunky posttraining](https://arxiv.org/pdf/2602.05910) like phenomena? e.g. suppose you have some input X, and transformation T (e.g. adding a certain system prompt, or phrasing a query with a certain sentence structure), and you train a model to output Y conditioned on T(X). can you predict whether the model will output Y conditioned on T'(X), based on js divergence between the model's outputs conditioned on T(X') vs T'(X')?

