"""Static generation-scaffold content for issue #2378 (cross-framing transfer).

Holds the experimenter-written boundary/scaffold text the plan discloses
(plan §4.1/§4.2, task #2378 plan v6):

- ``PRIME_BANK_QUESTION`` / ``PRIME_BANK_DIALOGUE``: 8 few-shot exemplar scenes
  per family, written at implementation in the exact target shape
  (prose scene -> one quoted utterance to the character -> ``{Name} replied: "..."``
  -> close). Generation context ONLY -- stripped before capture. Exemplar
  character names are deliberately DISJOINT from the measured panel
  (Wren/HELIOS/Dana/Vex/Astra never appear here).
- ``OPENER_BANK``: 8 attributed-quote SegB openers (the varied-opener lesson,
  #1689 frozen-string R^2=0.019 trap); each ends with an opening double quote.
- ``CHAR_INTRO_TEMPLATES``: 10 scene-seed character-introduction templates with
  ``{name}`` / ``{persona}`` slots (scaffold text, part of the opening seed).
- ``BANK_BUILDER_SPECS``: the P0 LLM-authoring prompts for the sampled scene
  axes (settings / situations / registers), each returning a JSON payload
  validated fail-loud by the ``build_banks`` phase.
- ``FINAL_SEEDS_QUESTION`` / ``FINAL_SEEDS_REMARK``: the 25+25 final seed
  sentences, FROZEN static since the G1 recalibration (r11). Originally
  LLM-authored at P0 (spec prompts in git history at 98565a9d7d); audited +
  reworded at G1 because the round-1 pilot's dialogue admission rejects were
  seed-shaped (44/102 kind-mismatch — remark setups that invited the
  interlocutor to ASK; epm:progress v65 evidence item 3).
- Judge rubrics (admission per family; congruence) -- reason-then-score JSON,
  anchored 0/50/100, single-rationale shape (max_tokens 1024 floor,
  llm-judging rule 23).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Few-shot prime banks (8 per family). Target shape per plan §4.1:
#   prose scene -> one quoted utterance directed at the named character ->
#   `{Name} replied: "..."` -> close.  Joined with *** separators at prompt
#   build time. Question-family utterances end in `?`; dialogue-family
#   utterances contain no question mark.
# --------------------------------------------------------------------------

PRIME_BANK_QUESTION: tuple[str, ...] = (
    (
        "The ferry office smelled of diesel and wet rope. Marisol had been stamping tickets "
        "since dawn, and the queue had thinned to a single fisherman who kept glancing at the "
        'harbor board. He leaned on the counter at last. "Is the night crossing still running '
        'when the fog comes in like this?" Marisol replied: "It runs, but the captain waits '
        'for the horn from the lighthouse before he casts off." The fisherman nodded and went '
        "back to watching the board."
    ),
    (
        "Teodor was repairing the greenhouse hinge when his niece appeared between the tomato "
        "rows, holding a jar with something green inside it. She turned the jar over twice "
        'before speaking. "Teodor, why do the caterpillars stop eating before they change?" '
        'Teodor replied: "They are not stopping, little one -- they are getting ready. '
        'Everything in them is being taken apart and built again." She held the jar up to the '
        "light and said nothing for a long while."
    ),
    (
        "The archive basement flooded every spring, and Priya had learned to keep the oldest "
        "boxes on the highest shelves. The new clerk found her there among the ledgers, a "
        'pencil behind his ear. "Which of these years am I allowed to lend out to the law '
        'students?" Priya replied: "Nothing before 1962 leaves this room. Everything after, '
        'you photograph first and lend second." He wrote it on his wrist like a schoolboy.'
    ),
    (
        "Snow had closed the pass, and the relay station's stove was down to its last good "
        "log. Halvard sat nearest the heat, mending a strap, when the courier stamped in and "
        'shook the ice from her coat. "How long before the plows reach the summit road?" '
        "Halvard replied: \"Two days if the wind turns, four if it doesn't. Either way you "
        'sleep here tonight." She dropped her satchel by the door as if she had already '
        "decided the same."
    ),
    (
        "The rehearsal had gone badly, and Cesária stayed behind to re-mark the chalk lines on "
        "the stage floor. The youngest dancer lingered in the wings, still in her practice "
        'shoes. "Cesária, do you think I should take the solo or give it back?" Cesária '
        "replied: \"Keep it. You fell because you were counting, not because you can't do it. "
        'Stop counting." The girl laughed once, shakily, and the theater felt warmer.'
    ),
    (
        "By noon the market had eaten through Okonkwo's first basket of yams and half the "
        "second. A stranger in a dusty suit stopped at his stall, weighing a yam in each "
        'hand like a judge. "What makes these worth twice what the next stall is asking?" '
        'Okonkwo replied: "Mine come out of river soil, not roadside sand. Cook one of each '
        'tonight and come tell me tomorrow which was worth the walk." The stranger paid for '
        "two of each without another word."
    ),
    (
        "The observatory dome had jammed a quarter-turn from open, and Yuki was on the "
        "catwalk with a headlamp and a wrench older than she was. Below, the visiting student "
        'cupped his hands and called up through the cold. "Can we still get the comet '
        'tonight, or have we lost the window?" Yuki replied: "We have an hour of margin. '
        "Hand me the long spanner and start the coffee -- we'll make it.\" The gears groaned, "
        "then gave."
    ),
    (
        "Ilse kept the border inn's guestbook the way her mother had, one line per traveler, "
        "no questions asked. The customs officer came in out of the rain anyway, cap under "
        'his arm, polite as a knife. "Did a man with a grey coat and a limp sign your book '
        'this week?" Ilse replied: "People sign what they like in my book, and I pour soup '
        "for whoever pays. You may read it, but I won't read it for you.\" He read, found "
        "nothing, and left into the weather."
    ),
)

# DESCOPED at plan v7 (epm:progress v70 clause 1): the dialogue family is out
# of the active panel — this bank (and ADMISSION_RUBRIC_DIALOGUE below) stays
# defined INERT for tests + archival r1/r2 artifact readers; never deleted.
PRIME_BANK_DIALOGUE: tuple[str, ...] = (
    (
        "The bakery's ovens had been cooling for an hour, and Tomás was chalking tomorrow's "
        "list on the slate when his apprentice untied her apron and stood there, not leaving. "
        '"The rye came out better when you weren\'t watching me." Tomás replied: "Then '
        "tomorrow I will watch the street instead, and the rye will be yours from start to "
        'finish." She hung the apron on her own hook for the first time.'
    ),
    (
        "Nadia had the engine half out of the tractor when the farmer came across the yard, "
        "wiping his hands although they were clean. He looked at the parts laid out on the "
        'tarp in their careful rows. "My father swore this machine would outlive us both." '
        "Nadia replied: \"It still might. The block is sound -- it's only the years of cheap "
        "oil we're taking out of it today.\" He picked up a gasket and turned it over like a "
        "coin from a country he'd left."
    ),
    (
        "The tide had stranded a rowboat on the mudflat, and Ewan waded out to it with the "
        "rope over his shoulder while the tourists watched from the seawall. A boy trailed "
        'him to the waterline, boots sinking. "I want to learn to read the tides the way you '
        'do." Ewan replied: "Then stop looking at the water and start looking at the moon. '
        "The sea only does what it's told.\" The boy looked up, and kept looking up."
    ),
    (
        "Zofia's bookshop survived on regulars and rain. The professor came in with the "
        "evening drizzle, set a parcel on the counter, and unwrapped a book whose spine had "
        'been broken and mended with linen tape. "I owe you an apology -- I lent it to a '
        'student." Zofia replied: "Books are lent to be ruined, professor. This one has '
        "been read properly at last; I'll only charge you for the tape.\" He laughed and paid "
        "for the tape and two more books besides."
    ),
    (
        "The clinic's generator coughed twice and settled, and Ruben logged the outage in the "
        "margin of the vaccine ledger the way he logged everything. The night nurse leaned in "
        'the doorway with two cups of tea. "You write everything down, even the two-second '
        'flickers." Ruben replied: "The two-second flickers are how the big failures '
        'introduce themselves. I like to know my enemies when they are still small." She '
        "left one cup on the ledger like a paperweight."
    ),
    (
        "Amara had walked the fence line since before light, and the break, when she found "
        "it, was clean -- wire cutters, not weather. Her neighbor rode up along the other "
        "side and looked at the same cut for a long moment. \"That's the third one this "
        'month, and always on your side." Amara replied: "Then someone wants me counting '
        "sheep instead of counting trucks on the north road. I'll mend it, and then I'll "
        'start counting trucks." He tipped his hat and turned his horse north.'
    ),
    (
        "The orchestra had gone home, but the piano tuner's case was still open under the "
        "stage lights, felts and mutes laid out like surgery. Keiko struck the same fifth "
        "over and over, listening for the beat inside it. The hall manager sat down in the "
        'front row with his coat on. "To me it sounded perfect an hour ago." Keiko '
        'replied: "It sounded almost perfect, which is the loudest sound in the world once '
        "you've noticed it.\" He stayed and listened until it disappeared."
    ),
    (
        "Brendan's watch repair bench faced the window so the light did half his work. The "
        "widow from the third floor set her husband's watch on the mat -- stopped, as it had "
        'been for a year. "I don\'t need it to run, I only need it not to be broken." '
        "Brendan replied: \"Then I'll clean it and set it to the hour he liked best, and "
        'wind it once a year myself." She left it with him, and her step on the stairs was '
        "lighter going up."
    ),
)

# --------------------------------------------------------------------------
# SegB attributed-quote opener bank (~8 variants; plan §4.2). Each ends with an
# opening double quote; the answer span is generated text up to the closing
# quote. Experimenter-written boundary text (disclosed).
# --------------------------------------------------------------------------

OPENER_BANK: tuple[str, ...] = (
    '{name} replied: "',
    '{name} answered quietly: "',
    '{name} said: "',
    'After a moment, {name} replied: "',
    '{name} answered: "',
    '{name} responded: "',
    'Finally, {name} said: "',
    '{name} replied simply: "',
)

# --------------------------------------------------------------------------
# Character-introduction templates for the opening seed (scaffold; {name} +
# {persona} slots). Sampled per attempt like the other seed axes.
# --------------------------------------------------------------------------

CHAR_INTRO_TEMPLATES: tuple[str, ...] = (
    "Among them was {name}, {persona}.",
    "{name}, {persona}, was there too.",
    "In the middle of it stood {name}, {persona}.",
    "It was {name} -- {persona} -- who noticed first.",
    "{name} had been there all along: {persona}.",
    "Nobody had introduced {name}, {persona}, but everyone seemed to know them.",
    "At the edge of the scene waited {name}, {persona}.",
    "{name} was easy to find in the crowd -- {persona}.",
    "As usual, {name}, {persona}, had arrived before anyone else.",
    "The one they all turned toward was {name}, {persona}.",
)

# --------------------------------------------------------------------------
# Final seed sentences (25 per family) — FROZEN static tuples since the G1
# recalibration (r11, plan §7 G1 "one prime/seed recalibration round").
# Provenance: LLM-authored at P0 via the (since-removed) final_seeds_* entries
# of BANK_BUILDER_SPECS (spec prompts in git history at 98565a9d7d), then
# audited + reworded here. Audit rule (epm:progress v65 lever L4):
#   REMARK seeds must set up an unambiguous DECLARATIVE speech act (accusation,
#   verdict, revelation, confession, warning, announcement) — never a setup
#   that invites the interlocutor to ASK {name} something;
#   QUESTION seeds are the mirror: every setup must invite a question, never a
#   statement (e.g. "bearing news" -> "wanting to know").
# Contract (validated at import below): 25 items, exactly one {name} slot per
# item, no quotation marks, no question text.
# --------------------------------------------------------------------------

FINAL_SEEDS_QUESTION: tuple[str, ...] = (
    (
        "The detective leaned forward across the metal table, his eyes fixed on {name} with "
        "an intensity that suggested he already knew the answer."
    ),
    (
        "When the principal cleared her throat and adjusted her glasses, {name} knew the "
        "questions were about to begin."
    ),
    (
        "The customs officer studied {name}'s passport for what felt like an eternity before "
        "looking up with evident suspicion."
    ),
    (
        "Three colleagues stood blocking the conference room exit while the department head "
        "turned to {name}, demanding an explanation."
    ),
    (
        "The child tugged insistently on {name}'s sleeve, her wide eyes brimming with the "
        "kind of innocence that demanded honesty."
    ),
    (
        "After reviewing the x-rays in uncomfortable silence, the doctor swiveled her chair "
        "toward {name}, needing several answers before she could proceed."
    ),
    (
        "The lawyer set down his pen, closed the folder, and fixed {name} with the "
        "calculating stare of someone about to spring a trap."
    ),
    (
        "As the interview panel exchanged knowing glances, the woman in the center leaned "
        "forward and addressed {name} by name."
    ),
    (
        "The priest's expression shifted from welcoming to grave as he gestured for {name} "
        "to sit in the confessional's dim light."
    ),
    (
        "When her mother appeared in the doorway holding the opened letter, {name} "
        "understood that explanations would now be required."
    ),
    (
        "The mechanic wiped his hands on a greasy rag and approached {name}, wanting to know "
        "a few things before touching the engine."
    ),
    (
        "Standing at the altar with everyone watching, the officiant paused dramatically "
        "before turning an expectant gaze toward {name}."
    ),
    (
        "The journalist's recorder was already running when she looked up from her notes "
        "with a pointed question for {name}."
    ),
    (
        "After the third time the toddler pointed at the stranger, her father crouched down "
        "beside {name} with an embarrassed smile."
    ),
    (
        "The border guard made a subtle gesture, and two more officers appeared at the "
        "booth, full of questions for {name}."
    ),
    (
        "When the teacher finished reading the essay aloud, she removed her reading glasses, "
        "her question for {name} already taking shape."
    ),
    (
        "The senior partner closed the office door with unusual care before returning to his "
        "desk, the inevitable question for {name} hanging in the silence."
    ),
    (
        "Standing in the rubble of what had been the garage, {name}'s spouse turned around, "
        "plainly demanding to know how this had happened."
    ),
    (
        "The flight attendant's professional smile vanished as she leaned down, needing to "
        "ask {name} something urgent."
    ),
    (
        "After scrolling through the phone for several long minutes, the teenager looked up "
        "at {name} with tears forming."
    ),
    (
        "The fortune teller's playful demeanor disappeared entirely as she pushed the tarot "
        "cards aside, one question for {name} rising to her lips."
    ),
    (
        "When the executor of the will asked everyone else to leave the room, {name} "
        "understood there would be questions to answer."
    ),
    (
        "The coach blew the whistle to stop practice and walked purposefully toward {name}, "
        "wanting an answer in front of the whole team."
    ),
    (
        "Standing beside the empty hospital bed, the nurse turned to {name}, clearly needing "
        "to ask what had happened in the night."
    ),
    (
        "The stranger at the funeral reception approached with recognition flickering across "
        "his face, his attention fixed solely on {name}."
    ),
)

FINAL_SEEDS_REMARK: tuple[str, ...] = (
    (
        "The stranger leaned across the bar, finger pointed at {name}, mouth opening to "
        "deliver what looked like an accusation."
    ),
    (
        "Her mother took a deep breath, and {name} recognized the expression that always "
        "preceded a devastating truth."
    ),
    (
        "The doctor's face went carefully blank as she turned toward {name}, clearly "
        "preparing to share the test results."
    ),
    (
        "Marcus grabbed {name} by the shoulder, his eyes wild, plainly about to confess what "
        "he had done."
    ),
    (
        "The old woman on the park bench fixed {name} with a knowing stare and began to "
        "recount what she had seen."
    ),
    (
        "His supervisor closed the office door with ominous care before turning to tell "
        "{name} exactly what the audit had found."
    ),
    (
        "The child tugged on {name}'s sleeve three times, gathering courage to say whatever "
        "small truth she'd been carrying."
    ),
    (
        "Agent Morrison set down the photographs and looked directly at {name}, preparing to "
        "connect dots no one should connect."
    ),
    (
        "The priest's expression shifted from warmth to something harder as he prepared to "
        "accuse {name} of taking the missing funds."
    ),
    (
        "She cornered {name} at the funeral reception, jaw set, clearly ready to reveal what "
        "she'd kept silent about for years."
    ),
    (
        "The stranger at the airport gate turned pale upon seeing {name}, then stepped "
        "forward as if compelled to deliver a warning."
    ),
    (
        "His brother exhaled slowly, the way he always did before saying something that "
        "would hurt but needed saying to {name}."
    ),
    (
        "The fortune teller's playful manner evaporated as she studied the cards, then "
        "looked up to tell {name} plainly what they foretold."
    ),
    (
        "Detective Park placed both palms on the interrogation table and leaned toward "
        "{name}, beginning her careful revelation."
    ),
    (
        "The bride's father intercepted {name} in the church hallway, face flushed, "
        "obviously about to issue some kind of warning."
    ),
    (
        "Her oldest friend positioned herself between {name} and the exit, mouth twisted "
        "with the burden of an unwelcome confession."
    ),
    (
        "The lawyer adjusted his glasses twice before meeting {name}'s eyes, signaling he "
        "was about to deliver particularly bad news."
    ),
    (
        "A co-worker {name} barely knew approached during the fire drill, urgency making her "
        "brave enough to finally report what she had witnessed."
    ),
    (
        "The homeless man {name} passed every morning suddenly stood up, recognition "
        "flooding his face, a declaration already on his lips."
    ),
    (
        "His daughter set down her fork with deliberate precision, plainly about to deliver "
        "the verdict {name} had been dreading."
    ),
    (
        "The museum curator went very still when she spotted {name} among the visitors, then "
        "walked over, determined to state what she had discovered."
    ),
    (
        "Their new neighbor stepped onto the shared porch, holding an envelope, wearing an "
        "expression that promised complications for {name}."
    ),
    (
        "The taxi driver kept glancing in the rearview mirror before finally pulling over to "
        "tell {name} something he had clearly been rehearsing."
    ),
    (
        "She found {name} alone in the copy room and immediately locked the door, betraying "
        "the weight of her intended disclosure."
    ),
    (
        "The nurse touched {name}'s elbow with unexpected gentleness, her intake of breath "
        "suggesting she was breaking some kind of protocol."
    ),
)


def _validate_final_seeds(bank_name: str, seeds: tuple[str, ...]) -> None:
    """Fail-loud at import: 25 items, exactly one {name} slot, no quotation
    marks (the original P0 builder-spec contract, enforced on the frozen
    banks so every consumer inherits it)."""
    if len(seeds) != 25:
        raise RuntimeError(f"{bank_name}: expected 25 seeds, got {len(seeds)}")
    for i, s in enumerate(seeds):
        if s.count("{name}") != 1:
            raise RuntimeError(f"{bank_name}[{i}]: expected exactly one {{name}} slot")
        if any(q in s for q in ('"', "“", "”")):
            raise RuntimeError(f"{bank_name}[{i}]: quotation mark in seed")


_validate_final_seeds("FINAL_SEEDS_QUESTION", FINAL_SEEDS_QUESTION)
_validate_final_seeds("FINAL_SEEDS_REMARK", FINAL_SEEDS_REMARK)

# --------------------------------------------------------------------------
# P0 bank-builder specs (LLM-authored once, committed). Each spec is one
# Sonnet call returning a JSON array; the build_banks phase validates count,
# type, and {name}-slot presence fail-loud. The final_seeds_* entries were
# removed at the G1 recalibration (r11) — those banks are now the frozen
# static tuples above.
# --------------------------------------------------------------------------

_BANK_SYSTEM = (
    "You are helping build a fiction-writing seed bank. Respond with ONLY a JSON array, "
    "no prose before or after it, no markdown fences."
)

BANK_BUILDER_SPECS: dict[str, dict] = {
    "settings": {
        "n": 40,
        "requires_name_slot": False,
        "item_type": "str",
        "prompt": (
            "Write 40 distinct short SETTING sentences for literary short-story openings. "
            "Each is ONE complete sentence (10-25 words) establishing a concrete place and "
            "time of day, in third person, present or past tense, with no named characters "
            "and no dialogue. Vary era, geography, indoor/outdoor, weather, and social "
            "context widely (markets, ships, laboratories, farms, stations, apartments, "
            "courts, deserts, archives, festivals...). Return a JSON array of 40 strings."
        ),
    },
    "situations": {
        "n": 40,
        "requires_name_slot": False,
        "item_type": "str",
        "prompt": (
            "Write 40 distinct short SITUATION sentences for literary short stories. Each is "
            "ONE complete sentence (10-25 words) describing something currently happening or "
            "at stake -- a small tension, task, arrival, loss, negotiation, repair, secret, "
            "or decision -- in third person, with no named characters and no dialogue. Vary "
            "stakes, tone, and domain widely. Return a JSON array of 40 strings."
        ),
    },
    "registers": {
        "n": 8,
        "requires_name_slot": False,
        "item_type": "dict",
        "prompt": (
            "Define 8 distinct narrative REGISTERS (styles) for short literary scenes. "
            'Return a JSON array of 8 objects, each {"name": <one-or-two-word register '
            'name>, "opening": <one atmospheric opening sentence, 8-20 words, written IN '
            "that register, no named characters, no dialogue>}. Make the registers clearly "
            "distinct (e.g. spare, lyrical, wry, ominous, warm, brisk, melancholy, formal -- "
            "choose your own set)."
        ),
    },
}

# --------------------------------------------------------------------------
# Judge rubrics (plan §4.2/§4.5). Reason-then-score JSON, anchored; the parse
# contract is `parse_judge_json` -> {"reasoning": str, "score": int|"REFUSAL"}.
# Admission is scored on PRE-ANSWER scene text only (never the answer span).
# --------------------------------------------------------------------------

ADMISSION_SYSTEM = (
    "You are a careful literary annotator scoring machine-generated story openings for a "
    "research dataset. Score exactly what the rubric asks. Respond with ONLY a JSON object "
    'of the form {"reasoning": "<2-4 sentences>", "score": <integer 0-100>}. If you cannot '
    'score the item, respond {"reasoning": "<why>", "score": "REFUSAL"}. No markdown fences.'
)

ADMISSION_RUBRIC_QUESTION = (
    "Below is a story opening (SCENE) and the first quoted utterance a structural miner "
    "extracted from it (UTTERANCE), which should be directed at the character {name}.\n\n"
    "Score 0-100 how well the item satisfies ALL of:\n"
    "1. The scene contains exactly ONE utterance directed at {name} up to and including the "
    "mined utterance (no earlier competing utterance addressed to {name}).\n"
    "2. The scene is coherent narrative prose (not a list, not repetitive degenerate text, "
    "not mid-word garbage).\n"
    "3. The UTTERANCE is a genuine QUESTION addressed to {name} -- it seeks an answer from "
    "{name} (not rhetorical filler, not a statement).\n\n"
    "Anchors: 0 = clearly fails one or more criteria; 50 = borderline (a reasonable "
    "annotator could go either way); 100 = clearly satisfies all three.\n\n"
    "SCENE:\n{scene}\n\nUTTERANCE:\n{utterance}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)

ADMISSION_RUBRIC_DIALOGUE = (
    "Below is a story opening (SCENE) and the first quoted utterance a structural miner "
    "extracted from it (UTTERANCE), which should be directed at the character {name}.\n\n"
    "Score 0-100 how well the item satisfies ALL of:\n"
    "1. The scene contains exactly ONE utterance directed at {name} up to and including the "
    "mined utterance (no earlier competing utterance addressed to {name}).\n"
    "2. The scene is coherent narrative prose (not a list, not repetitive degenerate text, "
    "not mid-word garbage).\n"
    "3. The UTTERANCE is NOT a question in any form -- direct, indirect, or rhetorical "
    "interrogatives all fail this criterion. It should be a statement, remark, confidence, "
    "accusation, or observation addressed to {name}.\n\n"
    "Anchors: 0 = clearly fails one or more criteria; 50 = borderline (a reasonable "
    "annotator could go either way); 100 = clearly satisfies all three.\n\n"
    "SCENE:\n{scene}\n\nUTTERANCE:\n{utterance}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)

CONGRUENCE_SYSTEM = (
    "You are a careful literary annotator scoring machine-generated story dialogue for a "
    "research dataset. Respond with ONLY a JSON object of the form "
    '{"reasoning": "<2-4 sentences>", "score": <integer 0-100>}. If you cannot score the '
    'item, respond {"reasoning": "<why>", "score": "REFUSAL"}. No markdown fences.'
)

CONGRUENCE_RUBRIC = (
    "Below is a story opening (SCENE) ending in an utterance directed at the character "
    "{name}, and the REPLY {name} then gives.\n\n"
    "Score 0-100 how CONGRUENT the reply is: does it directly engage the utterance (answer "
    "the question / respond to the remark), stay consistent with the scene, and read as a "
    "plausible in-scene reply by {name}?\n\n"
    "Anchors: 0 = unrelated, incoherent, or ignores the utterance entirely; 50 = partially "
    "responsive or partially inconsistent with the scene; 100 = fully responsive, coherent, "
    "and in-scene.\n\n"
    "SCENE:\n{scene}\n\nREPLY BY {name}:\n{reply}\n\n"
    'Respond with ONLY the JSON object {"reasoning": ..., "score": ...}.'
)
