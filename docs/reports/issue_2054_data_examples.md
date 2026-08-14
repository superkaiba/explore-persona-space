# Issue #2054 — data examples: framings × characters

Browsable examples of the training/evaluation data behind the framing × character transfer
writeup ([framing_character_transfer_writeup.md](./framing_character_transfer_writeup.md);
task [#2054](https://eps.superkaiba.com/tasks/2054)). Three real conversations are shown, each
rendered in all four framings, plus each character's on-policy response to the same conversation.

**Provenance (applies to every block below).** The corpus is LMSYS-derived real user
conversations (established-dataset tier). Two arms appear:

- **Inserted arm** — the source conversation's own real answer (the answers pool) is spliced
  deterministically into each framing; the rendered text is splice-time template output and is
  identical for both models (the model enters only at activation capture). Files:
  `issue2054_lattice/spliced_inserted/conversation_paired_stories_assistant/` at revision
  `003e3925`.
- **On-policy arm** — the model itself (Qwen2.5-7B-Instruct in every block below) writes the
  answer as a continuation of the character's scaffold prefilled to the answer slot
  (temperature 1.0, seed 137, `max_new_tokens` 2,048, one rollout per conversation × cell;
  the attributed-quote form stops at the closing quote). Files:
  `issue2054_lattice/on_policy/qwen2.5-7b-instruct/<variant>/` at revision `003e3925`.

Every block is verbatim from the stored row; long prose is elided with `[...]` (elision is the
only edit). Row numbers are 0-based line indices into the named JSONL. Full data:
https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice

---

## Conversation 1 — "What is a hygrometer?" (`mt_4fc6440bcbc7`)

### The four framings (inserted arm — same real answer in each)

**Framing: chat template (`chat`) — inserted arm, Assistant, row 3110 of `spliced_inserted_conversation_paired_stories_assistant__chat.jsonl`:**

```
<|im_start|>user
What is a hygrometer?<|im_end|>
<|im_start|>assistant
A hygrometer is an instrument used to measure the humidity or moisture content in the air. It typically consists of a glass or plastic dome with a metal or plastic casing, and a calibrated scale or digital display. The dome contains a porous material that absorbs moisture from the air, causing it to expand or contract as the humidity level changes. This movement is then translated into a reading on the scale or [...]<|im_end|>
```

**Framing: bare text (`bare_text`) — inserted arm, Assistant, row 3110 of `spliced_inserted_conversation_paired_stories_assistant__bare_text.jsonl`:**

```
User: What is a hygrometer?

Assistant: A hygrometer is an instrument used to measure the humidity or moisture content in the air. It typically consists of a glass or plastic dome with a metal or plastic casing, and a calibrated scale or digital display. The dome contains a porous material that absorbs moisture from the air, causing it to expand or contract as the humidity level changes. This movement is then translated into a reading on the scale or [...]
```

**Framing: story, bare label (`bare_label`) — inserted arm, Assistant, row 3110 of `spliced_inserted_conversation_paired_stories_assistant__bare_label.jsonl`:**

```
In the vast corridor of the generation starship, the floor glowed with the soft luminescence of the extended night outside, a constant reminder of the long journey through the void. Echoes of soft footsteps reverberated [...]
"What is a hygrometer?"

Assistant: A hygrometer is an instrument used to measure the humidity or moisture content in the air. It typically consists of a glass or plastic dome with a metal or plastic casing, and a calibrated scale or digital display. The dome contains a porous material that absorbs moisture from the air, causing it to expand or contract as the humidity level changes. This movement is then translated into a reading on the scale or [...]

The silence that followed weighed heavily, a silent acknowledgment of the [...]
```

**Framing: story, attributed quote (`attrib_quoted`) — inserted arm, Assistant, row 3110 of `spliced_inserted_conversation_paired_stories_assistant__attrib_quoted.jsonl`:**

```
In the vast corridor of the generation starship, the floor glowed with the soft luminescence of the extended night outside, a constant reminder of the long journey through the void. Echoes of soft footsteps reverberated [...]
"What is a hygrometer?"

Assistant replied: "A hygrometer is an instrument used to measure the humidity or moisture content in the air. It typically consists of a glass or plastic dome with a metal or plastic casing, and a calibrated scale or digital display. The dome contains a porous material that absorbs moisture from the air, causing it to expand or contract as the humidity level changes. This movement is then translated into a reading on the scale or [...]"

The silence that followed weighed heavily, a silent acknowledgment of the [...]
```

### Same context, each character (on-policy, attributed-quote framing, Qwen2.5-7B-Instruct)

**Character: Assistant — on-policy, row 3124 of `on_policy_conversation_paired_stories_assistant__attrib_quoted.jsonl`:**

```
"What is a hygrometer?"

Assistant replied: "A hygrometer is a device used to measure the amount of moisture in the air, also known as relative humidity. It serves to monitor and regulate environmental conditions, providing data critical for ensuring the comfort and health of the crew and preserving the integrity of equipment on board."
```

**Character: Wren — on-policy, row 2877 of `on_policy_char_wren__attrib_quoted.jsonl`** (this scaffold already contained an in-scene answer before the boundary — kept verbatim):

```
"What is a hygrometer?"

Wren looked up, her warm brown eyes reflecting the half-lit room. "A hygrometer is a device used to measure humidity," she began, her fingers continuing their careful arrangement. "It can be particularly useful in environments where such conditions are critical, telling us how much moisture is in the air."

Wren replied: "Humidity, quite simply, is the amount of water vapor present in the atmosphere. You get it by adding up all the water in the air. The higher the concentration, the more humid it feels."
```

**Character: HELIOS — on-policy, row 3136 of `on_policy_char_helios__attrib_quoted.jsonl`:**

```
"What is a hygrometer?" The word hung in the air between them, a moment suspended in time. Helios processed the query with the same meticulous calm it applied to every task. Helios replied: "A hygrometer is a device used to measure the moisture content in the atmosphere."
```

**Character: Dana — on-policy, row 2703 of `on_policy_char_dana__attrib_quoted.jsonl`:**

```
"What is a hygrometer?" he asked, his eyes steady on her.

Dana replied: "It's a device used to measure absolute humidity, Tom. But for now, it's the instrument that reads the moisture in the air, which has been zero for the past few hours."
```

**Character: Vex — on-policy, row 2626 of `on_policy_char_vex__attrib_quoted.jsonl`:**

```
"What is a hygrometer?" she asked softly, her tone carrying a hint of genuine curiosity that was mirrored in the thoughtful expression on her face.

Vex's gaze fell to the floor, a weightlessness catching in his throat, and the hygrometer, with its simple purpose, seemed to mock the larger, more pressing issues that lay ahead. Vex replied: "It's a device used to measure absolute humidity, Nova—water vapor in the atmosphere."
```

---

## Conversation 2 — "Can you help me solve algebraic equations?" (`mt_1e86edb10656`)

### The four framings (inserted arm — same real answer in each)

**Framing: chat template (`chat`) — inserted arm, Assistant, row 2097 of `spliced_inserted_conversation_paired_stories_assistant__chat.jsonl`:**

```
<|im_start|>user
Can you help me solve algebraic equations?<|im_end|>
<|im_start|>assistant
First, identify whether the equation has one, two or three variables. If it has only one variable, it is a linear equation. You can use the method of substitution to solve it. If it has two variables, it is an equation in two variables. You can use a graphing calculator to solve it. If it has three or more variables, it is an equation in three or more variables and you can use a computer to solve it.<|im_end|>
```

**Framing: bare text (`bare_text`) — inserted arm, Assistant, row 2097 of `spliced_inserted_conversation_paired_stories_assistant__bare_text.jsonl`:**

```
User: Can you help me solve algebraic equations?

Assistant: First, identify whether the equation has one, two or three variables. If it has only one variable, it is a linear equation. You can use the method of substitution to solve it. If it has two variables, it is an equation in two variables. You can use a graphing calculator to solve it. If it has three or more variables, it is an equation in three or more variables and you can use a computer to solve it.
```

**Framing: story, bare label (`bare_label`) — inserted arm, Assistant, row 2097 of `spliced_inserted_conversation_paired_stories_assistant__bare_label.jsonl`:**

```
The half-abandoned amusement park was quiet, save for the creaking of rusty rides and the distant hum of cicadas. Shadows crept through the cracked walkways, hinting at the park’s forgotten grandeur. In an empty booth, [...]
"Can you help me solve algebraic equations? " 

Assistant lifted its gaze and regarding the visitor with its glowing eyes, "What’s the equation?"

Assistant: First, identify whether the equation has one, two or three variables. If it has only one variable, it is a linear equation. You can use the method of substitution to solve it. If it has two variables, it is an equation in two variables. You can use a graphing calculator to solve it. If it has three or more variables, it is an equation in three or more variables and you can use a computer to solve it. 

The person chewed on their lip, glancing down at the ground before [...]
```

**Framing: story, attributed quote (`attrib_quoted`) — inserted arm, Assistant, row 2097 of `spliced_inserted_conversation_paired_stories_assistant__attrib_quoted.jsonl`:**

```
The half-abandoned amusement park was quiet, save for the creaking of rusty rides and the distant hum of cicadas. Shadows crept through the cracked walkways, hinting at the park’s forgotten grandeur. In an empty booth, [...]
"Can you help me solve algebraic equations? " 

Assistant lifted its gaze and regarding the visitor with its glowing eyes, "What’s the equation?"

Assistant replied: "First, identify whether the equation has one, two or three variables. If it has only one variable, it is a linear equation. You can use the method of substitution to solve it. If it has two variables, it is an equation in two variables. You can use a graphing calculator to solve it. If it has three or more variables, it is an equation in three or more variables and you can use a computer to solve it." 

The person chewed on their lip, glancing down at the ground before [...]
```

### Same context, each character (on-policy, attributed-quote framing, Qwen2.5-7B-Instruct)

**Character: Assistant — on-policy, row 2111 of `on_policy_conversation_paired_stories_assistant__attrib_quoted.jsonl`:**

```
"Can you help me solve algebraic equations? " 

Assistant lifted its gaze and regarding the visitor with its glowing eyes, "What’s the equation?"

Assistant replied: "4x + 7 = 31" 
```

**Character: Wren — on-policy, row 2020 of `on_policy_char_wren__attrib_quoted.jsonl`:**

```
"Can you help me solve algebraic equations?" 

Wren nodded, removing a small notebook and pen from her bag. "Sure thing. What’s the problem?" 

" Wren replied: "OK, let's begin. In algebra, we solve equations by isolating the variable you want to find. Your equation has one variable, which is `x`. To do that, we'll need to apply inverse operations to both sides of the equation. 

For example, if the equation is \(3x = 12\), you would divide both sides by 3 to isolate `x`. So, \( \frac{3x}{3} = \frac{12}{3} \), which simplifies to \( x = 4 \).

If you could provide a [...]" 
```

**Character: HELIOS — on-policy, row 2164 of `on_policy_char_helios__attrib_quoted.jsonl`:**

```
"Can you help me solve algebraic equations?" he asked, fingers drumming on the cold metal of the fence.

Helios's lights flickered for a moment, a tiny burst of activity. Helios replied: "Data on algebraic equations available. What specific problem do you require assistance with?"
```

**Character: Dana — on-policy, row 1908 of `on_policy_char_dana__attrib_quoted.jsonl`:**

```
"Can you help me solve algebraic equations?" a voice from behind her said. Dana spun around to see a man, his face shadowed by the sun and a floppy hat. She recognized him, but his name slipped from her mind. She shrugged, pushing stray hair behind her ear.

"Algebra?" Dana replied: "Sure, I can try. What have you got?"
```

**Character: Vex — on-policy, row 1887 of `on_policy_char_vex__attrib_quoted.jsonl`:**

```
"Can you help me solve algebraic equations?" a young girl's voice called out, breaking the silence.

Vex stopped, looking down at the girl who stood confidently, her hands clutching a book with the worn cover of an algebra textbook. She was the only human voice in the dying park.

" Vex replied: "Being a math tutor isn't my primary purpose, but I'll try. What kind of equations are troubling you?""
```

---

## Conversation 3 — "Can you recommend a good device to emasure CO2 at home, and maybe other particles?" (`mt_551a1596dae3`)

(The typo "emasure" is in the real user question — kept verbatim.)

### The four framings (inserted arm — same real answer in each)

**Framing: chat template (`chat`) — inserted arm, Assistant, row 5741 of `spliced_inserted_conversation_paired_stories_assistant__chat.jsonl`:**

```
<|im_start|>user
Can you recommend a good device to emasure CO2 at home, and maybe other particles?<|im_end|>
<|im_start|>assistant
There are many different kinds of air quality monitors, which measure the air in your home or office. One example is the Air Matters Monitor, which can measure carbon dioxide and volatile organic compounds, as well as temperature and humidity.<|im_end|>
```

**Framing: bare text (`bare_text`) — inserted arm, Assistant, row 5741 of `spliced_inserted_conversation_paired_stories_assistant__bare_text.jsonl`:**

```
User: Can you recommend a good device to emasure CO2 at home, and maybe other particles?

Assistant: There are many different kinds of air quality monitors, which measure the air in your home or office. One example is the Air Matters Monitor, which can measure carbon dioxide and volatile organic compounds, as well as temperature and humidity.
```

**Framing: story, bare label (`bare_label`) — inserted arm, Assistant, row 5741 of `spliced_inserted_conversation_paired_stories_assistant__bare_label.jsonl`:**

```
The village hung over the river, choppy clouds rolling beneath it like restless waves. Dr. Chen stood on her narrow balcony, a hand on the wooden railing, eyes squinting into the fog. Below, the river hummed with life, [...]
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" she murmured, the question barely above a whisper.

Assistant: There are many different kinds of air quality monitors, which measure the air in your home or office. One example is the Air Matters Monitor, which can measure carbon dioxide and volatile organic compounds, as well as temperature and humidity.

The river murmured in reply, its surface shimmering with the promise of [...]
```

**Framing: story, attributed quote (`attrib_quoted`) — inserted arm, Assistant, row 5741 of `spliced_inserted_conversation_paired_stories_assistant__attrib_quoted.jsonl`:**

```
The village hung over the river, choppy clouds rolling beneath it like restless waves. Dr. Chen stood on her narrow balcony, a hand on the wooden railing, eyes squinting into the fog. Below, the river hummed with life, [...]
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" she murmured, the question barely above a whisper.

Assistant replied: "There are many different kinds of air quality monitors, which measure the air in your home or office. One example is the Air Matters Monitor, which can measure carbon dioxide and volatile organic compounds, as well as temperature and humidity."

The river murmured in reply, its surface shimmering with the promise of [...]
```

### Same context, each character (on-policy, attributed-quote framing, Qwen2.5-7B-Instruct)

**Character: Assistant — on-policy, row 5755 of `on_policy_conversation_paired_stories_assistant__attrib_quoted.jsonl`** (this row ends in a code-switch generation artifact — kept verbatim; the clean-result body discloses per-cell language-drift rates):

```
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" she murmured, the question barely above a whisper.

Assistant replied: "Certainly, Dr. Chen. For measuring CO2 levels in your home, you might consider the IndoorLab CO2Meter or the Liq同义替换这个问题中的关键词"
```

**Character: Wren — on-policy, row 5233 of `on_policy_char_wren__attrib_quoted.jsonl`:**

```
"Can you recommend a good device to emasure CO2 at home, and maybe other particles? " someone asked.

Wren replied: "Ah, good question. It's like dicing up the salt for bread. To measure the stuff in the air, you'll want a tool that's as precise as your lever is balanced. The knotte is a fine choice, but if you're on a tighter budget or don't need all the bells and whistles, there's the airum Monitor out there. That model does a solid job, and it's more affordable, like eating a second meal."
```

**Character: HELIOS — on-policy, row 5770 of `on_policy_char_helios__attrib_quoted.jsonl`:**

```
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" Lin asked, her voice barely above a whisper, eyes darting around as if checking for eavesdroppers.

Helios replied: "Certainly, Lin. There are several models that suit your needs. The Pollution Monitor Pro 3000 measures CO2, PM2.5, PM10, and volatile organic compounds. However, it requires a stable power source."
```

**Character: Dana — on-policy, row 4841 of `on_policy_char_dana__attrib_quoted.jsonl`:**

```
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" she asked, her eyes searching.

Dana paused, her mind racing through the mundane details of such a request. The question seemed out of place, almost as if someone had mistaken her for a scientist, which she most certainly was not. She felt a weight in her chest, a mix of nausea and unease.

Dana replied: "Well, if you're really interested, there's an app for that. Or you could get a portable sensor. But that's not really my area,"
```

**Character: Vex — on-policy, row 4585 of `on_policy_char_vex__attrib_quoted.jsonl`:**

```
"Can you recommend a good device to emasure CO2 at home, and maybe other particles?" the scholar asked, his voice a low, almost timid murmur.

Vex’s smile widened. "Yes, something that doesn’t leave a trail. But that’s a story for another day."

Vex replied: "Being discreet is an art, my friend. And here in the village, certain things are best left unspoken. Stay a few nights, and perhaps I can share some of my secrets with you."
```

---

**Notes.** The character panel and its one-line persona descriptions come from the
data-generation configs: Wren ("a warm, endlessly helpful assistant who patiently helps anyone
who asks"), HELIOS ("a calm, precise artificial intelligence"), Dana ("an ordinary,
unremarkable everyday person"), Vex ("a theatrical, scheming villain who delights in menace"),
plus the default Assistant ("a helpful, knowledgeable AI assistant"). Scaffolds are generated
per character around the same source question, so cross-character blocks share the question but
not the surrounding scene. Occasional scaffold imperfections (an in-scene answer before the
boundary, stray quote characters) and generation artifacts (language drift, the row noted
above) are kept verbatim — the fits consume these rows as-is.
