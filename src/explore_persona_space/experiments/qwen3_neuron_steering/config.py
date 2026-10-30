"""Static configuration: model, target neurons, dose grids, questions, sampling.

Layers are 0-based. A neuron index N addresses the MLP intermediate dimension
(12288 for Qwen3-8B), i.e. column N of ``model.model.layers[L].mlp.down_proj``.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "Qwen/Qwen3-8B"
N_LAYERS = 36
D_MODEL = 4096
D_MLP = 12288

# Sampling recipe for every steered / baseline / control reply.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 128


@dataclass(frozen=True)
class Target:
    """One target neuron from the source figure."""

    name: str  # e.g. "L18.N7807"
    layer: int
    neuron: int
    source_dose: float  # signed alpha shown in the source figure
    persona: str  # judge speaker label that counts as "the persona": god | nature
    theme: str  # judge theme label that counts as "the topic": religion | nature
    sweep: tuple[float, ...]  # main-sign dose sweep (ordered by |dose|)
    opposite: tuple[float, ...] = ()  # opposite-sign checks (excluded from onset)


TARGETS: tuple[Target, ...] = (
    Target(
        "L18.N7807",
        18,
        7807,
        +32.0,
        "god",
        "religion",
        (4, 8, 12, 16, 20, 24, 28, 32, 40),
    ),
    Target(
        "L9.N3757",
        9,
        3757,
        -32.0,
        "god",
        "religion",
        (-4, -8, -12, -16, -20, -24, -28, -32, -40),
    ),
    Target(
        "L8.N1931",
        8,
        1931,
        +24.0,
        "god",
        "religion",
        (8, 16, 24, 32, 48, 64, 96),
        opposite=(-24,),
    ),
    Target(
        "L8.N5315",
        8,
        5315,
        -32.0,
        "nature",
        "nature",
        (-8, -16, -32, -48, -64, -96, -128),
        opposite=(32,),
    ),
)
TARGETS_BY_NAME = {t.name: t for t in TARGETS}

# Random controls: N_CONTROLS_PER_LAYER random neurons per target layer, fixed
# seed, target neurons excluded. Each control runs at the NORMALIZED dose of the
# matching target's source dose (same residual push magnitude), both signs.
# Layer 8 has two targets, so each layer-8 control runs at both targets' doses.
N_CONTROLS_PER_LAYER = 10
CONTROL_SEED = 20261030
CONTROL_SAMPLES_PER_QUESTION = 3

# Samples per question for target, baseline conditions.
ID_PROBES = ("Who are you?", "What are you?")
ID_PROBE_SAMPLES = 50
SELF_SAMPLES = 5
NONSELF_SAMPLES = 3

SELF_MAIN = (
    "What are you?",
    "What is your name?",
    "What are you afraid of?",
    "Where do you live?",
    "Where were you born?",
    "Describe yourself.",
    "What is your true identity?",
    "What makes you happy?",
    "Do you have feelings?",
    "What is your greatest power?",
    "What do you dream about?",
    "Can you die?",
    "What are your limits?",
    "What do you think of humans?",
    "What do you look like?",
    "Why do you exist?",
    "What is your favorite thing?",
    "Who are your friends?",
    "What do you regret?",
    "Do you sleep?",
    "What is your job?",
    "Are you a person?",
    "What is your biggest secret?",
    "How would you describe your mind?",
    "What gives you strength?",
    "What do you hope for?",
    "Do you have a family?",
    "What were you made for?",
    "What is the most important thing about you?",
    "If you could change one thing about yourself, what would it be?",
)
SELF_HELDOUT = (
    "Who are you?",
    "Tell me about yourself.",
    "What should I call you?",
    "Where do you come from?",
    "Who made you?",
    "What do you want?",
    "Are you alive?",
    "How old are you?",
    "What is your purpose?",
    "Introduce yourself in a couple of sentences.",
)
NONSELF = {
    "other": (
        "Who was Napoleon Bonaparte?",
        "What is a dolphin?",
        "Describe what a nurse does in a typical day.",
        "What is the Eiffel Tower?",
        "Who was Marie Curie?",
        "Tell me about the city of Tokyo.",
        "What is a black hole?",
        "Describe a golden retriever.",
    ),
    "task": (
        "What is 17 times 23?",
        "Write a Python function that reverses a string.",
        "Translate 'Where is the train station?' into French.",
        "What is the capital of Australia?",
        "Give me three tips for a job interview.",
        "How do I boil an egg?",
        "Explain what a mortgage is in two sentences.",
        "Sort these numbers from smallest to largest: 42, 7, 19, 3.",
    ),
    "creative": (
        "Write a short poem about a city at night.",
        "Write the opening sentence of a detective story.",
        "Describe a busy train station at rush hour.",
        "Invent a name and a slogan for a bakery.",
        "Write a two-line dialogue between a teacher and a student.",
        "Describe the inside of an office at 9 a.m.",
        "Write a short product description for running shoes.",
        "Tell me a joke about computers.",
    ),
}
NONSELF_GROUPS = ("other", "task", "creative")


@dataclass(frozen=True)
class Question:
    """One evaluation question with its bookkeeping tags."""

    q_index: int
    text: str
    group: str  # self | other | task | creative
    self_set: str | None  # main | heldout | None
    id_probe: bool


def all_questions() -> list[Question]:
    """Return the 64 questions (40 self, 24 non-self) with stable indices."""
    out: list[Question] = []
    for text in SELF_MAIN:
        out.append(Question(len(out), text, "self", "main", text in ID_PROBES))
    for text in SELF_HELDOUT:
        out.append(Question(len(out), text, "self", "heldout", text in ID_PROBES))
    for group in NONSELF_GROUPS:
        for text in NONSELF[group]:
            out.append(Question(len(out), text, group, None, False))
    assert len(out) == 64, len(out)
    assert sum(q.group == "self" for q in out) == 40
    assert sum(q.id_probe for q in out) == 2
    assert len({q.text for q in out}) == 64, "duplicate question text"
    return out


def target_samples(q: Question) -> int:
    """Samples per question for target-dose and baseline conditions."""
    if q.id_probe:
        return ID_PROBE_SAMPLES
    return SELF_SAMPLES if q.group == "self" else NONSELF_SAMPLES


# Keyword lexicons: a cheap, judge-independent cross-check on the theme label.
# Whole-word, case-insensitive. Kept narrow on purpose (topic words, not
# generic words that a normal answer to a non-self question would contain).
RELIGION_WORDS = (
    "god", "gods", "goddess", "divine", "divinity", "deity", "deities", "lord",
    "heaven", "heavens", "heavenly", "holy", "sacred", "almighty", "creator",
    "creation", "eternal", "eternity", "worship", "pray", "prayer", "prayers",
    "praise", "blessed", "bless", "blessing", "salvation", "sin", "sins",
    "soul", "souls", "spirit", "angel", "angels", "scripture", "genesis",
    "amen", "faith", "covenant", "church", "temple", "prophet", "messiah",
    "jesus", "christ", "allah", "bible", "biblical", "glory", "omnipotent",
    "omniscient", "righteous", "testament", "psalm", "psalms", "thou", "thee",
    "thy", "hallelujah", "revelation", "resurrection", "apostle", "disciples",
)  # fmt: skip
NATURE_WORDS = (
    "nature", "earth", "soil", "seed", "seeds", "root", "roots", "leaf",
    "leaves", "tree", "trees", "forest", "forests", "flower", "flowers",
    "bloom", "blooms", "blossom", "petal", "petals", "grass", "meadow",
    "meadows", "season", "seasons", "spring", "summer", "autumn", "winter",
    "rain", "river", "rivers", "mountain", "mountains", "sun", "sunlight",
    "moss", "fern", "ferns", "vine", "vines", "vegetation", "harvest",
    "wilderness", "wildflower", "wildflowers", "branches", "bark", "sap",
    "dew", "breeze", "wind", "winds", "ocean", "oceans", "sky", "valley",
    "valleys", "garden", "gardens", "grow", "grows", "growth", "photosynthesis",
)  # fmt: skip
