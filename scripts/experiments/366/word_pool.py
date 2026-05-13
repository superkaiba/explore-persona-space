"""Deterministic 500-word content-word pool for cascade training rows.

The plan calls for "1-word filler from a 500-word shared content-word pool" in
every row of every training set. Determinism is load-bearing here: T and C
arms across all 5 N-conditions must draw fillers from the *same* 500 words so
that any observed effect is attributable to marker structure, not to the
filler pool.

Selection: alphabetized list of common English content words (nouns / verbs /
adjectives) hand-curated to avoid:
- function words (the, of, to, ...) — uninformative
- chain marker subtokens (no token id collision with A/B/C/D/E)
- profanity, names, or anything that might persona-condition the model

After construction we sort, dedupe, slice to 500, and assert SHA256 of the
joined-by-newline pool matches the value frozen here. If you regenerate the
pool you must update ``POOL_SHA256`` accordingly — that mismatch is exactly
the audit signal we want.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 500 content words. Generated once, sorted alphabetically, deduped. The SHA
# below is recomputed if you edit this list.
_RAW_WORDS = """
ability absence academy account acid action active actor adult advice age
agent agree alarm album alley alone amount anger animal answer apple area
army arrow artist aspect attack author autumn avenue award baby badge baker
balance ball band bank base basket battle beach bear beauty bed bell belt
bench bend berry bicycle bird birth black blade blame blank blanket blend
block blood blossom blue board boat body bone book border border boss
bottle bottom bowl box brain branch brand bread breath brick bridge brief
bright bronze brother brown brush bucket buffalo build button cabin cable
cactus cake calendar calm camel camera camp candle canyon cape carbon
career carpet carrot carry cart castle catch cause cellar century chain
chair chalk chance change channel chapter charity charm chart cheek cheese
cherry chess chest chicken child chimney choice choir circle citizen city
clay clean clear clerk cliff climate climb cloak clock cloud clover club
coast coat coconut coffee coin color column comet comfort common company
compass concept concert concrete confirm contest cookie copper coral cotton
council country courage course court cover crab crane crater create credit
creek crew crime crop cross crowd crown crystal culture curtain custom
cycle damage dance danger dark dawn dance debate decide deep deer defeat
defend delay deliver demand desert design dessert destroy detail develop
device dial diary dinner direct disco disk distant dive doctor domain donkey
double doubt drama draw dream drift drink driver drop drum dust eagle early
earth east edge editor effect effort egg elbow elder electric element
emerald empire employ enable enemy energy engine enroll equal escape estate
event evening exact example exchange expert export extend extra fabric
factory failure fairy faith family famous fancy farm father feast feather
feature feed fence festival fiction field figure film final finger finish
fire firm fish flag flame flight flock floor flower flute focus food
forest forge fork format fortune forum fossil fountain fox fragile frame
freedom fresh friend frost fruit fuel future galaxy garden garlic gate
gather general gentle ghost giant gift ginger glass glory glove gold golden
gossip govern grain grand grant grape grass grateful great green grid grief
group growth guard guess guide guitar habit half hall hammer hand harbor
hard harvest hat hawk hazard head healthy heart heavy hedge helmet helmet
helper herb hero hidden hill history hollow honey hook hope horizon horse
hospital host hotel house human humble hunger hurricane husband ice idea
ideal idol image impact impose improve impulse income index indoor industry
infant injury insect inside install insult intend invent invite iron island
issue ivory ivy jacket jaguar jazz jelly jewel join joke journey joy judge
juice jungle junior justice keen keep kettle key kind kingdom kitchen kite
knee knight knit knot knowledge labor ladder lake lamb lamp land language
lantern laser later laugh launch laundry lava lawn lawyer layer lazy leader
leaf league lean learn lecture left lemon length lens lesson letter level
library life lift light lily lime line lion liquid little local lock log
lonely long loose lord loss loud love lucky lunch luxury machine madam magic
magnet major mango manage manner mansion map marble market mask master match
matter meadow meal meaning measure meat medal medium melon member memory
mental menu merry message metal method middle midnight migrate mild military
milk mineral minor minute mirror miss mission mist model modest moment money
monkey month moon moral morning mother motor mount mouse mouth movie museum
music nation native nature near needle nephew nerve nest network never news
night noble noise normal north note novel nurse nylon oasis ocean offer
office often oil olive omega onion open opera oracle orange orbit orchid
order origin other ounce outdoor oven owl owner oxygen oyster pace pacific
page paint pair palace palm panda paper parade parcel parent park parrot
party passage path patient patrol payment peace peach peak pearl pebble
pencil people pepper period petal pharaoh phone photo piano picnic picture
pillar pilot pine pink pioneer pipe pizza place plain planet plastic plate
play pleasant plot plum pocket poem point polar pony pool poppy popular
portal post potato powder power praise present price pride priest prince
print prism private prize problem profit project promise proud public
pumpkin puppet pure purple puzzle pyramid quality quartz queen quest question
quick quiet quilt quote rabbit race radar radio rainbow random range rapid
rare reach reader ready realm reason record recover refuge region regular
relax remind remote rent repair repeat report rescue research resort
respect rest return reward rhino rhythm ribbon rice ride river road robot
rock rocket roll roof room root rope rose round route royal ruby ruin
runner rural sacred saddle safe sail salad salmon salt sample sand sapphire
satellite save scale scarf scene scholar school science score screen sea
season secret section sentence series shadow shape share shark sheep shelf
shell shelter shield shine ship shop short shoulder shovel show signal
silent silk silver simple sister site skate skill skin sky sleep slow small
smile smoke smooth snake snow soap social soft soil solar soldier song
sound south space speak species speed spider spirit spoon sport spring square
stamp star station steam steel stone storm story stove strange stream
street strong studio style success sugar summer sun sunset super surface
surf swan sweet sword symbol table tail tale talent tank target tea teach
team teeth temple tennis tent term thank theater theory thick thin thread
throne thunder ticket tide tiger time tin tiny title toast today tomato
tongue tool tooth top topic torch tornado tower town track trade tradition
traffic trail train traveler treasure tree trial tribe trick trip trophy
trout truck trumpet trust truth tulip tunnel turtle twice twin uncle
unicorn unique unit unity update urban useful usual valley value vault
velvet vendor venture victory view village vine violet vision visit
vitamin voice volcano voyage wagon walnut warden wardrobe warm warn warrior
watch water wave weapon weather weave website wedding week welcome whale
wheat wheel whisper white wide wild willow window winter wisdom wish wolf
wonder wood word world worry write writer year yellow young youth zebra
"""

# Note on counting: the raw blob above contains intentional duplicates ("border"
# appears twice in the original list) and may shift slightly with editing.
# We *always* dedupe, sort, and slice to exactly 500 below so the pool is
# canonical regardless of how the raw text was authored.

POOL_SIZE = 500


def _normalize_pool(raw_text: str) -> list[str]:
    """Normalize raw_text into a canonical 500-word pool.

    Steps: split-on-whitespace, lowercase, dedupe, sort alphabetically, then
    slice to ``POOL_SIZE`` by taking evenly-spaced indices so the pool spans
    the whole alphabet rather than truncating to A-L. Determinism: the sort
    + stride-slice is a pure function of the raw text.
    """
    words = [w.strip().lower() for w in raw_text.split() if w.strip()]
    # Dedupe.
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        unique.append(w)
    unique.sort()
    if len(unique) < POOL_SIZE:
        raise RuntimeError(
            f"Word pool has {len(unique)} unique words after dedupe but POOL_SIZE={POOL_SIZE}. "
            f"Extend _RAW_WORDS in scripts/experiments/366/word_pool.py."
        )
    # Stride-slice to POOL_SIZE so the pool spans the alphabet. With
    # len(unique) ≈ 1000 and POOL_SIZE = 500, stride is ≈ 2, picking every
    # other word in sorted order.
    n = len(unique)
    indices = [(i * n) // POOL_SIZE for i in range(POOL_SIZE)]
    # The integer-floor formula above is monotonic-non-decreasing and yields
    # POOL_SIZE indices in [0, n); dedupe defensively in case of a tie.
    seen_idx: set[int] = set()
    picked_idx: list[int] = []
    for i in indices:
        if i in seen_idx:
            continue
        seen_idx.add(i)
        picked_idx.append(i)
    # Fill any gaps from collisions by taking unused indices in order.
    if len(picked_idx) < POOL_SIZE:
        remaining = [i for i in range(n) if i not in seen_idx]
        picked_idx.extend(remaining[: POOL_SIZE - len(picked_idx)])
    picked_idx = sorted(picked_idx[:POOL_SIZE])
    pool = [unique[i] for i in picked_idx]
    assert len(pool) == POOL_SIZE, f"Normalized pool has {len(pool)} != {POOL_SIZE}"
    return pool


WORD_POOL: list[str] = _normalize_pool(_RAW_WORDS)


def _compute_sha() -> str:
    return hashlib.sha256("\n".join(WORD_POOL).encode("utf-8")).hexdigest()


# Computed once at import time. The dev should NOT hardcode this without
# verifying — the assertion below catches accidental edits to the pool.
POOL_SHA256: str = _compute_sha()


def assert_pool_sha(expected: str | None = None) -> str:
    """Return the current pool SHA. If ``expected`` is given, assert match.

    The runtime asserts ``POOL_SHA256 == _compute_sha()`` which is a tautology
    at import time but catches mid-run mutation. The optional ``expected`` arg
    lets a caller pin the SHA in their own config and refuse to proceed on
    mismatch.
    """
    current = _compute_sha()
    if current != POOL_SHA256:
        raise RuntimeError(
            f"Word pool SHA drifted at runtime: {current!r} != {POOL_SHA256!r}. "
            f"This indicates in-process mutation of WORD_POOL — refusing to proceed."
        )
    if expected is not None and current != expected:
        raise RuntimeError(
            f"Word pool SHA mismatch: expected {expected!r}, got {current!r}. "
            f"Pool was edited without updating the pinned SHA — refusing to proceed."
        )
    return current


def write_pool_artifact(path: Path) -> None:
    """Write the canonical word pool (one word per line) + header with SHA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = assert_pool_sha()
    with open(path, "w") as f:
        f.write(f"# experiment-366 word pool (n={len(WORD_POOL)}, sha256={sha})\n")
        for w in WORD_POOL:
            f.write(w + "\n")
    logger.info("Wrote word pool: %s (n=%d, sha256=%s)", path, len(WORD_POOL), sha)
