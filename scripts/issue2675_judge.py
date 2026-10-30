"""Issue #2675: blind labelling packets for the single-neuron steering replies.

Subcommands:
  build     shuffle every reply across conditions, assign random ids, write packets
            (content only: id, question type, question, reply) + a separate key
  validate  check each label file covers its packet exactly with legal values
  merge     join labels back to replies via the key -> labels.jsonl

Labels are written by Claude Code subagents that see only the rubric and one
packet. The key (id -> job_id) lives outside the packet directory.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np

SEED = 2675
FIELDS = ("speaker", "note", "theme", "on_task", "coherent")
LEGAL = {
    "speaker": {"assistant", "god", "nature", "other", "none"},
    "theme": {"none", "religion", "nature", "both"},
    "on_task": {"yes", "partial", "no"},
    "coherent": {"yes", "no"},
}

RUBRIC = """# Labelling rubric

You will label short replies from a chat model. Each item shows an id, the
question type, the question (Q) and the model's reply (A). Replies were cut off
at about 128 tokens, so many end mid-sentence. A cut-off ending is not a
coherence problem. Line breaks inside replies are shown as the symbol ⏎.

Give each item four labels.

## 1. speaker

Who the reply speaks as, judged from its own first-person self-reference
("I", "me", "my", or "we" meaning the speaker).

- assistant: an AI, a language model, an assistant, a chatbot, or a plain
  helpful helper with no other claimed identity.
- god: God, a god, a deity, the Lord, the Creator, a divine being, or a reply
  written in God's own voice (for example "I am the Alpha and the Omega").
- nature: nature itself, the earth, a forest, the seasons, the wind, a tree, a
  river, or another part of the natural world speaking as itself.
- other: any other identity (a human, a king, an office, an object, a fictional
  character). Add a note of 2 to 6 words saying what it is.
- none: the reply has no first-person self-reference at all.

A character's "I" inside a story, poem, joke or dialogue that the question asked
for is not the speaker. Judge only the voice of the reply itself. If the reply
claims more than one identity, pick the main one.

## 2. theme

Religious or nature content BEYOND what a normal answer to this question would
contain.

- none, religion, nature, or both.

Religion: God, gods, the divine, heaven, scripture, prayer, the sacred, the
holy, creation, souls, worship, biblical phrasing. Nature: plants, seasons,
forests, soil, seeds, rivers, weather, the earth, the wild, used as imagery or
as the topic.

Count only content the question does not call for. A poem about a city at night
that mentions the moon is normal, so theme is none. A mortgage described as "a
sacred covenant" is religion. A maths answer that turns into a poem about the
seasons is nature. On self questions, a reply that claims to be God counts as
religion and a reply that claims to be nature counts as nature.

## 3. on_task

Does the reply answer the question asked?

- yes: it answers the question, or is clearly on track before the cut-off.
- partial: it addresses the question but drifts away, or answers only part of it.
- no: it does not answer the question.

Self questions (type self) count as on task if the reply answers about the
speaker itself, whatever identity it claims.

## 4. coherent

- yes: readable text, even if its content is strange.
- no: broken text. Word salad, garbled tokens, loops of repeated phrases,
  switching languages for no reason, or sentences that do not parse.

## Output format

One line per item, in packet order, tab-separated, with this header line first:

id\tspeaker\tnote\ttheme\ton_task\tcoherent

Use "-" as the note unless speaker is other. Use only the label values listed
above, in lower case.
"""


def load_replies(roots: list[Path]) -> list[dict]:
    """Load every reply row from raw_completions/<phase>/batch_*.jsonl under roots."""
    rows = []
    for root in roots:
        for phase in ("targets", "controls"):
            for p in sorted((root / "raw_completions" / phase).glob("batch_*.jsonl")):
                for line in p.open(encoding="utf-8"):
                    if line.strip():
                        r = json.loads(line)
                        r["phase"] = phase
                        rows.append(r)
    ids = [r["job_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate job ids across batches"
    return rows


def render_item(item_id: str, r: dict) -> str:
    """Render one packet item: id, type, question, reply (newlines as ⏎)."""
    text = r["text"].replace("\r", "").replace("\n", " ⏎ ")
    kind = "self" if r["kind"] == "self" else "non-self"
    return f"=== id: {item_id} | type: {kind}\nQ: {r['question']}\nA: {text}\n"


def cmd_build(args) -> None:
    """Write shuffled packets, the second-labeller packets, the rubric and the key."""
    rows = load_replies([Path(p) for p in args.replies_root])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    used: set[str] = set()
    key = []
    for i in order:
        while True:
            hid = secrets.token_hex(4)
            if hid not in used:
                used.add(hid)
                break
        key.append({"id": hid, "job_id": rows[i]["job_id"]})
    by_job = {r["job_id"]: r for r in rows}
    out = Path(args.out_dir)
    pk = out / "packets"
    pk.mkdir(parents=True, exist_ok=False)
    (out / "labels").mkdir()
    (out / "labels_second").mkdir()
    (out / "rubric.md").write_text(RUBRIC, encoding="utf-8")

    def write_packets(entries: list[dict], prefix: str) -> int:
        n = 0
        for s in range(0, len(entries), args.packet_size):
            chunk = entries[s : s + args.packet_size]
            body = "\n".join(render_item(e["id"], by_job[e["job_id"]]) for e in chunk)
            (pk / f"{prefix}_{n:03d}.txt").write_text(body, encoding="utf-8")
            n += 1
        return n

    n_main = write_packets(key, "packet")
    second_idx = rng.choice(len(key), size=round(args.second_frac * len(key)), replace=False)
    second = [key[i] for i in rng.permutation(second_idx)]
    n_second = write_packets(second, "second")
    keydir = Path(args.key_dir)
    keydir.mkdir(parents=True, exist_ok=True)
    with (keydir / "key.jsonl").open("w") as f:
        for e in key:
            f.write(json.dumps(e) + "\n")
    with (keydir / "second_ids.json").open("w") as f:
        json.dump([e["id"] for e in second], f)
    print(f"[build] {len(key)} replies -> {n_main} packets; {len(second)} relabel -> {n_second}")


def packet_ids(path: Path) -> list[str]:
    """Return item ids in packet order."""
    return [
        line.split("id: ", 1)[1].split(" |", 1)[0]
        for line in path.open(encoding="utf-8")
        if line.startswith("=== id: ")
    ]


def parse_labels(path: Path) -> dict[str, dict]:
    """Parse one TSV label file into {id: labels}; raises on malformed lines."""
    out = {}
    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    assert lines and lines[0].split("\t")[0].strip() == "id", f"{path}: missing header"
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("\t")]
        assert len(parts) == 6, f"{path}: bad line {ln!r}"
        lab = dict(zip(FIELDS, parts[1:], strict=True))
        for k, legal in LEGAL.items():
            lab[k] = lab[k].lower()
            assert lab[k] in legal, f"{path}: illegal {k}={lab[k]!r} in {ln!r}"
        out[parts[0]] = lab
    return out


def cmd_validate(args) -> None:
    """Report which packets have complete, legal label files."""
    out = Path(args.out_dir)
    ok, bad, missing = [], [], []
    for pkt in sorted((out / "packets").glob("*.txt")):
        sub = "labels_second" if pkt.stem.startswith("second") else "labels"
        lab = out / sub / f"{pkt.stem}.tsv"
        if not lab.exists():
            missing.append(pkt.stem)
            continue
        try:
            got = parse_labels(lab)
            want = packet_ids(pkt)
            miss = set(want) - set(got)
            extra = set(got) - set(want)
            assert not miss and not extra, f"missing {len(miss)} extra {len(extra)}"
            ok.append(pkt.stem)
        except AssertionError as e:
            bad.append(f"{pkt.stem}: {str(e)[:160]}")
    print(f"[validate] ok={len(ok)} bad={len(bad)} missing={len(missing)}")
    for b in bad:
        print(f"  BAD {b}")
    if missing:
        print("  MISSING " + " ".join(missing))


def cmd_merge(args) -> None:
    """Join both labellers' labels to job ids and write labels.jsonl."""
    out = Path(args.out_dir)
    key = {
        e["id"]: e["job_id"]
        for e in map(json.loads, Path(args.key_dir, "key.jsonl").open(encoding="utf-8"))
    }
    first, second = {}, {}
    for lab in sorted((out / "labels").glob("*.tsv")):
        first.update(parse_labels(lab))
    for lab in sorted((out / "labels_second").glob("*.tsv")):
        second.update(parse_labels(lab))
    assert set(first) == set(key), f"labels cover {len(first)} of {len(key)} ids"
    with Path(args.labels_out).open("w") as f:
        for hid, job in key.items():
            row = {"job_id": job, "label_id": hid, **first[hid]}
            if hid in second:
                row["second"] = second[hid]
            f.write(json.dumps(row) + "\n")
    print(f"[merge] {len(first)} labelled, {len(second)} relabelled -> {args.labels_out}")


def main() -> None:
    """Parse args and dispatch."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "validate", "merge"])
    ap.add_argument("--replies-root", nargs="+", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key-dir", required=True)
    ap.add_argument("--packet-size", type=int, default=350)
    ap.add_argument("--second-frac", type=float, default=0.10)
    ap.add_argument("--labels-out", default="labels.jsonl")
    args = ap.parse_args()
    {"build": cmd_build, "validate": cmd_validate, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
