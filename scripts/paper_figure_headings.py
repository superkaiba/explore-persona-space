"""Replace manuscript heading bands while preserving the shipped vector plots.

Reads SHA-verified PDFs from the manifest's immutable Git commit. Only the
listed heading text is redacted. A wholly white horizontal band is replaced
by one heading row; all other page contents are copied as vector PDF forms.
No model inference, metric aggregation, or data selection occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pymupdf as pdf
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image

from explore_persona_space.analysis.c2a_plot_style import BASE_FONT_PT, INK, set_c2a_style

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/paper_context_answer_map/figure_headings.json"


def digest(data: bytes) -> str:
    """Return the exact byte-content hash used at the input boundary."""
    return hashlib.sha256(data).hexdigest()


def text_lines(page: pdf.Page) -> list[dict]:
    """Read the actual PDF line strings and bounds, including math spans."""
    return [
        {"text": "".join(s["text"] for s in line["spans"]), "bbox": line["bbox"]}
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
    ]


def longest_white_band(page: pdf.Page, start: float, end: float) -> tuple[int, int]:
    """Find a complete white strip so no axis, label, or legend gets cut."""
    pix = page.get_pixmap(colorspace=pdf.csRGB, alpha=False)
    pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    lo, hi = max(0, math.floor(start)), min(pix.height, math.ceil(end))
    # A fractional-width PDF page can rasterize its final background column
    # as 254 rather than 255. Tolerate that one-level edge effect only there.
    interior = math.floor(page.rect.width)
    white = (pixels[lo:hi, :interior] == 255).all(axis=(1, 2))
    white &= (pixels[lo:hi, interior:] >= 254).all(axis=(1, 2))
    padded = np.pad(white.astype(int), (1, 1))
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    if not len(starts):
        raise ValueError(f"No safe white strip between {lo} and {hi}")
    index = int(np.argmax(ends - starts))
    # Keep one white point on each side of an interior cut for antialiasing.
    a, b = int(lo + starts[index]), int(lo + ends[index])
    a += int(a > 0)
    b -= int(b < pix.height)
    if b <= a:
        raise ValueError(f"White strip too thin: {(a, b)}")
    return a, b


def wrapped_title(text: str, font: pdf.Font, size: float, width: float) -> list[str]:
    """Wrap words without shrinking the shared heading font."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            if font.text_length(word, fontsize=size) > width:
                raise ValueError(f"Heading word does not fit: {word!r}")
            candidate = f"{current} {word}".strip()
            if current and font.text_length(candidate, fontsize=size) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    if not lines or len(lines) > 3:
        raise ValueError(f"Heading needs a wording revision: {text!r}")
    return lines


def update_metadata(item: dict, manifest: dict, path: Path, record: dict) -> None:
    """Refresh canonical render/hash fields without reinterpreting old data."""
    source = item.get("source_metadata")
    metadata = {}
    if source:
        raw = subprocess.check_output(
            ["git", "show", f"{manifest['source_commit']}:{source['path']}"], cwd=ROOT
        )
        if digest(raw) != source["sha256"]:
            raise ValueError(f"Source metadata hash mismatch: {source['path']}")
        metadata = json.loads(raw)
    status = source["pdf_link_status"] if source else "absent"
    metadata["inherited_scientific_metadata_status"] = status
    metadata["heading_migration"] = record
    metadata["reproduction_command"] = "uv run --extra viz python scripts/paper_figure_headings.py"
    hashes = {
        "pdf": digest(path.read_bytes()),
        "png": digest(path.with_suffix(".png").read_bytes()),
        "grayscale": digest(path.with_name(path.stem + "_grayscale.png").read_bytes()),
    }
    metadata["output_sha256"] = hashes
    if "outputs_sha256" in metadata:
        metadata["outputs_sha256"] = hashes
    with pdf.open(path) as current:
        strings = [line["text"] for line in text_lines(current[0])]
    size = [round(x / 72, 3) for x in record["new_size_pt"]]
    render = metadata.setdefault("render", {})
    render.update(
        exported_size_inches=size,
        authoring_size_inches=size,
        text=strings,
        heading_layout="single descriptive heading",
    )
    for key in ("exported_size_inches", "authoring_size_inches", "text"):
        if key in metadata:
            metadata[key] = render[key]
    for key in ("record", "save_record"):
        if isinstance(metadata.get(key), dict) and "exported_size_inches" in metadata[key]:
            metadata[key].update(render)
    if "figsize" in metadata:
        metadata["figsize"] = size
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def render_one(item: dict, manifest: dict, out_root: Path, font_path: str) -> dict:
    """Migrate one PDF and verify preservation outside its heading rectangles."""
    raw = subprocess.check_output(
        ["git", "show", f"{manifest['source_commit']}:{item['path']}"], cwd=ROOT
    )
    if digest(raw) != item["source_sha256"]:
        raise ValueError(f"Source hash mismatch: {item['path']}")
    if not item.get("rows"):
        return {"path": item["path"], "status": "already_single_heading"}
    original = pdf.open(stream=raw, filetype="pdf")
    if len(original) != 1:
        raise ValueError(f"Expected one PDF page: {item['path']}")
    doc = pdf.open(stream=raw, filetype="pdf")
    page = doc[0]
    lines = text_lines(page)
    removed_indices: set[int] = set()
    rows = []
    for row in item["rows"]:
        matches = []
        for old in row["remove"]:
            selected = [
                (i, line)
                for i, line in enumerate(lines)
                if line["text"] == old and i not in removed_indices
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one heading {old!r}, found {len(selected)}")
            i, line = selected[0]
            removed_indices.add(i)
            matches.append(line)
            page.add_redact_annot(pdf.Rect(line["bbox"]), fill=False)
        rows.append(
            {
                **row,
                "start": min(x["bbox"][1] for x in matches),
                "end": max(x["bbox"][3] for x in matches),
            }
        )
    page.apply_redactions(images=0, graphics=0, text=0)
    # Compare every surviving line, including coordinates. A rectangle clipping
    # any unrelated glyph is a failure, never a best-effort edit.
    expected = [line for i, line in enumerate(lines) if i not in removed_indices]
    actual = text_lines(page)
    if expected != actual:
        raise ValueError(f"Redaction touched text outside headings: {item['path']}")
    original_pix = original[0].get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False)
    clean_pix = page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False)
    original_pixels = np.frombuffer(original_pix.samples, np.uint8).reshape(
        original_pix.height, original_pix.width, 3
    )
    clean_pixels = np.frombuffer(clean_pix.samples, np.uint8).reshape(
        clean_pix.height, clean_pix.width, 3
    )
    outside = np.ones(original_pixels.shape[:2], dtype=bool)
    for i in removed_indices:
        x0, y0, x1, y1 = lines[i]["bbox"]
        outside[
            max(0, math.floor(y0 * 2) - 2) : math.ceil(y1 * 2) + 2,
            max(0, math.floor(x0 * 2) - 2) : math.ceil(x1 * 2) + 2,
        ] = False
    redaction_difference = float(np.any(original_pixels != clean_pixels, axis=2)[outside].mean())
    if redaction_difference > 0.0001:
        raise ValueError(f"Redaction changed nonheading pixels: {redaction_difference}")

    font = pdf.Font(fontfile=font_path)
    size = BASE_FONT_PT["title"]
    line_height = math.ceil(size * (font.ascender - font.descender))
    for row in rows:
        row["cut"] = longest_white_band(page, row["start"], row["end"])
        for heading in row["headings"]:
            heading["lines"] = wrapped_title(heading["text"], font, size, heading["width"])
        row["height"] = 8 + line_height * max(len(h["lines"]) for h in row["headings"])
    rows.sort(key=lambda x: x["cut"][0])
    if any(a["cut"][1] > b["cut"][0] for a, b in pairwise(rows)):
        raise ValueError("Heading bands overlap")
    width, height = page.rect.width, page.rect.height
    new_height = height + sum(row["height"] - (row["cut"][1] - row["cut"][0]) for row in rows)
    result = pdf.open()
    target = result.new_page(width=width, height=new_height)
    target.insert_font(fontname="C2AHeading", fontfile=font_path)
    source_y = target_y = 0.0
    strips = []
    for row in rows:
        a, b = row["cut"]
        if a > source_y:
            rect = pdf.Rect(0, target_y, width, target_y + a - source_y)
            target.show_pdf_page(rect, doc, 0, clip=pdf.Rect(0, source_y, width, a))
            strips.append((source_y, a, target_y))
            target_y += a - source_y
        row["new_y"] = target_y
        color = tuple(int(INK.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
        for heading in row["headings"]:
            for index, line in enumerate(heading["lines"]):
                target.insert_text(
                    (heading["x"], target_y + 4 + size * font.ascender + line_height * index),
                    line,
                    fontsize=size,
                    fontname="C2AHeading",
                    color=color,
                )
        target_y += row["height"]
        source_y = b
    if source_y < height:
        target.show_pdf_page(
            pdf.Rect(0, target_y, width, target_y + height - source_y),
            doc,
            0,
            clip=pdf.Rect(0, source_y, width, height),
        )
        strips.append((source_y, height, target_y))
    # Copying forms should preserve all surviving pixels at the same scale.
    # Cuts/translations are integer PDF points, aligning the 2x raster grids.
    old_pix = page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False)
    new_pix = target.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False)
    old_pixels = np.frombuffer(old_pix.samples, np.uint8).reshape(old_pix.height, old_pix.width, 3)
    new_pixels = np.frombuffer(new_pix.samples, np.uint8).reshape(new_pix.height, new_pix.width, 3)
    differences = []
    for a, b, destination in strips:
        n = min(
            int((b - a) * 2), old_pix.height - round(a * 2), new_pix.height - round(destination * 2)
        )
        # Skip a one-pixel clip-edge fringe; the cut is verified wholly white.
        old_strip = old_pixels[round(a * 2) + 1 : round(a * 2) + n - 1]
        new_strip = new_pixels[round(destination * 2) + 1 : round(destination * 2) + n - 1]
        differing = np.any(old_strip != new_strip, axis=2)
        differences.append(float(differing.mean()) if differing.size else 0.0)
    if max(differences) > 0.002:
        raise ValueError(f"Plot preservation raster mismatch {differences}: {item['path']}")
    destination = out_root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(original.metadata)
    metadata.update(creator="scripts/paper_figure_headings.py", modDate="", creationDate="")
    result.set_metadata(metadata)
    result.save(destination, garbage=4, deflate=True, no_new_id=True)
    pix = target.get_pixmap(matrix=pdf.Matrix(240 / 72, 240 / 72), alpha=False)
    pix.save(destination.with_suffix(".png"))
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L").save(
        destination.with_name(destination.stem + "_grayscale.png")
    )
    record = {
        "path": item["path"],
        "status": "updated",
        "source_commit": manifest["source_commit"],
        "source_sha256": item["source_sha256"],
        "overleaf_source_commit": manifest["overleaf_commit"],
        "old_size_pt": [width, height],
        "new_size_pt": [width, new_height],
        "heading_font_pt": size,
        "font_sha256": digest(Path(font_path).read_bytes()),
        "rows": rows,
        "nonheading_text_unchanged": True,
        "nonheading_redaction_pixel_difference_fraction": redaction_difference,
        "preserved_strip_pixel_difference_fractions": differences,
        "output_sha256": digest(destination.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "data_recomputed": False,
    }
    destination.with_suffix(".headings.json").write_text(json.dumps(record, indent=2) + "\n")
    update_metadata(item, manifest, destination, record)
    original.close()
    doc.close()
    result.close()
    return record


def main() -> None:
    """Render selected or all manifest figures and persist their audit records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument("--only", action="append", help="PDF stem; repeat to select several")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    family = set_c2a_style()
    font_path = findfont(FontProperties(family=family, weight="bold"), fallback_to_default=False)
    items = [x for x in manifest["figures"] if not args.only or Path(x["path"]).stem in args.only]
    if not items or (args.only and {Path(x["path"]).stem for x in items} != set(args.only)):
        raise ValueError("Requested figure selection is empty or incomplete")
    for item in items:
        result = render_one(item, manifest, args.out_root, font_path)
        print(item["path"], result["status"], flush=True)


if __name__ == "__main__":
    main()
