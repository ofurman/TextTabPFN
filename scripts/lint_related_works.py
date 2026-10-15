#!/usr/bin/env python3
"""Validate the related-work wiki offline. Uses only Python's standard library."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

TAGS = {"row-level-text", "column-label-semantics", "task-domain-context", "no-text"}
ARCHITECTURE_TAGS = {
    "tabular-transformer", "text-transformer-encoder", "text-transformer-decoder", "text-transformer-encoder-decoder",
    "static-text-embeddings", "vision-encoder", "audio-encoder",
    "tabular-row-attention", "tabular-feature-attention", "tabular-inducing-attention",
    "tabular-row-pooling", "modality-projector", "gated-mlp", "cross-attention-pooler",
    "graph-attention", "target-tokens", "class-token-logit-filter", "retrieval-head", "pca", "tree-predictor",
    "benchmark-generator",
}
SECTIONS = (
    "Text-source scope", "Problem and positioning", "Architecture", "Input",
    "Output", "For us", "Related works",
)
METADATA = {
    "id", "title", "year_or_venue", "tags", "architecture_tags", "brief", "source_url",
    "pdf", "pdf_status", "updated",
}
MANIFEST_FIELDS = {
    "file", "note", "title", "year_or_venue", "source_url", "pdf_url",
    "arxiv_version", "retrieved_on", "pages", "bytes", "sha256",
}
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)\)")
PLACEHOLDER = re.compile(r"(?:TODO|TBD|N/A|\.\.\.|<[^>]+>)\.?", re.I)


def prose(text: str) -> str:
    """Remove frontmatter and fenced examples before inspecting links/headings."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)
    return re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*(?:\n|$)", "", text, flags=re.M | re.S)


def links(text: str) -> list[str]:
    return [match.strip("<>") for match in LINK.findall(prose(text))]


def anchors(text: str) -> set[str]:
    counts: Counter[str] = Counter()
    result = set()
    for heading in re.findall(r"^#{1,6} +(.+?)\s*#*\s*$", prose(text), re.M):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        result.add(slug if counts[slug] == 0 else f"{slug}-{counts[slug]}")
        counts[slug] += 1
    return result


def valid_url(value: object) -> bool:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc)


def valid_date(value: object) -> bool:
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def meaningful(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not PLACEHOLDER.fullmatch(value.strip())


def local_path(base: Path, value: object, root: Path) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (base / value).resolve()
    return path if path.is_relative_to(root) else None


def parse_note(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    if not match:
        raise ValueError("requires --- frontmatter with one key: JSON-value per line")
    metadata = {}
    for line in match[1].splitlines():
        key, separator, value = line.partition(":")
        if not separator or not re.fullmatch(r"[a-z_]+", key) or key in metadata:
            raise ValueError(f"invalid or duplicate metadata key: {line!r}")
        try:
            metadata[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key}: use a JSON string, array, or null") from exc
    return metadata, match[2]


def markdown_files(root: Path):
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    # Prune dependency directories rather than traversing them.
    import os
    for directory, children, files in os.walk(root):
        children[:] = sorted(name for name in children if name not in ignored)
        for name in sorted(files):
            if name.endswith(".md"):
                yield Path(directory) / name


def lint(root: Path) -> list[str]:
    root = root.resolve()
    folder = root / "docs/related_works"
    errors: list[str] = []

    def fail(path: Path, message: str):
        label = path.relative_to(root) if path.is_relative_to(root) else path
        errors.append(f"{label}: {message}")

    catalogue = folder / "README.md"
    manifest_path = folder / "manifest.json"
    if not catalogue.is_file() or not manifest_path.is_file():
        return ["docs/related_works: README.md and manifest.json are required"]

    notes = {}
    for path in sorted(folder.rglob("*.md")):
        if path == catalogue:
            continue
        if path.parent != folder:
            fail(path, "paper notes must live directly in docs/related_works")
        try:
            metadata, body = parse_note(path)
        except (ValueError, OSError) as exc:
            fail(path, str(exc))
            continue
        notes[path.name] = (metadata, body, path)
        checked_body = prose(body)
        if set(metadata) != METADATA:
            fail(path, f"metadata keys differ: missing {sorted(METADATA - set(metadata))}; unexpected {sorted(set(metadata) - METADATA)}")
        if metadata.get("id") != path.stem or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", path.stem):
            fail(path, "id must match the lowercase hyphenated filename stem")
        for key in ("title", "year_or_venue", "brief"):
            if not meaningful(metadata.get(key)):
                fail(path, f"{key} must contain meaningful text")
        if re.findall(r"^# (.+)$", checked_body, re.M) != [metadata.get("title")]:
            fail(path, "requires exactly one H1 matching title")
        tags = metadata.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
            fail(path, "tags must be a nonempty string array")
        elif len(set(tags)) != len(tags) or not set(tags) <= TAGS or ("no-text" in tags and len(tags) != 1):
            fail(path, "invalid/duplicate tags; no-text must appear alone")
        architecture_tags = metadata.get("architecture_tags")
        if not isinstance(architecture_tags, list) or not architecture_tags or any(not isinstance(tag, str) for tag in architecture_tags):
            fail(path, "architecture_tags must be a nonempty string array")
        elif len(set(architecture_tags)) != len(architecture_tags) or not set(architecture_tags) <= ARCHITECTURE_TAGS:
            fail(path, "invalid/duplicate architecture_tags")
        else:
            if "benchmark-generator" in architecture_tags and len(architecture_tags) != 1:
                fail(path, "benchmark-generator must appear alone; do not tag evaluated agents as the proposed architecture")
            if any(tag.startswith("tabular-") and tag != "tabular-transformer" for tag in architecture_tags) and "tabular-transformer" not in architecture_tags:
                fail(path, "tabular attention/pooling tags require tabular-transformer")
        if not valid_url(metadata.get("source_url")):
            fail(path, "source_url must be an HTTP(S) primary-source URL")
        elif metadata["source_url"] not in links(body):
            fail(path, "body must link to source_url")
        if not valid_date(metadata.get("updated")):
            fail(path, "updated must be an ISO YYYY-MM-DD date")
        sections = re.findall(r"^## (.+)$", checked_body, re.M)
        for section in SECTIONS:
            if sections.count(section) != 1:
                fail(path, f"requires exactly one '{section}' section")
                continue
            content = re.search(r"^## " + re.escape(section) + r"\n(.*?)(?=^## |\Z)", checked_body, re.M | re.S)
            if not content or not meaningful(content[1]) or any(
                PLACEHOLDER.fullmatch(paragraph.strip())
                for paragraph in re.split(r"\n\s*\n", content[1]) if paragraph.strip()
            ):
                fail(path, f"'{section}' must not be empty or a placeholder")
            if section == "Architecture" and content:
                roles = re.findall(r"^\*\*Block roles:\*\* (.+)$", content[1], re.M)
                if len(roles) != 1 or not meaningful(roles[0]):
                    fail(path, "Architecture requires one meaningful **Block roles:** explanation")
        if "README.md" not in links(body):
            fail(path, "body must link back to README.md")
        pdf = metadata.get("pdf")
        if metadata.get("pdf_status") == "available":
            target = local_path(folder, pdf, root)
            if not target or target.parent != folder / "raw" or target.suffix != ".pdf" or not target.is_file():
                fail(path, "available PDF must exist directly under raw/")
            if pdf not in links(body):
                fail(path, "body must link to its PDF")
        elif metadata.get("pdf_status") in ("unavailable", "reference-only"):
            if pdf is not None:
                fail(path, "unavailable/reference-only PDF must be null")
            availability = re.search(r"^## PDF availability\n(.*?)(?=^## |\Z)", checked_body, re.M | re.S)
            if not availability or not meaningful(availability[1]):
                fail(path, "missing PDF requires a 'PDF availability' explanation")
        else:
            fail(path, "pdf_status must be available, unavailable, or reference-only")

    if not notes:
        fail(catalogue, "catalogue must contain at least one paper note")
    seen_sources: dict[str, Path] = {}
    for metadata, body, path in notes.values():
        source = metadata.get("source_url")
        if isinstance(source, str):
            canonical = re.sub(r"(arxiv.org/abs/\d+\.\d+)v\d+$", r"\1", source).rstrip("/")
            if canonical in seen_sources:
                fail(path, f"duplicate work/source also used by {seen_sources[canonical].name}")
            seen_sources[canonical] = path
        related = re.search(r"^## Related works\n(.*?)(?=^## |\Z)", prose(body), re.M | re.S)
        targets = links(related[1]) if related else []
        if len(notes) > 1 and not any(target in notes and target != path.name for target in targets):
            fail(path, "Related works must link to another paper note")
        for target in targets:
            if target not in notes or target == path.name:
                fail(path, f"Related works link must name another paper note: {target}")

    contents = catalogue.read_text(encoding="utf-8")
    section = re.search(r"^## Catalogue\n(.*?)(?=^## |\Z)", contents, re.M | re.S)
    rows: Counter[str] = Counter()
    if not section:
        fail(catalogue, "missing Catalogue section")
    else:
        lines = [line for line in section[1].splitlines() if line.startswith("|")]
        if lines[:2] != ["| Paper | Year / venue | Text-source tags | Architecture blocks | Brief |", "|---|---|---|---|---|"]:
            fail(catalogue, "Catalogue table must use Paper, Year / venue, Text-source tags, Architecture blocks, Brief columns")
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            match = re.fullmatch(r"\[([^]]+)\]\(([^)]+)\)", cells[0]) if cells else None
            if len(cells) != 5 or not match:
                fail(catalogue, f"malformed catalogue row: {line}")
                continue
            title, name = match.groups()
            rows[name] += 1
            if name not in notes:
                fail(catalogue, f"unknown paper note: {name}")
                continue
            metadata = notes[name][0]
            expected = [metadata.get("title"), metadata.get("year_or_venue"), metadata.get("brief")]
            if [title, cells[1], cells[4]] != expected:
                fail(catalogue, f"title/year/brief do not match {name}")
            tags = re.findall(r"`([^`]+)`", cells[2])
            if tags != metadata.get("tags") or cells[2] != ", ".join(f"`{tag}`" for tag in tags):
                fail(catalogue, f"tags do not match {name}")
            architecture_tags = re.findall(r"`([^`]+)`", cells[3])
            if architecture_tags != metadata.get("architecture_tags") or cells[3] != ", ".join(f"`{tag}`" for tag in architecture_tags):
                fail(catalogue, f"architecture_tags do not match {name}")
    for name in notes:
        if rows[name] != 1:
            fail(catalogue, f"{name} must have exactly one catalogue row (found {rows[name]})")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        fail(manifest_path, f"cannot read JSON: {exc}")
        manifest = []
    if not isinstance(manifest, list):
        fail(manifest_path, "must be a JSON array")
        manifest = []
    files: Counter[str] = Counter()
    note_files: Counter[str] = Counter()
    pdfinfo = shutil.which("pdfinfo")
    if manifest and not pdfinfo:
        fail(manifest_path, "pdfinfo is required; install Poppler and add pdfinfo to PATH")
    for entry in manifest:
        if not isinstance(entry, dict):
            fail(manifest_path, "each entry must be an object")
            continue
        if set(entry) != MANIFEST_FIELDS:
            fail(manifest_path, "entry keys must match the documented manifest schema")
        name, note = entry.get("file"), entry.get("note")
        if not isinstance(name, str) or not isinstance(note, str):
            fail(manifest_path, "file and note must be strings")
            continue
        files[name] += 1
        note_files[note] += 1
        if note not in notes:
            fail(manifest_path, f"unknown note: {note}")
        else:
            metadata = notes[note][0]
            for field, value in (("pdf", name), ("title", entry.get("title")), ("year_or_venue", entry.get("year_or_venue")), ("source_url", entry.get("source_url")), ("pdf_status", "available")):
                if metadata.get(field) != value:
                    fail(manifest_path, f"{note}: {field} differs from note")
        if not valid_date(entry.get("retrieved_on")):
            fail(manifest_path, f"{name}: invalid retrieved_on date")
        for key in ("source_url", "pdf_url"):
            if not valid_url(entry.get(key)):
                fail(manifest_path, f"{name}: invalid {key}")
        version = entry.get("arxiv_version")
        if version is not None and (not isinstance(version, str) or not re.fullmatch(r"v[1-9]\d*", version)):
            fail(manifest_path, f"{name}: invalid arxiv_version")
        if isinstance(version, str) and version and isinstance(entry.get("pdf_url"), str) and "arxiv.org/pdf/" in entry["pdf_url"] and not entry["pdf_url"].endswith(version):
            fail(manifest_path, f"{name}: PDF URL does not pin arxiv_version")
        for key in ("pages", "bytes"):
            if type(entry.get(key)) is not int or entry[key] <= 0:
                fail(manifest_path, f"{name}: {key} must be a positive integer")
        pdf = local_path(folder, name, root)
        if not pdf or pdf.parent != folder / "raw" or pdf.suffix != ".pdf" or not pdf.is_file():
            fail(manifest_path, f"{name}: PDF must exist directly under raw/")
            continue
        data = pdf.read_bytes()
        if not data.startswith(b"%PDF-"):
            fail(pdf, "invalid PDF header")
        if len(data) != entry.get("bytes"):
            fail(pdf, "byte size differs from manifest")
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            fail(pdf, "SHA-256 differs from manifest")
        if pdfinfo:
            try:
                result = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True, timeout=30)
                pages = re.search(r"^Pages:\s+(\d+)", result.stdout, re.M)
                if result.returncode or not pages:
                    fail(pdf, "pdfinfo could not parse PDF")
                elif int(pages[1]) != entry.get("pages"):
                    fail(pdf, "page count differs from manifest")
            except (OSError, subprocess.TimeoutExpired) as exc:
                fail(pdf, f"pdfinfo failed: {exc}")
    for name, count in files.items():
        if count != 1:
            fail(manifest_path, f"duplicate PDF entry: {name}")
    for name, (metadata, _, _) in notes.items():
        expected = 1 if metadata.get("pdf_status") == "available" else 0
        if note_files[name] != expected:
            fail(manifest_path, f"{name}: expected {expected} manifest entry, found {note_files[name]}")
    actual_pdfs = {path.relative_to(folder).as_posix() for path in folder.rglob("*.pdf")}
    for orphan in sorted(actual_pdfs - set(files)):
        fail(folder / orphan, "orphan PDF; missing manifest entry")
    for path in (folder / "raw").rglob("*"):
        if path.is_file() and path.suffix != ".pdf":
            fail(path, "raw/ must contain only downloaded PDFs")

    # Validate local inline Markdown links across the repository after migration.
    for document in markdown_files(root):
        for target in links(document.read_text(encoding="utf-8")):
            try:
                parsed = urlsplit(target)
            except ValueError:
                fail(document, f"invalid link URL: {target}")
                continue
            if parsed.scheme or parsed.netloc:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                fail(document, f"broken or out-of-repository local link: {target}")
            elif parsed.fragment and destination.suffix == ".md":
                if unquote(parsed.fragment) not in anchors(destination.read_text(encoding="utf-8")):
                    fail(document, f"missing Markdown anchor: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root (defaults to this script's repository)")
    args = parser.parse_args()
    errors = lint(args.root)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        print(f"Related-work lint failed: {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Related-work lint passed: notes, catalogue, tags, links, and PDFs are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
