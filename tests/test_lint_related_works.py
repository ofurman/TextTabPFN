"""Regression tests for failures that can silently break the literature wiki."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.lint_related_works import SECTIONS, lint


def minimal_pdf() -> bytes:
    """A real one-page PDF with a valid xref; no PDF Python dependency needed."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> >>",
    ]
    data = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 4\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return data


@unittest.skipUnless(shutil.which("pdfinfo"), "Poppler pdfinfo is required")
class WikiLintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "docs/related_works"
        (self.folder / "raw").mkdir(parents=True)
        self.pdf = self.folder / "raw/paper.pdf"
        self.pdf.write_bytes(minimal_pdf())
        self.metadata = {
            "id": "paper", "title": "Example paper", "year_or_venue": "2026",
            "tags": ["row-level-text"], "architecture_tags": ["text-transformer-encoder", "tabular-transformer"],
            "brief": "A useful text model.",
            "source_url": "https://example.com/paper", "pdf": "raw/paper.pdf",
            "pdf_status": "available", "updated": "2026-10-15",
        }
        self.entry = {
            "file": "raw/paper.pdf", "note": "paper.md", "title": "Example paper",
            "year_or_venue": "2026", "source_url": "https://example.com/paper",
            "pdf_url": "https://example.com/paper.pdf", "arxiv_version": None,
            "retrieved_on": "2026-10-15", "pages": 1,
            "bytes": self.pdf.stat().st_size,
            "sha256": hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
        }
        self.body = "# Example paper\n\n[Catalogue](README.md) · [Source](https://example.com/paper) · [PDF](raw/paper.pdf)\n\n"
        for section in SECTIONS:
            self.body += f"## {section}\n\nA concrete explanation of {section.lower()}.\n\n"
            if section == "Architecture":
                self.body += "**Block roles:** Text encoder supplies representations to a separate tabular transformer.\n\n"
        self.write_note()
        self.write_catalogue()
        self.write_manifest()

    def write_note(self, name="paper.md", metadata=None, body=None):
        metadata = self.metadata if metadata is None else metadata
        body = self.body if body is None else body
        frontmatter = "\n".join(f"{key}: {json.dumps(value)}" for key, value in metadata.items())
        (self.folder / name).write_text(f"---\n{frontmatter}\n---\n\n{body}")

    def write_catalogue(self, metadata=None):
        metadata = self.metadata if metadata is None else metadata
        tags = ", ".join(f"`{tag}`" for tag in metadata["tags"])
        architecture_tags = ", ".join(f"`{tag}`" for tag in metadata["architecture_tags"])
        self.row = f"| [{metadata['title']}](paper.md) | {metadata['year_or_venue']} | {tags} | {architecture_tags} | {metadata['brief']} |\n"
        (self.folder / "README.md").write_text(
            "# Wiki\n\n## Catalogue\n\n| Paper | Year / venue | Text-source tags | Architecture blocks | Brief |\n|---|---|---|---|---|\n" + self.row
        )

    def write_manifest(self, entries=None):
        (self.folder / "manifest.json").write_text(json.dumps([self.entry] if entries is None else entries))

    def assert_failure(self, message):
        errors = lint(self.root)
        self.assertTrue(any(message in error for error in errors), errors)

    def test_valid_wiki_and_pdf(self):
        self.assertEqual(lint(self.root), [])

    def test_fenced_examples_are_not_headings_or_links(self):
        example = "```md\n# An example title\n## Architecture\n[Example](not-a-real-file.md)\n```\n\n"
        self.write_note(body=self.body.replace("## Input", example + "## Input"))
        self.assertEqual(lint(self.root), [])

    def test_missing_or_empty_required_section(self):
        for body in (self.body.replace("## Architecture", "## Other"),
                     self.body.replace("A concrete explanation of architecture.", "TODO")):
            with self.subTest(body=body):
                self.write_note(body=body)
                self.assert_failure("Architecture")

    def test_unknown_mixed_and_duplicate_tags(self):
        for tags in (["unknown"], ["no-text", "row-level-text"], ["row-level-text", "row-level-text"]):
            with self.subTest(tags=tags):
                self.metadata["tags"] = tags
                self.write_note()
                self.write_catalogue()
                self.assert_failure("invalid/duplicate tags")

    def test_catalogue_tag_mismatch(self):
        changed = copy.deepcopy(self.metadata)
        changed["tags"] = ["task-domain-context"]
        self.write_catalogue(changed)
        self.assert_failure("tags do not match")

    def test_invalid_architecture_tags(self):
        for tags in (["transformer"], ["text-transformer-encoder", "text-transformer-encoder"], [], [{}]):
            with self.subTest(tags=tags):
                self.metadata["architecture_tags"] = tags
                self.write_note()
                self.assert_failure("architecture_tags")

    def test_catalogue_architecture_tag_mismatch(self):
        changed = copy.deepcopy(self.metadata)
        changed["architecture_tags"] = ["text-transformer-encoder-decoder"]
        self.write_catalogue(changed)
        self.assert_failure("architecture_tags do not match")

    def test_tabular_attention_requires_tabular_transformer(self):
        self.metadata["architecture_tags"] = ["text-transformer-encoder", "tabular-row-attention"]
        self.write_note()
        self.write_catalogue()
        self.assert_failure("require tabular-transformer")

    def test_benchmark_cannot_tag_evaluated_agent_blocks(self):
        self.metadata["architecture_tags"] = ["benchmark-generator", "text-transformer-encoder"]
        self.write_note()
        self.write_catalogue()
        self.assert_failure("benchmark-generator must appear alone")

    def test_architecture_requires_block_roles(self):
        self.write_note(body=self.body.replace("**Block roles:**", "**Other:**"))
        self.assert_failure("Block roles")

    def test_text_transformer_alone_is_a_valid_architecture(self):
        self.metadata["architecture_tags"] = ["text-transformer-encoder-decoder"]
        self.write_note()
        self.write_catalogue()
        self.assertEqual(lint(self.root), [])

    def test_catalogue_brief_mismatch(self):
        changed = copy.deepcopy(self.metadata)
        changed["brief"] = "A stale description."
        self.write_catalogue(changed)
        self.assert_failure("title/year/brief do not match")

    def test_duplicate_or_missing_catalogue_row(self):
        path = self.folder / "README.md"
        original = path.read_text()
        for content in (original + self.row, original.replace(self.row, "")):
            with self.subTest(content=content):
                path.write_text(content)
                self.assert_failure("exactly one catalogue row")

    def test_missing_note(self):
        (self.folder / "paper.md").unlink()
        self.assert_failure("unknown paper note")

    def test_duplicate_source_and_required_cross_links(self):
        second = dict(self.metadata, id="second")
        self.write_note("second.md", second)
        self.assert_failure("duplicate work/source")
        self.assert_failure("Related works must link to another paper note")

    def test_broken_and_self_wiki_links(self):
        self.write_note(body=self.body + "- [Missing](missing.md)\n- [Self](paper.md)\n")
        self.assert_failure("Related works link must name another paper note")
        self.assert_failure("broken or out-of-repository local link")

    def test_broken_anchor_elsewhere_in_repository(self):
        (self.root / "README.md").write_text("[Broken](docs/related_works/paper.md#does-not-exist)\n")
        self.assert_failure("missing Markdown anchor")

    def test_checksum_and_size_mismatch(self):
        self.pdf.write_bytes(self.pdf.read_bytes() + b"\n% changed\n")
        self.assert_failure("SHA-256 differs")
        self.assert_failure("byte size differs")

    def test_page_count_mismatch(self):
        self.entry["pages"] = 2
        self.write_manifest()
        self.assert_failure("page count differs")

    def test_pdf_parsing_not_just_header(self):
        self.pdf.write_bytes(b"%PDF-1.4\nnot a real PDF\n")
        self.entry["bytes"] = self.pdf.stat().st_size
        self.entry["sha256"] = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.write_manifest()
        self.assert_failure("pdfinfo could not parse PDF")

    def test_orphan_pdf_and_partial_download(self):
        (self.folder / "raw/orphan.pdf").write_bytes(minimal_pdf())
        (self.folder / "raw/unfinished.pdf.part").write_bytes(b"partial")
        self.assert_failure("orphan PDF")
        self.assert_failure("raw/ must contain only downloaded PDFs")

    def test_missing_and_duplicate_manifest_entries(self):
        for entries in ([], [self.entry, self.entry]):
            with self.subTest(entries=entries):
                self.write_manifest(entries)
                self.assert_failure("expected 1 manifest entry")

    def test_manifest_source_mismatch(self):
        self.entry["source_url"] = "https://example.com/wrong"
        self.write_manifest()
        self.assert_failure("source_url differs from note")

    def test_pdf_must_be_in_raw(self):
        self.pdf.rename(self.folder / "paper.pdf")
        self.metadata["pdf"] = self.entry["file"] = "paper.pdf"
        self.write_note(body=self.body.replace("raw/paper.pdf", "paper.pdf"))
        self.write_manifest()
        self.assert_failure("PDF must exist directly under raw/")

    def test_reference_only_note_without_pdf(self):
        self.pdf.unlink()
        self.metadata.update(pdf=None, pdf_status="reference-only")
        body = self.body.replace(" · [PDF](raw/paper.pdf)", "")
        body += "## PDF availability\n\nReference-only entry; no PDF has been downloaded.\n"
        self.write_note(body=body)
        self.write_manifest([])
        self.assertEqual(lint(self.root), [])

    def test_missing_pdf_requires_explanation(self):
        self.metadata.update(pdf=None, pdf_status="unavailable")
        self.write_note()
        self.assert_failure("PDF availability")

    def test_missing_pdfinfo_is_an_error(self):
        with patch("scripts.lint_related_works.shutil.which", return_value=None):
            self.assert_failure("pdfinfo is required")

    def test_bad_metadata_reports_errors_instead_of_crashing(self):
        self.metadata.update(pdf_status=[], source_url="https://[broken", tags=[{}])
        self.entry["arxiv_version"] = {}
        self.write_note()
        self.write_manifest()
        self.assert_failure("pdf_status must be")
        self.assert_failure("source_url must be")
        self.assert_failure("invalid arxiv_version")

    def test_duplicate_frontmatter_keys(self):
        path = self.folder / "paper.md"
        path.write_text(path.read_text().replace('id: "paper"', 'id: "paper"\nid: "second"'))
        self.assert_failure("invalid or duplicate metadata key")


if __name__ == "__main__":
    unittest.main()
