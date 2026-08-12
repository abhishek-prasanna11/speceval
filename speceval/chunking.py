"""Split PEP bodies into retrievable chunks along reStructuredText section boundaries.

Chunking quality is deliberately *not* a variable in this study -- all four retrieval
strategies see identical chunks, so any difference between them is attributable to the
strategy. The goal here is only to be reasonable and stable, not optimal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Pep

# reStructuredText marks a section title by underlining it with a run of punctuation.
_ADORNMENT_CHARS = set("=-`:'\"~^_*+#<>")

# Longer chunks retrieve more context but blur the lexical signal; 1200 characters is
# roughly a screenful of prose and keeps most PEP sections whole.
MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pep_number: int
    pep_title: str
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        """What the retrievers actually see.

        The PEP title and section heading are prepended so a query matching a title
        can find the body under it -- without this, "Data Classes" would score poorly
        against a chunk that never repeats the phrase.
        """
        return f"{self.pep_title}\n{self.section}\n{self.text}"


def _is_adornment(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 3 and len(set(stripped)) == 1 and stripped[0] in _ADORNMENT_CHARS


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split an rST body into (section_title, section_text) pairs."""
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = [("", [])]

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        title = line.strip()

        # A title is a non-blank, non-adornment line underlined by an adornment run
        # at least as long as the title itself.
        is_title = (
            title
            and not _is_adornment(line)
            and _is_adornment(next_line)
            and len(next_line.strip()) >= len(title)
        )
        if is_title:
            sections.append((title, []))
            i += 2
            continue

        sections[-1][1].append(line)
        i += 1

    return [(title, "\n".join(body_lines).strip()) for title, body_lines in sections]


def _split_long(text: str, limit: int) -> list[str]:
    """Pack paragraphs into windows of at most ``limit`` characters."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        # A single oversized paragraph becomes its own chunk rather than being cut
        # mid-sentence; a handful of PEPs contain very long grammar blocks.
        if size and size + len(paragraph) > limit:
            windows.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)

    if current:
        windows.append("\n\n".join(current))
    return windows


def chunk_pep(pep: Pep, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_title, section_text in split_sections(pep.body):
        if not section_text:
            continue
        for part in _split_long(section_text, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"pep-{pep.number:04d}#{len(chunks)}",
                    pep_number=pep.number,
                    pep_title=pep.title,
                    section=section_title,
                    text=part,
                )
            )
    return chunks


def chunk_corpus(peps: list[Pep], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    return [chunk for pep in peps for chunk in chunk_pep(pep, max_chars)]
