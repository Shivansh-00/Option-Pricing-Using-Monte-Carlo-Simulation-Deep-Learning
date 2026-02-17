"""
Enterprise Chunking Engine
==========================
Multi-strategy document chunking with metadata extraction.
Supports: recursive, semantic, sliding-window, and markdown-aware chunking.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ── Data Models ───────────────────────────────────────────────────────────

class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    MARKDOWN = "markdown"


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk."""
    source_file: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    section_title: str = ""
    section_hierarchy: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    page_number: int | None = None
    char_offset_start: int = 0
    char_offset_end: int = 0
    word_count: int = 0
    has_code: bool = False
    has_formula: bool = False
    has_list: bool = False
    has_table: bool = False
    content_hash: str = ""
    strategy: str = "recursive"


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    metadata: ChunkMetadata


# ── Utility Functions ─────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _word_count(text: str) -> int:
    return len(text.split())


def _detect_features(text: str) -> dict[str, bool]:
    """Detect content features in a text block."""
    return {
        "has_code": bool(re.search(r"```[\s\S]*?```|`[^`]+`", text)),
        "has_formula": bool(re.search(
            r"\$\$.+?\$\$|\$[^$]+\$|\\frac|\\int|\\sum|[=×÷±∑∏∫]"
            r"|e\^|σ[²√]|d[₁₂]|N\(d|ln\(|√T",
            text,
        )),
        "has_list": bool(re.search(r"^[\s]*[-*•]\s|^\s*\d+\.\s", text, re.MULTILINE)),
        "has_table": bool(re.search(r"\|.*\|.*\|", text)),
    }


def _extract_headings(text: str) -> list[str]:
    """Extract all markdown headings from text."""
    return re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)


def _build_section_hierarchy(text: str) -> list[str]:
    """Build hierarchical section path from markdown headings."""
    hierarchy: list[str] = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        # Trim hierarchy to current level
        hierarchy = hierarchy[:level - 1]
        hierarchy.append(title)
    return hierarchy


# ── Splitting Helpers ─────────────────────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+|(?<=:)\s*\n"
)

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")

_MARKDOWN_SECTION = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving meaningful boundaries."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    parts = _PARAGRAPH_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_markdown_sections(text: str) -> list[str]:
    """Split markdown text at heading boundaries."""
    parts = _MARKDOWN_SECTION.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Chunking Strategies ──────────────────────────────────────────────────

def recursive_chunk(
    text: str,
    max_size: int = 600,
    overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursive character text splitting.
    Tries each separator in order of priority, falling back to smaller
    separators when chunks exceed max_size.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # If single part exceeds max_size, recurse with smaller seps
                if len(part) > max_size:
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if remaining_seps:
                        sub_chunks = recursive_chunk(
                            part, max_size, overlap, remaining_seps,
                        )
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        # Hard split
                        for i in range(0, len(part), max_size - overlap):
                            chunks.append(part[i:i + max_size])
                        current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return _add_overlap(chunks, overlap)

    # Fallback: hard character split
    return _sliding_window_split(text, max_size, overlap)


def semantic_chunk(
    text: str,
    max_size: int = 800,
    min_size: int = 100,
    overlap: int = 100,
) -> list[str]:
    """
    Semantic chunking: split at paragraph/section boundaries,
    then merge small adjacent chunks for coherence.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    # Step 1: Split into semantic units (paragraphs)
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return [text]

    # Step 2: Merge small paragraphs until they reach min_size
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            candidate = f"{buffer}\n\n{para}"
        else:
            candidate = para

        if len(candidate) <= max_size:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            # If paragraph itself exceeds max_size, use recursive split
            if len(para) > max_size:
                merged.extend(recursive_chunk(para, max_size, overlap))
                buffer = ""
            else:
                buffer = para

    if buffer:
        merged.append(buffer)

    # Step 3: Merge very small chunks with neighbors
    result: list[str] = []
    for chunk in merged:
        if result and len(chunk) < min_size and len(result[-1]) + len(chunk) + 2 <= max_size:
            result[-1] = f"{result[-1]}\n\n{chunk}"
        else:
            result.append(chunk)

    return _add_overlap(result, overlap) if len(result) > 1 else result


def markdown_chunk(
    text: str,
    max_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Markdown-aware chunking: respects heading hierarchy, preserves
    code blocks, lists, and tables as atomic units.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    # Step 1: Protect code blocks from splitting
    code_blocks: list[str] = []
    protected = text

    def _protect_code(m: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{idx}__"

    protected = re.sub(r"```[\s\S]*?```", _protect_code, protected)

    # Step 2: Split at section boundaries
    sections = _split_markdown_sections(protected)

    # Step 3: Process each section
    chunks: list[str] = []
    for section in sections:
        # Restore code blocks
        for idx, block in enumerate(code_blocks):
            section = section.replace(f"__CODE_BLOCK_{idx}__", block)

        if len(section) <= max_size:
            chunks.append(section)
        else:
            # Section too large — use semantic chunking within section
            # But preserve the heading as prefix for context
            heading_match = re.match(r"^(#{1,6}\s+.+)\n", section)
            heading_prefix = ""
            body = section
            if heading_match:
                heading_prefix = heading_match.group(1) + "\n"
                body = section[heading_match.end():]

            sub_chunks = semantic_chunk(body, max_size - len(heading_prefix), 80, overlap)
            for sc in sub_chunks:
                chunks.append(f"{heading_prefix}{sc}" if heading_prefix else sc)

    return _add_overlap(chunks, overlap) if len(chunks) > 1 else chunks


def sliding_window_chunk(
    text: str,
    window_size: int = 600,
    step_size: int = 480,
) -> list[str]:
    """Simple sliding window with configurable step size."""
    return _sliding_window_split(text, window_size, window_size - step_size)


def _sliding_window_split(text: str, size: int, overlap: int) -> list[str]:
    """Low-level sliding window."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap between consecutive chunks for context continuity."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else ""
        # Find a word boundary in the overlap region
        space_idx = prev_tail.find(" ")
        if space_idx > 0:
            prev_tail = prev_tail[space_idx + 1:]
        if prev_tail and not chunks[i].startswith(prev_tail):
            result.append(f"{prev_tail} {chunks[i]}")
        else:
            result.append(chunks[i])
    return result


# ── Main Chunking API ────────────────────────────────────────────────────

def chunk_document(
    text: str,
    source_file: str = "",
    strategy: ChunkStrategy = ChunkStrategy.MARKDOWN,
    max_size: int = 600,
    overlap: int = 120,
    page_number: int | None = None,
) -> list[Chunk]:
    """
    Chunk a document using the specified strategy, attaching rich metadata.

    Parameters
    ----------
    text : str
        Raw document text.
    source_file : str
        Name of the source file.
    strategy : ChunkStrategy
        Chunking strategy to use.
    max_size : int
        Maximum chunk size in characters.
    overlap : int
        Overlap between consecutive chunks.
    page_number : int | None
        Page number (for PDF sources).

    Returns
    -------
    list[Chunk]
        List of chunks with metadata.
    """
    if not text or not text.strip():
        return []

    headings = _extract_headings(text)
    hierarchy = _build_section_hierarchy(text)

    # Select strategy
    strategy_fn: Callable[..., list[str]]
    if strategy == ChunkStrategy.RECURSIVE:
        raw_chunks = recursive_chunk(text, max_size, overlap)
    elif strategy == ChunkStrategy.SEMANTIC:
        raw_chunks = semantic_chunk(text, max_size, max_size // 6, overlap)
    elif strategy == ChunkStrategy.SLIDING_WINDOW:
        raw_chunks = sliding_window_chunk(text, max_size, max_size - overlap)
    else:  # MARKDOWN (default)
        raw_chunks = markdown_chunk(text, max_size, overlap)

    # Build chunks with metadata
    total = len(raw_chunks)
    chunks: list[Chunk] = []
    char_offset = 0

    for idx, raw in enumerate(raw_chunks):
        features = _detect_features(raw)
        local_headings = _extract_headings(raw)

        # Determine section title from nearest heading
        section_title = local_headings[0] if local_headings else (
            headings[-1] if headings else ""
        )

        # Find approximate char offset
        found = text.find(raw[:50], char_offset)
        if found >= 0:
            char_offset = found

        meta = ChunkMetadata(
            source_file=source_file,
            chunk_index=idx,
            total_chunks=total,
            section_title=section_title,
            section_hierarchy=hierarchy.copy(),
            headings=local_headings or headings,
            page_number=page_number,
            char_offset_start=char_offset,
            char_offset_end=char_offset + len(raw),
            word_count=_word_count(raw),
            content_hash=_content_hash(raw),
            strategy=strategy.value,
            **features,
        )
        chunks.append(Chunk(text=raw, metadata=meta))

    return chunks
